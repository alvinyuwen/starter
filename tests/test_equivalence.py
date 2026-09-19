"""CPU equivalence check for the optimized engine.

There is no GPU here, so the only way to catch a logic bug before spending a
real run is to compare against the reference on a small random model: same
architecture, same code path, tiny dimensions.

This proves structure - RoPE positions, per-head q/k norm, cache indexing,
chunk masking, cross-call cache reset, GQA mapping. It cannot prove BF16 cast
placement, which is argued from the reference source instead.

Run:  .venv/Scripts/python.exe tests/test_equivalence.py
"""

import sys
import pathlib

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import engine as engine_mod  # noqa: E402

# No CUDA on this machine; the engine reads its device from the module.
engine_mod.DEVICE = "cpu"
engine_mod.LOAD_DTYPE = torch.float32

DTYPE = torch.float32
torch.manual_seed(0)


def build_model(tmp, layers=3, hidden=320, heads=8, kv_heads=2, head_dim=128, vocab=256):
    """A miniature Qwen3 that keeps the parts of the real geometry that the
    engine is sensitive to.

    head_dim is 128 as in the real checkpoint, because RoPE's frequency table
    and the per-head norm both depend on it. The group ratio is 4 query heads
    per KV head, as in the real 32/8, so the GQA mapping is exercised rather
    than trivially satisfied. hidden is deliberately not heads*head_dim, which
    is also true of the real model (2560 vs 4096) and is the case a reshape
    bug would sail through.
    """
    config = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        max_position_embeddings=2048,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation="sdpa",
        # The real checkpoint has no sliding window; without this the config
        # default turns one on and the comparison stops being meaningful.
        use_sliding_window=False,
        sliding_window=None,
    )
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(DTYPE).eval()
    # Random weights that are not all near zero, so differences show up.
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.05)
    model.save_pretrained(tmp)
    return config


@torch.inference_mode()
def reference_generate(model, input_ids, max_new_tokens):
    """The starter's baseline loop, verbatim in behaviour."""
    current = torch.tensor(input_ids, dtype=torch.int64)
    cache = None
    out = []
    for _ in range(max_new_tokens):
        result = model(
            input_ids=current, past_key_values=cache, use_cache=True,
            logits_to_keep=1, return_dict=True,
        )
        current = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cache = result.past_key_values
        out.append(current[:, 0].tolist())
    return out


def run_case(path, name, batch, prompt_len, new_tokens, vocab, chunk=None):
    original = engine_mod.PREFILL_CHUNK_TOKENS
    if chunk is not None:
        engine_mod.PREFILL_CHUNK_TOKENS = chunk

    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True
    ).eval()
    eng = engine_mod.Engine(str(path))

    torch.manual_seed(batch * 1000 + prompt_len)
    ids = torch.randint(0, vocab, (batch, prompt_len)).tolist()

    want = reference_generate(reference, ids, new_tokens)
    got = list(eng.generate(ids, new_tokens))

    engine_mod.PREFILL_CHUNK_TOKENS = original

    ok = want == got
    detail = ""
    if not ok:
        for i, (a, b) in enumerate(zip(want, got)):
            if a != b:
                detail = f" first mismatch at step {i}: want {a} got {b}"
                break
    steps_ok = len(got) == new_tokens
    print(f"{'PASS' if ok and steps_ok else 'FAIL'}  {name}"
          f" (b={batch} prompt={prompt_len} new={new_tokens} steps={len(got)}){detail}")
    return ok and steps_ok


def run_reset_case(path, vocab):
    """Two consecutive generate calls with different prompts, same shape.

    This is the one that catches a cache that is not reset between calls: the
    second call must not see any of the first call's keys.
    """
    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True
    ).eval()
    eng = engine_mod.Engine(str(path))

    torch.manual_seed(7)
    first = torch.randint(0, vocab, (2, 24)).tolist()
    second = torch.randint(0, vocab, (2, 24)).tolist()

    list(eng.generate(first, 5))  # warms and fills the cache
    want = reference_generate(reference, second, 5)
    got = list(eng.generate(second, 5))

    ok = want == got
    print(f"{'PASS' if ok else 'FAIL'}  cache reset across calls"
          f"{'' if ok else f' want {want[:2]} got {got[:2]}'}")
    return ok


def run_logit_delta(path, vocab):
    """How much of the 2.0-logit tie margin does this engine actually spend?

    Argmax agreement alone can hide an engine sitting just inside the margin on
    every token. Comparing raw logits says how much headroom is left before a
    reordering starts flipping genuine near-ties.
    """
    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True
    ).eval()
    eng = engine_mod.Engine(str(path))

    torch.manual_seed(11)
    ids = torch.randint(0, vocab, (2, 40)).tolist()
    tensor = torch.tensor(ids, dtype=torch.int64)

    eng._ensure_shape(2, 40, 4)
    with torch.inference_mode():
        mine = eng._forward_prefill(tensor, 40)
        want = reference(input_ids=tensor, use_cache=True, logits_to_keep=1, return_dict=True).logits

    delta = (mine.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    ok = delta < 0.05 * max(scale, 1.0)
    print(f"{'PASS' if ok else 'FAIL'}  prefill logit delta {delta:.2e} "
          f"(logit magnitude ~{scale:.2f}; tie margin is 2.0)")
    return ok


def run_shape_reuse(path, vocab):
    """Same capacity, different prompt length.

    _ensure_shape keys on (batch, capacity), so this reuses buffers built for a
    different split of prompt and output. The position bookkeeping has to come
    from seq_len on each call, not from whatever built the cache.
    """
    reference = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, attn_implementation="sdpa", local_files_only=True
    ).eval()
    eng = engine_mod.Engine(str(path))

    torch.manual_seed(23)
    first = torch.randint(0, vocab, (2, 30)).tolist()
    second = torch.randint(0, vocab, (2, 26)).tolist()

    list(eng.generate(first, 6))          # capacity 36
    want = reference_generate(reference, second, 10)
    got = list(eng.generate(second, 10))  # capacity 36 again, different split

    ok = want == got
    print(f"{'PASS' if ok else 'FAIL'}  shape reuse at equal capacity"
          f"{'' if ok else f' want {want[:2]} got {got[:2]}'}")
    return ok


def main():
    import tempfile

    vocab = 256
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mini"
        build_model(path, vocab=vocab)

        results = [
            run_case(path, "batch 1, single-chunk prefill", 1, 16, 6, vocab),
            run_case(path, "batch 4, GQA broadcast", 4, 20, 5, vocab),
            run_case(path, "batch 3, longer prompt", 3, 64, 8, vocab),
            run_case(path, "single output token", 2, 12, 1, vocab),
            # Force multi-chunk prefill: chunk tokens below batch*prompt.
            run_case(path, "chunked prefill (mask path)", 2, 48, 6, vocab, chunk=32),
            run_case(path, "chunked prefill, uneven tail", 3, 50, 4, vocab, chunk=36),
            run_case(path, "long decode run (drift)", 2, 32, 24, vocab),
            run_reset_case(path, vocab),
            run_shape_reuse(path, vocab),
            run_logit_delta(path, vocab),
        ]

    print()
    if all(results):
        print(f"ALL {len(results)} EQUIVALENCE CHECKS PASSED")
        return 0
    print(f"{sum(1 for r in results if not r)}/{len(results)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
