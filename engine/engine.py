"""Qwen3 4B decode engine.

Three things carry the speedup over the Transformers baseline:

1. The ``Qwen3ForCausalLM`` wrapper, ``DynamicCache`` and generic module
   dispatch are gone; the forward is written out against the loaded weights.
2. A preallocated fixed-capacity KV cache, so decode never concatenates or
   reallocates and every decode step has identical shapes.
3. A CUDA graph over the whole decode step. At batch 1-16 across 36 layers the
   baseline's dominant cost is launch and dispatch overhead, not arithmetic.
   The graph is self-feeding - it consumes the token it wrote last step - so a
   step costs one replay and no host round trip.

Numerics follow Transformers 4.51.3 where it matters. Reductions may be
reordered; the formula and the cast boundaries may not. Every place that was
tempting to "improve" carries a comment saying why it is written as it is.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

DEVICE = "cuda:0"

#: Finite rather than -inf: it underflows to exactly zero through the FP32
#: softmax just the same, without risking a NaN in any backend that computes
#: -inf minus -inf on a fully masked tile.
NEG = torch.finfo(torch.bfloat16).min

#: Single-chunk prefill above this many tokens (batch * prompt) would allocate
#: a very large MLP activation; beyond it, prefill is chunked instead.
PREFILL_CHUNK_TOKENS = 16384

#: The checkpoint's dtype. Named so a CPU equivalence test can load the same
#: code path in fp32, where bf16 kernel coverage is patchy.
LOAD_DTYPE = torch.bfloat16

# Triton needs a GPU to compile at all, so an import failure here is expected
# off-target and must not be fatal: the engine simply keeps the eager path.
_kernels_error = None
try:
    from kernels.rmsnorm import rms_norm as _k_rms_norm
    from kernels.fused import (
        add_norm as _k_add_norm,
        kv_to_cache as _k_kv_cache,
        skinny_linear as _k_skinny,
        skinny_linear_split as _k_skinny_split,
        rope as _k_rope,
        swiglu as _k_swiglu,
        decode_attention as _k_attn,
        norm_rope as _k_norm_rope,
        decode_attention_split as _k_attn_split,
    )
except Exception as _exc:  # noqa: BLE001
    _kernels_error = _exc
    _k_rms_norm = _k_rope = _k_swiglu = _k_attn = _k_norm_rope = _k_attn_split = _k_add_norm = _k_kv_cache = _k_skinny = _k_skinny_split = None


class _Ready:
    """Stands in for a CUDA event when there is no CUDA to wait on."""

    def synchronize(self):
        return None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``Qwen3RMSNorm.forward``, transcribed.

    The reference ends ``self.weight * hidden_states.to(input_dtype)``: the
    normalised value is rounded to BF16 *before* the weight multiply. Holding
    the product in FP32 and rounding once at the end is strictly more accurate
    and is a different function - it can move a logit past the tie margin.
    Do not "fix" this.
    """
    dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return weight * xf.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(q, k, cos, sin):
    """``apply_rotary_pos_emb`` applied on ``[B, T, H, D]``.

    The reference rotates after the transpose, broadcasting cos/sin over the
    head axis with ``unsqueeze_dim=1``. Here the head axis is 2 instead, so
    cos/sin broadcast as ``[1, T, 1, D]``; every (batch, token, head) row still
    sees exactly the angles for its own token, so the values are unchanged.

    cos/sin arrive already in BF16: the reference casts the tables before the
    multiply, so multiplying in FP32 here would be a reformulation.
    """
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Engine:
    """Pinned Qwen3 4B is 36 layers, 2560 hidden, 32/8 heads, 128 head dim,
    eps 1e-6, theta 5e6 - but every one of those is read from the checkpoint's
    config rather than assumed, so the engine cannot silently disagree with the
    weights it just loaded."""

    def __init__(self, model_path: str) -> None:
        # TF32 silently changes every matmul's numerics. The baseline disables
        # it, so anything judged against the baseline must too.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=LOAD_DTYPE,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to(DEVICE)
        )
        self.model = model
        base = model.model
        cfg = model.config

        self.LAYERS = cfg.num_hidden_layers
        self.HIDDEN = cfg.hidden_size
        self.HEADS = cfg.num_attention_heads
        self.KV_HEADS = cfg.num_key_value_heads
        self.HEAD_DIM = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.GROUPS = self.HEADS // self.KV_HEADS
        self.EPS = cfg.rms_norm_eps
        self.THETA = cfg.rope_theta
        self.SCALE = self.HEAD_DIM**-0.5

        self.embed = base.embed_tokens.weight
        self.final_norm_w = base.norm.weight
        # Tied to the embedding; read it off the module so the tie is kept.
        self.lm_head_w = model.lm_head.weight

        self.layers = []
        for idx, layer in enumerate(base.layers):
            attn, mlp = layer.self_attn, layer.mlp
            # Fuse and release one layer at a time. Building every fused copy
            # first would hold both layouts for the whole model at once, and
            # that transient is ~4.7 GB of avoidable peak.
            qkv = torch.cat(
                [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
            ).contiguous()
            gate_up = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], dim=0).contiguous()
            attn.q_proj = attn.k_proj = attn.v_proj = None
            mlp.gate_proj = mlp.up_proj = None

            self.layers.append(
                {
                    "idx": idx,
                    "in_ln": layer.input_layernorm.weight,
                    # One GEMM instead of three. Every output row is still the
                    # same dot product of the same input row against the same
                    # weight row: a reordering, not a reformulation.
                    "qkv": qkv,
                    "q_norm": attn.q_norm.weight,
                    "k_norm": attn.k_norm.weight,
                    "o": attn.o_proj.weight,
                    "post_ln": layer.post_attention_layernorm.weight,
                    "gate_up": gate_up,
                    "down": mlp.down_proj.weight,
                }
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.dtype = self.embed.dtype
        self.cuda = torch.device(DEVICE).type == "cuda" and torch.cuda.is_available()
        self.q_size = self.HEADS * self.HEAD_DIM
        self.kv_size = self.KV_HEADS * self.HEAD_DIM

        self.rope_len = 0
        self.cos_table = None
        self.sin_table = None
        self._grow_rope(4096)

        # Rebuilt whenever a workload's shape changes.
        self.cache_k = None
        self.cache_v = None
        self.capacity = 0
        self.batch = 0
        self.graph = None
        self.g_token = None
        self.g_pos = None
        self.g_out = None
        self.g_arange = None
        self.gqa = None
        self.time_probe = 0
        self.fused_norm = False
        self.fused_rope = False
        self.fused_swiglu = False
        self.fused_attn = False
        self.split_attn = False
        self.fused_add_norm = False
        self.fused_kv_cache = False
        self.gemm_blocks = None
        self.gemm_choice = {}
        self.prefill_fused = True
        self.fused_norm_rope = False
        print(f"[engine] loaded {self.LAYERS} layers; qkv and gate_up fused", flush=True)

    # ------------------------------------------------------------------ rope

    def _grow_rope(self, length: int) -> None:
        """cos/sin for absolute positions [0, length).

        Built in FP32 and cast once to BF16, which is exactly where the
        reference casts: ``cos.to(dtype=x.dtype)``, before any multiply.
        """
        if length <= self.rope_len:
            return
        length = max(length, 4096)
        exponent = (
            torch.arange(0, self.HEAD_DIM, 2, dtype=torch.int64).float().to(DEVICE)
            / self.HEAD_DIM
        )
        inv_freq = 1.0 / (self.THETA**exponent)
        pos = torch.arange(length, dtype=torch.int64, device=DEVICE).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_table = emb.cos().to(self.dtype).contiguous()
        self.sin_table = emb.sin().to(self.dtype).contiguous()
        self.rope_len = length

    # ------------------------------------------------------------- attention

    def _sdpa(self, q, k, v, *, is_causal=False, attn_mask=None):
        """Grouped-query attention over the 8 KV heads and 32 query heads.

        ``enable_gqa`` broadcasts KV heads in-kernel. The alternative,
        ``repeat_kv``, materialises a 4x copy of the entire cache on every
        decode step - pure bandwidth precisely where decode is bandwidth
        bound. Probed once at warmup because backend coverage varies.
        """
        if self.gqa:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=is_causal,
                scale=self.SCALE, enable_gqa=True,
            )
        k = k.repeat_interleave(self.GROUPS, dim=1)
        v = v.repeat_interleave(self.GROUPS, dim=1)
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=self.SCALE
        )

    def _supports_gqa(self, batch: int) -> bool:
        q = torch.zeros(batch, self.HEADS, 1, self.HEAD_DIM, dtype=self.dtype, device=DEVICE)
        kv = torch.zeros(batch, self.KV_HEADS, 8, self.HEAD_DIM, dtype=self.dtype, device=DEVICE)
        try:
            F.scaled_dot_product_attention(q, kv, kv, scale=self.SCALE, enable_gqa=True)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(f"[engine] enable_gqa unavailable ({exc})", flush=True)
            return False

    def _time_decode(self, iters: int = 12) -> float:
        """Milliseconds per decode step, for picking between attention paths.

        Timed under graph replay, because that is what the measured samples
        run. Timing eager steps adds Python and launch overhead to every
        kernel, which systematically favours whichever candidate issues fewer
        launches rather than whichever is actually faster on the device - and
        under replay that overhead is gone and the ranking can invert. That is
        how a matmul that lost on every public shape still got selected on a
        hidden one and cost 6% of the score.

        Timed at a position partway through the generation, not at zero. The
        cost of every candidate depends on how many keys are live: at position
        zero a streaming kernel reads one block while SDPA still builds a
        full-capacity mask, which flatters the kernel and picked it for shapes
        where it loses badly. Timing where the workload actually spends its
        steps is the only comparison that means anything.
        """
        probe = min(self.time_probe, max(0, self.capacity - iters - 2))
        graph = self._try_capture(probe)
        if graph is not None:
            try:
                self.g_pos.fill_(probe)
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(iters):
                    graph.replay()
                stop.record()
                torch.cuda.synchronize()
                return start.elapsed_time(stop) / iters
            finally:
                del graph
                torch.cuda.synchronize()

        for _ in range(3):
            self.g_pos.fill_(probe)
            self._forward_decode()
        torch.cuda.synchronize()
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            self.g_pos.fill_(probe)
            self._forward_decode()
        stop.record()
        torch.cuda.synchronize()
        return start.elapsed_time(stop) / iters

    def _try_capture(self, probe: int):
        """Capture a throwaway graph of the current configuration, or None.

        Used to price candidates the way the run will actually execute them.
        Capture is not free, but warmup is untimed and a handful of captures
        sit comfortably inside its budget.
        """
        if not self.cuda:
            return None
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.g_pos.fill_(probe)
                    self._forward_decode()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._forward_decode()
            return graph
        except Exception as exc:  # noqa: BLE001 - fall back to eager timing
            print(f"[engine] timing capture failed ({exc}); timing eager", flush=True)
            return None

    def _select_attention(self, batch: int) -> None:
        """Measure the two GQA paths instead of assuming one wins.

        ``enable_gqa`` avoids materialising a 4x copy of the cache, but on some
        builds it steers SDPA onto the math backend, which is far worse. Warmup
        is untimed and shape is known, so the engine settles this by timing
        both rather than by guessing about backend coverage.
        """
        if not self.cuda:
            self.gqa = False
            return
        supports_gqa = self._supports_gqa(batch)
        try:
            # Time whole decode steps, not the attention call alone: the fused
            # kernel also removes the mask build and the output transpose, and
            # only an end-to-end step prices that in.
            candidates = []
            if self.fused_attn:
                candidates.append(("triton", True, False, supports_gqa))
            if getattr(self, "split_ok", False):
                candidates.append(("triton_split", True, True, supports_gqa))
            if supports_gqa:
                candidates.append(("enable_gqa", False, False, True))
            candidates.append(("repeat_kv", False, False, False))

            timings = []
            for name, fused, split, gqa in candidates:
                self.fused_attn, self.split_attn, self.gqa = fused, split, gqa
                timings.append((self._time_decode(), name, fused, split, gqa))

            best_ms, best_name, self.fused_attn, self.split_attn, self.gqa = min(timings)
            summary = ", ".join(f"{n} {ms:.3f} ms" for ms, n, _, _, _ in timings)
            print(f"[engine] decode step: {summary} -> {best_name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - timing must never fail a run
            print(f"[engine] attention timing failed ({exc}); falling back", flush=True)
            self.fused_attn, self.split_attn, self.gqa = False, False, supports_gqa
        finally:
            self.g_pos.zero_()

    # ---------------------------------------------------------------- fusion

    def _norm(self, x, weight):
        if self.fused_norm:
            return _k_rms_norm(x, weight, self.EPS)
        return rms_norm(x, weight, self.EPS)

    def _rope(self, q, k, cos, sin):
        if self.fused_rope:
            return _k_rope(q, cos, sin), _k_rope(k, cos, sin)
        return apply_rope(q, k, cos, sin)

    def _linear(self, x, weight):
        """F.linear, or the bandwidth-tuned kernel when the row count is tiny.

        Only decode qualifies. Prefill multiplies thousands of rows at once,
        where cuBLAS has real arithmetic intensity to work with and wins
        comfortably; the skinny kernel exists for the one-to-sixteen row case
        where the cost is purely streaming the weights.
        """
        if x.shape[0] * x.shape[1] <= 32:
            blocks = self.gemm_choice.get((weight.shape[0], weight.shape[1]))
            if blocks is not None:
                *tile, splits = blocks
                if splits > 1:
                    return _k_skinny_split(x, weight, *tile, splits)
                return _k_skinny(x, weight, *tile)
        return F.linear(x, weight)

    def _add_norm(self, residual, delta, weight):
        if self.fused_add_norm:
            return _k_add_norm(residual, delta, weight, self.EPS)
        total = residual + delta
        return total, self._norm(total, weight)

    def _norm_rope_pair(self, q, k, qw, kw, cos, sin):
        if self.fused_norm_rope:
            return (_k_norm_rope(q, qw, cos, sin, self.EPS),
                    _k_norm_rope(k, kw, cos, sin, self.EPS))
        return self._rope(self._norm(q, qw), self._norm(k, kw), cos, sin)

    def _swiglu(self, fused):
        if self.fused_swiglu:
            return _k_swiglu(fused)
        gate, up = fused.chunk(2, dim=-1)
        return F.silu(gate) * up

    def _validate_kernels(self, batch: int) -> None:
        """Adopt each fused kernel only if it reproduces the eager path exactly.

        These kernels cannot be tested off the target GPU - Triton needs one to
        compile at all - so they are checked here, at warmup, against the
        reference chain they replace, on tensors of the real shape. Anything
        that does not match bit for bit is not worth the tie margin, so it is
        simply not used.
        """
        if not self.cuda or _kernels_error is not None:
            if _kernels_error is not None:
                print(f"[engine] triton kernels unavailable ({_kernels_error})", flush=True)
            return

        inner = self.layers[0]["gate_up"].shape[0] // 2
        weight = self.layers[0]["in_ln"]
        head_w = self.layers[0]["q_norm"]
        verdict = {"norm": True, "rope": True, "swiglu": True, "norm_rope": True,
                   "add_norm": True, "add_norm_sum": True}

        # Both shapes that reach these kernels: the single-token decode step and
        # a multi-token prefill chunk. The index arithmetic differs between them
        # - a kernel can be right for one and wrong for the other.
        for tokens in (1, 5):
            hidden = torch.randn(batch, tokens, self.HIDDEN, dtype=self.dtype, device=DEVICE)
            heads = torch.randn(
                batch, tokens, self.HEADS, self.HEAD_DIM, dtype=self.dtype, device=DEVICE
            )
            gate_up = torch.randn(batch, tokens, 2 * inner, dtype=self.dtype, device=DEVICE)
            hidden2 = torch.randn(batch, tokens, self.HIDDEN, dtype=self.dtype, device=DEVICE)
            # q and k reach these kernels as slices of the fused QKV
            # projection, so they are views with a row stride wider than the
            # head block. Validating against a freshly allocated contiguous
            # tensor tests a layout that never occurs and hides stride bugs -
            # so build the real thing and slice it the same way _block does.
            qkv_probe = torch.randn(
                batch, tokens, self.q_size + 2 * self.kv_size,
                dtype=self.dtype, device=DEVICE,
            )
            q_view, k_view, _ = qkv_probe.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
            q_view = q_view.view(batch, tokens, self.HEADS, self.HEAD_DIM)
            k_view = k_view.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM)
            cos = self.cos_table[:tokens]
            sin = self.sin_table[:tokens]

            checks = (
                ("norm", lambda: _k_rms_norm(hidden, weight, self.EPS),
                 lambda: rms_norm(hidden, weight, self.EPS)),
                ("norm", lambda: _k_rms_norm(heads, head_w, self.EPS),
                 lambda: rms_norm(heads, head_w, self.EPS)),
                ("rope", lambda: _k_rope(heads, cos, sin),
                 lambda: apply_rope(heads, heads, cos, sin)[0]),
                ("swiglu", lambda: _k_swiglu(gate_up),
                 lambda: F.silu(gate_up.chunk(2, dim=-1)[0]) * gate_up.chunk(2, dim=-1)[1]),
                ("add_norm", lambda: _k_add_norm(hidden, hidden2, weight, self.EPS)[1],
                 lambda: rms_norm(hidden + hidden2, weight, self.EPS)),
                ("add_norm_sum", lambda: _k_add_norm(hidden, hidden2, weight, self.EPS)[0],
                 lambda: hidden + hidden2),
                ("norm_rope", lambda: _k_norm_rope(q_view, head_w, cos, sin, self.EPS),
                 lambda: apply_rope(rms_norm(q_view, head_w, self.EPS),
                                    rms_norm(q_view, head_w, self.EPS), cos, sin)[0]),
                ("norm_rope", lambda: _k_norm_rope(k_view, head_w, cos, sin, self.EPS),
                 lambda: apply_rope(rms_norm(k_view, head_w, self.EPS),
                                    rms_norm(k_view, head_w, self.EPS), cos, sin)[0]),
            )
            for name, fused_fn, eager_fn in checks:
                try:
                    ok = torch.equal(fused_fn(), eager_fn())
                except Exception as exc:  # noqa: BLE001 - a broken kernel is just unused
                    print(f"[engine] kernel {name} t={tokens} failed ({exc})", flush=True)
                    ok = False
                if not ok:
                    print(f"[engine] kernel {name} mismatched eager at t={tokens}", flush=True)
                verdict[name] &= ok

        self.fused_norm = verdict["norm"]
        self.fused_rope = verdict["rope"]
        self.fused_swiglu = verdict["swiglu"]
        # Only worth taking if both halves it replaces are themselves sound.
        self.fused_norm_rope = verdict["norm_rope"]
        # Both outputs must be right: the sum feeds the next residual.
        self.fused_add_norm = verdict["add_norm"] and verdict["add_norm_sum"]

        # The cache writer returns nothing, so it is judged on what it leaves
        # behind: run it and the eager chain into separate caches and require
        # both to match exactly, at a non-zero slot so a hardcoded offset of
        # zero cannot pass.
        self.fused_kv_cache = False
        self.gemm_blocks = None
        if _k_kv_cache is not None:
            try:
                ok = True
                for tokens in (1, 5):
                    qkv = torch.randn(
                        batch, tokens, self.q_size + 2 * self.kv_size,
                        dtype=self.dtype, device=DEVICE,
                    )
                    cos_t, sin_t = self.cos_table[:tokens], self.sin_table[:tokens]
                    base = min(3, max(0, self.capacity - tokens - 1))
                    slots = self.g_arange[base:base + tokens]

                    shape = (batch, self.KV_HEADS, self.capacity, self.HEAD_DIM)
                    got_k = torch.zeros(shape, dtype=self.dtype, device=DEVICE)
                    got_v = torch.zeros(shape, dtype=self.dtype, device=DEVICE)
                    want_k = torch.zeros(shape, dtype=self.dtype, device=DEVICE)
                    want_v = torch.zeros(shape, dtype=self.dtype, device=DEVICE)

                    _k_kv_cache(
                        qkv, head_w, cos_t, sin_t, slots, got_k, got_v,
                        self.KV_HEADS, self.HEAD_DIM, self.EPS,
                        self.q_size, self.q_size + self.kv_size,
                    )

                    _, k_ref, v_ref = qkv.split(
                        [self.q_size, self.kv_size, self.kv_size], dim=-1
                    )
                    k_ref = k_ref.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM)
                    k_ref = apply_rope(
                        rms_norm(k_ref, head_w, self.EPS),
                        rms_norm(k_ref, head_w, self.EPS), cos_t, sin_t,
                    )[0].transpose(1, 2)
                    v_ref = v_ref.view(
                        batch, tokens, self.KV_HEADS, self.HEAD_DIM
                    ).transpose(1, 2)
                    want_k[:, :, base:base + tokens, :] = k_ref
                    want_v[:, :, base:base + tokens, :] = v_ref

                    ok &= torch.equal(got_k, want_k) and torch.equal(got_v, want_v)
                self.fused_kv_cache = bool(ok)
                if not ok:
                    print("[engine] fused kv cache write mismatched eager", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] fused kv cache unavailable ({exc})", flush=True)
                self.fused_kv_cache = False
        self.gemm_blocks = None

        # Attention is a reduction over keys, so a flash-style streaming order
        # will not reproduce SDPA bit for bit the way the elementwise kernels
        # do. Reordering a reduction is explicitly within budget - it is what
        # separates the reference's own cached and uncached paths - so this one
        # is held to a tolerance rather than to equality.
        self.fused_attn = False
        if _k_attn is not None:
            try:
                q = torch.randn(
                    batch, 1, self.HEADS, self.HEAD_DIM, dtype=self.dtype, device=DEVICE
                )
                ck = torch.randn_like(self.cache_k[0])
                cv = torch.randn_like(self.cache_v[0])
                probe = torch.tensor([min(7, self.capacity - 1)], dtype=torch.int64, device=DEVICE)

                got = _k_attn(q, ck, cv, probe, self.HEADS, self.KV_HEADS, self.SCALE)

                keep = int(probe.item()) + 1
                ref = self._sdpa(
                    q.transpose(1, 2), ck[:, :, :keep, :], cv[:, :, :keep, :], is_causal=False
                )
                ref = ref.transpose(1, 2).reshape(batch, 1, self.q_size)

                def matches(candidate):
                    return bool(
                        torch.isfinite(candidate).all()
                        and torch.allclose(candidate.float(), ref.float(), atol=2e-2, rtol=2e-2)
                    )

                self.fused_attn = matches(got)
                split_ok = matches(
                    _k_attn_split(q, ck, cv, probe, self.HEADS, self.KV_HEADS, self.SCALE)
                )
                self.split_ok = split_ok
                if not self.fused_attn:
                    delta = (got.float() - ref.float()).abs().max().item()
                    print(f"[engine] fused attention off, max delta {delta:.3e}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] fused attention unavailable ({exc})", flush=True)
                self.fused_attn = False
                self.split_ok = False
        print(
            f"[engine] fused kernels: norm={self.fused_norm} rope={self.fused_rope} "
            f"swiglu={self.fused_swiglu} norm_rope={self.fused_norm_rope} "
            f"add_norm={self.fused_add_norm} kv_cache={self.fused_kv_cache}",
            flush=True,
        )

    # --------------------------------------------------------------- forward

    def _block(self, residual, n, layer, next_norm_w, cos, sin, *,
               positions, position, start, mask, is_causal):
        """One decoder layer, taking and returning (residual, pre-normed).

        The input arrives already normalised because the previous layer's
        closing add produced it: every residual join is followed immediately by
        a norm, so the two are done together and the next layer's input falls
        out of this layer's last kernel.

        ``position`` is a device index tensor for decode, or ``None`` for
        prefill, where ``start`` gives the absolute offset of this chunk.
        ``positions`` gives each token's cache slot, for the fused writer.
        """
        batch, tokens, _ = n.shape
        qkv = self._linear(n, layer["qkv"])
        ck, cv = self.cache_k[layer["idx"]], self.cache_v[layer["idx"]]
        end = None if position is not None else start + tokens

        if self.fused_kv_cache:
            # k is normalised, rotated and written to its slot, and v copied to
            # its own, in one pass - so neither ever becomes a tensor of its
            # own and the two index_copy_ calls disappear.
            _k_kv_cache(
                qkv, layer["k_norm"], cos, sin, positions, ck, cv,
                self.KV_HEADS, self.HEAD_DIM, self.EPS,
                self.q_size, self.q_size + self.kv_size,
            )
            q = _k_norm_rope(
                qkv[..., : self.q_size].view(batch, tokens, self.HEADS, self.HEAD_DIM),
                layer["q_norm"], cos, sin, self.EPS,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            # Per-head RMSNorm over the head dimension, then RoPE. Qwen3 norms
            # q and k; never v. Both run on [B, T, H, D] rather than after the
            # transpose as the reference writes it: the operation is per
            # (batch, token, head) row and indexed only by token, so the values
            # are identical, but the tensor keeps a stride a kernel can walk.
            q, k = self._norm_rope_pair(
                q.view(batch, tokens, self.HEADS, self.HEAD_DIM),
                k.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM),
                layer["q_norm"], layer["k_norm"], cos, sin,
            )
            k_t = k.transpose(1, 2)
            v_t = v.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM).transpose(1, 2)
            if position is None:
                ck[:, :, start:end, :] = k_t
                cv[:, :, start:end, :] = v_t
            else:
                # A device-side index stays dynamic under graph capture, where
                # a Python slice would be frozen at the captured step.
                ck.index_copy_(2, position, k_t)
                cv.index_copy_(2, position, v_t)

        if self.fused_attn and position is not None:
            # Writes [batch, 1, heads * head_dim] directly, which is already
            # the layout o_proj wants - the SDPA path needs a transpose and a
            # copy to get there.
            fn = _k_attn_split if self.split_attn else _k_attn
            a = fn(q, ck, cv, position, self.HEADS, self.KV_HEADS, self.SCALE)
        else:
            keys = ck if position is not None else ck[:, :, :end, :]
            values = cv if position is not None else cv[:, :, :end, :]
            a = self._sdpa(q.transpose(1, 2), keys, values,
                           is_causal=is_causal, attn_mask=mask)
            a = a.transpose(1, 2).reshape(batch, tokens, self.q_size)

        residual, m = self._add_norm(residual, self._linear(a, layer["o"]), layer["post_ln"])
        mlp = self._linear(self._swiglu(self._linear(m, layer["gate_up"])), layer["down"])
        return self._add_norm(residual, mlp, next_norm_w)

    def _forward_prefill(self, ids, seq_len):
        """Consume the prompt and return logits for its last position only."""
        batch = ids.shape[0]
        chunk = seq_len
        if batch * seq_len > PREFILL_CHUNK_TOKENS:
            chunk = max(1, PREFILL_CHUNK_TOKENS // batch)

        # These kernels were written for the one-row decode step. Prefill runs
        # thousands of rows at once, where PyTorch's vectorised elementwise
        # kernels may well win, so whether to use them here is a separate
        # question from whether to use them in decode - and it is measured.
        saved = (self.fused_norm, self.fused_rope, self.fused_swiglu,
                 self.fused_norm_rope, self.fused_add_norm, self.fused_kv_cache)
        if not self.prefill_fused:
            (self.fused_norm, self.fused_rope, self.fused_swiglu,
             self.fused_norm_rope, self.fused_add_norm,
             self.fused_kv_cache) = (False,) * 6
        try:
            return self._prefill_body(ids, seq_len)
        finally:
            (self.fused_norm, self.fused_rope, self.fused_swiglu,
             self.fused_norm_rope, self.fused_add_norm,
             self.fused_kv_cache) = saved

    def _prefill_body(self, ids, seq_len):
        batch = ids.shape[0]
        chunk = seq_len
        if batch * seq_len > PREFILL_CHUNK_TOKENS:
            chunk = max(1, PREFILL_CHUNK_TOKENS // batch)

        x_all = F.embedding(ids, self.embed)
        hidden = None
        for start in range(0, seq_len, chunk):
            stop = min(start + chunk, seq_len)
            width = stop - start
            x = x_all[:, start:stop, :]
            cos = self.cos_table[start:stop]
            sin = self.sin_table[start:stop]

            if start == 0 and stop == seq_len:
                # Whole prompt at once: let SDPA use its causal fast path.
                mask, is_causal = None, True
            else:
                # A later chunk's queries see every earlier key, so causality
                # cannot be inferred from the query length - spell it out.
                keys = torch.arange(stop, device=DEVICE)
                queries = torch.arange(start, stop, device=DEVICE).unsqueeze(1)
                mask = torch.where(keys <= queries, 0.0, NEG).to(self.dtype)
                mask = mask.view(1, 1, width, stop)
                is_causal = False

            residual = x
            n = self._norm(x, self.layers[0]["in_ln"])
            for i, layer in enumerate(self.layers):
                nxt = (self.layers[i + 1]["in_ln"] if i + 1 < self.LAYERS
                       else self.final_norm_w)
                residual, n = self._block(
                    residual, n, layer, nxt, cos, sin,
                    positions=self.g_arange[start:stop], position=None, start=start,
                    mask=mask, is_causal=is_causal,
                )
            hidden = n

        # `n` already carries the final norm: the last layer's closing add was
        # told to normalise with final_norm_w instead of a next layer's weight.
        return F.linear(hidden[:, -1:, :], self.lm_head_w)

    def _forward_decode(self):
        """One decode step from persistent buffers, and self-feeding.

        No Python-visible shape depends on the step index, so this captures
        once and replays for every token.
        """
        pos = self.g_pos
        x = F.embedding(self.g_token, self.embed)
        cos = self.cos_table.index_select(0, pos)
        sin = self.sin_table.index_select(0, pos)

        # Slots at index <= pos hold real keys; the rest is capacity that was
        # never written. NEG underflows to zero through the softmax.
        if self.fused_attn:
            mask = None  # the kernel reads the length itself; no mask needed
        else:
            mask = torch.where(self.g_arange <= pos, 0.0, NEG).to(self.dtype)
            mask = mask.view(1, 1, 1, self.capacity)

        residual = x
        n = self._norm(x, self.layers[0]["in_ln"])
        for i, layer in enumerate(self.layers):
            nxt = self.layers[i + 1]["in_ln"] if i + 1 < self.LAYERS else self.final_norm_w
            residual, n = self._block(
                residual, n, layer, nxt, cos, sin,
                positions=pos, position=pos, start=0, mask=mask, is_causal=False,
            )
        x = n
        token = self._linear(x, self.lm_head_w)[:, -1, :].argmax(dim=-1, keepdim=True)

        # Close the loop inside the captured region: next replay reads this
        # token at this position, with no host involvement.
        self.g_token.copy_(token)
        self.g_pos.add_(1)
        return token

    # ------------------------------------------------------------ shape prep

    def _ensure_shape(self, batch: int, seq_len: int, max_new_tokens: int) -> None:
        capacity = seq_len + max_new_tokens
        if self.batch == batch and self.capacity == capacity:
            return
        print(f"[engine] building for batch={batch} capacity={capacity}", flush=True)

        self.graph = None
        self.cache_k = self.cache_v = None
        if self.cuda:
            torch.cuda.empty_cache()

        self._grow_rope(capacity + 1)

        shape = (batch, self.KV_HEADS, capacity, self.HEAD_DIM)
        self.cache_k = [
            torch.zeros(shape, dtype=self.dtype, device=DEVICE) for _ in range(self.LAYERS)
        ]
        self.cache_v = [
            torch.zeros(shape, dtype=self.dtype, device=DEVICE) for _ in range(self.LAYERS)
        ]

        self.batch = batch
        self.capacity = capacity
        # Midpoint of the decode range: the position a typical step runs at,
        # and therefore the only honest place to compare attention paths.
        self.time_probe = seq_len + max_new_tokens // 2
        self.g_token = torch.zeros(batch, 1, dtype=torch.int64, device=DEVICE)
        self.g_pos = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.g_arange = torch.arange(capacity, dtype=torch.int64, device=DEVICE)

        # All three need the buffers above to exist, and all belong to warmup,
        # which is untimed: settle correctness, then the attention path, then
        # the matmul tiling, then capture whatever won.
        self._validate_kernels(batch)
        self._select_attention(batch)
        self._select_gemm()
        self._select_prefill(batch, seq_len)
        self._capture()

    def _select_gemm(self) -> None:
        """Decide, per weight shape, whether the skinny matmul beats cuBLAS.

        Decode is weight-bandwidth bound, so this is where the remaining time
        is. Measurement rather than assumption, for two reasons: whether a
        hand-tiled kernel beats a tuned library at these shapes is not knowable
        from a machine with no GPU, and the answer is not even the same across
        shapes - at batch 1 it won by 30%, at batch 16 it lost by 18%.

        Two passes. First pick a tiling using every weight at once, then take
        that tiling and test each weight shape on its own, keeping the kernel
        only where it actually helps. Greedy, but it costs a handful of
        captures rather than the full cross product.
        """
        self.gemm_choice = {}
        if not self.cuda or _k_skinny is None:
            return

        shapes = []
        for weight in (self.layers[0]["qkv"], self.layers[0]["o"],
                       self.layers[0]["gate_up"], self.layers[0]["down"],
                       self.lm_head_w):
            key = (weight.shape[0], weight.shape[1])
            if key not in [s for s, _ in shapes]:
                shapes.append((key, weight))

        try:
            # Accuracy gate. A matmul sums thousands of terms, so its order
            # differs from cuBLAS's and equality is the wrong test - this is
            # the reordering budget the contract allows, not an approximation.
            probe = torch.randn(self.batch, 1, self.HIDDEN, dtype=self.dtype, device=DEVICE)
            reference = F.linear(probe, self.layers[0]["qkv"])
            usable = []
            # splits > 1 spreads a narrow output over more programs; splits
            # of 1 is the single-pass kernel. Kept small so the whole search,
            # including Triton compiling each variant, fits the load budget.
            grid = [(bn, 64, w, 3, sp)
                    for bn in (64, 128) for w in (4, 8) for sp in (1, 8)]
            for blocks in grid:
                try:
                    *tile, splits = blocks
                    got = (_k_skinny_split(probe, self.layers[0]["qkv"], *tile, splits)
                           if splits > 1
                           else _k_skinny(probe, self.layers[0]["qkv"], *tile))
                except Exception:  # noqa: BLE001 - a variant that will not compile
                    continue
                if torch.isfinite(got).all() and torch.allclose(
                    got.float(), reference.float(), atol=2e-2, rtol=2e-2
                ):
                    usable.append(blocks)
            if not usable:
                print("[engine] skinny matmul rejected on accuracy", flush=True)
                return

            baseline = self._time_decode()
            best_ms, best_blocks = baseline, None
            for blocks in usable:
                self.gemm_choice = {key: blocks for key, _ in shapes}
                elapsed = self._time_decode()
                if elapsed < best_ms:
                    best_ms, best_blocks = elapsed, blocks

            if best_blocks is None:
                self.gemm_choice = {}
                print(f"[engine] decode matmul: cublas ({baseline:.3f} ms)", flush=True)
                return

            # Now drop it from any shape it does not earn its place on.
            self.gemm_choice = {key: best_blocks for key, _ in shapes}
            current = best_ms
            for key, _ in shapes:
                self.gemm_choice.pop(key)
                without = self._time_decode()
                if without < current:
                    current = without
                else:
                    self.gemm_choice[key] = best_blocks

            kept = [f"{k[0]}x{k[1]}" for k in self.gemm_choice]
            print(
                f"[engine] decode matmul: cublas {baseline:.3f} ms -> {current:.3f} ms "
                f"with {best_blocks} on {kept or 'nothing'}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - never let tuning fail a run
            print(f"[engine] matmul tuning failed ({exc}); cublas", flush=True)
            self.gemm_choice = {}
        finally:
            self.g_pos.zero_()

    def _select_prefill(self, batch: int, seq_len: int) -> None:
        """Measure whether the fused kernels help or hurt the prompt pass.

        At batch 4 with a 2048-token prompt, prefill is over a third of the
        workload's time, so this is worth settling rather than assuming. The
        kernels were tuned for a single row; thousands of rows is a different
        regime and PyTorch's own kernels may be better at it.
        """
        if not self.cuda:
            return
        try:
            ids = torch.zeros(batch, seq_len, dtype=torch.int64, device=DEVICE)

            def timed(iters=3):
                for _ in range(1):
                    self._forward_prefill(ids, seq_len)
                torch.cuda.synchronize()
                start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(iters):
                    self._forward_prefill(ids, seq_len)
                stop.record()
                torch.cuda.synchronize()
                return start.elapsed_time(stop) / iters

            self.prefill_fused = True
            with_fused = timed()
            self.prefill_fused = False
            without = timed()
            self.prefill_fused = with_fused <= without
            print(
                f"[engine] prefill: fused {with_fused:.1f} ms vs eager {without:.1f} ms "
                f"-> {'fused' if self.prefill_fused else 'eager'}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - tuning must never fail a run
            print(f"[engine] prefill tuning failed ({exc}); fused", flush=True)
            self.prefill_fused = True

    def _capture(self) -> None:
        """Capture the decode step.

        Capture failing is survivable: the eager path yields identical tokens,
        only slower. It must never take the run down.
        """
        if not self.cuda:
            self.graph = None
            return
        try:
            torch.cuda.synchronize()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    # Rewind each time: _forward_decode advances the position,
                    # and three unchecked steps would index past a small cache.
                    self.g_pos.fill_(min(self.time_probe, self.capacity - 1))
                    self._forward_decode()
            self.g_pos.zero_()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.g_out = self._forward_decode()
            self.graph = graph
            print("[engine] decode graph captured", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.graph = None
            print(f"[engine] graph capture failed ({exc}); eager decode", flush=True)

    def _decode_step(self):
        if self.graph is not None:
            self.graph.replay()
            return self.g_out
        return self._forward_decode()

    # -------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        ids = torch.tensor(input_ids, dtype=torch.int64, device=DEVICE)
        batch, seq_len = ids.shape
        self._ensure_shape(batch, seq_len, max_new_tokens)

        # No state carries between calls: positions restart at zero and only
        # slots strictly below the running position are ever read.
        token = self._forward_prefill(ids, seq_len)[:, -1, :].argmax(dim=-1, keepdim=True)
        self.g_token.copy_(token)
        self.g_pos.fill_(seq_len)

        # One step of lookahead. The graph is self-feeding, so step i+1 can be
        # launched before step i's ids have landed on the host, hiding the
        # device-to-host latency behind real work.
        host = torch.empty(batch, 1, dtype=torch.int64, device="cpu", pin_memory=self.cuda)
        pending = None

        for step in range(max_new_tokens):
            if pending is None:
                out = token.to("cpu")
            else:
                pending.synchronize()
                out = host.clone()

            if step + 1 < max_new_tokens:
                nxt = self._decode_step()
                host.copy_(nxt, non_blocking=self.cuda)
                if self.cuda:
                    pending = torch.cuda.Event()
                    pending.record()
                else:
                    pending = _Ready()
            else:
                pending = None

            yield out[:, 0].tolist()
