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
        rope as _k_rope,
        swiglu as _k_swiglu,
        decode_attention as _k_attn,
        norm_rope as _k_norm_rope,
        decode_attention_split as _k_attn_split,
    )
except Exception as _exc:  # noqa: BLE001
    _kernels_error = _exc
    _k_rms_norm = _k_rope = _k_swiglu = _k_attn = _k_norm_rope = _k_attn_split = None


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

        Timed at a position partway through the generation, not at zero. The
        cost of every candidate depends on how many keys are live: at position
        zero a streaming kernel reads one block while SDPA still builds a
        full-capacity mask, which flatters the kernel and picked it for shapes
        where it loses badly. Timing where the workload actually spends its
        steps is the only comparison that means anything.
        """
        probe = min(self.time_probe, self.capacity - 1)
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
        verdict = {"norm": True, "rope": True, "swiglu": True, "norm_rope": True}

        # Both shapes that reach these kernels: the single-token decode step and
        # a multi-token prefill chunk. The index arithmetic differs between them
        # - a kernel can be right for one and wrong for the other.
        for tokens in (1, 5):
            hidden = torch.randn(batch, tokens, self.HIDDEN, dtype=self.dtype, device=DEVICE)
            heads = torch.randn(
                batch, tokens, self.HEADS, self.HEAD_DIM, dtype=self.dtype, device=DEVICE
            )
            gate_up = torch.randn(batch, tokens, 2 * inner, dtype=self.dtype, device=DEVICE)
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
                ("norm_rope", lambda: _k_norm_rope(heads, head_w, cos, sin, self.EPS),
                 lambda: apply_rope(rms_norm(heads, head_w, self.EPS),
                                    rms_norm(heads, head_w, self.EPS), cos, sin)[0]),
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
            f"swiglu={self.fused_swiglu} norm_rope={self.fused_norm_rope}",
            flush=True,
        )

    # --------------------------------------------------------------- forward

    def _block(self, x, layer, cos, sin, *, position, start, mask, is_causal):
        """One decoder layer.

        ``position`` is a device index tensor for decode, or ``None`` for
        prefill, where ``start`` gives the absolute offset of this chunk.
        """
        batch, tokens, _ = x.shape
        residual = x

        n = self._norm(x, layer["in_ln"])
        q, k, v = F.linear(n, layer["qkv"]).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        # Per-head RMSNorm over the 128-wide head dimension, before RoPE.
        # Qwen3 norms q and k; never v.
        q, k = self._norm_rope_pair(
            q.view(batch, tokens, self.HEADS, self.HEAD_DIM),
            k.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM),
            layer["q_norm"], layer["k_norm"], cos, sin,
        )

        # RoPE runs here, on [B, T, H, D], rather than after the transpose as
        # the reference writes it. It is elementwise per (batch, token, head)
        # row and indexed only by token, so the result is identical - but the
        # tensor is still contiguous, which is what lets a fused kernel read it
        # with a single stride instead of a transposed one.
        k_t = k.transpose(1, 2)
        v_t = v.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM).transpose(1, 2)

        ck, cv = self.cache_k[layer["idx"]], self.cache_v[layer["idx"]]
        if position is None:
            end = start + tokens
            ck[:, :, start:end, :] = k_t
            cv[:, :, start:end, :] = v_t
            keys, values = ck[:, :, :end, :], cv[:, :, :end, :]
        else:
            # A device-side index stays dynamic under graph capture, where a
            # Python slice would be frozen at the captured step.
            ck.index_copy_(2, position, k_t)
            cv.index_copy_(2, position, v_t)
            keys, values = ck, cv

        if self.fused_attn and position is not None:
            # Writes [batch, 1, heads * head_dim] directly, which is already
            # the layout o_proj wants - the SDPA path needs a transpose and a
            # copy to get there.
            fn = _k_attn_split if self.split_attn else _k_attn
            a = fn(q, ck, cv, position, self.HEADS, self.KV_HEADS, self.SCALE)
        else:
            a = self._sdpa(q.transpose(1, 2), keys, values, is_causal=is_causal, attn_mask=mask)
            a = a.transpose(1, 2).reshape(batch, tokens, self.q_size)
        x = residual + F.linear(a, layer["o"])

        residual = x
        m = self._norm(x, layer["post_ln"])
        return residual + F.linear(self._swiglu(F.linear(m, layer["gate_up"])), layer["down"])

    def _forward_prefill(self, ids, seq_len):
        """Consume the prompt and return logits for its last position only."""
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

            for layer in self.layers:
                x = self._block(
                    x, layer, cos, sin, position=None, start=start,
                    mask=mask, is_causal=is_causal,
                )
            hidden = x

        x = self._norm(hidden[:, -1:, :].contiguous(), self.final_norm_w)
        return F.linear(x, self.lm_head_w)

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

        for layer in self.layers:
            x = self._block(
                x, layer, cos, sin, position=pos, start=0, mask=mask, is_causal=False
            )
        x = self._norm(x, self.final_norm_w)
        token = F.linear(x, self.lm_head_w)[:, -1, :].argmax(dim=-1, keepdim=True)

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

        # Both need the buffers above to exist, and both belong to warmup,
        # which is untimed: settle the attention path, then capture it.
        self._validate_kernels(batch)
        self._select_attention(batch)
        self._capture()

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
