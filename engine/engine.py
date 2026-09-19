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
    """``apply_rotary_pos_emb`` with ``unsqueeze_dim=1``.

    cos/sin arrive already in BF16: the reference casts the tables before the
    multiply, so multiplying in FP32 here would be a reformulation.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
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
            self.layers.append(
                {
                    "idx": idx,
                    "in_ln": layer.input_layernorm.weight,
                    # One GEMM instead of three. Every output row is still the
                    # same dot product of the same input row against the same
                    # weight row: a reordering, not a reformulation.
                    "qkv": torch.cat(
                        [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0
                    ).contiguous(),
                    "q_norm": attn.q_norm.weight,
                    "k_norm": attn.k_norm.weight,
                    "o": attn.o_proj.weight,
                    "post_ln": layer.post_attention_layernorm.weight,
                    "gate_up": torch.cat(
                        [mlp.gate_proj.weight, mlp.up_proj.weight], dim=0
                    ).contiguous(),
                    "down": mlp.down_proj.weight,
                }
            )

        # The unfused originals are dead weight once copied; drop them so the
        # fused copies do not double peak memory.
        for layer in base.layers:
            layer.self_attn.q_proj = None
            layer.self_attn.k_proj = None
            layer.self_attn.v_proj = None
            layer.mlp.gate_proj = None
            layer.mlp.up_proj = None
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

    def _probe_gqa(self, batch: int) -> None:
        if self.gqa is not None:
            return
        q = torch.zeros(batch, self.HEADS, 1, self.HEAD_DIM, dtype=self.dtype, device=DEVICE)
        kv = torch.zeros(batch, self.KV_HEADS, 8, self.HEAD_DIM, dtype=self.dtype, device=DEVICE)
        try:
            F.scaled_dot_product_attention(q, kv, kv, scale=self.SCALE, enable_gqa=True)
            self.gqa = True
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            print(f"[engine] enable_gqa unavailable ({exc}); using repeat_kv", flush=True)
            self.gqa = False

    # --------------------------------------------------------------- forward

    def _block(self, x, layer, cos, sin, *, position, start, mask, is_causal):
        """One decoder layer.

        ``position`` is a device index tensor for decode, or ``None`` for
        prefill, where ``start`` gives the absolute offset of this chunk.
        """
        batch, tokens, _ = x.shape
        residual = x

        n = rms_norm(x, layer["in_ln"], self.EPS)
        q, k, v = F.linear(n, layer["qkv"]).split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        # Per-head RMSNorm over the 128-wide head dimension, before RoPE and
        # before the transpose. Qwen3 norms q and k; never v.
        q = q.view(batch, tokens, self.HEADS, self.HEAD_DIM)
        k = k.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM)
        q = rms_norm(q, layer["q_norm"], self.EPS).transpose(1, 2)
        k = rms_norm(k, layer["k_norm"], self.EPS).transpose(1, 2)
        v = v.view(batch, tokens, self.KV_HEADS, self.HEAD_DIM).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        ck, cv = self.cache_k[layer["idx"]], self.cache_v[layer["idx"]]
        if position is None:
            end = start + tokens
            ck[:, :, start:end, :] = k
            cv[:, :, start:end, :] = v
            keys, values = ck[:, :, :end, :], cv[:, :, :end, :]
        else:
            # A device-side index stays dynamic under graph capture, where a
            # Python slice would be frozen at the captured step.
            ck.index_copy_(2, position, k)
            cv.index_copy_(2, position, v)
            keys, values = ck, cv

        a = self._sdpa(q, keys, values, is_causal=is_causal, attn_mask=mask)
        a = a.transpose(1, 2).reshape(batch, tokens, self.q_size)
        x = residual + F.linear(a, layer["o"])

        residual = x
        m = rms_norm(x, layer["post_ln"], self.EPS)
        gate, up = F.linear(m, layer["gate_up"]).chunk(2, dim=-1)
        return residual + F.linear(F.silu(gate) * up, layer["down"])

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
            cos = self.cos_table[start:stop].unsqueeze(0)
            sin = self.sin_table[start:stop].unsqueeze(0)

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

        x = rms_norm(hidden[:, -1:, :], self.final_norm_w, self.EPS)
        return F.linear(x, self.lm_head_w)

    def _forward_decode(self):
        """One decode step from persistent buffers, and self-feeding.

        No Python-visible shape depends on the step index, so this captures
        once and replays for every token.
        """
        pos = self.g_pos
        x = F.embedding(self.g_token, self.embed)
        cos = self.cos_table.index_select(0, pos).unsqueeze(0)
        sin = self.sin_table.index_select(0, pos).unsqueeze(0)

        # Slots at index <= pos hold real keys; the rest is capacity that was
        # never written. NEG underflows to zero through the softmax.
        mask = torch.where(self.g_arange <= pos, 0.0, NEG).to(self.dtype)
        mask = mask.view(1, 1, 1, self.capacity)

        for layer in self.layers:
            x = self._block(
                x, layer, cos, sin, position=pos, start=0, mask=mask, is_causal=False
            )
        x = rms_norm(x, self.final_norm_w, self.EPS)
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
        self._probe_gqa(batch)

        shape = (batch, self.KV_HEADS, capacity, self.HEAD_DIM)
        self.cache_k = [
            torch.zeros(shape, dtype=self.dtype, device=DEVICE) for _ in range(self.LAYERS)
        ]
        self.cache_v = [
            torch.zeros(shape, dtype=self.dtype, device=DEVICE) for _ in range(self.LAYERS)
        ]

        self.batch = batch
        self.capacity = capacity
        self.g_token = torch.zeros(batch, 1, dtype=torch.int64, device=DEVICE)
        self.g_pos = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        self.g_arange = torch.arange(capacity, dtype=torch.int64, device=DEVICE)
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
                    self.g_pos.zero_()
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
