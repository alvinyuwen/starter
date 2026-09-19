"""Fused elementwise kernels for the decode path.

With the decode step captured in a CUDA graph, launch overhead is gone but
per-kernel execution overhead is not: a batch-1 step runs ~2000 kernels across
36 layers, most of them elementwise ops on a single token's worth of data, and
that fixed cost - not bandwidth - is what sets time per output token.

Each kernel here collapses a chain of those into one launch. The arithmetic is
the same arithmetic, including where values round to BF16, because PyTorch's
elementwise kernels compute in FP32 and store the result back in the input
dtype at every step. Skipping one of those intermediate roundings computes a
different function and spends tie margin for nothing, so each is reproduced
explicitly.

Nothing here is trusted on faith: the engine compares every kernel against the
eager path during warmup and falls back if the outputs are not identical.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr, out_ptr, cos_ptr, sin_ptr,
    tokens, heads, head_dim, half,
    HALF_BLOCK: tl.constexpr,
):
    """One program per (batch, token, head) row of `head_dim` values.

    RoPE pairs element j with j+half, so a program loads both halves at once
    and writes both, which needs one pass over the row instead of the four the
    eager chain makes (slice, negate, concatenate, multiply-add).
    """
    row = tl.program_id(0)
    # Rows are laid out [batch, token, head], so row // heads is the flattened
    # (batch, token) index. The angle table has only `tokens` rows, so the
    # batch component has to come back out - without the modulo every sequence
    # after the first reads past the end of the table.
    token = (row // heads) % tokens

    lane = tl.arange(0, HALF_BLOCK)
    mask = lane < half

    lo_off = row * head_dim + lane
    hi_off = lo_off + half

    x_lo = tl.load(x_ptr + lo_off, mask=mask, other=0.0)
    x_hi = tl.load(x_ptr + hi_off, mask=mask, other=0.0)

    angle = token * head_dim + lane
    cos_lo = tl.load(cos_ptr + angle, mask=mask, other=0.0)
    sin_lo = tl.load(sin_ptr + angle, mask=mask, other=0.0)
    cos_hi = tl.load(cos_ptr + angle + half, mask=mask, other=0.0)
    sin_hi = tl.load(sin_ptr + angle + half, mask=mask, other=0.0)

    out_dtype = out_ptr.dtype.element_ty

    # rotate_half puts -x[half:] in the low lanes and x[:half] in the high ones.
    # Each product rounds to the working dtype before the sum, matching the two
    # separate PyTorch multiplies that the eager path performs.
    lo_a = (x_lo.to(tl.float32) * cos_lo.to(tl.float32)).to(out_dtype)
    lo_b = ((-x_hi).to(tl.float32) * sin_lo.to(tl.float32)).to(out_dtype)
    hi_a = (x_hi.to(tl.float32) * cos_hi.to(tl.float32)).to(out_dtype)
    hi_b = (x_lo.to(tl.float32) * sin_hi.to(tl.float32)).to(out_dtype)

    tl.store(out_ptr + lo_off, (lo_a.to(tl.float32) + lo_b.to(tl.float32)).to(out_dtype), mask=mask)
    tl.store(out_ptr + hi_off, (hi_a.to(tl.float32) + hi_b.to(tl.float32)).to(out_dtype), mask=mask)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE over ``[batch, tokens, heads, head_dim]`` against ``[tokens, head_dim]``.

    ``x`` must be contiguous. Returns a new tensor of the same shape.
    """
    batch, tokens, heads, head_dim = x.shape
    half = head_dim // 2
    out = torch.empty_like(x)
    _rope_kernel[(batch * tokens * heads,)](
        x, out, cos, sin,
        tokens, heads, head_dim, half,
        HALF_BLOCK=triton.next_power_of_2(half),
        num_warps=4,
    )
    return out


@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, inner, n_elements, BLOCK: tl.constexpr):
    """``silu(gate) * up`` straight off the fused gate/up projection.

    Reading the [gate | up] halves directly avoids the two strided views that
    ``chunk`` produces, and folds silu and the multiply into one pass.
    """
    start = tl.program_id(0) * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    row = offsets // inner
    col = offsets % inner
    base = row * (2 * inner) + col

    gate = tl.load(x_ptr + base, mask=mask, other=0.0)
    up = tl.load(x_ptr + base + inner, mask=mask, other=0.0)

    out_dtype = out_ptr.dtype.element_ty
    g = gate.to(tl.float32)
    # F.silu rounds its result back to the input dtype before the multiply.
    silu = (g / (1.0 + tl.exp(-g))).to(out_dtype)
    value = (silu.to(tl.float32) * up.to(tl.float32)).to(out_dtype)
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _add_norm_kernel(
    res_ptr, delta_ptr, sum_ptr, out_ptr, w_ptr, n_cols, eps, BLOCK: tl.constexpr,
):
    """residual + delta, then RMSNorm of the sum, emitting both.

    Every residual join is immediately followed by a norm, and the sum is
    needed twice - once as the next residual, once as the norm's input - so
    computing it in the norm's own pass removes a launch and a round trip at
    each of the two joins per layer.

    The add happens in the working dtype, matching the eager tensor add, and
    the norm keeps the reference's cast placement.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * n_cols + cols
    out_dtype = out_ptr.dtype.element_ty

    residual = tl.load(res_ptr + offsets, mask=mask, other=0.0)
    delta = tl.load(delta_ptr + offsets, mask=mask, other=0.0)
    total = (residual.to(tl.float32) + delta.to(tl.float32)).to(out_dtype)
    tl.store(sum_ptr + offsets, total, mask=mask)

    xf = total.to(tl.float32)
    variance = tl.sum(xf * xf, axis=0) / n_cols
    normed = (xf * tl.math.rsqrt(variance + eps)).to(out_dtype)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, normed * weight, mask=mask)


def add_norm(residual, delta, weight, eps):
    """Returns ``(residual + delta, rms_norm(residual + delta, weight))``."""
    n_cols = residual.shape[-1]
    rows = residual.numel() // n_cols
    total = torch.empty_like(residual)
    out = torch.empty_like(residual)
    _add_norm_kernel[(rows,)](
        residual, delta, total, out, weight, n_cols, eps,
        BLOCK=triton.next_power_of_2(n_cols),
        num_warps=8,
    )
    return total, out


@triton.jit
def _norm_rope_kernel(
    x_ptr, out_ptr, w_ptr, cos_ptr, sin_ptr,
    tokens, heads, head_dim, half, eps, row_stride,
    HALF_BLOCK: tl.constexpr,
):
    """Per-head RMSNorm followed by RoPE, in one pass over the row.

    These always run back to back on q and on k, and both are row-local, so
    reading the row once and doing both saves a launch and a round trip to
    memory at each of the four sites per layer.

    RoPE pairs lane j with j + half, so the row is loaded in halves; the
    variance reduction simply sums both halves, which is the same sum over the
    same values.
    """
    row = tl.program_id(0)
    bt = row // heads               # flattened (batch, token)
    head = row % heads
    token = bt % tokens

    lane = tl.arange(0, HALF_BLOCK)
    mask = lane < half
    # q and k arrive as slices of the fused QKV projection, so consecutive
    # tokens are row_stride apart, not heads*head_dim. Assuming the packed
    # layout reads the wrong memory for every token after the first.
    src = bt * row_stride + head * head_dim
    lo_off = src + lane
    hi_off = lo_off + half
    dst = row * head_dim + lane

    lo = tl.load(x_ptr + lo_off, mask=mask, other=0.0)
    hi = tl.load(x_ptr + hi_off, mask=mask, other=0.0)
    out_dtype = out_ptr.dtype.element_ty

    # RMSNorm: reduce in FP32, round the normalised value to the working dtype,
    # then multiply by the weight - the reference's cast placement exactly.
    f_lo = lo.to(tl.float32)
    f_hi = hi.to(tl.float32)
    variance = (tl.sum(f_lo * f_lo, axis=0) + tl.sum(f_hi * f_hi, axis=0)) / head_dim
    inv = tl.math.rsqrt(variance + eps)
    w_lo = tl.load(w_ptr + lane, mask=mask, other=0.0)
    w_hi = tl.load(w_ptr + lane + half, mask=mask, other=0.0)
    n_lo = (f_lo * inv).to(out_dtype) * w_lo
    n_hi = (f_hi * inv).to(out_dtype) * w_hi

    angle = token * head_dim + lane
    cos_lo = tl.load(cos_ptr + angle, mask=mask, other=0.0)
    sin_lo = tl.load(sin_ptr + angle, mask=mask, other=0.0)
    cos_hi = tl.load(cos_ptr + angle + half, mask=mask, other=0.0)
    sin_hi = tl.load(sin_ptr + angle + half, mask=mask, other=0.0)

    a_lo = (n_lo.to(tl.float32) * cos_lo.to(tl.float32)).to(out_dtype)
    b_lo = ((-n_hi).to(tl.float32) * sin_lo.to(tl.float32)).to(out_dtype)
    a_hi = (n_hi.to(tl.float32) * cos_hi.to(tl.float32)).to(out_dtype)
    b_hi = (n_lo.to(tl.float32) * sin_hi.to(tl.float32)).to(out_dtype)

    tl.store(out_ptr + dst, (a_lo.to(tl.float32) + b_lo.to(tl.float32)).to(out_dtype), mask=mask)
    tl.store(out_ptr + dst + half, (a_hi.to(tl.float32) + b_hi.to(tl.float32)).to(out_dtype), mask=mask)


def norm_rope(x, weight, cos, sin, eps):
    """RMSNorm over each head then RoPE, on ``[batch, tokens, heads, head_dim]``.

    ``x`` may be a non-contiguous slice of a wider tensor - which is exactly
    what it is in practice, being q or k carved out of the fused QKV
    projection - so its row stride is read from the tensor rather than assumed.
    The output is always freshly packed, so everything downstream can rely on
    it being contiguous.
    """
    batch, tokens, heads, head_dim = x.shape
    half = head_dim // 2

    # Triton is handed x.data_ptr(), which already accounts for the view's
    # storage offset, so only the strides matter here. Heads and lanes must be
    # packed; the token stride may be wider than heads*head_dim, which is the
    # normal case when x is q or k sliced out of the fused QKV projection.
    batch_stride, token_stride, head_stride, lane_stride = x.stride()
    if (head_stride != head_dim or lane_stride != 1
            or batch_stride != tokens * token_stride):
        x = x.contiguous()
        batch_stride, token_stride, head_stride, lane_stride = x.stride()

    out = torch.empty(batch, tokens, heads, head_dim, dtype=x.dtype, device=x.device)
    _norm_rope_kernel[(batch * tokens * heads,)](
        x, out, weight, cos, sin,
        tokens, heads, head_dim, half, eps, token_stride,
        HALF_BLOCK=triton.next_power_of_2(half),
        num_warps=4,
    )
    return out


@triton.jit
def _decode_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, pos_ptr,
    heads, kv_heads, capacity, head_dim, groups, scale,
    BLOCK_N: tl.constexpr, D_BLOCK: tl.constexpr,
):
    """Flash-style decode attention for a single query token.

    One program per (batch, query head), streaming the cache in blocks with an
    online softmax. Two things matter more than the fusion itself:

    * The valid length is read from device memory, not baked in. That is what
      lets one captured graph serve every step - a Python slice would freeze
      the length at capture, and the alternative, a full-capacity additive
      mask, makes SDPA read every unused slot and pushes it onto a backend
      that materialises the whole score row.
    * The loop therefore runs to the real length, so the cache tail that has
      not been written yet is never touched at all.
    """
    pid = tl.program_id(0)
    batch_id = pid // heads
    head_id = pid % heads
    kv_head = head_id // groups

    lane = tl.arange(0, D_BLOCK)
    lane_mask = lane < head_dim

    q = tl.load(q_ptr + (batch_id * heads + head_id) * head_dim + lane,
                mask=lane_mask, other=0.0).to(tl.float32)

    # pos is the slot written this step, so keys 0..pos inclusive are live.
    length = tl.load(pos_ptr) + 1
    kv_base = (batch_id * kv_heads + kv_head) * capacity * head_dim

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)

    for start in range(0, length, BLOCK_N):
        idx = start + tl.arange(0, BLOCK_N)
        key_mask = idx < length
        offsets = kv_base + idx[:, None] * head_dim + lane[None, :]
        both = key_mask[:, None] & lane_mask[None, :]

        k = tl.load(k_ptr + offsets, mask=both, other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(key_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v = tl.load(v_ptr + offsets, mask=both, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(out_ptr + (batch_id * heads + head_id) * head_dim + lane,
             (acc / l_i).to(out_ptr.dtype.element_ty), mask=lane_mask)


@triton.jit
def _kv_cache_kernel(
    qkv_ptr, ck_ptr, cv_ptr, w_ptr, cos_ptr, sin_ptr, pos_ptr,
    tokens, kv_heads, head_dim, half, eps, row_stride, k_offset, v_offset,
    capacity, HALF_BLOCK: tl.constexpr, D_BLOCK: tl.constexpr,
):
    """Normalise and rotate k, copy v, and land both in the cache.

    k and v come from the same fused projection row and end up in the same slot
    of their respective caches, so the two cache writes and k's
    normalise-and-rotate become one pass. Writing straight into the slot is
    what removes the separate index_copy_ for each.

    Programs below kv_heads handle a key head; the rest copy a value head,
    which Qwen3 never normalises.
    """
    row = tl.program_id(0)
    bt = row // (2 * kv_heads)
    slot_id = row % (2 * kv_heads)
    token = bt % tokens
    batch_id = bt // tokens

    lane = tl.arange(0, D_BLOCK)
    lane_mask = lane < head_dim
    out_dtype = ck_ptr.dtype.element_ty
    slot = tl.load(pos_ptr + token)

    if slot_id < kv_heads:
        head = slot_id
        src = bt * row_stride + k_offset + head * head_dim
        dst = ((batch_id * kv_heads + head) * capacity + slot) * head_dim

        hl = tl.arange(0, HALF_BLOCK)
        hmask = hl < half
        lo = tl.load(qkv_ptr + src + hl, mask=hmask, other=0.0)
        hi = tl.load(qkv_ptr + src + hl + half, mask=hmask, other=0.0)

        f_lo = lo.to(tl.float32)
        f_hi = hi.to(tl.float32)
        variance = (tl.sum(f_lo * f_lo, axis=0) + tl.sum(f_hi * f_hi, axis=0)) / head_dim
        inv = tl.math.rsqrt(variance + eps)
        n_lo = (f_lo * inv).to(out_dtype) * tl.load(w_ptr + hl, mask=hmask, other=0.0)
        n_hi = (f_hi * inv).to(out_dtype) * tl.load(w_ptr + hl + half, mask=hmask, other=0.0)

        angle = token * head_dim + hl
        cos_lo = tl.load(cos_ptr + angle, mask=hmask, other=0.0)
        sin_lo = tl.load(sin_ptr + angle, mask=hmask, other=0.0)
        cos_hi = tl.load(cos_ptr + angle + half, mask=hmask, other=0.0)
        sin_hi = tl.load(sin_ptr + angle + half, mask=hmask, other=0.0)

        a_lo = (n_lo.to(tl.float32) * cos_lo.to(tl.float32)).to(out_dtype)
        b_lo = ((-n_hi).to(tl.float32) * sin_lo.to(tl.float32)).to(out_dtype)
        a_hi = (n_hi.to(tl.float32) * cos_hi.to(tl.float32)).to(out_dtype)
        b_hi = (n_lo.to(tl.float32) * sin_hi.to(tl.float32)).to(out_dtype)

        tl.store(ck_ptr + dst + hl,
                 (a_lo.to(tl.float32) + b_lo.to(tl.float32)).to(out_dtype), mask=hmask)
        tl.store(ck_ptr + dst + hl + half,
                 (a_hi.to(tl.float32) + b_hi.to(tl.float32)).to(out_dtype), mask=hmask)
    else:
        head = slot_id - kv_heads
        src = bt * row_stride + v_offset + head * head_dim
        dst = ((batch_id * kv_heads + head) * capacity + slot) * head_dim
        tl.store(cv_ptr + dst + lane,
                 tl.load(qkv_ptr + src + lane, mask=lane_mask, other=0.0), mask=lane_mask)


def kv_to_cache(qkv, weight, cos, sin, positions, cache_k, cache_v,
                kv_heads, head_dim, eps, k_offset, v_offset):
    """Write this step's k and v into the cache, k normalised and rotated.

    ``qkv`` is the fused projection ``[batch, tokens, width]``; ``k_offset``
    and ``v_offset`` locate the k and v blocks inside a row. ``positions`` is a
    device int64 tensor of length ``tokens`` giving each token's cache slot.
    """
    batch, tokens, _ = qkv.shape
    if not qkv.is_contiguous():
        qkv = qkv.contiguous()
    _kv_cache_kernel[(batch * tokens * 2 * kv_heads,)](
        qkv, cache_k, cache_v, weight, cos, sin, positions,
        tokens, kv_heads, head_dim, head_dim // 2, eps,
        qkv.stride(1), k_offset, v_offset, cache_k.shape[2],
        HALF_BLOCK=triton.next_power_of_2(head_dim // 2),
        D_BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


@triton.jit
def _split_attn_kernel(
    q_ptr, k_ptr, v_ptr, acc_ptr, stat_ptr, pos_ptr,
    heads, kv_heads, capacity, head_dim, groups, scale, splits,
    BLOCK_N: tl.constexpr, D_BLOCK: tl.constexpr,
):
    """One program per (batch, query head, split) over a slice of the keys.

    The single-program kernel launches batch * query_heads programs, which is
    32 at batch 1 - a quarter of this device's SMs, most of them idle while a
    few stream the whole cache. Splitting the key range gives the scheduler
    splits times as much to place, at the cost of a second pass to merge the
    partial softmaxes.

    Each program emits its partial accumulator plus the running max and sum
    that the merge needs to reweight it.
    """
    pid = tl.program_id(0)
    split = pid % splits
    head_id = (pid // splits) % heads
    batch_id = pid // (splits * heads)
    kv_head = head_id // groups

    lane = tl.arange(0, D_BLOCK)
    lane_mask = lane < head_dim

    q = tl.load(q_ptr + (batch_id * heads + head_id) * head_dim + lane,
                mask=lane_mask, other=0.0).to(tl.float32)

    length = tl.load(pos_ptr) + 1
    # Ceiling division, so the final split takes the short remainder.
    per_split = (length + splits - 1) // splits
    begin = split * per_split
    end = tl.minimum(begin + per_split, length)

    kv_base = (batch_id * kv_heads + kv_head) * capacity * head_dim
    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([D_BLOCK], dtype=tl.float32)

    for start in range(begin, end, BLOCK_N):
        idx = start + tl.arange(0, BLOCK_N)
        key_mask = idx < end
        offsets = kv_base + idx[:, None] * head_dim + lane[None, :]
        both = key_mask[:, None] & lane_mask[None, :]

        k = tl.load(k_ptr + offsets, mask=both, other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(key_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v = tl.load(v_ptr + offsets, mask=both, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    tl.store(acc_ptr + pid * head_dim + lane, acc, mask=lane_mask)
    tl.store(stat_ptr + pid * 2, m_i)
    tl.store(stat_ptr + pid * 2 + 1, l_i)


@triton.jit
def _merge_splits_kernel(
    acc_ptr, stat_ptr, out_ptr, head_dim, splits, D_BLOCK: tl.constexpr,
):
    """Combine the per-split partial softmaxes into one output row.

    Standard rescale: shift every split onto the global max, weight its
    accumulator by exp(m_split - m_global), and divide by the summed mass. A
    split that saw no keys contributes m = -inf and l = 0, which drops out.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, D_BLOCK)
    lane_mask = lane < head_dim

    m_global = float("-inf")
    for s in range(0, splits):
        m_global = tl.maximum(m_global, tl.load(stat_ptr + (pid * splits + s) * 2))

    acc = tl.zeros([D_BLOCK], dtype=tl.float32)
    total = 0.0
    for s in range(0, splits):
        m_s = tl.load(stat_ptr + (pid * splits + s) * 2)
        l_s = tl.load(stat_ptr + (pid * splits + s) * 2 + 1)
        weight = tl.where(l_s > 0.0, tl.exp(m_s - m_global), 0.0)
        part = tl.load(acc_ptr + (pid * splits + s) * head_dim + lane,
                       mask=lane_mask, other=0.0)
        acc += part * weight
        total += l_s * weight

    tl.store(out_ptr + pid * head_dim + lane,
             (acc / total).to(out_ptr.dtype.element_ty), mask=lane_mask)


def decode_attention_split(q, cache_k, cache_v, pos, heads, kv_heads, scale, splits=8):
    """Split-K decode attention, for shapes too small to fill the device."""
    batch, _, _, head_dim = q.shape
    capacity = cache_k.shape[2]
    programs = batch * heads * splits

    partial = torch.empty(programs, head_dim, dtype=torch.float32, device=q.device)
    stats = torch.empty(programs, 2, dtype=torch.float32, device=q.device)
    out = torch.empty(batch, 1, heads * head_dim, dtype=q.dtype, device=q.device)

    d_block = triton.next_power_of_2(head_dim)
    _split_attn_kernel[(programs,)](
        q, cache_k, cache_v, partial, stats, pos,
        heads, kv_heads, capacity, head_dim, heads // kv_heads, scale, splits,
        BLOCK_N=64, D_BLOCK=d_block, num_warps=4,
    )
    _merge_splits_kernel[(batch * heads,)](
        partial, stats, out, head_dim, splits, D_BLOCK=d_block, num_warps=4,
    )
    return out


def decode_attention(q, cache_k, cache_v, pos, heads, kv_heads, scale):
    """Attention for one decode token against a fixed-capacity cache.

    ``q`` is ``[batch, 1, heads, head_dim]`` contiguous, the caches are
    ``[batch, kv_heads, capacity, head_dim]``, and ``pos`` is a device scalar
    holding the slot just written. Returns ``[batch, 1, heads * head_dim]``,
    already in the layout the output projection wants, which also removes the
    transpose-and-copy the SDPA path needs.
    """
    batch, _, _, head_dim = q.shape
    capacity = cache_k.shape[2]
    out = torch.empty(batch, 1, heads * head_dim, dtype=q.dtype, device=q.device)
    _decode_attn_kernel[(batch * heads,)](
        q, cache_k, cache_v, out, pos,
        heads, kv_heads, capacity, head_dim, heads // kv_heads, scale,
        BLOCK_N=64,
        D_BLOCK=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return out


def swiglu(fused: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` where ``fused`` is ``[..., 2 * inner]`` contiguous."""
    inner = fused.shape[-1] // 2
    rows = fused.numel() // fused.shape[-1]
    out = torch.empty(*fused.shape[:-1], inner, dtype=fused.dtype, device=fused.device)
    n_elements = rows * inner
    block = 1024
    _swiglu_kernel[(triton.cdiv(n_elements, block),)](
        fused, out, inner, n_elements, BLOCK=block, num_warps=4,
    )
    return out
