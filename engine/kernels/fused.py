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
