"""Re-derive the Triton kernels' index arithmetic in PyTorch.

Triton will not compile without a GPU, so the kernels themselves cannot run
here. What can run is their addressing: the same program_id decomposition, the
same pointer offsets, the same rounding sequence, expressed as tensor ops. That
catches the class of bug these kernels are actually prone to - a row index that
silently means the wrong thing once batch > 1 - long before it costs a run.

Run:  .venv/Scripts/python.exe tests/test_kernel_indexing.py
"""

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))

import engine as engine_mod  # noqa: E402

DTYPE = torch.float32  # exercise the index maths; dtype rounding is separate


def simulate_rope(x, cos, sin, heads, head_dim, tokens):
    """Mirror _rope_kernel: one program per (batch, token, head) row."""
    batch = x.shape[0]
    rows = batch * tokens * heads
    half = head_dim // 2
    flat = x.reshape(rows, head_dim)
    out = torch.empty_like(flat)

    row = torch.arange(rows)
    # This is the line under test. Rows are laid out [batch, token, head], so
    # row // heads is the flattened (batch, token) index; the angle table has
    # only `tokens` rows, so the batch component has to come back out.
    token = (row // heads) % tokens

    lane = torch.arange(half)
    x_lo = flat[:, :half]
    x_hi = flat[:, half:]
    cos_lo = cos[token][:, lane]
    sin_lo = sin[token][:, lane]
    cos_hi = cos[token][:, lane + half]
    sin_hi = sin[token][:, lane + half]

    out[:, :half] = x_lo * cos_lo + (-x_hi) * sin_lo
    out[:, half:] = x_hi * cos_hi + x_lo * sin_hi
    return out.reshape(x.shape)


def simulate_swiglu(fused):
    """Mirror _swiglu_kernel: flat offsets over [rows, inner], gate at base,
    up at base + inner."""
    inner = fused.shape[-1] // 2
    rows = fused.numel() // fused.shape[-1]
    flat = fused.reshape(rows * 2 * inner)
    n_elements = rows * inner

    offsets = torch.arange(n_elements)
    row = offsets // inner
    col = offsets % inner
    base = row * (2 * inner) + col

    gate = flat[base]
    up = flat[base + inner]
    out = torch.nn.functional.silu(gate) * up
    return out.reshape(*fused.shape[:-1], inner)


def main():
    torch.manual_seed(0)
    results = []

    for batch, tokens, heads, head_dim in [(1, 1, 8, 128), (4, 1, 8, 128),
                                           (3, 7, 4, 128), (16, 1, 2, 64)]:
        x = torch.randn(batch, tokens, heads, head_dim, dtype=DTYPE)
        cos = torch.randn(tokens, head_dim, dtype=DTYPE)
        sin = torch.randn(tokens, head_dim, dtype=DTYPE)

        want, _ = engine_mod.apply_rope(x, x, cos, sin)
        got = simulate_rope(x, cos, sin, heads, head_dim, tokens)
        ok = torch.allclose(want, got, atol=0, rtol=0)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  rope indexing b={batch} t={tokens} "
              f"h={heads} d={head_dim}"
              f"{'' if ok else f'  maxdiff {(want - got).abs().max():.3e}'}")

    for rows, inner in [(1, 64), (5, 128), (32, 9728 // 8)]:
        fused = torch.randn(rows, 2 * inner, dtype=DTYPE)
        gate, up = fused.chunk(2, dim=-1)
        want = torch.nn.functional.silu(gate) * up
        got = simulate_swiglu(fused)
        ok = torch.allclose(want, got, atol=0, rtol=0)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  swiglu indexing rows={rows} inner={inner}")

    print()
    if all(results):
        print(f"ALL {len(results)} KERNEL INDEXING CHECKS PASSED")
        return 0
    print(f"{sum(1 for r in results if not r)}/{len(results)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
