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


def simulate_decode_attention(q, cache_k, cache_v, pos, heads, kv_heads, scale, block_n=64):
    """Mirror _decode_attn_kernel: one program per (batch, query head), online
    softmax over cache blocks, length read from `pos`.

    Reproduces the pointer arithmetic and the streaming accumulation, so a
    wrong KV-head mapping or a mis-strided cache offset shows up here rather
    than as a failed run.
    """
    batch, _, _, head_dim = q.shape
    capacity = cache_k.shape[2]
    groups = heads // kv_heads
    length = int(pos.item()) + 1

    k_flat = cache_k.reshape(-1)
    v_flat = cache_v.reshape(-1)
    q_flat = q.reshape(-1)
    out = torch.empty(batch * heads * head_dim, dtype=torch.float32)

    for pid in range(batch * heads):
        batch_id = pid // heads
        head_id = pid % heads
        kv_head = head_id // groups

        qb = q_flat[(batch_id * heads + head_id) * head_dim:
                    (batch_id * heads + head_id) * head_dim + head_dim].float()
        kv_base = (batch_id * kv_heads + kv_head) * capacity * head_dim

        m_i = float("-inf")
        l_i = 0.0
        acc = torch.zeros(head_dim, dtype=torch.float32)

        for start in range(0, length, block_n):
            idx = torch.arange(start, min(start + block_n, length))
            offs = kv_base + idx[:, None] * head_dim + torch.arange(head_dim)[None, :]
            k = k_flat[offs.reshape(-1)].reshape(len(idx), head_dim).float()
            v = v_flat[offs.reshape(-1)].reshape(len(idx), head_dim).float()

            scores = (qb[None, :] * k).sum(dim=1) * scale
            m_new = max(m_i, float(scores.max()))
            alpha = torch.tensor(m_i - m_new).exp().item() if m_i != float("-inf") else 0.0
            p = (scores - m_new).exp()
            acc = acc * alpha + (p[:, None] * v).sum(dim=0)
            l_i = l_i * alpha + float(p.sum())
            m_i = m_new

        out[pid * head_dim:(pid + 1) * head_dim] = acc / l_i

    return out.reshape(batch, 1, heads * head_dim)


def simulate_norm_rope(x, weight, cos, sin, eps, heads, head_dim, tokens):
    """Mirror _norm_rope_kernel, including its stride handling.

    The kernel reads through the tensor's real token stride rather than
    assuming a packed layout, because in production x is q or k sliced out of
    the fused QKV projection and its rows are wider than heads*head_dim.
    Gathering by explicit offsets here reproduces that.
    """
    batch = x.shape[0]
    rows = batch * tokens * heads
    half = head_dim // 2

    # The kernel decomposes its program id into (batch, token, head) and then
    # walks the tensor's real token stride. Gathering by that decomposition
    # reproduces what it reads, for a packed tensor and a strided view alike.
    row = torch.arange(rows)
    bt, head = row // heads, row % heads
    b, t = bt // tokens, bt % tokens
    flat = x[b, t, head, :]

    lo, hi = flat[:, :half].float(), flat[:, half:].float()
    variance = ((lo * lo).sum(1) + (hi * hi).sum(1)) / head_dim
    inv = torch.rsqrt(variance + eps).unsqueeze(1)
    n_lo = (lo * inv).to(x.dtype) * weight[:half]
    n_hi = (hi * inv).to(x.dtype) * weight[half:]

    token = (torch.arange(rows) // heads) % tokens
    c_lo, s_lo = cos[token][:, :half], sin[token][:, :half]
    c_hi, s_hi = cos[token][:, half:], sin[token][:, half:]

    out = torch.empty_like(flat)
    out[:, :half] = n_lo * c_lo + (-n_hi) * s_lo
    out[:, half:] = n_hi * c_hi + n_lo * s_hi
    return out.reshape(x.shape)


def check_norm_rope():
    results = []
    cases = [(1, 1, 8, 128, False), (4, 1, 8, 128, False), (2, 6, 4, 128, False),
             # The layout production actually passes: a slice of a wider row.
             (2, 5, 8, 128, True), (4, 1, 8, 128, True), (1, 7, 2, 128, True)]
    for batch, tokens, heads, head_dim, strided in cases:
        torch.manual_seed(batch * 7 + tokens + int(strided))
        if strided:
            wide = torch.randn(batch, tokens, heads * head_dim * 2 + 64, dtype=DTYPE)
            x = wide[..., :heads * head_dim].view(batch, tokens, heads, head_dim)
            assert not x.is_contiguous(), "case should exercise the strided path"
        else:
            x = torch.randn(batch, tokens, heads, head_dim, dtype=DTYPE)
        w = torch.randn(head_dim, dtype=DTYPE)
        cos = torch.randn(tokens, head_dim, dtype=DTYPE)
        sin = torch.randn(tokens, head_dim, dtype=DTYPE)
        eps = 1e-6

        normed = engine_mod.rms_norm(x, w, eps)
        want, _ = engine_mod.apply_rope(normed, normed, cos, sin)
        got = simulate_norm_rope(x, w, cos, sin, eps, heads, head_dim, tokens)

        ok = torch.allclose(want, got, atol=1e-5, rtol=1e-5)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  norm+rope b={batch} t={tokens} h={heads}"
              f"{' strided' if strided else ''}"
              f"{'' if ok else f'  maxdiff {(want - got).abs().max():.3e}'}")
    return results


def simulate_split_attention(q, cache_k, cache_v, pos, heads, kv_heads, scale,
                             splits=8, block_n=64):
    """Mirror _split_attn_kernel plus _merge_splits_kernel.

    Checks the two things the split version can get wrong that the single
    version cannot: the key range each split owns (ceiling division, so the
    last split takes a short remainder and an empty split is possible), and
    the rescale that merges partial softmaxes onto a common maximum.
    """
    batch, _, _, head_dim = q.shape
    capacity = cache_k.shape[2]
    groups = heads // kv_heads
    length = int(pos.item()) + 1

    k_flat, v_flat, q_flat = cache_k.reshape(-1), cache_v.reshape(-1), q.reshape(-1)
    programs = batch * heads * splits
    partial = torch.zeros(programs, head_dim, dtype=torch.float32)
    stats = torch.zeros(programs, 2, dtype=torch.float32)

    per_split = (length + splits - 1) // splits
    for pid in range(programs):
        split = pid % splits
        head_id = (pid // splits) % heads
        batch_id = pid // (splits * heads)
        kv_head = head_id // groups

        qb = q_flat[(batch_id * heads + head_id) * head_dim:
                    (batch_id * heads + head_id) * head_dim + head_dim].float()
        kv_base = (batch_id * kv_heads + kv_head) * capacity * head_dim

        begin = split * per_split
        end = min(begin + per_split, length)
        m_i, l_i = float("-inf"), 0.0
        acc = torch.zeros(head_dim, dtype=torch.float32)

        for start in range(begin, end, block_n):
            idx = torch.arange(start, min(start + block_n, end))
            offs = kv_base + idx[:, None] * head_dim + torch.arange(head_dim)[None, :]
            k = k_flat[offs.reshape(-1)].reshape(len(idx), head_dim).float()
            v = v_flat[offs.reshape(-1)].reshape(len(idx), head_dim).float()
            scores = (qb[None, :] * k).sum(dim=1) * scale
            m_new = max(m_i, float(scores.max()))
            alpha = torch.tensor(m_i - m_new).exp().item() if m_i != float("-inf") else 0.0
            p = (scores - m_new).exp()
            acc = acc * alpha + (p[:, None] * v).sum(dim=0)
            l_i = l_i * alpha + float(p.sum())
            m_i = m_new

        partial[pid] = acc
        stats[pid, 0], stats[pid, 1] = m_i, l_i

    out = torch.empty(batch * heads, head_dim, dtype=torch.float32)
    for pid in range(batch * heads):
        block = slice(pid * splits, (pid + 1) * splits)
        m_global = float(stats[block, 0].max())
        weight = torch.where(stats[block, 1] > 0,
                             (stats[block, 0] - m_global).exp(),
                             torch.zeros(splits))
        out[pid] = (partial[block] * weight[:, None]).sum(0) / float((stats[block, 1] * weight).sum())
    return out.reshape(batch, 1, heads * head_dim)


def check_split_attention():
    results = []
    for batch, heads, kv_heads, head_dim, capacity, pos, splits in [
        (1, 8, 2, 128, 600, 511, 8),
        (4, 8, 2, 128, 128, 40, 8),
        (2, 4, 4, 64, 64, 3, 8),    # fewer keys than splits: empty splits
        (1, 8, 2, 64, 64, 0, 8),    # single key
    ]:
        torch.manual_seed(batch * 13 + pos)
        q = torch.randn(batch, 1, heads, head_dim, dtype=DTYPE)
        ck = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        cv = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        p = torch.tensor([pos], dtype=torch.int64)
        scale = head_dim**-0.5

        got = simulate_split_attention(q, ck, cv, p, heads, kv_heads, scale, splits)
        keep = pos + 1
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            ck[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            cv[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            scale=scale,
        ).transpose(1, 2).reshape(batch, 1, heads * head_dim)

        ok = torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  split attn b={batch} hq={heads} "
              f"pos={pos} splits={splits}"
              f"{'' if ok else f'  maxdiff {(got - ref).abs().max():.3e}'}")
    return results


def simulate_split_matmul(x, w, splits, block_n=64):
    """Mirror _skinny_split_kernel + _reduce_splits_kernel.

    The partitioning is ceiling division over K, so the last split takes a
    short remainder and, when splits exceed K blocks, some splits own nothing
    at all. Those have to contribute exactly zero rather than reading past the
    end or double-counting.
    """
    rows, k = x.shape
    n = w.shape[0]
    per_split = -(-k // splits)
    part = torch.zeros(splits, rows, n, dtype=torch.float32)

    for pid_k in range(splits):
        begin = pid_k * per_split
        end = min(begin + per_split, k)
        if end <= begin:
            continue
        part[pid_k] = x[:, begin:end].float() @ w[:, begin:end].float().T

    return part.sum(0)


def simulate_gqa_attention(q, cache_k, cache_v, pos, heads, kv_heads, scale, block_n=64):
    """Mirror _decode_attn_gqa_kernel: one program per KV head covering all of
    its query heads.

    The claim under test is that query heads kv_head*groups + 0..groups-1 are
    exactly the heads that map to kv_head under h // groups. If that mapping is
    off, every head still produces a plausible-looking vector - it is just
    attending with the wrong keys - so this is checked against SDPA directly.
    """
    batch, _, _, head_dim = q.shape
    capacity = cache_k.shape[2]
    groups = heads // kv_heads
    length = int(pos.item()) + 1
    out = torch.zeros(batch * heads, head_dim, dtype=torch.float32)

    for pid in range(batch * kv_heads):
        batch_id = pid // kv_heads
        kv_head = pid % kv_heads
        rows = [batch_id * heads + kv_head * groups + g for g in range(groups)]
        qg = q.reshape(-1, head_dim)[rows].float()          # [groups, D]

        k_all = cache_k[batch_id, kv_head, :length, :].float()
        v_all = cache_v[batch_id, kv_head, :length, :].float()

        m_i = torch.full((groups,), float("-inf"))
        l_i = torch.zeros(groups)
        acc = torch.zeros(groups, head_dim)

        for start in range(0, length, block_n):
            k = k_all[start:start + block_n]
            v = v_all[start:start + block_n]
            scores = (qg @ k.T) * scale
            m_new = torch.maximum(m_i, scores.max(dim=1).values)
            alpha = torch.where(torch.isinf(m_i), torch.zeros_like(m_i), (m_i - m_new).exp())
            p = (scores - m_new[:, None]).exp()
            acc = acc * alpha[:, None] + p @ v
            l_i = l_i * alpha + p.sum(dim=1)
            m_i = m_new

        out[rows] = acc / l_i[:, None]

    return out.reshape(batch, 1, heads * head_dim)


def check_gqa_attention():
    results = []
    for batch, heads, kv_heads, head_dim, capacity, pos in [
        (1, 8, 2, 128, 64, 40),
        (4, 32, 8, 128, 600, 511),   # the real head ratio
        (2, 8, 8, 64, 40, 9),        # no grouping at all
        (3, 8, 1, 64, 32, 0),        # every head shares one, single key
    ]:
        torch.manual_seed(batch * 41 + pos)
        q = torch.randn(batch, 1, heads, head_dim, dtype=DTYPE)
        ck = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        cv = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        p = torch.tensor([pos], dtype=torch.int64)
        scale = head_dim**-0.5

        got = simulate_gqa_attention(q, ck, cv, p, heads, kv_heads, scale)
        keep = pos + 1
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            ck[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            cv[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            scale=scale,
        ).transpose(1, 2).reshape(batch, 1, heads * head_dim)

        ok = torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  gqa attn b={batch} hq={heads} hkv={kv_heads} "
              f"pos={pos}{'' if ok else f'  maxdiff {(got - ref).abs().max():.3e}'}")
    return results


def check_split_matmul():
    results = []
    for rows, n, k, splits in [(1, 2560, 9728, 8), (16, 6144, 2560, 8),
                               (4, 2560, 4096, 4), (1, 64, 64, 8),
                               (2, 128, 5, 8)]:  # more splits than K: empty splits
        torch.manual_seed(rows * 17 + k)
        x = torch.randn(rows, k, dtype=DTYPE)
        w = torch.randn(n, k, dtype=DTYPE)

        got = simulate_split_matmul(x, w, splits)
        ref = torch.nn.functional.linear(x, w).float()

        ok = torch.allclose(got, ref, atol=1e-3, rtol=1e-3)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  split matmul rows={rows} n={n} k={k} "
              f"splits={splits}"
              f"{'' if ok else f'  maxdiff {(got - ref).abs().max():.3e}'}")
    return results


def check_decode_attention():
    results = []
    for batch, heads, kv_heads, head_dim, capacity, pos in [
        (1, 8, 2, 128, 40, 11),
        (4, 8, 2, 128, 40, 23),
        (2, 4, 4, 64, 80, 64),   # no grouping
        (3, 8, 1, 64, 32, 0),    # single key, single KV head
    ]:
        torch.manual_seed(batch * 31 + pos)
        q = torch.randn(batch, 1, heads, head_dim, dtype=DTYPE)
        ck = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        cv = torch.randn(batch, kv_heads, capacity, head_dim, dtype=DTYPE)
        p = torch.tensor([pos], dtype=torch.int64)
        scale = head_dim**-0.5

        got = simulate_decode_attention(q, ck, cv, p, heads, kv_heads, scale)

        keep = pos + 1
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            ck[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            cv[:, :, :keep, :].repeat_interleave(heads // kv_heads, dim=1),
            scale=scale,
        )
        ref = ref.transpose(1, 2).reshape(batch, 1, heads * head_dim)

        ok = torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  decode attn b={batch} hq={heads} "
              f"hkv={kv_heads} d={head_dim} cap={capacity} pos={pos}"
              f"{'' if ok else f'  maxdiff {(got - ref).abs().max():.3e}'}")
    return results


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

    results.extend(check_norm_rope())
    results.extend(check_decode_attention())
    results.extend(check_split_attention())
    results.extend(check_split_matmul())
    results.extend(check_gqa_attention())

    print()
    if all(results):
        print(f"ALL {len(results)} KERNEL INDEXING CHECKS PASSED")
        return 0
    print(f"{sum(1 for r in results if not r)}/{len(results)} FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
