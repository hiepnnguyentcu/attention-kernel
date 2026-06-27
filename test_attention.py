import torch
import torch.nn.functional as F
import attention_kernel

B, H, N, D = 2, 8, 128, 64
Q = torch.randn(B, H, N, D, device="cuda")
K = torch.randn(B, H, N, D, device="cuda")
V = torch.randn(B, H, N, D, device="cuda")

def naive_attention(Q, K, V):
    scale = Q.size(-1) ** -0.5
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N] — full matrix in HBM
    return torch.matmul(torch.softmax(scores, dim=-1), V)

# ── Correctness ───────────────────────────────────────────────────────────────
out_ours  = attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous())
out_ref   = F.scaled_dot_product_attention(Q, K, V)
out_naive = naive_attention(Q, K, V)

match    = torch.allclose(out_ours, out_ref, atol=1e-3)
max_err  = (out_ours - out_ref).abs().max().item()
print(f"Correctness (non-causal): {'PASS' if match else 'FAIL'}  (max error: {max_err:.6f})")

out_causal_ours = attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous(), causal=True)
out_causal_ref  = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
match_c   = torch.allclose(out_causal_ours, out_causal_ref, atol=1e-3)
max_err_c = (out_causal_ours - out_causal_ref).abs().max().item()
print(f"Correctness (causal):     {'PASS' if match_c else 'FAIL'}  (max error: {max_err_c:.6f})")

# ── Benchmark ─────────────────────────────────────────────────────────────────
def bench(fn, label, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    print(f"{label:30s}  {ms:.3f} ms")

print()
bench(lambda: naive_attention(Q, K, V),                                                          "naive (O(N²) memory)")
bench(lambda: attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous()),          "ours (tiled)")
bench(lambda: attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous(), True),    "ours (causal)")
bench(lambda: F.scaled_dot_product_attention(Q, K, V),                                          "pytorch sdpa")
