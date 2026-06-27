import torch
import torch.nn.functional as F
import attention_kernel

B, H, N, D = 2, 8, 128, 64
Q = torch.randn(B, H, N, D, device="cuda")
K = torch.randn(B, H, N, D, device="cuda")
V = torch.randn(B, H, N, D, device="cuda")

# ── Correctness ───────────────────────────────────────────────────────────────
out_ours = attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous())
out_ref  = F.scaled_dot_product_attention(Q, K, V)

match = torch.allclose(out_ours, out_ref, atol=1e-3)
max_err = (out_ours - out_ref).abs().max().item()
print(f"Correctness: {'PASS' if match else 'FAIL'}  (max error: {max_err:.6f})")

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

bench(lambda: attention_kernel.forward(Q.contiguous(), K.contiguous(), V.contiguous()),
      "ours")
bench(lambda: F.scaled_dot_product_attention(Q, K, V),
      "pytorch sdpa")
