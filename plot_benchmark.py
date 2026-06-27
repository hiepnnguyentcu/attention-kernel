import torch
import torch.nn.functional as F
import attention_kernel
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

B, H, D = 2, 8, 64
N_values = [128, 256, 512, 1024, 2048]

results = {k: [] for k in ["naive", "ours", "causal", "sdpa"]}

def bench(fn, iters=100):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def naive_attention(Q, K, V):
    s = Q.size(-1) ** -0.5
    return torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * s, dim=-1), V)

print("Benchmarking...")
for N in N_values:
    Q = torch.randn(B, H, N, D, device="cuda")
    K = torch.randn(B, H, N, D, device="cuda")
    V = torch.randn(B, H, N, D, device="cuda")
    Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()

    results["naive"].append(bench(lambda: naive_attention(Q, K, V)))
    results["ours"].append(bench(lambda: attention_kernel.forward(Qc, Kc, Vc)))
    results["causal"].append(bench(lambda: attention_kernel.forward(Qc, Kc, Vc, True)))
    results["sdpa"].append(bench(lambda: F.scaled_dot_product_attention(Q, K, V)))
    print(f"  N={N} done")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(N_values, results["naive"],  "o--", color="#e74c3c", label="Naive O(N²)")
ax.plot(N_values, results["ours"],   "o-",  color="#3498db", label="Ours (tiled, fp32)", linewidth=2)
ax.plot(N_values, results["causal"], "o-",  color="#2ecc71", label="Ours (causal)", linewidth=2)
ax.plot(N_values, results["sdpa"],   "o-",  color="#9b59b6", label="PyTorch SDPA (FlashAttn-2)", linewidth=2)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks(N_values)
ax.set_xlabel("Sequence length N", fontsize=12)
ax.set_ylabel("Time (ms)", fontsize=12)
ax.set_title("Attention kernel scaling  ·  B=2, H=8, D=64  ·  T4 GPU", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig("benchmark.png", dpi=150)
print("Saved benchmark.png")
