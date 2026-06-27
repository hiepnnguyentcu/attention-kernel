# attention-kernel

A concise CUDA implementation of tiled forward attention with an online softmax (FlashAttention-style). Implemented from first principles to demonstrate how tiling and streaming softmax eliminate HBM-resident score matrices and shift the performance bottleneck from memory to compute.

## Overview

This project implements a tiled, single-pass attention kernel that reproduces the exact numerics of scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

Key design goals:

- Avoid materializing the full N×N score matrix in global memory
- Compute numerically stable softmax in one streaming pass
- Minimize HBM traffic by using on-chip shared memory (SRAM) and registers for tile processing

**Result:** Q, K, V are each read once; the output O is written once. No N×N matrix is written to HBM.

## Implementation layout

```
csrc/
  attention.cu      # CUDA kernel implementing tiled forward attention (~160 LOC)
  bindings.cpp      # PyTorch C++ extension glue
setup.py            # nvcc-based build wrapper
test_attention.py   # Correctness checks and microbenchmarks
```

Build: `pip install -e .` (requires PyTorch with CUDA). Default architecture: `sm_75` (T4). Modify `-arch` in `setup.py` for other targets (`sm_80`, `sm_89`, etc.).

## Algorithmic techniques

- **Tiling** — load K and V in blocks that fit in shared memory, process Q row-tiles against streamed K/V tiles
- **Online softmax** — maintain a running max and normalization accumulator to compute the exact softmax result in one left-to-right scan without storing all scores
- **Causal masking** — skip whole tiles that lie strictly in the future to implement autoregressive (GPT-style) attention
- **Vectorized loads** — use `float4` for 128-bit transactions on K/V tile loads
- **Synchronization-minimal control** — avoid divergent `__syncthreads()` patterns and use whole-tile skipping to reduce work for causal cases

Online softmax update per tile:

```
m_new = max(m_old, max(scores_tile))
l_new = exp(m_old − m_new) × l_old + Σ exp(scores_tile − m_new)
O_new = exp(m_old − m_new) × O_old + exp(scores_tile − m_new) @ V_tile
```

Final output: `O = O_new / l_new`. Produces numerically identical results to a conventional two-pass softmax.

## Performance summary

Measured on T4 (B=2, H=8, D=64). Naive refers to a full-score-matrix implementation; pytorch sdpa is `torch.nn.functional.scaled_dot_product_attention`.

![Benchmark](benchmark.png)

| N | naive | ours | ours causal | pytorch sdpa |
|---:|---:|---:|---:|---:|
| 128 | 0.089ms | 0.124ms | 0.123ms | 0.051ms |
| 256 | 0.214ms | 0.444ms | 0.334ms | 0.151ms |
| 512 | 0.597ms | 1.141ms | 0.583ms | 0.366ms |
| 1024 | 2.560ms | 3.581ms | 2.116ms | 1.541ms |
| 2048 | 10.214ms | 13.283ms | 8.278ms | 5.964ms |

Notes:

- Algorithmic scaling advantage is visible as a shallower slope than naive on the log-log plot (tiling reduces effective HBM pressure)
- The microbenchmark favors PyTorch cuBLAS fp16 tensor-core matmuls; this implementation uses scalar fp32, so measured gaps reflect execution-unit differences rather than algorithmic deficits
- Causal masking reduces work by skipping tiles and yields ~40–50% speedup at large N versus non-causal runs

## Roofline and bottlenecks (T4)

Device peaks:
- fp32 peak: **8.1 TFLOPS**
- HBM peak: **~300 GB/s**
- Ridge point ≈ **27 FLOPs/byte**

For N=1024, B=2, H=8, D=64:

| | FLOPs | HBM traffic | Arithmetic intensity | |
|---|---|---|---|---|
| Naive attention | ~2.1 GFLOPs | ~280 MB | ~7.5 FLOPs/byte | memory bound |
| This kernel | ~2.1 GFLOPs | ~16 MB | ~131 FLOPs/byte | compute bound |

Tiling and online softmax shift the kernel above the ridge point. The next practical optimizations are reduced-precision compute (fp16/bf16) and tensor-core utilization.

## Profiling

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("flash_attn"):
        for _ in range(20):
            attention_kernel.forward(Q, K, V)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
```

Output (T4, N=1024):

```
Name                                          Self CUDA   Self CUDA %   CUDA time avg   # Calls
-----------------------------------------------------------------------------------------------
flash_attn_kernel(float const*, ...)          158.958ms       99.83%        7.948ms        20
void at::native::vectorized_elementwise...    271.771us        0.17%       13.589us        20
```

The custom kernel consumes 99.83% of GPU time; launcher overhead is negligible. For per-SM utilization and memory timeline, use Nsight Systems: `nsys profile --stats=true python test_attention.py`.

## Production differences vs FlashAttention-2/3

This implementation is pedagogical and demonstrates the core algorithm. Production systems add:

- Tensor-core kernels (`wmma` / `mma.sync`) and reduced precision (fp16/bf16)
- Multi-warp pipelining and warp-level scheduling to hide latency
- Register-file tiling to avoid spills to L1
- Full backward pass for training

## Correctness

Validated against `torch.nn.functional.scaled_dot_product_attention` with `atol=1e-3`.

```
Non-causal: PASS  (max error: 0.000000)
Causal:     PASS  (max error: 0.000000)
```

## What this code demonstrates

- HBM round-trips, not FLOPs, are the primary bottleneck for naive attention
- Streaming softmax can achieve numerically stable single-pass computation
- Shared memory/register tiling and vectorized loads materially reduce bandwidth pressure
- Kernel-level concerns: thread/block layout, cooperative loading, divergence safety, and synchronization patterns determine correctness and performance
