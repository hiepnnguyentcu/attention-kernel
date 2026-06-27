#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr int HEAD_DIM   = 64;
static constexpr int TILE_SIZE  = 64;
static constexpr int BLOCK_SIZE = 32;

__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int   N,
    float scale
);

torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);

    auto O = torch::zeros_like(Q);

    dim3 grid(B * H, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    const size_t smem_bytes = 2 * TILE_SIZE * HEAD_DIM * sizeof(float);
    const float scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));

    flash_attn_kernel<<<grid, block, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        N,
        scale
    );

    TORCH_CHECK(
        cudaGetLastError() == cudaSuccess,
        "flash_attn_kernel launch failed: ",
        cudaGetErrorString(cudaGetLastError())
    );

    return O;
}

__global__ void flash_attn_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    int   N,
    float scale
) {
    // ── Block 4: shared memory + thread/position setup ───────────────────────
    extern __shared__ float smem[];
    float* K_tile = smem;
    float* V_tile = smem + TILE_SIZE * HEAD_DIM;

    const int bh    = blockIdx.x;
    const int q_idx = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    const int tid   = threadIdx.x;

    if (q_idx >= N) return;

    const float* Q_bh = Q + bh * N * HEAD_DIM;
    const float* K_bh = K + bh * N * HEAD_DIM;
    const float* V_bh = V + bh * N * HEAD_DIM;
    float*       O_bh = O + bh * N * HEAD_DIM;

    // Block 5: load Q into registers
    // Block 6: initialize accumulators
    // Block 7: tile loop
    // Block 8: finalize and write output
}
