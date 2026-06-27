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

    // ── Block 5: load Q into registers ───────────────────────────────────────
    float q_reg[HEAD_DIM];
    const float* q_ptr = Q_bh + q_idx * HEAD_DIM;
    for (int i = 0; i < HEAD_DIM; i++)
        q_reg[i] = q_ptr[i];

    // ── Block 6: initialize accumulators ─────────────────────────────────────
    float m = -INFINITY;
    float l = 0.0f;
    float o_reg[HEAD_DIM] = {};

    // ── Block 7: tile loop ────────────────────────────────────────────────────
    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        const int tile_len = min(TILE_SIZE, N - tile_start);

        // 7a+7b: all threads in the block cooperate to load one K tile and one V tile
        for (int i = tid; i < TILE_SIZE * HEAD_DIM; i += BLOCK_SIZE) {
            const int row        = i / HEAD_DIM;
            const int col        = i % HEAD_DIM;
            const int global_row = tile_start + row;
            K_tile[i] = (global_row < N) ? K_bh[global_row * HEAD_DIM + col] : 0.0f;
            V_tile[i] = (global_row < N) ? V_bh[global_row * HEAD_DIM + col] : 0.0f;
        }
        __syncthreads();

        // 7c: each thread computes dot(q_reg, K_tile[j]) for every key in the tile
        float scores[TILE_SIZE];
        for (int j = 0; j < tile_len; j++) {
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                dot += q_reg[d] * K_tile[j * HEAD_DIM + d];
            scores[j] = dot * scale;
        }

        // 7d: online softmax — find new max across this tile's scores
        float m_new = m;
        for (int j = 0; j < tile_len; j++)
            m_new = fmaxf(m_new, scores[j]);

        // 7d: rescale old accumulations to the new max, then fold in this tile
        const float exp_diff = expf(m - m_new);
        float l_new = l * exp_diff;

        for (int d = 0; d < HEAD_DIM; d++)
            o_reg[d] *= exp_diff;

        for (int j = 0; j < tile_len; j++) {
            const float p = expf(scores[j] - m_new);
            l_new += p;
            for (int d = 0; d < HEAD_DIM; d++)
                o_reg[d] += p * V_tile[j * HEAD_DIM + d];
        }

        m = m_new;
        l = l_new;

        // 7e: barrier before the next tile overwrites shared memory
        __syncthreads();
    }

    // ── Block 8: finalize and write output ───────────────────────────────────
    const float inv_l = 1.0f / l;
    float* out_ptr = O_bh + q_idx * HEAD_DIM;
    for (int d = 0; d < HEAD_DIM; d++)
        out_ptr[d] = o_reg[d] * inv_l;
}
