#include <torch/extension.h>

// Forward declaration — defined in attention.cu
// "Hey compiler, trust me this function exists, it'll be linked in later."
// Same idea as declaring a function signature in a .h file before defining it.
torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

// This is the function Python will actually call.
// It's like an Express.js route handler — validate inputs, call the real logic.
torch::Tensor attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    // --- Input validation (like zod/joi on an API endpoint) ---

    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    // All three must have shape [batch, heads, seq_len, head_dim]
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, N, d]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, N, d]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, N, d]");

    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(V.scalar_type() == torch::kFloat32, "Only float32 supported");

    // Shapes must be compatible
    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0), "Batch size mismatch");
    TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1), "Head count mismatch");
    TORCH_CHECK(Q.size(2) == K.size(2) && Q.size(2) == V.size(2), "Seq length mismatch");
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3), "Head dim mismatch");

    // Hand off to the CUDA kernel
    return flash_attention_forward(Q, K, V);
}

// PYBIND11_MODULE is the "export default" of C++ extensions.
// It tells Python: "this module has one function called 'forward'."
// m.def("forward", &attention_forward) is like: module.exports = { forward }
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &attention_forward,
        "Tiled flash attention forward pass (CUDA)"
    );
}
