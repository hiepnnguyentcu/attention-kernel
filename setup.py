from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDAExtension knows how to compile .cu files using nvcc (the NVIDIA compiler)
# and .cpp files using your regular C++ compiler, then link them together into
# a single .so file that Python can import.
#
# Think of this like webpack — it takes your source files, compiles them, and
# bundles the output into something the runtime (Python) can load.

setup(
    name="attention_kernel",
    ext_modules=[
        CUDAExtension(
            name="attention_kernel",          # import name in Python
            sources=[
                "csrc/attention.cu",          # the actual GPU kernel
                "csrc/attention.cpp",         # the Python↔C++ glue
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "-arch=sm_80",            # Ampere (A100/RTX 3090); change to sm_86 for RTX 3080
                    "--use_fast_math",        # approximate exp/sqrt — fine for attention
                    "-lineinfo",              # keeps line numbers in GPU profiler output
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
