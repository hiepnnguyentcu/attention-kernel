from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attention_kernel",
    ext_modules=[
        CUDAExtension(
            name="attention_kernel",          
            sources=[
                "csrc/attention.cu",
                "csrc/bindings.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "-arch=sm_75",
                    "--use_fast_math",        
                    "-lineinfo",              
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
