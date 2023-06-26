from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension




setup(
    name='rrrc_cpp',
    ext_modules=[
        CUDAExtension('rrrc_cpp', [
            'rrrc.cpp',
            'rrrc_cuda_kernel.cu',
        ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)