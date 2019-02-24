from distutils.core import setup, Extension 
import sys
import torch
from torch.utils.cpp_extension import CppExtension, BuildExtension

_C = Extension('ReplayMemory._C',
    sources=[
        'src/_C/SegmentTree.cpp',
        'src/_C/Transition.cpp',
        'src/_C/wrapper.cpp'
    ],
    include_dirs=torch.utils.cpp_extension.include_paths(),
    extra_compile_args=['-D_GLIBCXX_USE_CXX11_ABI=0', '-DTORCH_API_INCLUDE_EXTENSION_H'],
    language='c++')

setup(name='ReplayMemory',
      version='0.1',
      description='ReplayMemory',
      package_dir={'': 'src'},
      install_requires=[
          'torch',
      ],
      packages=['ReplayMemory'],
      ext_modules=[_C])