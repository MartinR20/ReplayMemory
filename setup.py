from distutils.core import setup, Extension 
import sys

_C = Extension('Snake._C',
                        sources=['src/_C/_C.cpp'],
                        include_dirs=['/usr/local/include'],
                        library_dirs=['/usr/local/lib/boost'],
                        runtime_library_dirs=['/usr/local/lib/boost'],
                        libraries=['boost_python3', 'boost_numpy3'])

setup(name='Snake',
      version='0.1',
      description='Snake',
      package_dir={'': 'src'},
      packages=['Snake'],
      install_requires=[
          'numpy',
          'gym'
      ],
      ext_modules=[_C])
