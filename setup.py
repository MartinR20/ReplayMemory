from distutils.core import setup, Extension
import sys

test_module = Extension('pkg.test_ext',
                        sources=['src/test_ext/test_ext.cpp'],
                        include_dirs=['/usr/local/include'],
                        library_dirs=['/usr/local/lib/boost'],
                        runtime_library_dirs=['/usr/local/lib/boost'],
                        libraries=['boost_python3'])

setup(name='Snake',
      version='0.1',
      description='Snake',
      package_dir={'': 'src'},
      packages=['pkg'],
      ext_modules=[test_module])
