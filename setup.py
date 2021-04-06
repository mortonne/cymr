from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize(['src/cymr/operations.pyx', 'src/cymr/lba.pyx']))
