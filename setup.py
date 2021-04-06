import os
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_bases = ['src/cymr/operations', 'src/cymr/lba']
ext_list = []
for ext_base in ext_bases:
    ext_name = 'cymr.' + os.path.basename(ext_base)
    if os.path.exists(ext_base + '.pyx'):
        # for simplicity, always use .pyx if available
        ext = cythonize(ext_base + '.pyx')[0]
    elif os.path.exists(ext_base + '.c'):
        # if no .pyx and have .c, create an extension from that
        ext = Extension(ext_name, [ext_base + '.c'])
    else:
        raise IOError(f'No file for extension {ext_name} matching {ext_base}.*')
    ext_list.append(ext)

setup(ext_modules=ext_list)
