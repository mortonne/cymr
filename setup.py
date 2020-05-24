from setuptools import setup, find_packages
from Cython.Build import cythonize


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='cymr',
    version='0.4.0',
    description='CMR implemented using Cython.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Neal Morton',
    author_email='mortonne@gmail.com',
    license='GPLv3',
    url='http://github.com/mortonne/cymr',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=cythonize('src/cymr/operations.pyx'),
    extras_require={
        'docs': ['sphinx', 'sphinx-bootstrap-theme', 'ipython'],
        'test': ['pytest', 'codecov', 'pytest-cov'],
    },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.8',
    ]
)
