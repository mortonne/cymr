# cymr
Experimental implementation of the CMR model using Cython.

## Installation

```bash
python setup.py install
```

or, to install for development:

```bash
python setup.py develop
```

This will set links to the package modules so that you can edit the source code and have changes be reflected in your installation.

## Unit tests

First, install pytest:

```bash
pip install pytest
```

To run all tests:

```bash
cd cymr
pytest
```

## Benchmark

To run a speed benchmark test, first install snakeviz (`pip install snakeviz`). To run likelihood calculation with a sample dataset and then display an html report:

```bash
./benchmark
```
