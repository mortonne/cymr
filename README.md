# cymr
[![Documentation Status](https://readthedocs.org/projects/cymr/badge/?version=latest)](https://cymr.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/mortonne/cymr.svg?branch=master)](https://travis-ci.com/mortonne/cymr)
[![codecov](https://codecov.io/gh/mortonne/cymr/branch/master/graph/badge.svg)](https://codecov.io/gh/mortonne/cymr)

Experimental implementation of the CMR model using Cython.

See the [website](https://cymr.readthedocs.io/en/latest/) for full
documentation.

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

## Design

* `cymr.network` - core model code. The network class is flexible and can be used to implement different model versions. Also includes functions for working with model "patterns" that can be used to initialize weight matrices.
* `cymr.fit` - code for fitting and simulating free recall data. The general framework is applicable for multiple models of free recall, including but not limited to CMR.
* `cymr.models` - library of model implementations. They inherit from cymr.fit.Recall, which handles some of the complications in fitting and simulating data. Implementing a new model requires only writing methods to calculate likelihood of free recall data for one subject, and to generate data for one subject.
