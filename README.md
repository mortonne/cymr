# cymr
[![PyPI version](https://badge.fury.io/py/cymr.svg)](https://badge.fury.io/py/cymr)
[![Documentation Status](https://readthedocs.org/projects/cymr/badge/?version=latest)](https://cymr.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/mortonne/cymr.svg?branch=master)](https://travis-ci.com/mortonne/cymr)
[![codecov](https://codecov.io/gh/mortonne/cymr/branch/master/graph/badge.svg)](https://codecov.io/gh/mortonne/cymr)

Package for fitting and simulating free recall data. Includes a fast 
implementation of the context maintenance and retrieval (CMR) model 
using Cython.

See the [website](https://cymr.readthedocs.io/en/latest/) for full
documentation.

## Installation

For the latest stable version:

```bash
pip install cymr
```

To get the development version:

```bash
pip install git+https://github.com/mortonne/cymr
```

To install for development, clone the repository and run: 

```bash
python setup.py develop
```

This will set links to the package modules so that you can edit the 
source code and have changes be reflected in your installation.

## Quickstart

Fit the context maintenance and retrieval model (CMR) to sample data: 

```python
from cymr import fit, cmr
# load sample data
data = fit.sample_data('Morton2013_mixed').query('subject <= 3')

# define model weights
n_items = 768
param_def, patterns = cmr.config_loc_cmr(n_items)
param_def.set_fixed(
    T=0.1, Lfc=0.15, Lcf=0.15, Afc=0, Acf=0, Dfc=0.85, Dcf=0.85, 
    P1=0.2, P2=2, B_start=0.3, B_rec=0.9, X1=0.001, X2=0.25
)
param_def.set_free(B_enc=(0, 1))

# fit the model to sample data
model = cmr.CMR()
results = model.fit_indiv(data, param_def, patterns=patterns, tol=0.1)
```

See the [documentation](https://cymr.readthedocs.io/en/latest/) for details.

## Unit tests

First, install extra packages needed for testing:

```bash
pip install -r requirements.txt .[test]
```

To run all tests (from the main repository directory)

```bash
pytest
```

## Benchmark

To run a speed benchmark test, first install snakeviz (`pip install snakeviz`). 
To run likelihood calculation with a sample dataset and then display an html 
report:

```bash
./benchmark
```
