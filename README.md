# ALPUD: Active Learning from Positive and Unlabeled Data

This repository contains Python and MATLAB implementations of the ALPUD algorithm (Active Learning from Positive and Unlabeled Data), introduced and presented in the following paper:

- __Ghasemi, Alireza__, Hamid R. Rabiee, Mohsen Fadaee, Mohammad T. Manzuri, and Mohammad H. Rohban. _"Active learning from positive and unlabeled data."_ In Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference on, pp. 244-250. IEEE, 2011.

The paper can be found on [IEEE Xplore](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6137386) and downloaded from [here](https://arxiv.org/pdf/1602.07495.pdf).

## Installation

### Python

To install the Python package, you can use pip:

```bash
pip install -r requirements.txt
```

Or install the package directly:

```bash
pip install .
```

### Dependencies

The Python implementation requires:
- NumPy (>=1.19.0)
- SciPy (>=1.5.0)

## Usage

### Python

```python
import numpy as np
from alpud import compute_informativeness_with_known_prior, compute_informativeness_with_unknown_prior

# Example with 1D data
positive_data = np.array([1.0, 1.2, 1.1, 1.3, 1.15])
unlabeled_data = np.array([0.5, 2.0, 1.5, 0.8, 2.5])
query_samples = np.array([0.9, 1.0, 1.5, 2.0])

# With known prior
P = 0.3  # Prior probability of positive class
informativeness = compute_informativeness_with_known_prior(
    positive_data, unlabeled_data, P, query_samples
)
print("Informativeness (known prior):", informativeness)

# With unknown prior
informativeness = compute_informativeness_with_unknown_prior(
    positive_data, unlabeled_data, query_samples
)
print("Informativeness (unknown prior):", informativeness)
```

### MATLAB

The original MATLAB implementation is also available. Use the functions:
- `computeInformativenessWithKnownPrior.m`
- `computeInformativenessWithUnknownPrior.m`

## License

The work is published under MIT license. 

Copyright (c) 2016 Alireza Ghasemi
 
