# MVG-Mechansim

A module to assist with the implementation of the Matrix-Variate Gaussian (MVG) mechanism for differential privacy under matrix-valued query. Please see the following paper for the detail on the MVG mechanism.

*Thee Chanyaswad, Alex Dytso, H. Vincent Poor, and Prateek Mittal. "MVG Mechanism: Differential Privacy under Matrix-Valued Query." 2018 ACM SIGSAC Conference on Computer and Communications Security (CCS'18), 2018.* (https://dl.acm.org/citation.cfm?doid=3243734.3243750)



## Functionality

The module currently implements three main functionalities:

1) `compute_precision_budget` implements the calculation of the *precision budget*. It takes the following input parameters:
	- the dimension of the matrix-valued query function: `m` x `n`,
	- `gamma` as defined in the paper,
	- the sensitivity `s2` as defined in the paper,
	- and the privacy parameters, `epsilon` and `delta`.
2) `generate_mvg_noise_via_affine_tx` implements the MVG sampler based on the affine transformation method.
3) `generate_mvg_noise_via_multivariate_gaussian` implements the MVG sampler based on the equivalence to the multivariate Gaussian.

Both implementations of the MVG sampler take the row-wise and column-wise covariance matrices as the inputs, and return an instance of the desired random matrix noise as the output.



### Prerequisites

The module is implemented in Python 2. and uses `numpy`.

## Author

* **Thee Chanyaswad**
