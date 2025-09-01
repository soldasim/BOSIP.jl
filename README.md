[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://soldasim.github.io/BOSIP.jl/stable/)

# BOSIP (Bayesian Optimization for Simulator Inverse Problems)

The BOSIP method solves Bayesian inverse problems with the forward model represented by a simulator by learning the whole posterior distribution. A probabilistic surrogate model is used to alleviate the need to evaluate the expensive simulation, and the Bayesian optimization procedure is used to minimize the number of required simulations.

Similar approaches have been discussed in [1,2,3,4,5].

See the [documentation](https://soldasim.github.io/BOSIP.jl/) for more information.

# References

[1] M. J¨arvenp¨a¨a, M. U. Gutmann, A. Vehtari, P. Marttinen, Parallel gaussian process surrogate bayesian inference
with noisy likelihood evaluations (2021).

[2] O. S¨urer, Batch sequential experimental design for calibration of stochastic simulation models, Technometrics (just-
accepted) (2025) 1–22.

[3] P. Villani, J. Unger, M. Weiser, Adaptive gaussian process regression for bayesian inverse problems, arXiv preprint
arXiv:2404.19459 (2024).

[4] H. Wang, J. Li, Adaptive gaussian process approximation for bayesian inference with expensive likelihood func-
tions, Neural computation 30 (11) (2018) 3072–3094.

[5] A. L. Teckentrup, Convergence of gaussian process regression with estimated hyper-parameters and applications in
bayesian inverse problems, SIAM/ASA Journal on Uncertainty Quantification 8 (4) (2020) 1310–1337.

# Citation

If you use this package, please cite it using the provided `CITATION.cff` file.
