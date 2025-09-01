
# BOSIP.jl

BOSIP stands for "Bayesian Optimization for Simulator Inverse Problems". BOSIP.jl provides a general algorithm, which uses the Bayesian optimization procedure to solve Bayesian inverse problems. The package is inspired by the papers [1,2,3,4,5], which explored the idea of using Bayesian optimization and Gaussian processes for solving simulator inverse problems.

The BOSIP method is based on Bayesian optimization. BOSIP.jl depends heavily on the BOSS.jl package which handles the underlying Bayesian optimization. As such, the [BOSS.jl documentation](https://soldasim.github.io/BOSS.jl/stable/) can also be a useful resource when working with BOSIP.jl.

## References

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
