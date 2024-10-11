
# BOLFI.jl

BOLFI stands for "Bayesian Optimization Likelihood-Free Inference". BOLFI.jl provides a high-level algorithm, which uses Bayesian optimization-like procedure to solve likelihood-free inference problems. The package is inspired by the papers [1,2,3], which explored the idea of using Bayesian optimization for solving LFI problems. Notably, the core of BOLFI.jl follows the method described in [3].

Additionally, BOLFI.jl provides a method to solve likelihood-free sensor selection (LFSS) problems. This is a newly formulated problem based on LFI.

The BOLFI method is based on Bayesian optimization. BOLFI.jl depends heavily on the BOSS.jl package which handles underlying Bayesian optimization. As such, the [BOSS.jl documentation](https://soldasim.github.io/BOSS.jl/stable/) can also be a useful resource when working with BOLFI.jl.

## References

[1] Edward Meeds and Max Welling. “GPS-ABC: Gaussian process surrogate approximate Bayesian computation”. In: arXiv preprint arXiv:1401.2838 (2014).

[2] Michael U Gutmann, Jukka Cor, et al. “Bayesian optimization for likelihood-free
inference of simulator-based statistical models”. In: Journal of Machine Learning
Research 17.125 (2016), pp. 1–47.

[3] Bach Do and Makoto Ohsaki. “Bayesian optimization-assisted approximate Bayesian
computation and its application to identifying cyclic constitutive law of structural
steels”. In: Computers & Structures 286 (2023), p. 107111.


