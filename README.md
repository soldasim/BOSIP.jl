[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://soldasim.github.io/BOLFI.jl/stable/)

# BOLFI (Bayesian Optimization Likelihood-Free Inference)

The BOLFI method performs Likelihood-Free Inference by approximating the likelihood function with a Gaussian Process and greatly reduces the number of required evaluations by employing the Bayesian Optimization procedure to select new evaluation points.

The BOLFI method has been introduced in [1]. Similar approaches have been discussed in [2], [3].

See the [documentation](https://soldasim.github.io/BOLFI.jl/) for more information.

# References

[1] Michael U Gutmann, Jukka Cor, et al. “Bayesian optimization for likelihood-free
inference of simulator-based statistical models”. In: Journal of Machine Learning
Research 17.125 (2016), pp. 1–47.

[2] Bach Do and Makoto Ohsaki. “Bayesian optimization-assisted approximate Bayesian
computation and its application to identifying cyclic constitutive law of structural
steels”. In: Computers & Structures 286 (2023), p. 107111.

[3] Edward Meeds and Max Welling. “GPS-ABC: Gaussian process surrogate approxi-
mate Bayesian computation”. In: arXiv preprint arXiv:1401.2838 (2014).
