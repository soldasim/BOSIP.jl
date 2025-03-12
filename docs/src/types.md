# Data Types

## Problem & Model

The `BolfiProblem` structure contains all information about the inference problem, as well as the model hyperparameters.

```@docs
BolfiProblem
```

## Likelihood

The abstract type `Likelihood` represents the likelihood distribution of the observation `y_o`.

```@docs
Likelihood
```

The `NormalLikelihood` assumes that the observation `y_o` has been drawn from a Gaussian distribution with a known diagonal covariance matrix with the `std_obs` values on the diagonal. The simulator is used to learn the mean function.

```@docs
NormalLikelihood
```

The `LogNormalLikelihood` assumes that the observation `y_o` has been drawn from a log-normal distribution with a known diagonal covariance matrix with the `std_obs` values on the diagonal. The simulator is used to learn the mean function.

```@docs
LogNormalLikelihood
```

The `BinomialLikelihood` assumes that the observation `y_o` has been drawn from a Binomial distribution with a known number `trials`. The simulator is used to learn the probability parameter `p` as a function of the input parameters. The expectation over this likelihood (in case one wants to use `posterior_mean` and/or `posterior_variance`) is calculated via simple numerical integration on a predefined grid.

```@docs
BinomialLikelihood
```

The `GutmannNormalLikelihood` is implemented according to the equations of Gutmann et al. in [1,2]. It is defined in a different way to the other `Likelihood`s. It is provided mainly for a comparison with `NormalLikelihood`.

```@docs
GutmannNormalLikelihood
```

## Acquisition Function

The abstract type `BolfiAcquisition` represents the acquisition function.

PostVarAcq, MWMVAcq, InfoGain

```@docs
BolfiAcquisition
```

The `PostVarAcq` can be used to solve LFI problems. It maximizes the posterior variance to select the next evaluation point.

```@docs
PostVarAcq
```

The `MWMVAcq` can be used to solve LFSS problems. It maximizes the "mass-weighted mean variance" of the posteriors given by the different sensor sets.

```@docs
MWMVAcq
```

## Termination Condition

The abstract type `BolfiTermCond` represents the termination condition for the whole BOLFI procedure. Additionally, any `BOSS.TermCond` from the BOSS.jl package can be used with BOLFI.jl as well, and it will be automatically converted to a `BolfiTermCond`.

```@docs
BolfiTermCond
```

The most basic termination condition is the `BOSS.IterLimit`, which can be used to simply terminate the procedure after a predefined number of iterations.

BOLFI.jl provides two specialized termination conditions; the `AEConfidence`, and the `UBLBConfidence`. Both of them estimate the degree of convergence by comparing confidence regions given by two different approximations of the posterior.

```@docs
AEConfidence
UBLBConfidence
```

## Miscellaneous

The `BolfiOptions` structure can be used to define miscellaneous settings of BOLFI.jl.

```@docs
BolfiOptions
```

The abstract type `BolfiCallback` can be derived to define a custom callback, which will be called once before the BOLFI procedure starts, and subsequently in every iteration.

For an example usage of this functionality, see the [example](https://github.com/soldasim/BOLFI.jl/tree/master/examples/simple) in the package repository, where a custom callback is used to create the plots.

```@docs
BolfiCallback
```

# References

[1] Gutmann, Michael U., and Jukka Cor. "Bayesian optimization for likelihood-free inference of simulator-based statistical models." Journal of Machine Learning Research 17.125 (2016): 1-47.

[2] Järvenpää, Marko, et al. "Efficient acquisition rules for model-based approximate Bayesian computation." (2019): 595-622.
