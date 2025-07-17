# Data Types

## Problem & Model

The `BolfiProblem` structure contains all information about the inference problem, as well as the model hyperparameters.

```@docs
BolfiProblem
```

## Likelihood

The abstract type `Likelihood` represents the likelihood distribution of the observation `z_o`.

```@docs
Likelihood
```

The `NormalLikelihood` assumes that the observation `z_o` has been drawn from a Gaussian distribution with a known diagonal covariance matrix with the `std_obs` values on the diagonal. The simulator is used to learn the mean function.

```@docs
NormalLikelihood
```

The `LogNormalLikelihood` assumes that the observation `z_o` has been drawn from a log-normal distribution with a known diagonal covariance matrix with the `std_obs` values on the diagonal. The simulator is used to learn the mean function.

```@docs
LogNormalLikelihood
```

The `BinomialLikelihood` assumes that the observation `z_o` has been drawn from a Binomial distribution with a known number `trials`. The simulator is used to learn the probability parameter `p` as a function of the input parameters. The expectation over this likelihood (in case one wants to use `posterior_mean` and/or `posterior_variance`) is calculated via simple numerical integration on a predefined grid.

```@docs
BinomialLikelihood
```

The `ExpLikelihood` assumes that the function `f` of the [`BolfiProblem`](@ref) already maps the parameters ``x`` to the log-likelihood ``\log p(z_o|y)``. Thus, the `ExpLikelihood` only exponentiates the surrogate model output ``\delta`` to obtain the likelihood value.

```@docs
ExpLikelihood
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

## Samplers

The subtypes of `DistributionSampler` can be used to draw samples from the trained parameter posterior distribution.

```@docs
DistributionSampler
PureSampler
WeightedSampler
```

In particular, the following distribution samplers are currently provided.

```@docs
RejectionSampler
TuringSampler
AMISSampler
```

## Evaluation Metric

The subtypes of `DistributionMetric` can be used to evaluate the quality of the learned parameter posterior distribution.

```@docs
DistributionMetric
SampleMetric
PDFMetric
```

In particular, the following metrics are currently provided.

```@docs
MMDMetric
TVMetric
```

# References

[1] Gutmann, Michael U., and Jukka Cor. "Bayesian optimization for likelihood-free inference of simulator-based statistical models." Journal of Machine Learning Research 17.125 (2016): 1-47.

[2] Järvenpää, Marko, et al. "Efficient acquisition rules for model-based approximate Bayesian computation." (2019): 595-622.
