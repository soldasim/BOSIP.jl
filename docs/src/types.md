# Data Types

## Problem & Model

The `BosipProblem` structure contains all information about the inference problem, as well as the model hyperparameters.

```@docs
BosipProblem
```

## Likelihood

The abstract type `Likelihood` represents the likelihood distribution of the observation `z_o`.

```@docs
Likelihood
```

To implement a custom likelihood, either subtype `Likelihood` directly and implement its full interface, or alternatively subtype `MonteCarloLikelihood`, which provides a simplified interface. The full `Likelihood` interface can be used to define closed-form solutions for the integrals required to calculate the expected likelihood and its variance with respect to the surrogate model uncertainty. If one subtypes the `MonteCarloLikelihood`, these integrals are automatically approximated using MC integration.

```@docs
MonteCarloLikelihood
```

Alternatively, one can simply instantiate the `CustomLikelihood` and provide the mapping from the modeled variable to the log-likelihood. This is functionally equivalent to defining a new `MonteCarloLikelihood` subtype.

```@docs
CustomLikelihood
```

A list of some predefined likelihoods follows;

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

The `ExpLikelihood` assumes that the function `f` of the [`BosipProblem`](@ref) already maps the parameters ``x`` to the log-likelihood ``\log p(z_o|y)``. Thus, the `ExpLikelihood` only exponentiates the surrogate model output ``\delta`` to obtain the likelihood value.

```@docs
ExpLikelihood
```

## Acquisition Function

The abstract type `BosipAcquisition` represents the acquisition function.

```@docs
BosipAcquisition
```

The `MaxVar` can be used to solve LFI problems. It maximizes the posterior variance to select the next evaluation point.

```@docs
MaxVar
LogMaxVar
```

The `IMMD` acquisition maximizes the Integrated MMD as a proxy to the Expected Integrated Information Gain. That is; it attempts to minimize the entropy of the current distribution over the possible parameter posteriors (which is implicitly given by the surrogate model posterior). However, since calculating the KLD is too challenging, MMD is used instead. Beware, that there are no theoretical guarantees about this approximation though.

```@docs
IMMD
```

The `MWMV` can be used to solve LFSS problems. It maximizes the "mass-weighted mean variance" of the posteriors given by the different sensor sets.

```@docs
MWMV
```

## Termination Condition

The abstract type `BosipTermCond` represents the termination condition for the whole BOSIP procedure. Additionally, any `BOSS.TermCond` from the BOSS.jl package can be used with BOSIP.jl as well, and it will be automatically converted to a `BosipTermCond`.

```@docs
BosipTermCond
```

The most basic termination condition is the `BOSS.IterLimit`, which can be used to simply terminate the procedure after a predefined number of iterations.

BOSIP.jl provides two specialized termination conditions; the `AEConfidence`, and the `UBLBConfidence`. Both of them estimate the degree of convergence by comparing confidence regions given by two different approximations of the posterior.

```@docs
AEConfidence
UBLBConfidence
```

## Miscellaneous

The `BosipOptions` structure can be used to define miscellaneous settings of BOSIP.jl.

```@docs
BosipOptions
```

The abstract type `BosipCallback` can be derived to define a custom callback, which will be called once before the BOSIP procedure starts, and subsequently in every iteration.

For an example usage of this functionality, see the [example](https://github.com/soldasim/BOSIP.jl/tree/master/examples/simple) in the package repository, where a custom callback is used to create the plots.

```@docs
BosipCallback
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
OptMMDMetric
TVMetric
```

# References

[1] Gutmann, Michael U., and Jukka Cor. "Bayesian optimization for likelihood-free inference of simulator-based statistical models." Journal of Machine Learning Research 17.125 (2016): 1-47.

[2] Järvenpää, Marko, et al. "Efficient acquisition rules for model-based approximate Bayesian computation." (2019): 595-622.
