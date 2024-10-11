# Functions

This page contains documentation for all exported functions.

## Main Function

Call the main function `bolfi!` to run the BOLFI procedure. 

```@docs
bolfi!
```

## Parameter Posterior

This section contains function used to obtain the parameter posterior approximations.

The `approx_posterior` function can be used to obtain the (normalized) approximate
posterior obtained by substituting the predictive mean of the GPs directly as the discrepancies.

```@docs
approx_posterior
```

The `posterior_mean` function can be used to obtain the (normalized) expected posterior obtained by analytically calculating the expectation of the posterior function over the uncertainty in the prediction of the discrepancies.

```@docs
posterior_mean
```

The `posterior_variance` function can be used to obtain the (normalized) posterior variance caused by the uncertainty in the prediction of the discrepancies.

```@docs
posterior_variance
```

The `evidence` function can be used to approximate the evidence ``p(y_o)``
of a given posterior function by sampling. It is advisable to use this
estimate only in low parameter dimensions, as it will require many samples
to achieve reasonable precision on high-dimensional domains.

```@docs
evidence
```

## Predictive Posterior of the GPs

This section contains functions used to transform the GP predictive posterior.
They all receive the GP posterior as a vector-valued function `(x) -> (μ, σ)`,
and return some modiftication of this function.

The `gp_mean` ignores the prediction unceratinty by zeroing-out the predictive deviation.

```@docs
gp_mean
```

The `gp_bound` predicts the mean plus some (positive or negative) number of standard deviations.

```@docs
gp_bound
```

The `gp_quantile` predicts the `q`-th quantile of the predictive distribution.

```@docs
gp_quantile
```

## Confidence Sets

This section contains function used to extract approximate confidence sets from the posterior. It is advised to use these approximations only with low-dimensional parameter domains, as they will require many samples to reach reasonable precision in high-dimensional domains.

The `find_cutoff` function can be used to estimate some confidence set of a given posterior function.

```@docs
find_cutoff
```

The `approx_cutoff_area` function can be used to estimate the ratio of the area
of a confidence set given by sum cutoff constant (perhaps found by `find_cutoff`)
and the whole domain.

```@docs
approx_cutoff_area
```

The `set_iou` function can be used to estimate the intersection-over-union (IoU)
value between two sets.

```@docs
set_iou
```
