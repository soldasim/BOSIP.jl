# Functions

This page contains documentation for all exported functions.

## Training

Call the main function `bolfi!` to run the BOLFI procedure, which sequentially queries the expensive blackbox simulator to learn the parameter posterior efficiently.

```@docs
bolfi!
```

## Parameter Distributions

This section contains function used to obtain the trained parameter posterior/likelihood approximations.

The `approx_posterior` function can be used to obtain the (un)normalized approximate posterior
``p(\theta|y_o) \propto p(y_o|\theta) p(\theta)`` obtained by substituting the predictive means of the GPs directly as the discrepancies.

```@docs
approx_posterior
```

The `posterior_mean` function can be used to obtain the (un)normalized expected posterior
``\mathbb{E}\left[p(\theta|y_o)\right] \propto \mathbb{E}\left[p(y_o|\theta)p(\theta)\right]``
obtained by analytically calculating the expectation of the posterior function
over the uncertainty in the prediction of the discrepancies.

```@docs
posterior_mean
```

The `posterior_variance` function can be used to obtain the (un)normalized posterior variance
``\mathbb{V}\left[p(\theta|y_o)\right] \propto \mathbb{V}\left[p(y_o|\theta)p(\theta)\right]``
caused by the uncertainty in the prediction of the discrepancies.

```@docs
posterior_variance
```

The `approx_likelihood` function can be used to obtain the approximate likelihood
``p(y_o|\theta)`` obtained by substituting the predictive means of the GPs directly as the discrepancies.

```@docs
approx_likelihood
```

The `likelihood_mean` function can be used to obtain the expected likelihood
``\mathbb{E}\left[p(y_o|\theta)\right]`` obtained by analytically calculating the expectation
of the likelihood function over the uncertainty in the prediction of the discrepancies.

```@docs
likelihood_mean
```

The `likelihood_variance` function can be used to obtain the likelihood variance
``\mathbb{V}\left[p(y_o|\theta)\right]`` caused by the uncertainty in the prediction of the discrepancies.

```@docs
likelihood_variance
```

The `evidence` function can be used to approximate the evidence ``p(y_o)``
of a given posterior function by sampling. It is advisable to use this
estimate only in low parameter dimensions, as it will require many samples
to achieve reasonable precision on high-dimensional domains.

The evidence is the normalization constant needed to obtain the normalized posterior.
The `evidence` function is used to normalize the posterior if one calls
`approx_posterior`, `posterior_mean`, or `posterior_variance` with `normalize=true`.

```@docs
evidence
```

## Sampling

The `sample_posterior` function can be used to obtain approximate samples from the trained parameter posterior.

```@docs
sample_posterior
```

The sampling is performed via the Turing.jl package. The Turing.jl package is a quite heavy dependency, so it is not loaded by default. To sample from the posterior, one has to first load Turing.jl as `using Turing`, which will also compile the `sample_posterior` function.

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
