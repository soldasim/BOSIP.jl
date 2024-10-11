
# Likelihood-Free Inference Problem

Likelihood-free inference (LFI), also known as simulation-based inference (SBI), is methodology used to solve the inverse problem in cases where the evaluation of the forward model is prohibitively expensive. Also, LFI methods aim to learn the posterior distribution of the parameters (the target of inference) instead of finding single "optimal" parameter values.

This section formally introduces the general LFI problem as considered in BOLFI.jl.

## Definitions

Let ``\theta \in \mathbb{R}^n`` be parameters of interest, and ``y \in \mathbb{R}^m`` be some observable quantities.

Let the noisy experiment ``f`` be defined as
```math
f(\theta) = f_t(\theta) + \epsilon_f = y \;.
```
The experiment consists of a deterministic mapping ``f_t: \mathbb{R}^n \rightarrow \mathbb{R}^m``, and a random observation noise ``\epsilon_f \sim \mathcal{N}(0, \Sigma_f)``.

Let the simulator ``g`` be defined as
```math
g(\theta) = g_t(\theta) + \epsilon_g = y \;.
```
The simulator consists of a deterministic mapping ``g_t: \mathbb{R}^n \rightarrow \mathbb{R}^m``, and a random simulation noise ``\epsilon_g \sim \mathcal{N}(0, \Sigma_g)``. We assume that the simulator approximates the generative model up to the noise, i.e.
```math
g_t(\theta) \approx f_t(\theta) \;.
```

## The Problem

The problem is defined as follows. We have performed the experiment
```math
f(\theta^*) = y^* + \epsilon_f = y_o \;,
```
and obtained a noisy observation ``y_o``. Our goal is to infer the unknown parameters ``\theta^*``. Even better, we would like to learn the whole posterior parameter distribution ``p(\theta|y_o)``.

## Assumptions

We assume;
- The simulator approximates the experiment: ``g_t(\theta) \approx f_t(\theta)``.
- The observation dimensions to be independent. I.e. the noise covariance matrices ``\Sigma_f, \Sigma_g`` are diagonal. _(This is an additional assumption required by BOLFI.jl, which may often not hold. In case the observation dimensions are dependent, one could construct some summary statistics for each set of dependent dimensions in order to obtain fewer independent observations.)_
- Both the experiment noise and simulation noise to be Gaussian and homoscedastic.

We do not know;
- The true mappings ``f_t, g_t``.

We know;
- We can point-wise evaluate the simulator ``g``.
- The experiment noise covariances ``\Sigma_f``.
- The parameter prior ``p(\theta)``. _(Usually it is reasonable to use a weak prior. In case we have substantial expert knowledge about the domain, we can provide it via a stronger prior.)_

We may know;
- The simulation noise covariances ``\Sigma_g``. _(They can be estimated by BOLFI.jl, or provided.)_

## The BOLFI Method

Our goal is to learn the parameter posterior ``p(\theta|y_o)``. The posterior can be expressed using the likelihood ``p(y_o|\theta)``, prior ``p(\theta)``, and evidence ``p(y_o)`` via the Bayes' rule as
```math
p(\theta|y_o) = \frac{p(y_o|\theta) p(\theta)}{p(y_o)} \;.
```
The prior ``p(\theta)`` is known. The evidence ``p(y_o)`` is just a normalization constant, and is often unimportant. We mainly need to learn the likelihood
```math
p(y_o|\theta) = \mathcal{N}(f_t(\theta), \Sigma_f)|_{y_o} \;.
```
The covariances ``\Sigma_f`` are known, but we need to learn the mapping ``f_t``. We will approximate it based on data queried from the simulator ``g``, using the assumption ``g_t(\theta) \approx f_t(\theta)``. This way, we obtain an approximate posterior up to the normalization constant.

First, we rewrite the likelihood by abusing the assumption of a diagonal covariance matrix ``\Sigma_f`` as a product of the likelihoods of the individual observation dimensions ``j = 1,...,m``. This gives
```math
p(y_o|\theta) = \prod\limits_{j=1}^{m} \mathcal{N}(f_t(\theta)^{[j]} - y_o^{[j]}, (\sigma_f^{[j]})^2)|_0 \;,
```
where the superscript ``[j]`` refers to the ``j``-th observation dimension. _(For example, ``y_o^[j] \in \mathbb{R}`` is the ``j``-th element of the vector ``y_o``.)_

In order to approximate the likelihood we train ``m`` Gaussian processes to predict the discrepancies
```math
\delta^{[j]}(\theta) = g_t^{[j]}(\theta) - y_o^{[j]} \approx f_t^{[j]}(\theta) - y_o^{[j]}\;.
```
The newly defined stochastic functions ``\delta^{[j]}`` can be queried for new data as
```math
\delta(\theta) = g(\theta) - y_o \;,
```
by using the noisy simulator ``g``.

Given _infinite_ data, the predictive distribution of the ``j``-th GP would converge to
```math
\mathcal{N}(\mu_\delta^{[j]}(\theta), (\sigma_\delta^{[j]}(\theta))^2) \approx \mathcal{N}(g_t^{[j]}(\theta) - y_o, (\sigma_g^{[j]})^2) \;.
```
In other words, the predictive mean would approximate the true simulator outputs as
```math
\mu_\delta^{[j]}(\theta) \approx g_t^{[j]}(\theta) - y_o \;,
```
and the predictive deviation would approximate the simulation noise deviation as
```math
\sigma_\delta^{[j]}(\theta) \approx \sigma_g^{[j]} \;.
```
Thus we could approximate the likelihood by substituting ``\mu_\delta^{[j]}(\theta)`` into the equiation as
```math
p(y_o|\theta) \approx \prod\limits_{j=1}^{m} \mathcal{N}(\mu_\delta^{[j]}(\theta), (\sigma_f^{[j]})^2)|_0 \;,
```
because
```math
\mu_\delta^{[j]}(\theta) \approx g_t(\theta)^{[j]} - y_o^{[j]} \approx f_t(\theta)^{[j]} - y_o^{[j]}
```
holds. _(This posterior approximation can be obtained by calling the `approx_posterior` function.)_

However, we do not have infinite data. In case we have only a small dataset, the predictive deviation ``\sigma_\delta^{[j]}`` is not converged to the true experiment noise ``\sigma_g^{[j]}(\theta)``, as it also "contains" our uncertainty about the prediction ``\mu_\delta^{[j]}(\theta)``. Thus a more meaningful estimate of the likelihood is achieved by taking in consideration the uncertainty, and calculating the expected likelihood values
```math
\mathbb{E}\left[ p(y_o|\theta) \right] \approx \prod\limits_{j=1}^{m} \mathcal{N}(\mu_\delta^{[j]}(\theta), (\sigma_f^{[j]})^2 + (\sigma_\delta^{[j]}(\theta))^2)|_0 \;.
```
_(The derivation of the expression is skipped here. It can be, however, derived easily. This approximation of the posterior can be obtained by calling the `posterior_mean` function.)_

Then we obtain the expected parameter posterior ``\mathbb{E}\left[p(\theta|y_o)\right]`` (up to a normalization constant) simply by multiplying this expectation with the known prior ``p(\theta)``.

Similarly, we can estimate the uncertainty of our posterior approximation as the variance ``\mathbb{V}\left[p(\theta|y_o)\right]``, which can be used as a primitive acquisition function used to select new data. _(The derivation is skipped here. The posterior variance can be obtained by calling the `posterior_variance` function.)_
