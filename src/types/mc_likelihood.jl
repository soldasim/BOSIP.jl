"""
    MonteCarloLikelihood <: Likelihood

An abstract type for simplified definition of likelihoods in comparison to the default [`Likelihood`](@ref) interface.

Consider defining a custom likelihood by subtyping `Likelihood` and implementing the full interface
to provide closed-form solutions for the integrals in `log_likelihood_mean`, `log_sq_likelihood_mean`, `log_likelihood_variance`.

## Defining a Custom Monte Carlo Likelihood

Each subtype of `MonteCarloLikelihood` must implement:
- `loglike_marginal(::MonteCarloLikelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::AbstractVector{<:Real}`
- `δ_dim(::MonteCarloLikelihood) -> ::Int`
- `mc_samples(::MonteCarloLikelihood) -> ::Int`

_(Optionally, only implement `loglike` instead of `loglike_marginal`. The marginal-likelihood functions will not work then.)_

The rest of the `Likelihood` interface is already implemented via Monte Carlo integration.
"""
abstract type MonteCarloLikelihood <: Likelihood end

"""
    δ_dim(::MonteCarloLikelihood) -> ::Int

Return the dimension of the proxy variable `δ` used in the likelihood.
"""
function δ_dim end

"""
    mc_samples(::MonteCarloLikelihood) -> ::Int

Return the number of Monte Carlo samples to use when computing the integrals
in `log_likelihood_mean`, `log_sq_likelihood_mean`, and `log_likelihood_variance`.
"""
function mc_samples end
