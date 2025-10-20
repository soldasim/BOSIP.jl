"""
    MonteCarloLikelihood <: Likelihood

An abstract type for simplified definition of likelihoods in comparison to the default [`Likelihood`](@ref) interface.

Consider defining a custom likelihood by subtyping `Likelihood` and implementing the full interface
to provide closed-form solutions for the integrals in `log_likelihood_mean`, `log_sq_likelihood_mean`, `log_likelihood_variance`.

## Defining a Custom Monte Carlo Likelihood

Each subtype of `MonteCarloLikelihood` *should* implement:
- `loglike(::MonteCarloLikelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real`
- `mc_samples(::MonteCarloLikelihood) -> ::Int`

The rest of the `Likelihood` interface is already implemented via Monte Carlo integration.
"""
abstract type MonteCarloLikelihood <: Likelihood end

# function loglike end

"""
    mc_samples(::MonteCarloLikelihood) -> ::Int

Return the number of Monte Carlo samples to use when computing the integrals
in `log_likelihood_mean`, `log_sq_likelihood_mean`, and `log_likelihood_variance`.
"""
function mc_samples end
