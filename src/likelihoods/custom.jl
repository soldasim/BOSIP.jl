
"""
    CustomLikelihood(; log_ψ::Function)

A custom likelihood defined via providing the log-likelihood mapping ``log_ψ(δ, x) ↦ log p(z_o|δ)``,
where ``z_o`` is the observation, ``δ`` is the proxy variable modeled by the surrogate model,
and ``x`` are the input parameters (which will usually not be used for the calculation).

The parameters ``x`` are provided for special cases, where some transformation of the modeled variable
is used, which is based on the input parameters.

## Keywords
- `log_ψ::Function`: A function `log(ℓ) = log_ψ(δ, x)` computing the log-likelihood
        for a given model output `δ` and input parameters `x`.
        Here, `δ` is the proxy variable modeled by the surrogate model
        and `x` are the input parameters (which will usually not be used for the calculation).
- `mc_samples::Int = 1000`: Number of Monte Carlo samples to use when computing the expected log-likelihood
        and its variance.
"""
@kwdef struct CustomLikelihood <: MonteCarloLikelihood
    log_ψ::Function
    mc_samples::Int
end

function loglike(like::CustomLikelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return like.log_ψ(δ, x)
end

function mc_samples(like::CustomLikelihood)
    return like.mc_samples
end
