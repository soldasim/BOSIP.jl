
"""
    ExpLikelihood(; kwargs...)

Assumes the model approximates the log-likelihood directly (as a scalar).
Only exponentiates the model prediction.

# Keywords
- `opt`: An `AcquisitionMaximizer` used to maximizer the log-likelihood
        needed for GP posterior normalization.
"""
@kwdef struct ExpLikelihood <: Likelihood
    opt::AcquisitionMaximizer
end

function loglike(::ExpLikelihood, z::AbstractVector{<:Real})
    @assert length(z) == 1
    return z[1]
end

function approx_likelihood(::ExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function approx_like(x)
        μ_z, std_z = gp_post(x)
        μ = μ_z[1]
        return exp(μ)
    end
end

function likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function like_mean(x)
        μ_z, std_z = gp_post(x)
        μ, σ = μ_z[1], std_z[1]
        return exp(μ + 0.5 * σ^2)
    end
end

function sq_likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function sq_like_mean(x)
        μ_z, std_z = gp_post(x)
        μ, σ = μ_z[1], std_z[1]
        return exp(2 * μ + 2 * σ^2)
    end
end
