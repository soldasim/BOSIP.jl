
"""
    SqExpLikelihood(; kwargs...)

Assumes the model approximates the square root of the log-likelihood (as a scalar).
Squares and then exponentiates the model prediction to obtain the likelihood.

# Keywords
- `opt`: An `AcquisitionMaximizer` used to maximizer the log-likelihood
        needed for GP posterior normalization.
"""
@kwdef struct SqExpLikelihood <: Likelihood
    opt::AcquisitionMaximizer
end

function loglike(::SqExpLikelihood, z::AbstractVector{<:Real})
    @assert length(z) == 1
    return z[1]^2
end

function approx_likelihood(::SqExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function approx_like(x)
        μ_z, std_z = gp_post(x)
        μ = μ_z[1]
        return exp(μ^2)
    end
end

function likelihood_mean(::SqExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function like_mean(x)
        μ_z, std_z = gp_post(x)
        μ, σ = μ_z[1], std_z[1]
        return (1 / sqrt(1 + σ^2)) * exp(-(1/2) * (μ^2 / (1 + σ^2)))
    end
end

function sq_likelihood_mean(::SqExpLikelihood, bolfi::BolfiProblem, gp_post)
    mid = mean(bolfi.problem.domain.bounds)
    @assert gp_post(mid)[1] |> length == 1

    function sq_like_mean(x)
        μ_z, std_z = gp_post(x)
        μ, σ = μ_z[1], std_z[1]
        return (1 / sqrt(1 + 2 * σ^2)) * exp(-(1/2) * (μ^2 / (σ^2 * (1 + 2 * σ^2))))
    end
end
