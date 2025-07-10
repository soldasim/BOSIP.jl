
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

function loglike(::SqExpLikelihood, δ::AbstractVector{<:Real})
    @assert length(δ) == 1
    return δ[1]^2
end

function log_approx_likelihood(::SqExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_approx_like(x)
        μ_δ, std_δ = mean_and_std(model_post, x)
        μ = μ_δ[1]

        # return log( exp(μ^2) )
        return μ^2
    end
end

function log_likelihood_mean(::SqExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_like_mean(x)
        μ_δ, std_δ = mean_and_std(model_post, x)
        μ, σ = μ_δ[1], std_δ[1]

        # return log( (1 / sqrt(1 + σ^2)) * exp(-(1/2) * (μ^2 / (1 + σ^2))) )
        return (-(1/2) * log(1 + σ^2)) + (-(1/2) * (μ^2 / (1 + σ^2)))
    end
end

function log_sq_likelihood_mean(::SqExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_sq_like_mean(x)
        μ_δ, std_δ = mean_and_std(model_post, x)
        μ, σ = μ_δ[1], std_δ[1]

        # return log( (1 / sqrt(1 + 2 * σ^2)) * exp(-(1/2) * (μ^2 / (σ^2 * (1 + 2 * σ^2)))) )
        return (-(1/2) * log(1 + 2 * σ^2)) + (-(1/2) * (μ^2 / (σ^2 * (1 + 2 * σ^2))))
    end
end
