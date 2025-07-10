
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

function loglike(::ExpLikelihood, δ::AbstractVector{<:Real})
    @assert length(δ) == 1
    return δ[1]
end

function log_approx_likelihood(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_approx_like(x)
        μ_δ = mean(model_post, x)
        μ = μ_δ[1]

        # return log( exp(μ) )
        return μ
    end
end

function log_likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_like_mean(x)
        μ_δ, σ2_δ = mean_and_var(model_post, x)
        μ, σ2 = μ_δ[1], σ2_δ[1]

        # return log( exp(μ + 0.5 * σ2) )
        return μ + 0.5 * σ2
    end
end

# function log_sq_likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
#     mid = mean(bolfi.problem.domain.bounds)
#     @assert mean(model_post, mid) |> length == 1

#     function log_sq_like_mean(x)
#         μ_δ, σ2_δ = mean_and_var(model_post, x)
#         μ, σ2 = μ_δ[1], σ2_δ[1]

#         # return log( exp(2 * μ + 2 * σ2) )
#         return 2 * μ + 2 * σ2
#     end
# end

# This is a more numerically stable version, than using the `log_sq_likelihood_mean` function above.
function log_likelihood_variance(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_like_var(x::AbstractVector{<:Real})
        μ_δ, σ2_δ = mean_and_var(model_post, x)
        μ, σ2 = μ_δ[1], σ2_δ[1]

        # return log( exp(2 * (μ + σ2) + log(1 - exp(-σ2))) )
        return 2 * (μ + σ2) + log(1 - exp(-σ2))
    end
end
