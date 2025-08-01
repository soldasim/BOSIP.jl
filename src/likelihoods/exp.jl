
"""
    ExpLikelihood(; kwargs...)

Assumes the model approximates the log-likelihood directly (as a scalar).
Only exponentiates the model prediction.
"""
@kwdef struct ExpLikelihood <: Likelihood end

function loglike(::ExpLikelihood, y::AbstractVector{<:Real})
    @assert length(y) == 1
    return y[1]
end

function log_likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, σ2_y = mean_and_var(model_post, x)
        μ, σ2 = μ_y[1], σ2_y[1]

        # return log( exp(μ + 0.5 * σ2) )
        return μ + 0.5 * σ2
    end
end

# function log_sq_likelihood_mean(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
#     mid = mean(bolfi.problem.domain.bounds)
#     @assert mean(model_post, mid) |> length == 1

#     function log_sq_like_mean(x::AbstractVector{<:Real})
#         μ_y, σ2_y = mean_and_var(model_post, x)
#         μ, σ2 = μ_y[1], σ2_y[1]

#         # return log( exp(2 * μ + 2 * σ2) )
#         return 2 * μ + 2 * σ2
#     end
# end

# This is a more numerically stable version, than using the `log_sq_likelihood_mean` function above.
function log_likelihood_variance(::ExpLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    mid = mean(bolfi.problem.domain.bounds)
    @assert mean(model_post, mid) |> length == 1

    function log_like_var(x::AbstractVector{<:Real})
        μ_y, σ2_y = mean_and_var(model_post, x)
        μ, σ2 = μ_y[1], σ2_y[1]

        # return log( exp(2 * (μ + σ2) + log(1 - exp(-σ2))) )
        return 2 * (μ + σ2) + log(1 - exp(-σ2))
    end
end
