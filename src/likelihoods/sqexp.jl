
"""
    SqExpLikelihood(; kwargs...)

Assumes the model approximates the square root of the log-likelihood (as a scalar).
Squares and then exponentiates the model prediction to obtain the likelihood.
"""
@kwdef struct SqExpLikelihood <: Likelihood end

# defined directly through `loglike` (models the scalar sqrt-log-likelihood), not `loglike_marginal`
likelihood_kind(::SqExpLikelihood) = JointOnly()

function loglike(::SqExpLikelihood, y::AbstractVector{<:Real})
    @assert length(y) == 1
    return y[1]^2
end
function loglike(::SqExpLikelihood, Y::AbstractMatrix{<:Real})
    @assert size(Y, 1) == 1
    return Y[1,:] .^ 2
end

function log_likelihood_mean(::GaussianPredictive, ::SqExpLikelihood, model_post::ModelPosterior)
    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        @assert length(μ_y) == length(std_y) == 1
        μ, σ = μ_y[1], std_y[1]

        # return log( (1 / sqrt(1 + σ^2)) * exp(-(1/2) * (μ^2 / (1 + σ^2))) )
        return (-(1/2) * log(1 + σ^2)) + (-(1/2) * (μ^2 / (1 + σ^2)))
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end
function log_likelihood_mean(::SampledPredictive, ::SqExpLikelihood, model_post::ModelPosterior)
    function log_like_mean(x::AbstractVector{<:Real})
        ys, ws = only(per_dim_predictive_samples(model_post, x))
        return logsumexp((ys .^ 2) .+ log.(ws))
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end

function log_sq_likelihood_mean(::GaussianPredictive, ::SqExpLikelihood, model_post::ModelPosterior)
    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        @assert length(μ_y) == length(std_y) == 1
        μ, σ = μ_y[1], std_y[1]

        # return log( (1 / sqrt(1 + 2 * σ^2)) * exp(-(1/2) * (μ^2 / (σ^2 * (1 + 2 * σ^2)))) )
        return (-(1/2) * log(1 + 2 * σ^2)) + (-(1/2) * (μ^2 / (σ^2 * (1 + 2 * σ^2))))
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end
function log_sq_likelihood_mean(::SampledPredictive, ::SqExpLikelihood, model_post::ModelPosterior)
    function log_sq_like_mean(x::AbstractVector{<:Real})
        ys, ws = only(per_dim_predictive_samples(model_post, x))
        return logsumexp((2 .* ys .^ 2) .+ log.(ws))
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end
