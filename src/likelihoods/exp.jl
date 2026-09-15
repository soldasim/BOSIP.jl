
"""
    ExpLikelihood()

Assumes the model approximates the log-likelihood directly (as a scalar).
Only exponentiates the model prediction.
"""
@kwdef struct ExpLikelihood <: Likelihood end

# defined directly through `loglike` (models the scalar log-likelihood), not `loglike_marginal`
likelihood_kind(::ExpLikelihood) = JointOnly()

function loglike(::ExpLikelihood, y::AbstractVector{<:Real})
    @assert length(y) == 1
    return y[1]
end
function loglike(::ExpLikelihood, Y::AbstractMatrix{<:Real})
    @assert size(Y, 1) == 1
    return vec(Y[1,:])
end

function log_likelihood_mean(::GaussianPredictive, ::ExpLikelihood, model_post::ModelPosterior)
    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, σ2_y = mean_and_var(model_post, x)
        @assert length(μ_y) == length(σ2_y) == 1
        μ, σ2 = μ_y[1], σ2_y[1]

        # return log( exp(μ + 0.5 * σ2) )
        return μ + 0.5 * σ2
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end
function log_likelihood_mean(::SampledPredictive, ::ExpLikelihood, model_post::ModelPosterior)
    function log_like_mean(x::AbstractVector{<:Real})
        ys, ws = only(per_dim_predictive_samples(model_post, x))
        return logsumexp(ys .+ log.(ws))
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end

function log_likelihood_variance(::GaussianPredictive, ::ExpLikelihood, model_post::ModelPosterior)
    function log_like_var(x::AbstractVector{<:Real})
        μ_y, σ2_y = mean_and_var(model_post, x)
        @assert length(μ_y) == length(σ2_y) == 1
        μ, σ2 = μ_y[1], σ2_y[1]

        # return log( exp(2 * (μ + σ2) + log(1 - exp(-σ2))) )
        return 2 * (μ + σ2) + log(1 - exp(-σ2))
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end
function log_likelihood_variance(::SampledPredictive, ::ExpLikelihood, model_post::ModelPosterior)
    function log_like_var(x::AbstractVector{<:Real})
        ys, ws = only(per_dim_predictive_samples(model_post, x))
        log_mean = logsumexp(ys .+ log.(ws))
        log_sq_mean = logsumexp((2 .* ys) .+ log.(ws))
        return log_sq_mean + log1mexp(2 * log_mean - log_sq_mean)
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end
