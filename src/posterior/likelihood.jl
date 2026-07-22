# Default implementations for the `Likelihood` API.

const MAX_NEG_VAR = 1e-8

function log_approx_marginal_likelihood(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_approx_marginal_likelihood(predictive_kind(model_post), like, model_post; kwargs...)
end
function log_approx_marginal_likelihood(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    function log_approx_ml(x::AbstractVector{<:Real})
        μy = mean(model_post, x)
        return loglike_marginal(like, μy, x)
    end
    function log_approx_ml(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)
        return loglike_marginal(like, μY, X)
    end
    return log_approx_ml
end

function log_approx_likelihood(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_approx_likelihood(predictive_kind(model_post), like, model_post; kwargs...)
end
function log_approx_likelihood(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    function log_approx_like(x::AbstractVector{<:Real})
        μy = mean(model_post, x)
        return loglike(like, μy, x)
    end
    function log_approx_like(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)
        return loglike(like, μY, X)
    end
    return log_approx_like
end

function log_marginal_likelihood_mean(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_marginal_likelihood_mean(predictive_kind(model_post), like, model_post; kwargs...)
end
function log_marginal_likelihood_mean(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    predictive_kind(model_post) isa SampledPredictive || error(
        "No generic `log_marginal_likelihood_mean` default exists for `GaussianPredictive` models; " *
        "define `log_marginal_likelihood_mean(::GaussianPredictive, ::$(nameof(typeof(like))), ::ModelPosterior)`."
    )
    function log_ml_mean(x::AbstractVector{<:Real})
        dims = per_dim_predictive_samples(model_post, x) # one (atoms, weights) pair per dimension
        return _marginal_log_means(like, dims, x)
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        dims = per_dim_predictive_samples(model_post, X) # each (atoms, weights) pair: (K, n) matrices
        cols = [_marginal_log_means(like, [(ys[:, j], ws[:, j]) for (ys, ws) in dims], X[:, j])
                for j in axes(X, 2)]
        return reduce(hcat, cols)
    end
    return log_ml_mean
end

function log_likelihood_mean(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_likelihood_mean(predictive_kind(model_post), like, model_post; kwargs...)
end
# `log(E[L]) = Σᵢ log(E[Lᵢ])` requires `E[∏ᵢ Lᵢ] = ∏ᵢ E[Lᵢ]`, i.e. the model's dimensions must be
# independent given the (fixed) parameters `model_post` represents.
function log_likelihood_mean(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    @assert BOSS.dimension_independent_given_parameters(model_post)
    log_ml_mean = log_marginal_likelihood_mean(like, model_post)
    function log_like_mean(x::AbstractVector{<:Real})
        return sum(log_ml_mean(x))
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return vec(sum(log_ml_mean(X), dims=1))
    end
    return log_like_mean
end

function log_likelihood_variance(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_likelihood_variance(predictive_kind(model_post), like, model_post; kwargs...)
end
function log_likelihood_variance(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    log_like_mean = log_likelihood_mean(like, model_post)
    log_sq_like_mean = log_sq_likelihood_mean(like, model_post)

    function log_like_var(x::AbstractArray{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        log_sqL = log_sq_like_mean(x)
        log_L = log_like_mean(x)
        like_var = @. exp(log_sqL) - exp(2 * log_L)
        like_var = _assure_nonneg.(like_var)
        return log.(like_var)
    end
end

function _assure_nonneg(x::Real)
    (x >= 0) && return x
    (x >= -MAX_NEG_VAR) && return 0.0
    throw(DomainError(x, "Expected a non-negative value, but got $x."))
end

function log_sq_likelihood_mean(like::Likelihood, model_post::ModelPosterior; kwargs...)
    return log_sq_likelihood_mean(predictive_kind(model_post), like, model_post; kwargs...)
end
# Same reasoning for the `@assert` as in `log_likelihood_mean`.
function log_sq_likelihood_mean(::PredictiveKind, like::Likelihood, model_post::ModelPosterior)
    predictive_kind(model_post) isa SampledPredictive || error(
        "No generic `log_sq_likelihood_mean` default exists for `GaussianPredictive` models; " *
        "define `log_sq_likelihood_mean(::GaussianPredictive, ::$(nameof(typeof(like))), ::ModelPosterior)`."
    )
    @assert BOSS.dimension_independent_given_parameters(model_post)
    function log_sq_like_mean(x::AbstractVector{<:Real})
        dims = per_dim_predictive_samples(model_post, x)
        return sum(_marginal_log_means(like, dims, x; square=true))
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        dims = per_dim_predictive_samples(model_post, X)
        return [sum(_marginal_log_means(like, [(ys[:, j], ws[:, j]) for (ys, ws) in dims], X[:, j]; square=true))
                for j in axes(X, 2)]
    end
    return log_sq_like_mean
end

function _marginal_log_means(like::Likelihood, dims, x; square::Bool=false)
    D = length(dims)
    K = length(dims[1][1])
    acc = zeros(D)
    for k in 1:K
        δ = [dims[i][1][k] for i in 1:D]
        Lk = like_marginal(like, δ, x)
        for i in 1:D
            acc[i] += dims[i][2][k] * Lk[i]^(square ? 2 : 1)
        end
    end
    return log.(acc)
end

function per_dim_predictive_samples(model_post::BOSS.DefaultModelPosterior, x::AbstractVector{<:Real}; kwargs...)
    @assert BOSS.sliceable(model_post) "`predictive_samples` bundling per-dimension slices of `$(typeof(model_post))` requires the underlying model to be `sliceable`."
    return [predictive_samples(BOSS.slice(model_post, i), x; kwargs...) for i in eachindex(model_post.slices)]
end
function per_dim_predictive_samples(model_post::BOSS.DefaultModelPosterior, X::AbstractMatrix{<:Real}; kwargs...)
    @assert BOSS.sliceable(model_post) "`predictive_samples` bundling per-dimension slices of `$(typeof(model_post))` requires the underlying model to be `sliceable`."
    return [predictive_samples(BOSS.slice(model_post, i), X; kwargs...) for i in eachindex(model_post.slices)]
end
function per_dim_predictive_samples(model_post::ModelPosterior, x::AbstractVector{<:Real}; kwargs...)
    Ys, Ws = predictive_samples(model_post, x; kwargs...) # Ys: (y_dim, K), Ws: (1, K)
    ws = vec(Ws)
    return [(Ys[i, :], ws) for i in axes(Ys, 1)]
end
function per_dim_predictive_samples(model_post::ModelPosterior, X::AbstractMatrix{<:Real}; kwargs...)
    Ys, Ws = predictive_samples(model_post, X; kwargs...) # Ys: (y_dim, K, n), Ws: (1, K, n)
    Ws_ = Ws[1, :, :] # (K, n), the shared weights reused for every dimension
    return [(Ys[i, :, :], Ws_) for i in axes(Ys, 1)]
end
