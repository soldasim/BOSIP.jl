# Construction of posterior and likelihood estimates from an instance of `BosipProblem`.

"""
    log_approx_posterior(::BosipProblem)

Return *the log of* the unnormalized approximate posterior ``\\hat{p}(z_o|x) p(x)`` as a function of `x`.

# See Also

[`approx_posterior`](@ref),
[`log_approx_likelihood`](@ref),
[`log_posterior_mean`](@ref),
[`log_posterior_variance`](@ref),
"""
function log_approx_posterior(bosip::BosipProblem)
    x_prior = bosip.x_prior

    log_like = log_approx_likelihood(bosip)
    log_post(x) = _log_prior(x_prior, x) .+ log_like(x)
    return log_post
end

"""
    log_posterior_mean(::BosipProblem)

Return *the log of* the expectation of the unnormalized posterior ``\\mathbb{E}[\\hat{p}(z_o|x) p(x)]`` as a function of ``x``.

# See Also

[`posterior_mean`](@ref),
[`log_likelihood_mean`](@ref),
[`log_approx_posterior`](@ref),
[`log_posterior_variance`](@ref)
"""
function log_posterior_mean(bosip::BosipProblem)
    x_prior = bosip.x_prior

    log_like_mean = log_likelihood_mean(bosip)
    log_post_mean(x) = _log_prior(x_prior, x) .+ log_like_mean(x)
    return log_post_mean
end

"""
    log_posterior_variance(::BosipProblem)

Return *the log of* the variance of the unnormalized posterior ``\\mathbb{V}[\\hat{p}(z_o|x) p(x)]`` as a function of ``x``.

# See Also

[`posterior_variance`](@ref),
[`log_likelihood_variance`](@ref),
[`log_posterior_mean`](@ref),
[`log_approx_posterior`](@ref)
"""
function log_posterior_variance(bosip::BosipProblem)
    x_prior = bosip.x_prior

    log_like_var = log_likelihood_variance(bosip)
    log_post_var(x) = (2 .* _log_prior(x_prior, x)) .+ log_like_var(x)
    return log_post_var
end

"""
    log_approx_likelihood(::BosipProblem)
    log_approx_likelihood(::Likelihood, ::ModelPosterior)

Return *the log of* the approximate likelihood ``\\hat{p}(z_o|x)`` as a function of `x`.

# See Also

[`approx_likelihood`](@ref),
[`log_approx_posterior`](@ref),
[`log_likelihood_mean`](@ref),
[`log_likelihood_variance`](@ref)
"""
function log_approx_likelihood(bosip::BosipProblem)
    return log_approx_likelihood(typeof(bosip.problem.params), bosip)
end

function log_approx_likelihood(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like = log_approx_likelihood(bosip.likelihood, model_post)
    return log_like
end
function log_approx_likelihood(::Type{<:MultiFittedParams}, bosip::BosipProblem)
    model_posts = BOSS.model_posterior(bosip.problem)
    sample_count = length(model_posts)
    log_likes = log_approx_likelihood.(Ref(bosip.likelihood), model_posts)
    
    function log_like(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_likes) ./ sample_count)
    end
    return log_like
end

"""
    log_likelihood_mean(::BosipProblem)
    log_likelihood_mean(::Likelihood, ::ModelPosterior)

Return *the log of* the expectation of the likelihood approximation ``\\mathbb{E}[\\hat{p}(z_o|x)]`` as a function of ``x``.

# See Also

[`likelihood_mean`](@ref),
[`log_posterior_mean`](@ref),
[`log_likelihood_variance`](@ref),
[`log_approx_likelihood`](@ref)
"""
function log_likelihood_mean(bosip::BosipProblem)
    return log_likelihood_mean(typeof(bosip.problem.params), bosip)
end

function log_likelihood_mean(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like_mean = log_likelihood_mean(bosip.likelihood, model_post)
    return log_like_mean
end
function log_likelihood_mean(::Type{<:MultiFittedParams}, bosip::BosipProblem)
    model_posts = BOSS.model_posterior(bosip.problem)
    sample_count = length(model_posts)
    log_like_means = log_likelihood_mean.(Ref(bosip.likelihood), model_posts)
    
    function log_like_mean(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_like_means) ./ sample_count)
    end
    return log_like_mean
end

"""
    log_likelihood_variance(::BosipProblem)

Return *the log of* the variance of the likelihood approximation ``\\mathbb{V}[\\hat{p}(z_o|x)]`` as a function of ``x``.

# See Also

[`likelihood_variance`](@ref),
[`log_posterior_variance`](@ref),
[`log_likelihood_mean`](@ref),
[`log_approx_likelihood`](@ref)
"""
function log_likelihood_variance(bosip::BosipProblem)
    return log_likelihood_variance(typeof(bosip.problem.params), bosip)
end

function log_likelihood_variance(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like_var = log_likelihood_variance(bosip.likelihood, model_post)
    return log_like_var
end
function log_likelihood_variance(::Type{<:MultiFittedParams}, bosip::BosipProblem)
    model_posts = BOSS.model_posterior(bosip.problem)
    sample_count = length(model_posts)
    log_like_vars = log_likelihood_variance.(Ref(bosip.likelihood), model_posts)
    
    function log_like_var(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_like_vars) ./ sample_count)
    end
    return log_like_var
end

"""
    log_approx_marginal_likelihood(bosip::BosipProblem) -> ::Function
    log_approx_marginal_likelihood(like::Likelihood, model_post::ModelPosterior) -> ::Function

Return *the log of* the per-dimension approximate likelihoods ``\\log \\hat{p}(z_o^{(i)}|x)``
as a function of `x`, where each element corresponds to one observation dimension.
Summing the result recovers [`log_approx_likelihood`](@ref).

# See Also

[`log_approx_likelihood`](@ref),
[`log_marginal_likelihood_mean`](@ref)
"""
function log_approx_marginal_likelihood(bosip::BosipProblem)
    return log_approx_marginal_likelihood(typeof(bosip.problem.params), bosip)
end

function log_approx_marginal_likelihood(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    return log_approx_marginal_likelihood(bosip.likelihood, model_post)
end
function log_approx_marginal_likelihood(::Type{<:MultiFittedParams}, bosip::BosipProblem)
    model_posts = BOSS.model_posterior(bosip.problem)
    sample_count = length(model_posts)
    log_mls = log_approx_marginal_likelihood.(Ref(bosip.likelihood), model_posts)
    
    function log_ml(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_mls) ./ sample_count)
    end
    return log_ml
end

"""
    log_marginal_likelihood_mean(bosip::BosipProblem) -> ::Function
    log_marginal_likelihood_mean(like::Likelihood, model_post::ModelPosterior) -> ::Function

Return *the log of* the per-dimension likelihood expectation ``\\log \\mathbb{E}[\\hat{p}(z_o^{(i)}|x)]``
as a function of `x`, where each element corresponds to one observation dimension.
Summing the result recovers [`log_likelihood_mean`](@ref).

# See Also

[`log_likelihood_mean`](@ref),
[`log_approx_marginal_likelihood`](@ref)
"""
function log_marginal_likelihood_mean(bosip::BosipProblem)
    return log_marginal_likelihood_mean(typeof(bosip.problem.params), bosip)
end

function log_marginal_likelihood_mean(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    return log_marginal_likelihood_mean(bosip.likelihood, model_post)
end
function log_marginal_likelihood_mean(::Type{<:MultiFittedParams}, bosip::BosipProblem)
    model_posts = BOSS.model_posterior(bosip.problem)
    sample_count = length(model_posts)
    log_ml_means = log_marginal_likelihood_mean.(Ref(bosip.likelihood), model_posts)
    
    function log_ml_mean(x)
        return log.(mapreduce(f -> exp.(f(x)), .+, log_ml_means) ./ sample_count)
    end
    return log_ml_mean
end

_log_prior(x_prior::MultivariateDistribution, x::AbstractVector{<:Real}) = logpdf(x_prior, x)
_log_prior(x_prior::MultivariateDistribution, X::AbstractMatrix{<:Real}) = logpdf.(Ref(x_prior), eachcol(X))
