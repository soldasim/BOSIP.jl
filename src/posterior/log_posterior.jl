
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
    log_approx_likelihood(::BosipProblem)

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

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_approx_likelihood(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like = log_approx_likelihood(bosip.likelihood, bosip, model_post)
    return log_like
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
    log_likelihood_mean(::BosipProblem)

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

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_likelihood_mean(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like_mean = log_likelihood_mean(bosip.likelihood, bosip, model_post)
    return log_like_mean
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

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_likelihood_variance(::Type{<:UniFittedParams}, bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    
    log_like_var = log_likelihood_variance(bosip.likelihood, bosip, model_post)
    return log_like_var
end

_log_prior(x_prior::MultivariateDistribution, x::AbstractVector{<:Real}) = logpdf(x_prior, x)
_log_prior(x_prior::MultivariateDistribution, X::AbstractMatrix{<:Real}) = logpdf.(Ref(x_prior), eachcol(X))
