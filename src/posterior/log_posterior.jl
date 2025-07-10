
"""
    log_approx_posterior(::BolfiProblem)

Return *the log of* the unnormalized approximate posterior ``\\hat{p}(y_o|x) p(x)`` as a function of `x`.

# See Also

[`approx_posterior`](@ref),
[`log_approx_likelihood`](@ref),
[`log_posterior_mean`](@ref),
[`log_posterior_variance`](@ref),
"""
function log_approx_posterior(bolfi::BolfiProblem)
    x_prior = bolfi.x_prior

    log_like = log_approx_likelihood(bolfi)
    log_post(x) = _log_prior(x_prior, x) .+ log_like(x)
    return log_post
end

"""
    log_approx_likelihood(::BolfiProblem)

Return *the log of* the approximate likelihood ``\\hat{p}(y_o|x)`` as a function of `x`.

# See Also

[`approx_likelihood`](@ref),
[`log_approx_posterior`](@ref),
[`log_likelihood_mean`](@ref),
[`log_likelihood_variance`](@ref)
"""
function log_approx_likelihood(bolfi::BolfiProblem)
    return log_approx_likelihood(typeof(bolfi.problem.params), bolfi)
end

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_approx_likelihood(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    model_post = BOSS.model_posterior(bolfi.problem)
    
    log_like = log_approx_likelihood(bolfi.likelihood, bolfi, model_post)
    return log_like
end

"""
    log_posterior_mean(::BolfiProblem)

Return *the log of* the expectation of the unnormalized posterior ``\\mathbb{E}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

# See Also

[`posterior_mean`](@ref),
[`log_likelihood_mean`](@ref),
[`log_approx_posterior`](@ref),
[`log_posterior_variance`](@ref)
"""
function log_posterior_mean(bolfi::BolfiProblem)
    x_prior = bolfi.x_prior

    log_like_mean = log_likelihood_mean(bolfi)
    log_post_mean(x) = _log_prior(x_prior, x) .+ log_like_mean(x)
    return log_post_mean
end

"""
    log_likelihood_mean(::BolfiProblem)

Return *the log of* the expectation of the likelihood approximation ``\\mathbb{E}[\\hat{p}(y_o|x)]`` as a function of ``x``.

# See Also

[`likelihood_mean`](@ref),
[`log_posterior_mean`](@ref),
[`log_likelihood_variance`](@ref),
[`log_approx_likelihood`](@ref)
"""
function log_likelihood_mean(bolfi::BolfiProblem)
    return log_likelihood_mean(typeof(bolfi.problem.params), bolfi)
end

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_likelihood_mean(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    model_post = BOSS.model_posterior(bolfi.problem)
    
    log_like_mean = log_likelihood_mean(bolfi.likelihood, bolfi, model_post)
    return log_like_mean
end

"""
    log_posterior_variance(::BolfiProblem)

Return *the log of* the variance of the unnormalized posterior ``\\mathbb{V}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

# See Also

[`posterior_variance`](@ref),
[`log_likelihood_variance`](@ref),
[`log_posterior_mean`](@ref),
[`log_approx_posterior`](@ref)
"""
function log_posterior_variance(bolfi::BolfiProblem)
    x_prior = bolfi.x_prior

    log_like_var = log_likelihood_variance(bolfi)
    log_post_var(x) = (2 .* _log_prior(x_prior, x)) .+ log_like_var(x)
    return log_post_var
end

"""
    log_likelihood_variance(::BolfiProblem)

Return *the log of* the variance of the likelihood approximation ``\\mathbb{V}[\\hat{p}(y_o|x)]`` as a function of ``x``.

# See Also

[`likelihood_variance`](@ref),
[`log_posterior_variance`](@ref),
[`log_likelihood_mean`](@ref),
[`log_approx_likelihood`](@ref)
"""
function log_likelihood_variance(bolfi::BolfiProblem)
    return log_likelihood_variance(typeof(bolfi.problem.params), bolfi)
end

# the method for `MultiFittedParams` is implemented in `src/posterior/posterior.jl`
function log_likelihood_variance(::Type{<:UniFittedParams}, bolfi::BolfiProblem)
    model_post = BOSS.model_posterior(bolfi.problem)
    
    log_like_var = log_likelihood_variance(bolfi.likelihood, bolfi, model_post)
    return log_like_var
end

_log_prior(x_prior::MultivariateDistribution, x::AbstractVector{<:Real}) = logpdf(x_prior, x)
_log_prior(x_prior::MultivariateDistribution, X::AbstractMatrix{<:Real}) = logpdf.(Ref(x_prior), eachcol(X))
