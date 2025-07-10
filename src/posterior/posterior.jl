
"""
    approx_posterior(::BolfiProblem; kwargs...)

Return the MAP estimation of the unnormalized approximate posterior ``\\hat{p}(y_o|x) p(x)`` as a function of ``x``.

If `normalize=true`, the resulting posterior is approximately normalized.

The posterior is approximated by directly substituting the predictive means of the GPs
as the discrepancies from the true observation and ignoring both the uncertainty of the GPs
due to a lack of data and due to the simulator evaluation noise.

By using `approx_posterior` or `posterior_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized approximate posterior ``\\hat{p}(y_o|x) p(x) / \\hat{p}(y_o)``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`posterior_mean`](@ref),
[`posterior_variance`](@ref),
[`approx_likelihood`](@ref)
"""
function approx_posterior(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior
    log_post = log_approx_posterior(bolfi)

    if normalize
        py = evidence(x -> exp(log_post(x)), x_prior; xs, samples)
        log_py = log(py)
        post = x -> exp.(log_post(x) .- log_py)
    
    else
        post = x -> exp.(log_post(x))
    end

    return post
end

"""
    posterior_mean(::BolfiProblem; kwargs...)

Return the expectation of the unnormalized posterior ``\\mathbb{E}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

If `normalize=true`, the resulting expected posterior is approximately normalized.

The returned function maps parameters `x` to the expected posterior probability density value
integrated over the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By using `approx_posterior` or `posterior_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized expected posterior ``\\mathbb{E}[\\hat{p}(y_o|x) p(x)]``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`approx_posterior`](@ref),
[`posterior_variance`](@ref),
[`likelihood_mean`](@ref)
"""
function posterior_mean(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior
    log_post_mean = log_posterior_mean(bolfi)

    if normalize
        py = evidence(x -> exp(log_post_mean(x)), x_prior; xs, samples)
        log_py = log(py)
        post_mean = x -> exp.(log_post_mean(x) .- log_py)

    else
        post_mean = x -> exp.(log_post_mean(x))
    end

    return post_mean
end

"""
    posterior_variance(::BolfiProblem; kwargs...)

Return the variance of the unnormalized posterior ``\\mathbb{V}[\\hat{p}(y_o|x) p(x)]`` as a function of ``x``.

If `normalize=true`, the resulting posterior variance is approximately normalized.

The returned function maps parameters `x` to the variance of the posterior probability density value estimate
caused by the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to compute the variance over the uncertainty in the GP hyperparameters as well.

# Keywords

- `normalize::Bool`: If `normalize` is set to `true`, the evidence ``\\hat{p}(y_o)```
        is estimated by sampling and the normalized posterior variance ``\\mathbb{V}[\\hat{p}(y_o|x) p(x) / \\hat{p}(y_o)]``
        is returned instead of the unnormalized one.
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the parameter prior ``p(x)`` as a column-wise matrix.
        Only has an effect if `normalize == true`.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `normalize == true` and `isnothing(xs)`.

# See Also

[`approx_posterior`](@ref),
[`posterior_mean`](@ref),
[`likelihood_variance`](@ref)
"""
function posterior_variance(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    x_prior = bolfi.x_prior
    log_post_var = log_posterior_variance(bolfi)

    if normalize
        post_mean = posterior_mean(bolfi)
        py = evidence(post_mean, x_prior; xs, samples)
        log_py2 = 2 * log(py)
        post_var = x -> exp.(log_post_var(x) .- log_py2)

    else
        post_var = x -> exp.(log_post_var(x))
    end

    return post_var
end

"""
    approx_likelihood(::BolfiProblem)

Return the MAP estimation of the likelihood ``\\hat{p}(y_o|x)`` as a function of ``x``.

The likelihood is approximated by directly substituting the predictive means of the GPs
as the discrepancies from the true observation and ignoring both the uncertainty of the GPs
due to a lack of data and due to the simulator evaluation noise.

By using `approx_likelihood` or `likelihood_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# See Also

[`likelihood_mean`](@ref),
[`likelihood_variance`](@ref),
[`approx_posterior`](@ref)
"""
function approx_likelihood(args...)
    log_like = log_approx_likelihood(args...)
    like(x) = exp.(log_like(x))
    return like
end

# the method for `UniFittedParams` is implemented in `src/posterior/log_posterior.jl`
function approx_likelihood(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    model_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(model_posts)
    
    likes = approx_likelihood.(Ref(bolfi.likelihood), Ref(bolfi), model_posts)
    
    function exp_like(x)
        return mapreduce(l -> l(x), .+, likes) ./ sample_count
    end
end

"""
    likelihood_mean(::BolfiProblem)

Return the expectation of the likelihood approximation ``\\mathbb{E}[\\hat{p}(y_o|x)]`` as a function of ``x``.

The returned function maps parameters `x` to the expected likelihood probability density value
integrated over the uncertainty of the GPs due to a lack of data and due to the simulator evaluation noise.

By using `approx_likelihood` or `likelihood_mean` one controls,
whether to integrate over the uncertainty in the discrepancy estimate.
In addition to that, by providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to integrate over the uncertainty in the GP hyperparameters.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_variance`](@ref),
[`posterior_mean`](@ref)
"""
function likelihood_mean(args...)
    log_like_mean = log_likelihood_mean(args...)
    like_mean(x) = exp.(log_like_mean(x))
    return like_mean
end

# the method for `UniFittedParams` is implemented in `src/posterior/log_posterior.jl`
function likelihood_mean(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    model_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(model_posts)
    
    like_means = likelihood_mean.(Ref(bolfi.likelihood), Ref(bolfi), model_posts)
    
    function exp_like_mean(x)
        return mapreduce(l -> l(x), .+, like_means) ./ sample_count
    end
end

"""
    likelihood_variance(::BolfiProblem)

Return the variance of the likelihood approximation ``\\mathbb{V}[\\hat{p}(y_o|x)]`` as a function of ``x``.

The returned function maps parameters `x` to the variance of the likelihood probability density value estimate
caused by the uncertainty of the GPs due to a lack of data and the uncertainty of the simulator
due to the evaluation noise.

By providing a `ModelFitter{MAP}` or a `ModelFitter{BI}` to `bolfi!` one controls,
whether to compute the variance over the uncertainty in the GP hyperparameters as well.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_mean`](@ref),
[`posterior_variance`](@ref)
"""
function likelihood_variance(args...)
    log_like_var = log_likelihood_variance(args...)
    like_var(x) = exp.(log_like_var(x))
    return like_var
end

# the method for `UniFittedParams` is implemented in `src/posterior/log_posterior.jl`
function likelihood_variance(::Type{<:MultiFittedParams}, bolfi::BolfiProblem)
    model_posts = BOSS.model_posterior(bolfi.problem)
    sample_count = length(model_posts)

    like_vars = likelihood_variance.(Ref(bolfi.likelihood), Ref(bolfi), model_posts)

    function exp_like_var(x)
        return mapreduce(l -> l(x), .+, like_vars) ./ sample_count
    end
end
