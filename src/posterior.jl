
"""
    approx_posterior(::BolfiProblem; kwargs...)

Return the approximate posterior ``\\hat{p}(x|y_o)`` as a function of ``x``.

The posterior is approximated by directly substituting the predictive means of the GPs
as the discrepancies (and ignoring the uncertainty of the GPs). Thus, the probability
given by `approx_posterior` is the probability of the given parameters being the true ones
used to generate the observation while only taking into account the noise on the observation,
and assuming that the mean prediction of the GP gives the true noiseless observation for the parameters.

The unnormalized approximate posterior ``\\hat{p}(y_o|x) p(x)`` is returned by default.

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
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return approx_posterior(gp_post, bolfi.x_prior, std_obs(bolfi); normalize, xs, samples)
end

function approx_posterior(gp_post, x_prior, std_obs; normalize=false, xs=nothing, samples=10_000)
    apporx_like = approx_likelihood(gp_post, std_obs)
    approx_post(x) = apporx_like(x) * pdf(x_prior, x)

    if normalize
        py = evidence(approx_post, x_prior; xs, samples)
        return (x) -> approx_post(x) / py
    else
        return approx_post
    end
end

"""
    approx_likelihood(::BolfiProblem; kwargs...)

Return the approximate likelihood ``\\hat{p}(y_o|x)`` as a function of ``x``.

The likelihood is approximated by directly substituting the predictive means of the GPs
as the discrepancies (and ignoring the variance of the GPs).

# See Also

[`likelihood_mean`](@ref),
[`likelihood_variance`](@ref),
[`approx_posterior`](@ref)
"""
function approx_likelihood(bolfi::BolfiProblem)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return approx_likelihood(gp_post, std_obs(bolfi))
end
function approx_likelihood(gp_post, std_obs::AbstractVector{<:Real})
    function like_mean(x)
        pred = gp_post(x)
        μ_δ, _ = pred
        return pdf(MvNormal(μ_δ, std_obs), zero(μ_δ))
    end
end

"""
    posterior_mean(::BolfiProblem; kwargs...)

Return the expectation of the posterior approximation ``\\mathbb{E}[\\hat{p}(x|y_o)]`` as a function of ``x``.

In contrast to `approx_posterior`, the `posterior_mean` considers the uncertainty of the GPs as well.
Thus, the probability given by the `posterior_mean` is the probability of the given parameters
being the true ones used to generate the observation while taking into account the noise
on the observation, the noise of the simulator, and the uncertainty due to lack of data.

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
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_mean(gp_post, bolfi.x_prior, std_obs(bolfi); normalize, xs, samples)
end
function posterior_mean(gp_post, x_prior, std_obs; normalize=false, xs=nothing, samples=10_000)
    like_mean = likelihood_mean(gp_post, std_obs)
    post_mean(x) = pdf(x_prior, x) * like_mean(x)

    if normalize
        py = evidence(post_mean, x_prior; xs, samples)
        return (x) -> post_mean(x) / py
    else
        return post_mean
    end
end

"""
    likelihood_mean(::BolfiProblem; kwargs...)

Return the expectation of the likelihood approximation ``\\mathbb{E}[\\hat{p}(y_o|x)]`` as a function of ``x``.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_variance`](@ref),
[`posterior_mean`](@ref)
"""
function likelihood_mean(bolfi::BolfiProblem)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return likelihood_mean(gp_post, std_obs(bolfi))
end
function likelihood_mean(gp_post, std_obs::AbstractVector{<:Real})
    function like_mean(x)
        pred = gp_post(x)
        μ_δ, std_δ = pred
        return pdf(MvNormal(μ_δ, sqrt.(std_obs.^2 .+ std_δ.^2)), zero(μ_δ))
    end
end

"""
    posterior_variance(::BolfiProblem; kwargs...)

Return the variance of the posterior approximation ``\\mathbb{V}[\\hat{p}(x|y_o)]`` as a function of ``x``.

The returned variance is the variance of the predicted posterior probability of the given parameters
caused by the uncertainty of the GPs due to the simulator noise and a (possible) lack of data.

The variance of the unnormalized posterior ``\\mathbb{V}[\\hat{p}(y_o|x) p(x)]`` is returned by default.

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
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(gp_post, bolfi.x_prior, std_obs(bolfi); normalize, xs, samples)
end
function posterior_variance(gp_post, x_prior, std_obs; normalize=false, xs=nothing, samples=10_000)
    if normalize
        isnothing(xs) && (xs = rand(x_prior, samples))
        mean = posterior_mean(gp_post, x_prior, std_obs; normalize=false, xs)
        py = evidence(mean, x_prior; xs)
    else
        py = 1.
    end

    like_var = likelihood_variance(gp_post, std_obs)
    post_var(x) = (pdf(x_prior, x) / py)^2 * like_var(x)
end

"""
    likelihood_variance(::BolfiProblem; kwargs...)

Return the variance of the likelihood approximation ``\\mathbb{V}[\\hat{p}(y_o|x)]`` as a function of ``x``.

# See Also

[`approx_likelihood`](@ref),
[`likelihood_mean`](@ref),
[`posterior_variance`](@ref)
"""
function likelihood_variance(bolfi::BolfiProblem)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return likelihood_variance(gp_post, std_obs(bolfi))
end
function likelihood_variance(gp_post, std_obs::AbstractVector{<:Real})
    var_obs = std_obs .^ 2

    function like_var(x)
        pred = gp_post(x)
        μ_δ, std_δ = pred
        var_δ = std_δ .^ 2

        prodA = log.(A_.(μ_δ, var_δ, var_obs)) |> sum |> exp
        prodB = log.(B_.(μ_δ, var_δ, var_obs)) |> sum |> exp
        return prodA - prodB
    end
end

function A_(μ_δ, var_δ, var_obs)
    varA = var_obs + 2*var_δ
    return sqrt(varA / var_obs) * pdf(Normal(0, sqrt(varA)), μ_δ)^2
end
function B_(μ_δ, var_δ, var_obs)
    varB = var_obs + var_δ
    return pdf(Normal(0, sqrt(varB)), μ_δ)^2
end

"""
    evidence(post, x_prior; kwargs...)

Return the estimated evidence ``\\hat{p}(y_o)``.

# Arguments

- `post`: A function `::AbstractVector{<:Real} -> ::Real`
        representing the posterior ``p(x|y_o)``.
- `x_prior`: A multivariate distribution
        representing the prior ``p(x)``.

# Keywords

- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `x_prior` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `isnothing(xs)`.

"""
function evidence(post, x_prior; xs=nothing, samples=10_000)
    ll(x) = post(x) / pdf(x_prior, x)
    isnothing(xs) && (xs = rand(x_prior, samples))
    py = mean((ll(x) for x in eachcol(xs)))
    return py
end
