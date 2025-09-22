
"""
Represents the assumed likelihood of the experiment observation ``z_o``.

# Defining a Custom Likelihood

To define a custom likelihood, create a new subtype of `Likelihood`
and implement the following API;

Each subtype of `Likelihood` *should* implement:
- `loglike(::Likelihood, y::AbstractVector{<:Real})` where `y` is the simulator output
- `log_likelihood_mean(::Likelihood, ::BosipProblem, ::ModelPosterior)`

Each subtype of `Likelihood` *should* implement *at least one* of:
- `log_sq_likelihood_mean(::Likelihood, ::BosipProblem, ::ModelPosterior)`
- `log_likelihood_variance(::Likelihood, ::BosipProblem, ::ModelPosterior)`

Additionally, the following method is also necessary to implement
if `BosipProblem` where `!isnothing(problem.y_sets)` is used:
- `get_subset(::Likelihood, y_set::AsbtractVector{<:Bool})`:

The following additional methods are provided by default and *need not be implemented*:
- `log_approx_likelihood(::Likelihood, ::BosipProblem, ::ModelPosterior)`
- `like(::Likelihood, y:AbstractVector{<:Real})` where `y` is the simulator output
"""
abstract type Likelihood end

"""
    loglike(::Likelihood, y::AbstractVector{<:Real}) -> ::Real
    loglike(l::Likelihood, Y::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}

Return the log-likelihood of the observation given the simulator output `y`.
"""
function loglike end

"""
    like(::Likelihood, y::AbstractVector{<:Real}) -> ::Real
    like(l::Likelihood, Y::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}

Return the likelihood of the observation given the model output `y`.
"""
function like(l::Likelihood, y::AbstractVector{<:Real})
    return exp(loglike(l, y))
end
function like(l::Likelihood, Y::AbstractMatrix{<:Real})
    return exp.(loglike(l, Y))
end

"""
    log_approx_likelihood(::Likelihood, ::BosipProblem, ::ModelPosterior)

Returns a function `log_approx_like` mapping ``x`` to ``log \\hat{p}(z_o|x)``,
with the following two methods:
- `log_approx_like(x::AbstractVector{<:Real}) -> ::Real`
- `log_approx_like(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_approx_likelihood end

"""
    log_likelihood_mean(::Likelihood, ::BosipProblem, ::ModelPosterior)

Returns a function `log_like_mean` mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x) | GP ]``,
with the following two methods:
- `log_like_mean(x::AbstractVector{<:Real}) -> ::Real`
- `log_like_mean(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_likelihood_mean end

"""
    log_sq_likelihood_mean(::Likelihood, ::BosipProblem, ::ModelPosterior)

Returns a function `log_sq_like_mean` mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x)^2 | GP ]``,
with the following two methods:
- `log_sq_like_mean(x::AbstractVector{<:Real}) -> ::Real`
- `log_sq_like_mean(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_sq_likelihood_mean end

"""
    log_likelihood_variance(::Likelihood, ::BosipProblem, ::ModelPosterior)

Return a function `log_like_var` mapping ``x`` to ``log \\mathbb{V}[ \\hat{p}(z_o|x) | GP ]``,
with the following two methods:
- `log_like_var(x::AbstractVector{<:Real}) -> ::Real`
- `log_like_var(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_likelihood_variance end

"""
    get_subset(::Likelihood, y_set::AbstractVector{<:Bool}) -> ::Likelihood

Construct the likelihood of the observation dimensions specified by `y_set`.
"""
function get_subset end
