
"""
Represents the assumed likelihood of the experiment observation ``z_o``.

See [`MonteCarloLikelihood`](@ref) for a simplified interface for likelihoods.

See also [`CombinedLikelihood`](@ref), which facilitates combining multiple likelihoods.

## Defining a Custom Likelihood

To define a custom likelihood, create a new subtype of `Likelihood`
and implement the following API;

Each subtype of `Likelihood` *should* implement:
- `loglike(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstarctVector{<:Real}])`
- `log_likelihood_mean(::Likelihood, ::ModelPosterior)`

Additionally, subtypes of `Likelihood` *may* implement:
- `loglike(::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}])`

Each subtype of `Likelihood` *should* implement *at least one* of:
- `log_sq_likelihood_mean(::Likelihood, ::ModelPosterior)`
- `log_likelihood_variance(::Likelihood, ::ModelPosterior)`

The following method is necessary to implement
if `BosipProblem` where `!isnothing(problem.y_sets)` is used:
- `get_subset(::Likelihood, y_set::AbstractVector{<:Bool})`:

The following methods are provided by default and *need not be implemented*:
- `log_approx_likelihood(::Likelihood, ::ModelPosterior)`
- `like(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
"""
abstract type Likelihood end

"""
    loglike(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
    loglike(::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractVector{<:Real}

Return the log-likelihood of the observation given the proxy variable `δ`.
Rarely, some `Likelihood`s may require the input parameters `x` to compute the log-likelihood as well.
"""
loglike(l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = loglike(l, δ)

# default broadcasting
loglike(l::Likelihood, Δ::AbstractMatrix{<:Real}) = loglike.(Ref(l), eachcol(Δ))
loglike(l::Likelihood, Δ::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}) = loglike.(Ref(l), eachcol(Δ), eachcol(X))

"""
    like(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
    like(l::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractVector{<:Real}

Return the likelihood of the observation given the model output `δ`.
"""
like(args...) = exp.(loglike(args...))

"""
    log_approx_likelihood(::Likelihood, ::ModelPosterior)

Returns a function `log_approx_like` mapping ``x`` to ``log \\hat{p}(z_o|x)``,
with the following two methods:
- `log_approx_like(x::AbstractVector{<:Real}) -> ::Real`
- `log_approx_like(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_approx_likelihood end

"""
    log_likelihood_mean(::Likelihood, ::ModelPosterior)

Returns a function `log_like_mean` mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x) | GP ]``,
with the following two methods:
- `log_like_mean(x::AbstractVector{<:Real}) -> ::Real`
- `log_like_mean(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_likelihood_mean end

"""
    log_sq_likelihood_mean(::Likelihood, ::ModelPosterior)

Returns a function `log_sq_like_mean` mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x)^2 | GP ]``,
with the following two methods:
- `log_sq_like_mean(x::AbstractVector{<:Real}) -> ::Real`
- `log_sq_like_mean(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_sq_likelihood_mean end

"""
    log_likelihood_variance(::Likelihood, ::ModelPosterior)

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
