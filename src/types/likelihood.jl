
"""
Represents the assumed likelihood of the experiment observation ``z_o``.

See [`MonteCarloLikelihood`](@ref) for a simplified interface for likelihoods.

See also [`CombinedLikelihood`](@ref), which facilitates combining multiple likelihoods.

## The Likelihood API

Each subtype of `Likelihood` must implement the following API;

**Mandatory:** Define the log-likelihood and its expectation - ideally through the marginal contributions.
- `loglike_marginal(like::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
- `log_marginal_likelihood_mean(::Likelihood, ::ModelPosterior)`

_(Alternatively, implement the non-marginal versions: `loglike`, `log_likelihood_mean`.
It is better to implement the marginal versions, as then both are available.)_

**Optional:** (for performance optimization)
- `loglike_marginal(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}])` or `loglike(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}])`

**Mandatory at least one of:** Define the likelihood variance.
- `log_sq_likelihood_mean(::Likelihood, ::ModelPosterior)`
- `log_likelihood_variance(::Likelihood, ::ModelPosterior)`

_(You either define the likelihood variance directly
or it is computed by definition as the difference of the mean of the square and the squared mean.)_

**Necessary only if** `BosipProblem` where `!isnothing(problem.y_sets)` is used:
- `get_subset(::Likelihood, y_set::AbstractVector{<:Bool})`:

**Provided by default:**
- `loglike(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstarctVector{<:Real}])`
- `log_likelihood_mean(::Likelihood, ::ModelPosterior)`
- `log_approx_marginal_likelihood(::Likelihood, ::ModelPosterior)`
- `log_approx_likelihood(::Likelihood, ::ModelPosterior)`
- `like(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
"""
abstract type Likelihood end

"""
    loglike_marginal(like::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::AbstractVector{<:Real}
    loglike_marginal(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractMatrix{<:Real}

Return the per-dimension log-likelihoods of the observation given the proxy variable `δ`.
The result is a vector with one entry per observation dimension, such that
`sum(loglike_marginal(like, δ)) ≈ loglike(like, δ)`.

The batched method returns a matrix where each column corresponds to one sample in `Δ`.
Rarely, some `Likelihood`s may require the input parameters `x` to compute the log-likelihood as well.
"""
loglike_marginal(like::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = loglike_marginal(like, δ)

# default broadcasting
loglike_marginal(l::Likelihood, Δ::AbstractMatrix{<:Real}) = hcat(loglike_marginal.(Ref(l), eachcol(Δ))...)
loglike_marginal(l::Likelihood, Δ::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}) = hcat(loglike_marginal.(Ref(l), eachcol(Δ), eachcol(X))...)

"""
    loglike(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
    loglike(::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractVector{<:Real}

Return the log-likelihood of the observation given the proxy variable `δ`.
Rarely, some `Likelihood`s may require the input parameters `x` to compute the log-likelihood as well.
"""
loglike(l::Likelihood, δ::AbstractVector{<:Real}) = sum(loglike_marginal(l, δ))
loglike(l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = sum(loglike_marginal(l, δ, x))

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
    log_approx_marginal_likelihood(::Likelihood, ::ModelPosterior)

Returns a function `log_approx_ml` mapping ``x`` to the per-dimension ``log \\hat{p}(z_o^{(i)}|x)``,
with the following two methods:
- `log_approx_ml(x::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`
- `log_approx_ml(X::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`
"""
function log_approx_marginal_likelihood end

"""
    log_approx_likelihood(::Likelihood, ::ModelPosterior)

Returns a function `log_approx_like` mapping ``x`` to ``log \\hat{p}(z_o|x)``,
with the following two methods:
- `log_approx_like(x::AbstractVector{<:Real}) -> ::Real`
- `log_approx_like(X::AbstractMatrix{<:Real}) -> ::AbstractVector{<:Real}`
"""
function log_approx_likelihood end

"""
    log_marginal_likelihood_mean(::Likelihood, ::ModelPosterior)

Returns a function `log_ml_mean` mapping ``x`` to the per-dimension ``log \\mathbb{E}[ \\hat{p}(z_o^{(i)}|x) | GP ]``,
with the following two methods:
- `log_ml_mean(x::AbstractVector{<:Real}) -> ::AbstractVector{<:Real}`
- `log_ml_mean(X::AbstractMatrix{<:Real}) -> ::AbstractMatrix{<:Real}`
"""
function log_marginal_likelihood_mean end

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
