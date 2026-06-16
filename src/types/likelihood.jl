
"""
Represents the assumed likelihood of the experiment observation ``z_o``.

See [`MonteCarloLikelihood`](@ref) for a simplified interface for likelihoods.

See also [`CombinedLikelihood`](@ref), which facilitates combining multiple likelihoods.

## The Likelihood API

Each subtype of `Likelihood` must implement the following API;

**Mandatory:** Declare whether the likelihood is marginalizable via [`likelihood_kind`](@ref),
and define the log-likelihood and its expectation accordingly. One of:

- **Marginalizable** — factorizes over observation dimensions (preferred when possible, as both the
    marginal and joint quantities then become available):
    - `likelihood_kind(::MyLikelihood) = Marginalizable()`
    - `loglike_marginal(like::MyLikelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
    - `log_marginal_likelihood_mean(::MyLikelihood, ::ModelPosterior)`
- **Joint-only** — does not factorize (e.g. a full-covariance likelihood):
    - `likelihood_kind(::MyLikelihood) = JointOnly()`
    - `loglike(like::MyLikelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
    - `log_likelihood_mean(::MyLikelihood, ::ModelPosterior)`

The input parameters `x` are optional, it is preferable to define the method without `x` if it is not needed.
See [`LikelihoodKind`](@ref) for which fallbacks each kind provides.

**Optional:** For performance optimization, implement the corresponding matrix-variate methods.
- `loglike_marginal(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}])` or `loglike(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}])`

**Mandatory at least one of:** Define the likelihood variance via one of the following methods.
- `log_sq_likelihood_mean(::Likelihood, ::ModelPosterior)`
- `log_likelihood_variance(::Likelihood, ::ModelPosterior)`

_(You either define the likelihood variance directly
or it is computed by definition as the difference of the mean of the square and the squared mean.)_

**Necessary only if** `BosipProblem` where `!isnothing(problem.y_sets)` is used:
- `get_subset(::Likelihood, y_set::AbstractVector{<:Bool})`:

Additional provided functions which **need not be defined**:
- `log_approx_likelihood(::Likelihood, ::ModelPosterior)`
- `log_approx_marginal_likelihood(::Likelihood, ::ModelPosterior)` for marginalizable likelihoods
- `like(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])`
- `like_marginal(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}])` for marginalizable likelihoods
"""
abstract type Likelihood end

"""
    LikelihoodKind

Trait describing how a [`Likelihood`](@ref) defines its log-likelihood, and thus which fallbacks
apply. Each `Likelihood` subtype declares its kind via [`likelihood_kind`](@ref).

- [`Marginalizable`](@ref) defined through `loglike_marginal`. The `loglike` methods
    fall back on summing the per-dimension marginal contributions.
- [`JointOnly`](@ref) defined directly through `loglike` (e.g. a non-factorizable full-covariance
    likelihood, for which no per-dimension marginal exists). The `loglike_marginal` is not defined.
"""
abstract type LikelihoodKind end

"The likelihood is defined through `loglike_marginal`. See [`LikelihoodKind`](@ref)."
struct Marginalizable <: LikelihoodKind end

"The likelihood is defined directly through `loglike`. See [`LikelihoodKind`](@ref)."
struct JointOnly <: LikelihoodKind end

"""
    likelihood_kind(::Likelihood) -> ::LikelihoodKind

Return the [`LikelihoodKind`](@ref) trait of the likelihood. **Has no default**: every `Likelihood`
subtype must define it, as either [`Marginalizable`](@ref) (and implement `loglike_marginal`) or
[`JointOnly`](@ref) (and implement `loglike`).
"""
function likelihood_kind end

"""
    loglike(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
    loglike(::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractVector{<:Real}

Return the log-likelihood of the observation given the proxy variable `δ`.
Rarely, some `Likelihood`s may require the input parameters `x` to compute the log-likelihood as well.

For [`Marginalizable`](@ref) likelihoods the vector methods fall back on summing `loglike_marginal`
(with or without `x`). For [`JointOnly`](@ref) likelihoods the `x`-independent vector method is
implemented directly by the subtype, and the `x`-dependent one falls back on it. The matrix methods
broadcast the vector methods over the columns (subtypes may override them for efficiency).
"""
# vector methods are trait-dispatched
loglike(l::Likelihood, δ::AbstractVector{<:Real}) = _loglike(likelihood_kind(l), l, δ)
loglike(l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = _loglike(likelihood_kind(l), l, δ, x)

# marginalizable -> sum the marginals (with or without x)
_loglike(::Marginalizable, l::Likelihood, δ::AbstractVector{<:Real}) = sum(loglike_marginal(l, δ))
_loglike(::Marginalizable, l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = sum(loglike_marginal(l, δ, x))

# joint-only -> check if a method without x exists
_loglike(::JointOnly, l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = loglike(l, δ)

# default broadcasting (subtypes may override the matrix methods for efficiency)
loglike(l::Likelihood, Δ::AbstractMatrix{<:Real}) = loglike.(Ref(l), eachcol(Δ))
loglike(l::Likelihood, Δ::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}) = loglike.(Ref(l), eachcol(Δ), eachcol(X))

"""
    loglike_marginal(like::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::AbstractVector{<:Real}
    loglike_marginal(like::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractMatrix{<:Real}

Return the per-dimension log-likelihoods of the observation given the proxy variable `δ`.
The result is a vector with one entry per observation dimension, such that
`sum(loglike_marginal(like, δ)) ≈ loglike(like, δ)`.

The batched method returns a matrix where each column corresponds to one sample in `Δ`.
Rarely, some `Likelihood`s may require the input parameters `x` to compute the log-likelihood as well.

Only defined for [`Marginalizable`](@ref) likelihoods, which implement the `δ`-only method;
the rest is provided by the fallbacks below.
"""
loglike_marginal(l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) =
    _loglike_marginal(likelihood_kind(l), l, δ, x)

# marginalizable -> check if a method without x exists
_loglike_marginal(::Marginalizable, l::Likelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real}) =
    loglike_marginal(l, δ)

# default broadcasting (subtypes may override the matrix methods for efficiency)
loglike_marginal(l::Likelihood, Δ::AbstractMatrix{<:Real}) = hcat(loglike_marginal.(Ref(l), eachcol(Δ))...)
loglike_marginal(l::Likelihood, Δ::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real}) = hcat(loglike_marginal.(Ref(l), eachcol(Δ), eachcol(X))...)

"""
    like(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
    like(l::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractVector{<:Real}

Return the likelihood of the observation given the model output `δ`.
"""
like(args...) = exp.(loglike(args...))

"""
    like_marginal(::Likelihood, δ::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::AbstractVector{<:Real}
    like_marginal(l::Likelihood, Δ::AbstractMatrix{<:Real}, [X::AbstractMatrix{<:Real}]) -> ::AbstractMatrix{<:Real}

Return the per-dimension likelihoods of the observation given the model output `δ`.
Only defined for [`Marginalizable`](@ref) likelihoods.
"""
like_marginal(args...) = exp.(loglike_marginal(args...))

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
