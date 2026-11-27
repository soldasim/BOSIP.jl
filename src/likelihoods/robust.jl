
"""
    RobustLikelihood(; like, α, A)
    RobustLikelihood(; like, ε)

Wraps a [`Likelihood`](@ref) `ℓ` to bound how much a single wrongly-behaving likelihood
can affect the posterior.

It is mostly intended to be used on the individual likelihoods inside a [`CombinedLikelihood`](@ref).
Used alone, it has almost no effect on the posterior.

The resulting likelihood is always [`JointOnly`](@ref), as the mixture does not factorize over dimensions.

## Mixture parametrization (`α`, `A`)

With probability `α` the observation is considered corrupted and a uniform density `1/A` is used instead of `ℓ`:

`ℓ'(δ) = (1 - α) * ℓ(δ) + α / A`

The resulting likelihood is normalized.

### Keywords
- `like::Likelihood`: The wrapped likelihood.
- `α::Float64`: The probability of the observation being corrupted. Must be in `(0, 1)`.
- `A::Float64`: The volume of the domain of the observation modeled by `like`.

## Additive parametrization (`ε`)

A constant `ε` is directly added to the likelihood:

`ℓ'(δ) = ℓ(δ) + ε`

The resulting likelihood is not normalized. The posterior is identical to the one
of the mixture with `ε = α / ((1 - α) * A)`; only the absolute scale of the likelihood
(and of its variance) differs.

### Keywords
- `like::Likelihood`: The wrapped likelihood.
- `ε::Float64`: The constant added to the likelihood. Must be positive.
"""
struct RobustLikelihood{L<:Likelihood} <: Likelihood
    like::L
    log_w::Float64
    log_ε::Float64

    function RobustLikelihood(like::L, α::Real, A::Real) where {L<:Likelihood}
        @assert 0 < α < 1
        @assert A > 0
        new{L}(like, log1p(-α), log(α) - log(A))
    end
    function RobustLikelihood(like::L, ε::Real) where {L<:Likelihood}
        @assert ε > 0
        new{L}(like, 0.0, log(ε))
    end
end
function RobustLikelihood(; like, α=nothing, A=nothing, ε=nothing)
    if isnothing(ε)
        @assert !isnothing(α) && !isnothing(A) "Provide either `α` and `A`, or `ε`."
        return RobustLikelihood(like, α, A)
    else
        @assert isnothing(α) && isnothing(A) "Provide either `α` and `A`, or `ε`."
        return RobustLikelihood(like, ε)
    end
end

likelihood_kind(::RobustLikelihood) = JointOnly()

_logaddexp(a, b) = max(a, b) + log1p(exp(-abs(a - b)))

# `log(ℓ')` given `log(ℓ)`, or given `log(E[ℓ])`
_robust_log(like::RobustLikelihood, log_ℓ) = _logaddexp(like.log_w + log_ℓ, like.log_ε)

function loglike(like::RobustLikelihood, δ::AbstractVector{<:Real})
    return _robust_log(like, loglike(like.like, δ))
end
function loglike(like::RobustLikelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return _robust_log(like, loglike(like.like, δ, x))
end
function loglike(like::RobustLikelihood, Δ::AbstractMatrix{<:Real})
    return _robust_log.(Ref(like), loglike(like.like, Δ))
end
function loglike(like::RobustLikelihood, Δ::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Real})
    return _robust_log.(Ref(like), loglike(like.like, Δ, X))
end

# `E[ℓ'] = w * E[ℓ] + ε`
function log_likelihood_mean(::PredictiveKind, like::RobustLikelihood, model_post::ModelPosterior)
    log_like_mean = log_likelihood_mean(like.like, model_post)

    function log_like_mean_robust(x::AbstractVector{<:Real})
        return _robust_log(like, log_like_mean(x))
    end
    function log_like_mean_robust(X::AbstractMatrix{<:Real})
        return _robust_log.(Ref(like), log_like_mean(X))
    end
    return log_like_mean_robust
end

# `V[ℓ'] = w² * V[ℓ]`
function log_likelihood_variance(::PredictiveKind, like::RobustLikelihood, model_post::ModelPosterior)
    log_like_var = log_likelihood_variance(like.like, model_post)

    function log_like_var_robust(x::AbstractArray{<:Real})
        return (2 * like.log_w) .+ log_like_var(x)
    end
    return log_like_var_robust
end

"""
    make_robust(like::CombinedLikelihood, ε) -> ::CombinedLikelihood

Return a new [`CombinedLikelihood`](@ref) where each individual likelihood is wrapped
in a [`RobustLikelihood`](@ref) with the additive parameter `ε`.
Throws an `ArgumentError` if any of the individual likelihoods is already a `RobustLikelihood`.

## Arguments
- `like::CombinedLikelihood`: The likelihood to make robust.
- `ε::Real` or `ε::AbstractVector{<:Real}`: The constant added to each individual likelihood,
    either shared by all, or one value per individual likelihood.
"""
function make_robust(like::CombinedLikelihood, ε::Real)
    return make_robust(like, fill(ε, length(like.likelihoods)))
end
function make_robust(like::CombinedLikelihood, ε::AbstractVector{<:Real})
    @assert length(ε) == length(like.likelihoods)
    any(l -> l isa RobustLikelihood, like.likelihoods) &&
        throw(ArgumentError("Some individual likelihoods are already `RobustLikelihood`s."))
    return CombinedLikelihood(RobustLikelihood.(like.likelihoods, ε), like.δ_ranges)
end
