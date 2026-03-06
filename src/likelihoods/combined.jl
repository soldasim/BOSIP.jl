
"""
    CombinedLikelihood(Vector{Likelihood}, ::Vector{UnitRange{Int}})
    CombinedLikelihood(; kwargs...)

Combines multiple [`Likelihood`](@ref)s into a single likelihood by assuming independence.

## Keywords
- `likelihoods::Vector{Likelihood}`: A vector of likelihoods to be combined.
- `δ_ranges::Vector{UnitRange{Int}}`: A vector of ranges specifying the indices
    of the proxy variable `δ` corresponding to each likelihood.
"""
@kwdef struct CombinedLikelihood <: Likelihood
    likelihoods::Vector{Likelihood}
    δ_ranges::Vector{UnitRange{Int}}

    function CombinedLikelihood(likelihoods::Vector{Likelihood}, δ_ranges::Vector{UnitRange{Int}})
        @assert length(likelihoods) > 0
        @assert length(likelihoods) == length(δ_ranges)
        new(likelihoods, δ_ranges)
    end
end
function CombinedLikelihood(likelihoods, δ_ranges)
    likelihoods = convert(Vector{Likelihood}, likelihoods)
    δ_ranges = convert(Vector{UnitRange{Int}}, δ_ranges)
    return CombinedLikelihood(likelihoods, δ_ranges)
end

function loglike_marginal(like::CombinedLikelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return vcat([loglike_marginal(l, δ[rng], x) for (l, rng) in zip(like.likelihoods, like.δ_ranges)]...)
end

function log_marginal_likelihood_mean(like::CombinedLikelihood, model_post::ModelPosterior)
    @error "`CombinedLikelihood` only supports sliceable `SurrogateModel`s for now."
    throw(MethodError(log_marginal_likelihood_mean, (like, model_post)))
end
function log_marginal_likelihood_mean(like::CombinedLikelihood, model_post::BOSS.DefaultModelPosterior)
    model_posts = [BOSS.DefaultModelPosterior(model_post.slices[rng]) for rng in like.δ_ranges]
    ml_means = log_marginal_likelihood_mean.(like.likelihoods, model_posts)

    function log_ml_mean(x::AbstractVector{<:Real})
        return vcat([f(x) for f in ml_means]...)
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        return vcat([f(X) for f in ml_means]...)
    end
    return log_ml_mean
end

function loglike(like::CombinedLikelihood, δ::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return mapreduce((l, rng) -> loglike(l, δ[rng], x), +, like.likelihoods, like.δ_ranges)
end

function log_likelihood_mean(like::CombinedLikelihood, model_post::ModelPosterior)
    @error "`CombinedLikelihood` only supports sliceable `SurrogateModel`s for now."
    throw(MethodError(log_likelihood_mean, (like, model_post)))
end
function log_likelihood_mean(like::CombinedLikelihood, model_post::BOSS.DefaultModelPosterior)
    model_posts = [BOSS.DefaultModelPosterior(model_post.slices[rng]) for rng in like.δ_ranges]
    ll_means = log_likelihood_mean.(like.likelihoods, model_posts)

    function log_like_mean(x::AbstractVector{<:Real})
        return mapreduce(f -> f(x), +, ll_means)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return mapreduce(f -> f(X), .+, ll_means)
    end
    return log_like_mean
end

function log_likelihood_variance(like::CombinedLikelihood, model_post::ModelPosterior)
    @error "`CombinedLikelihood` only supports sliceable `SurrogateModel`s for now."
    throw(MethodError(log_likelihood_variance, (like, model_post)))
end
function log_likelihood_variance(like::CombinedLikelihood, model_post::BOSS.DefaultModelPosterior)
    model_posts = [BOSS.DefaultModelPosterior(model_post.slices[rng]) for rng in like.δ_ranges]
    ll_means = log_likelihood_mean.(like.likelihoods, model_posts)
    ll_vars = log_likelihood_variance.(like.likelihoods, model_posts)

    # numerically stable calculation without any subtraction or division
    # but the complexity is exponential w.r.t. the number of individual likelihoods

    # prepare iterators over all subsets with size in [1, n-1] and the complementary subsets
    n = length(like.likelihoods)
    subsets = powerset(1:n, 1, n-1) |> collect
    complements = reverse(subsets)
    subsets_length = length(subsets)

    # check that subsets returned by `powerset` are ordered
    # this is not guaranteed by the documentation, and may break
    indices = Set(1:n)   
    for (s, s_) in zip(subsets, complements)
        @assert Set(vcat(s, s_)) == indices
    end

    function log_like_var(x::AbstractVector{<:Real})
        log_σ2 = [f(x) for f in ll_vars]
        log_μ2 = [2 * f(x) for f in ll_means]

        log_terms = zeros(subsets_length + 1)
        for (i, (s, s_)) in enumerate(zip(subsets, complements))
            log_terms[i] = sum(log_σ2[s]) + sum(log_μ2[s_])
        end
        log_terms[end] = sum(log_σ2)

        return logsumexp(log_terms)
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end
