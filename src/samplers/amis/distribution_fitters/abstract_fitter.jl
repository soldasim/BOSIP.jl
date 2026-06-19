
"""
Abstract type for algorithms that fit a [`ProposalDistribution`](@ref)'s parameters to weighted samples.

Used by the [`AMISSampler`](@ref) to adapt its proposal distribution each iteration.
Subtypes implement `fit_distribution!`. See [`AnalyticalFitter`](@ref) and [`OptimizationFitter`](@ref).
"""
abstract type DistributionFitter end


# API

"""
Find the optimal parameters of the given `ProposalDistribution` that best fit the given data `xs`.
"""
function fit_distribution!(::DistributionFitter, ::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real};
    options::BosipOptions = BosipOptions(),
) end


# Default implementations

fit_distribution!(fitter::DistributionFitter, dist::ProposalDistribution, xs::AbstractMatrix{<:Real};
    options::BosipOptions = BosipOptions(),    
) = fit_distribution!(fitter, dist, xs, ones(size(xs, 2)); options)
