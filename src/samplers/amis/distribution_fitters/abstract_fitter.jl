
abstract type DistributionFitter end


# API

"""
Find the optimal parameters of the given `ProposalDistribution` that best fit the given data `xs`.
"""
function fit_distribution!(::DistributionFitter, ::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real};
    options::BolfiOptions = BolfiOptions(),
) end


# Default implementations

fit_distribution!(fitter::DistributionFitter, dist::ProposalDistribution, xs::AbstractMatrix{<:Real};
    options::BolfiOptions = BolfiOptions(),    
) = fit_distribution!(fitter, dist, xs, ones(size(xs, 2)); options)
