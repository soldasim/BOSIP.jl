
"""
Estimates the proposal distribution parameters analytically.

Only works with proposal distributions which implement `estimate_parameters!`.
"""
struct AnalyticalFitter <: DistributionFitter end

function fit_distribution!(::AnalyticalFitter, dist::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real};
    options::BolfiOptions = BolfiOptions(),
)
    estimate_parameters!(dist, xs, ws)
end
