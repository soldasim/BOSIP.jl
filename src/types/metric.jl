
"""
Subtypes of `DistributionMetric` are used to evaluate the quality of the posterior approximation.

Each subtype of `DistributionMetric` *should* implement the following method:
- `calculate_metric(::DistributionMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real}; kwargs...) -> ::Real`
"""
abstract type DistributionMetric end

"""
    score = calculate_metric(::DistributionMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real}; kwargs...)

Evaluate the accuracy of the approximate parameter posterior by comparing the samples
from the true and approximate posterior distributions.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.
"""
function calculate_metric end
