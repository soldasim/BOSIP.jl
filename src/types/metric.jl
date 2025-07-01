
"""
Subtypes of `DistributionMetric` are used to evaluate the quality of the posterior approximation.

The `DistributionMetric`s are grouped into two categories; [`SampleMetric`](@ref) and [`PDFMetric`](@ref).
"""
abstract type DistributionMetric end

"""
`SampleMetric` is a subtype of `DistributionMetric` that evaluates the quality of the posterior approximation
based on samples drawn from the true and approximate posteriors.

Each subtype of `SampleMetric` *should* implement:
- `calculate_metric(::DistributionMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real}; kwargs...) -> ::Real`

See also: [`DistributionMetric`](@ref), [`PDFMetric`](@ref)
"""
abstract type SampleMetric <: DistributionMetric end

"""
`PDFMetric` is a subtype of `DistributionMetric` that evaluates the quality of the posterior approximation
based on the probability density functions (pdfs) of the true and approximate posteriors.

Each subtype of `PDFMetric` *should* implement:
- `calculate_metric(::DistributionMetric, true_post::Function, approx_post::Function; kwargs...) -> ::Real`

See also: [`DistributionMetric`](@ref), [`SampleMetric`](@ref)
"""
abstract type PDFMetric <: DistributionMetric end

"""
    score = calculate_metric(::SampleMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real}; kwargs...)
    score = calculate_metric(::PDFMetric, true_post::Function, approx_post::Function; kwargs...)

Evaluate the accuracy of the approximate parameter posterior.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.
"""
function calculate_metric end
