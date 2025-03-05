
"""
    c = find_cutoff(target_pdf, xs, q)
    c = find_cutoff(target_pdf, xs, ws, q)

Estimate the cutoff value `c` such that the set `{x | post(x) >= c}`
contains `q` of the total probability mass.

The value `c` is estimated based on the provided samples `xs` sampled according to the `target_pdf`.

Alternatively, one can provide samples `xs` sampled according to some `proposal_pdf`
with corresponding importance weights `ws = target_pdf.(eachcol(xs)) ./ proposal_pdf.(eachcol(xs))`.

# See Also

[`approx_cutoff_area`](@ref)
[`set_iou`](@ref)
"""
function find_cutoff(target_pdf, xs::AbstractMatrix{<:Real}, q::Real)
    ws = ones(size(xs, 2))
    return find_cutoff(target_pdf, xs, ws, q)
end
function find_cutoff(target_pdf, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, q::Real)
    vals = target_pdf.(eachcol(xs))
    c = quantile(vals, Distributions.weights(ws), 1. - q)
    return c
end

"""
    V = approx_cutoff_area(target_pdf, xs, c)
    V = approx_cutoff_area(target_pdf, xs, ws, c)

Approximate the ratio of the area where `target_pdf(x) >= c`
relative to the whole support of `target_pdf`.

The are is estimated based on the provided samples `xs`
sampled uniformly from the whole support of `target_pdf`.

Alternatively, one can provide samples `xs` sampled according to some `proposal_pdf`
with corresponding importance weights `ws = 1 ./ proposal_pdf.(eachcol(xs))`.

# See Also

[`find_cutoff`](@ref)
[`set_iou`](@ref)
"""
function approx_cutoff_area(target_pdf, xs, c)
    ws = ones(size(xs, 2))
    return approx_cutoff_area(target_pdf, xs, ws, c)
end
function approx_cutoff_area(target_pdf, xs, ws, c)
    ws = deepcopy(ws)
    ws .= exp.( log.(ws) .- log(sum(ws)) ) # normalize
    V = sum(ws[target_pdf.(eachcol(xs)) .>= c])
    return V
end

"""
    iou = set_iou(in_A, in_B, x_prior, xs)

Approximate the intersection-over-union of two sets A and B.

The parameters `in_A`, `in_B` are binary arrays declaring which samples
from `xs` fall into the sets A and B. The column-wise matrix `xs` contains
the parameter samples. The samples have to be drawn from the common prior `x_prior`.

# See Also

[`find_cutoff`](@ref)
[`approx_cutoff_area`](@ref)
"""
function set_iou(in_A, in_B, x_prior, xs)
    isnothing(xs) && (xs = rand(x_prior, samples))

    ws = 1 ./ pdf.(Ref(x_prior), eachcol(xs))
    ws ./= sum(ws)

    V_intersect = sum(ws[in_A .&& in_B])
    V_union = sum(ws[in_A .|| in_B])
    return V_intersect / V_union
end
