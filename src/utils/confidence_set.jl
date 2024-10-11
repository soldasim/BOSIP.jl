
"""
    post, c = find_cutoff(post, x_prior, q; kwargs...)

Estimate the cutoff value `c` such that the set `{x | post(x) > c}`
contains `q` of the total probability mass.

The approximation is calculated by MC sampling from the `x_prior`.

The returned posterior `post` is unchanged.

# Keywords

- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `x_prior` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used for the approximation.
        Only has an effect if `isnothing(xs)`.

# See Also

[`approx_cutoff_area`](@ref)
[`set_iou`](@ref)
"""
function find_cutoff(post, x_prior, q; xs=nothing, samples=10_000)
    isnothing(xs) && (xs = rand(x_prior, samples))
    ws = post.(eachcol(xs)) ./ pdf.(Ref(x_prior), eachcol(xs))
    vals = post.(eachcol(xs))
    c = quantile(vals, Distributions.weights(ws), 1. - q)
    return post, c
end

"""
    V = approx_cutoff_area(post, x_prior, c; kwargs...)

Approximate the ratio of the area where `post(x) > c`
relative to the whole support of `post(x)`.

The approximation is calculated by MC sampling from the `x_prior`.

The prior `x_prior` must support the whole support of `post(x)`.

# Keywords

- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `x_prior` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used for the approximation.
        Only has an effect if `isnothing(xs)`.

# See Also

[`find_cutoff`](@ref)
[`set_iou`](@ref)
"""
function approx_cutoff_area(post, x_prior, c; xs=nothing, samples=10_000)
    if isnothing(xs)
        xs = rand(x_prior, samples)
    end
    ws = 1 ./ pdf.(Ref(x_prior), eachcol(xs))
    ws ./= sum(ws)
    V = sum(ws[post.(eachcol(xs)) .> c])
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
