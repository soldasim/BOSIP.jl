
"""
    evidence(post, x_prior; kwargs...)

Return the estimated evidence ``\\hat{p}(y_o)``.

# Arguments
- `post`: A function `::AbstractVector{<:Real} -> ::Real`
        representing the posterior ``p(x|y_o)``.
- `x_prior`: A multivariate distribution
        representing the prior ``p(x)``.

# Keywords
- `xs::Union{Nothing, <:AbstractMatrix{<:Real}}`: Can be used to provide a pre-sampled
        set of samples from the `x_prior` as a column-wise matrix.
- `samples::Int`: Controls the number of samples used to estimate the evidence.
        Only has an effect if `isnothing(xs)`.
"""
function evidence(post, x_prior; xs=nothing, samples=10_000)
    ll(x) = post(x) / pdf(x_prior, x)
    isnothing(xs) && (xs = rand(x_prior, samples))
    py = mean((ll(x) for x in eachcol(xs)))
    return py
end
