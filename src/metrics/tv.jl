
"""
    TVMetric(; kwargs...)

Measures the quality of the posterior approximation by approximating the Total Variation (TV) distance
based on a precomputed parameter grid.

# Keywords
- `grid::Matrix{Float64}`: The parameter grid used to approximate the TV integral.
- `ws::Vector{Float64}`: The weights for the grid points. Should be `1 / q(x)`,
        where `q(x)` is the probability density function of the distribution
        used to sample the grid points.
        (`1 / domain_area` is appropriate for an evenly distributed grid)
"""
@kwdef struct TVMetric <: PDFMetric
    grid::Matrix{Float64}
    ws::Vector{Float64}
end

function calculate_metric(tv::TVMetric, true_logpost::Function, approx_logpost::Function;
    options::BolfiOptions = BolfiOptions(),    
)
    return calc_tv(tv.grid, tv.ws, true_logpost, approx_logpost)
end

function calc_tv(grid::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, true_logpost::Function, approx_logpost::Function)
    true_post(x) = exp(true_logpost(x))
    approx_post(x) = exp(approx_logpost(x))
    
    # pdf values on grid
    true_vals = true_post.(eachcol(grid))
    approx_vals = approx_post.(eachcol(grid))

    # normalize by evidence estimated on the grid
    # true_vals ./= sum(true_vals)
    # approx_vals ./= sum(approx_vals)
    true_ev = mean(ws .* true_vals)
    approx_ev = mean(ws .* approx_vals)

    true_vals ./= true_ev
    approx_vals ./= approx_ev

    # calculate the total variation distance
    return (1/2) * sum(abs.(approx_vals .- true_vals))
end
