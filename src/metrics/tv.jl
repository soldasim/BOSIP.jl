
"""
    TVMetric(; kwargs...)

Measures the quality of the posterior approximation by approximating the Total Variation (TV) distance
based on a precomputed parameter grid.

# Keywords
- `grid::Matrix{Float64}`: The parameter grid used to approximate the TV integral.
- `log_ws::Vector{Float64}`: The log-weights for the grid points. Should be `1 / q(x)`,
        where `q(x)` is the probability density function of the distribution
        used to sample the grid points.
        (`1 / domain_area` is appropriate for an evenly distributed grid)
        (It is also possible to provide the non-logarithmic weights `ws` instead.)
"""
struct TVMetric <: PDFMetric
    grid::Matrix{Float64}
    log_ws::Vector{Float64}
end
function TVMetric(;
    grid,
    ws = nothing,
    log_ws = nothing,
)
    @assert xor(isnothing(ws), isnothing(log_ws))
    isnothing(log_ws) && (log_ws = log.(ws))
    return TVMetric(grid, log_ws)
end

function calculate_metric(tv::TVMetric, true_logpost::Function, approx_logpost::Function;
    options::BolfiOptions = BolfiOptions(),    
)
    return calc_tv(tv.grid, tv.log_ws, true_logpost, approx_logpost)
end

function calc_tv(grid::AbstractMatrix{<:Real}, log_ws::AbstractVector{<:Real}, true_logpost::Function, approx_logpost::Function)
    ### pdf values on grid
    # true_vals = exp.( true_logpost.(eachcol(grid)) )
    # approx_vals = exp.( approx_logpost.(eachcol(grid)) )
    true_logvals = true_logpost.(eachcol(grid))
    approx_logvals = approx_logpost.(eachcol(grid))

    ### estimate the evidence on the same grid
    # true_ev = mean(ws .* true_vals)
    # approx_ev = mean(ws .* approx_vals)
    true_logev = log( mean( exp.(log_ws .+ true_logvals) ) )
    approx_logev = log( mean( exp.(log_ws .+ approx_logvals) ) )

    ### normalize the pdf values
    # true_vals ./= true_ev
    # approx_vals ./= approx_ev
    true_logvals .-= true_logev
    approx_logvals .-= approx_logev

    # calculate the total variation distance
    # return (1/2) * sum(abs.(approx_vals .- true_vals))
    return (1/2) * sum(abs.( exp.(approx_logvals) .- exp.(true_logvals) ))
end
