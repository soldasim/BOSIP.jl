
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
- `true_logpost::Function`: The log-pdf of the true posterior distribution.
        If provided, the log-pdf values on the grid are cached, which greatly improves performance.
"""
struct TVMetric <: PDFMetric
    grid::Matrix{Float64}
    log_ws::Vector{Float64}
    true_logvals::Union{Nothing, Vector{Float64}}
end
function TVMetric(;
    grid,
    ws = nothing,
    log_ws = nothing,
    true_logpost = nothing,
)
    @assert xor(isnothing(ws), isnothing(log_ws))
    isnothing(log_ws) && (log_ws = log.(ws))
    
    if isnothing(true_logpost)
        true_logvals = nothing
    else
        true_logvals = true_logpost.(eachcol(grid))
        true_logev = _log_mean_exp(log_ws .+ true_logvals)
        true_logvals .-= true_logev
    end

    return TVMetric(grid, log_ws, true_logvals)
end

function calculate_metric(tv::TVMetric, true_logpost::Function, approx_logpost::Function;
    options::BosipOptions = BosipOptions(),    
)
    # true logpdf values
    if isnothing(tv.true_logvals)
        true_logvals = true_logpost.(eachcol(tv.grid))
        true_logev = _log_mean_exp(tv.log_ws .+ true_logvals)
        true_logvals .-= true_logev
    else
        true_logvals = tv.true_logvals
    end

    # approx. logpdf values
    approx_logvals = approx_logpost.(eachcol(tv.grid))
    approx_logev = _log_mean_exp(tv.log_ws .+ approx_logvals)
    approx_logvals .-= approx_logev

    return calc_tv(approx_logvals, true_logvals)
end

function calc_tv(approx_logvals, true_logvals)
    return (1/2) * sum(abs.( exp.(approx_logvals) .- exp.(true_logvals) ))
end

function _log_mean_exp(vals::AbstractVector{<:Real})
    M = maximum(vals)
    return M + log( mean( exp.(vals .- M) ) )
end
