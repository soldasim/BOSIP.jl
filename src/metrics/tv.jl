
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
- `true_logpost::Function`: The (potentially unnormalized) log-pdf of the true posterior distribution.
        If provided, the log-pdf values on the grid are cached, which greatly improves performance.
- `true_logvals::Vector{Float64}`: The (potentially unnormalized) log-pdf values of the true posterior distribution on the grid.
        If provided, these are used directly instead of computing them from `true_logpost`.
        This is useful when the true log-pdf values are pre-computed or can be computed more efficiently in a batch.
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
    true_logvals = nothing
)
    @assert xor(isnothing(ws), isnothing(log_ws))
    isnothing(log_ws) && (log_ws = log.(ws))
    
    if isnothing(true_logvals) && !isnothing(true_logpost)
        true_logvals = true_logpost.(eachcol(grid))
    end

    return TVMetric(grid, log_ws, true_logvals)
end

function calculate_metric(tv::TVMetric, true_logpost::Function, approx_logpost::Function;
    options::BosipOptions = BosipOptions(),    
)
    # true (unnormalized) logpdf values
    if isnothing(tv.true_logvals)
        true_logvals = true_logpost.(eachcol(tv.grid))
    else
        true_logvals = tv.true_logvals
    end

    # approx. (unnormalized) logpdf values
    approx_logvals = approx_logpost.(eachcol(tv.grid))

    return calc_tv(tv.log_ws, approx_logvals, true_logvals)
end

function calc_tv(log_ws, approx_logvals, true_logvals)
    # normalize the distributions
    ev_approx = logmeanexp(log_ws .+ approx_logvals)
    ev_true = logmeanexp(log_ws .+ true_logvals)

    # handle the case where the distributions are not normalizable (e.g. due to numerical issues)
    approx_logvals_norm = isinf(ev_approx) ? fill(-Inf, length(approx_logvals)) : approx_logvals .- ev_approx
    true_logvals_norm = isinf(ev_true) ? fill(-Inf, length(true_logvals)) : true_logvals .- ev_true

    # calculate tv
    diffs = @. abs( exp(approx_logvals_norm) - exp(true_logvals_norm) )
    tv = (1/2) * exp(logmeanexp(log_ws .+ log.(diffs)))

    return tv
end
