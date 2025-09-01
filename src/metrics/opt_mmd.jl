
"""
    OptMMDMetric(; kwargs...)

Measures the quality of the posterior approximation by sampling from the true posterior
and the approximate posterior and calculating the Maximum Mean Discrepancy (MMD).

In constrast to [`MMDMetric`](@ref), this metric optimizes the kernel lengthscales
automatically during each evaluation of the metric.

# Keywords
- `kernel::Kernel`: The kernel used to calculate the MMD.
    (Provide a kernel *without* lengthscales as they are optimized automatically.)
- `bounds::AbstractBounds`: The domain bounds of the `BosipProblem`.
- `algorithm`: The optimization algorithm used to optimize the kernel lengthscales.
- `kwargs...`: Additional keyword arguments passed to the optimization algorithm.
"""
struct OptMMDMetric <: SampleMetric
    kernel::Kernel
    bounds::AbstractBounds
    algorithm::Any
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptMMDMetric(;
    kernel,
    bounds,
    algorithm,
    kwargs...
)
    return OptMMDMetric(kernel, bounds, algorithm, kwargs)
end

function calculate_metric(mmd::OptMMDMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real};
    options::BosipOptions = BosipOptions(),    
)
    options.info && @info "Optimizing kernel lengthscales for the MMD metric ..."

    function _calc_mmd(λ::AbstractVector{<:Real})
        kernel = with_lengthscale(mmd.kernel, λ)
        val = calc_mmd(kernel, eachcol(true_samples), eachcol(approx_samples))
        return val
    end

    f(λ, _) = (-1) * _calc_mmd(λ) # maximize
    x0 = (mmd.bounds[2] .- mmd.bounds[1]) ./ 3

    prob = OptimizationProblem(f, x0, nothing;
        lb = [0., 0.],
        ub = mmd.bounds[2] .- mmd.bounds[1],
    )
    sol = solve(prob, mmd.algorithm)

    return _calc_mmd(sol.u)
end

# Implemented in "src/metrics/mmd.jl".
function calc_mmd end
