
"""
    OptimizationFitter(; algorithm, kwargs...)

A [`DistributionFitter`](@ref) that fits the [`ProposalDistribution`](@ref) parameters by numerically
maximizing the (weighted) sample log-likelihood via the Optimization.jl library.

## Keywords
- `algorithm`: The optimization algorithm passed to Optimization.jl.
- `multistart::Int`: The number of optimization restarts.
- `parallel::Bool`: If `true`, the restarts are run in parallel.
- `static_schedule::Bool`: If `true`, the `:static` schedule is used for parallelization.
- `autodiff`: The automatic differentiation type passed to the `OptimizationFunction`.
- `kwargs...`: Additional keyword arguments passed to the optimization algorithm.
"""
struct OptimizationFitter{A} <: DistributionFitter
    algorithm::A
    multistart::Int64
    parallel::Bool
    static_schedule::Bool
    autodiff::AbstractADType
    kwargs::Base.Pairs{Symbol, <:Any}
end
function OptimizationFitter(;
    algorithm,
    multistart = 200,
    parallel = false,
    static_schedule = false,
    autodiff = AutoForwardDiff(),
    kwargs...
)
    return OptimizationFitter(
        algorithm,
        multistart,
        parallel,
        static_schedule,
        autodiff,
        kwargs,
    )
end

function fit_distribution!(opt::OptimizationFitter, dist::ProposalDistribution, xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real};
    options::BosipOptions = BosipOptions(),
)
    init_θs = initial_params(dist, opt.multistart)
    ll = loglikelihood(dist, xs, ws)

    optimization_function = OptimizationFunction((θ, _) -> -ll(θ), opt.autodiff)
    optimization_problem = (init_θ) -> OptimizationProblem(optimization_function, init_θ, nothing)

    function optimize(init_θ)
        params = Optimization.solve(optimization_problem(init_θ), opt.algorithm; opt.kwargs...).u
        loglike = ll(params)
        return params, loglike
    end

    θ_opt, _ = BOSS.optimize_multistart(optimize, init_θs;
        opt.parallel,
        opt.static_schedule,
        options = create_boss_options(options),
    )
    set_params!(dist, θ_opt)
end
