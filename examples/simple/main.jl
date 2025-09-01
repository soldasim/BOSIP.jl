using BOSIP
using BOSS
using Distributions
using OptimizationPRIMA
using Turing
using KernelFunctions
using JLD2

# TODO rem
using CairoMakie
using Optimization
using OptimizationOptimJL

using Random
Random.seed!(555)

# TODO rem
log_posterior_estimate() = log_posterior_mean
# log_posterior_estimate() = log_approx_posterior

include("toy_problem.jl")
include("makie/makie.jl")

function main(;
    init_data = 3,
    kwargs...
)
    problem = ToyProblem.bosip_problem(init_data)
    return main(problem; kwargs...)
end

function main(problem;
    plots = true,
    metric = false,
    parallel = true,
)
    bounds = ToyProblem.get_bounds()

    ### algorithms
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel,
        rhoend = 1e-4,
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 24,
        parallel,
        rhoend = 1e-4,
    )
    term_cond = IterLimit(50)

    ### sampler
    # sampler = RejectionSampler(;
    #     logpdf_maximizer = LogpdfMaximizer(;
    #         algorithm = BOBYQA(),
    #         multistart = 24,
    #         parallel,
    #         static_schedule = true, # issues with PRIMA.jl
    #         rhoend = 1e-4,
    #     ),
    # )
    sampler = AMISSampler(;
        iters = 10,
        proposal_fitter = BOSIP.AnalyticalFitter(), # re-fit the proposal analytically
        # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
        #     algorithm = NEWUOA(),
        #     multistart = 6,
        #     parallel,
        #     rhoend = 1e-2,
        # ),
        # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
        gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
            algorithm = BOBYQA(),
            multistart = 24,
            parallel,
            cluster_ϵs = nothing,
            rel_min_weight = 1e-8,
            rhoend = 1e-4,
        ),
    )
    # sampler = TuringSampler(;
    #     sampler = NUTS(1000, 0.65),
    #     warmup = 400,
    #     chain_count = 6,
    #     leap_size = 5,
    #     parallel,
    # )

    ### metrics
    λ_count = 10
    ds = (bounds[2] .- bounds[1])
    λ_ranges = [range(0., d; length=λ_count+1)[2:end] for d in ds]
    ls = Iterators.product(λ_ranges...)
    ks = map(l -> with_lengthscale(GaussianKernel(), [l...]), ls)
    
    metric_cb = MetricCallback(;
        reference = ToyProblem.true_logpost,
        logpost_estimator = log_posterior_estimate(),
        sampler,
        sample_count = 200,
        # TODO
        # metric = MMDMetric(;
        #     kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
        #     # kernel = KernelSum(vec(ks)),
        # ),
        metric = OptMMDMetric(;
            kernel = GaussianKernel(),
            bounds,
            algorithm = BOBYQA(),
        ),
    )

    ### plotting
    plt = MakiePlots.PlotCallback(;
        title = "",
        plot_each = 10,
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        step = 0.05,
        sampler,
    )
    options = BosipOptions(;
        callback = CombinedCallback(
            metric ? metric_cb : BOSIP.NoCallback(),
            plots ? plt : BOSIP.NoCallback(),
        ),
    )

    # RUN
    MakiePlots.init_plotting(plt)
    bosip!(problem; model_fitter, acq_maximizer, term_cond, options)

    # final plot
    # plots && MakiePlots.plot_state(problem, nothing; plt, iter=term_cond.iter_max)

    # marginals
    # plot_marginals_int(problem; func=posterior_mean, lhc_grid_size=200, matrix_ops=false) # TODO

    ### save
    pname = "test"
    @save "./examples/data/results_0/$pname.jld2" score=metric_cb.score_history problem=problem
    
    # # TODO rem
    # fig = Figure(resolution = (600, 400))
    # ax = Axis(fig[1, 1], xlabel = "Iteration", ylabel = "Score", title = "Metric Score History")
    # lines!(ax, 1:length(metric_cb.score_history), metric_cb.score_history, color = :blue)
    # display(fig)

    return problem, metric_cb.score_history # TODO rem
end
