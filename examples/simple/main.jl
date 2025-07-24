using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA
using Turing
using KernelFunctions
using JLD2

using Random
Random.seed!(555)

# TODO rem
# log_posterior_estimate() = log_posterior_mean
log_posterior_estimate() = log_approx_posterior

include("toy_problem.jl")
include("makie/makie.jl")

function main(;
    init_data = 3,
    kwargs...
)
    problem = ToyProblem.bolfi_problem(init_data)
    return main(problem; kwargs...)
end

function main(problem;
    plots = true,
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
    sampler = RejectionSampler(;
        logpdf_maximizer = LogpdfMaximizer(;
            algorithm = BOBYQA(),
            multistart = 24,
            parallel,
            rhoend = 1e-4,
        ),
    )

    ### metrics
    metric_cb = MetricCallback(;
        reference = ToyProblem.true_logpost,
        logpost_estimator = log_posterior_estimate(),
        sampler,
        sample_count = 1000,
        metric = MMDMetric(;
            kernel = with_lengthscale(GaussianKernel(), (bounds[2] .- bounds[1]) ./ 3),
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
    options = BolfiOptions(;
        callback = CombinedCallback(
            metric_cb,
            plots ? plt : BOSS.NoCallback(),
        ),
    )

    # RUN
    MakiePlots.init_plotting(plt)
    bolfi!(problem; model_fitter, acq_maximizer, term_cond, options)

    # final plot
    plots && MakiePlots.plot_state(problem, nothing; plt, iter=term_cond.iter_max)

    # marginals
    plot_marginals_int(problem; func=posterior_mean, lhc_grid_size=20)

    return problem
end
