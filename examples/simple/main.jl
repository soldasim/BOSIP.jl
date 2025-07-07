using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA
using Turing

using Random
Random.seed!(555)

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

    ### algorithms
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 200,
        parallel,
        rhoend = 1e-4,
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 200,
        parallel,
        rhoend = 1e-4,
    )
    term_cond = IterLimit(50)

    ### plotting
    plt = MakiePlots.PlotCallback(;
        title = "",
        plot_each = 10,
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        step = 0.05,
    )
    options = BolfiOptions(;
        callback = plots ? plt : BOSS.NoCallback(),
    )

    # RUN
    MakiePlots.init_plotting(plt)
    bolfi!(problem; model_fitter, acq_maximizer, term_cond, options)
    
    # final plot
    plots && MakiePlots.plot_state(problem, nothing; plt, iter=term_cond.iter_max)

    # marginals
    plot_marginals_int(problem; func=posterior_mean, lhc_grid_size=20)

    # MakiePlots.plot_param_slices(plt, problem; options, samples=2_000)
    return problem
end
