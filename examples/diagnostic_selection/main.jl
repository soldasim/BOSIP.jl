using BOLFI
using BOSS
using Distributions

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot.jl")

function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    acquisition = SetsPostVariance()

    model_fitter = BOSS.SamplingMLE(;
        samples = 200,
        parallel = true,
    )
    acq_maximizer = BOSS.GridAM(;
        problem.problem,
        steps = [0.05, 0.05],
        parallel = true,
    )

    # term_cond = ConfidenceTermCond(;
    #     # samples = 10_000,
    #     xs = rand(problem.x_prior, 10_000),
    #     q = 0.95,
    #     r = 0.95,
    #     max_iters = 50,
    # )
    term_cond = UBLBConfidence(;
        # samples = 10_000,
        xs = rand(problem.x_prior, 10_000),
        n = 1.,
        q = 0.8,
        r = 0.8,
        max_iters = 30,
    )

    save_plots = true
    plot_dir = "./examples/diagnostic_selection/plots"

    options = BolfiOptions(;
        callback = PlotCallback(;
            plot_each = 5,
            term_cond,
            save_plots,
            plot_dir,
            put_in_scale = false,
        ),
    )

    init_plotting(; save_plots, plot_dir)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    return problem
end
