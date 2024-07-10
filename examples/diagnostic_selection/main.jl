using BOLFI
using BOSS
using Distributions

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot.jl")

# MAIN SCRIPT
# Change the experiment by changing the `const mode` in `toy_problem.jl`.
function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    acquisition = ToyProblem.acquisition()

    model_fitter = BOSS.SamplingMAP(;
        samples = 200,
        parallel = true,
    )
    acq_maximizer = BOSS.GridAM(;
        problem.problem,
        steps = [0.05, 0.05],
        parallel = true,
    )

    # EXPERIMENTAL TERMINATION CONDITIONS
    xs = rand(problem.x_prior, 10_000)
    max_iters = 50

    term_cond = AEConfidence(;
        xs,
        q = 0.8,
        r = 0.95,
        max_iters,
    )
    # term_cond = UBLBConfidence(;
    #     xs = rand(problem.x_prior, samples),
    #     n = 1.,
    #     q = 0.8,
    #     r = 0.8,
    #     max_iters
    # )
    # term_cond = IterLimit(25);

    save_plots = true
    plot_dir = "./examples/diagnostic_selection/plots/_new_"

    options = BolfiOptions(;
        callback = Plot.PlotCallback(;
            plot_each = 5,          # todo: integer
            term_cond,
            save_plots,
            plot_dir,
            put_in_scale = false,
            square_layout = false,   # todo: `true`, `false`
            ftype = "png",          # todo: `png`, `pdf`
        ),
    )

    Plot.init_plotting(; save_plots, plot_dir)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)

    # final state                   # todo: comment / uncomment
    plt = options.callback
    Plot.plot_state(problem, nothing; ftype=plt.ftype, square_layout=plt.square_layout, term_cond, iter=plt.iters, save_plots, plot_dir, plot_name="p_final", acquisition)
    
    return problem
end
