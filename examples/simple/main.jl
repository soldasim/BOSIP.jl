using BOLFI
using BOSS
using Distributions

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot.jl")

# MAIN SCRIPT
function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    acquisition = PostVarAcq()

    model_fitter = BOSS.SamplingMAP(;
        samples = 200,
        parallel = true,
    )
    acq_maximizer = BOSS.GridAM(;
        problem.problem,
        steps = [0.05, 0.05],
        parallel = true,
    )

    save_plots = true
    plot_dir = "./examples/simple/plots/_new_"
    options = BolfiOptions()

    ITER_TOTAL = 25
    PLOT_EACH = 5

    noise_std_true = ToyProblem.σe_true

    init_plotting(; save_plots, plot_dir)
    iters = 0
    while iters < ITER_TOTAL
        iters += PLOT_EACH
        term_cond = BOSS.IterLimit(PLOT_EACH)
        bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
        plot_state(problem; save_plots, plot_dir, plot_name="p_$iters", noise_std_true, acquisition)
    end
    return problem
end
