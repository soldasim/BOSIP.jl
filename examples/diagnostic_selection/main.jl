using BOLFI
using BOSS
using Distributions

include("toy_problem.jl")
include("plot.jl")

function script_bolfi(;
    init_data=3,
    save_plots=false,
    put_in_scale=false,  # puts the approximation in scale with the true posterior
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
    options = BOSS.BossOptions(;
        info = true,
        debug = true,
    )

    ITER_TOTAL = 50
    PLOT_EACH = 10

    noise_vars_true = ToyProblem.σe_true.^2

    iters = 0
    while iters < ITER_TOTAL
        iters += PLOT_EACH
        term_cond = BOSS.IterLimit(PLOT_EACH)
        bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
        plot_state(problem; save_plots, iters, put_in_scale, noise_vars_true, acquisition)
    end
    return problem
end
