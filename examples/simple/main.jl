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
    term_cond = BOSS.IterLimit(25)

    plt = Plot.PlotCallback(;
        plot_each = 5,
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        noise_std_true = ToyProblem.σe_true,
    )
    options = BolfiOptions(;
        callback = plt,
    )

    Plot.init_plotting(plt)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    Plot.plot_final(plt; acquisition, model_fitter, acq_maximizer, term_cond, options)
    
    Plot.plot_param_slices(plt; samples=2000)

    return problem
end
