using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA

using Random
Random.seed!(555)

include("toy_problem.jl")
include("makie/makie.jl")

# MAIN SCRIPT
function script_bolfi(;
    init_data=3,
)
    problem = ToyProblem.bolfi_problem(init_data)
    acquisition = PostVarAcq()

    model_fitter = SamplingMAP(;
        samples = 200,
        parallel = true,
    )
    acq_maximizer = GridAM(;
        problem.problem,
        steps = fill(0.05, ToyProblem.x_dim()),
        parallel = true,
    )
    term_cond = IterLimit(25)

    plt = MakiePlots.PlotCallback(;
        title = "",
        plot_each = 5,
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        step = 0.05,  # TODO
    )
    options = BolfiOptions(;
        callback = plt,
    )

    MakiePlots.init_plotting(plt)
    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    MakiePlots.plot_final(plt; acquisition, model_fitter, acq_maximizer, term_cond, options)
    
    # MakiePlots.plot_param_slices(plt, problem; options, samples=2_000)
    return problem
end
