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

    q = 0.95  # quantile of interest
    options = BolfiOptions(;
        callback = PlotCallback(;
            q,
            save_plots=false,
            put_in_scale=false,
        ),
    )
    term_cond = ConfidenceTermCond(;
        # samples = 10_000,
        xs = rand(problem.x_prior, 10_000),
        q,
        r = 0.95,
    )

    bolfi!(problem; acquisition, model_fitter, acq_maximizer, term_cond, options)
    return problem
end
