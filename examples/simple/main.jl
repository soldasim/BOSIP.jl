using BOLFI
using BOSS
using Distributions
using OptimizationPRIMA
using Turing
using BOSS.KernelFunctions
using CairoMakie
using JLD2
using Statistics
using LinearAlgebra

using Random
Random.seed!(555)

# TODO problem
# include("toy_problem.jl")
include("toy_problem_simple.jl")
include("makie/makie.jl")
include("metric_callback.jl")

function main(;
    init_data = 10, # TODO
    plots = true,
)
    problem = ToyProblem.bolfi_problem(init_data)
    return main(problem; plots)
end

function main(problem;
    plots = true,
)
    parallel = true

    ### algorithms
    model_fitter = OptimizationMAP(;
        algorithm = NEWUOA(),
        multistart = 24,
        parallel,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )
    acq_maximizer = OptimizationAM(;
        algorithm = BOBYQA(),
        multistart = 200,
        parallel,
        static_schedule = true, # issues with PRIMA.jl
        rhoend = 1e-4,
    )
    term_cond = IterLimit(50) # TODO

    ### plotting
    plt = MakiePlots.PlotCallback(;
        title = "",
        plot_each = 10,
        display = true,
        save_plots = true,
        plot_dir = "./examples/simple/plots/_new_",
        plot_name = "p",
        step = 0.5,
    )

    ### TODO metric
    metric_cb = MetricCallback(;
        # metric = MMDMetric(;
        #     kernel = with_lengthscale(GaussianKernel(), [10/3, 10/3]),
        # ),
        metric = TVMetric(ToyProblem.get_eval_grid()...),
        # sampler = RejectionSampler(;
        #     likelihood_maximizer = LikelihoodMaximizer(;
        #         algorithm = BOBYQA(),
        #         multistart = 24,
        #         parallel,
        #         static_schedule = true, # issues with PRIMA.jl
        #         rhoend = 1e-4,
        #     ),
        # ),
        sampler = AMISSampler(;
            iters = 10,
            proposal_fitter = BOLFI.AnalyticalFitter(), # re-fit the proposal analytically
            # proposal_fitter = OptimizationFitter(;      # re-fit the proposal by MAP optimization
            #     algorithm = NEWUOA(),
            #     multistart = 24,
            #     parallel,
            #     static_schedule = true, # issues with PRIMA.jl
            #     rhoend = 1e-2,
            # ),
            # gauss_mix_options = nothing,                # use Laplace approximation for the 0th iteration
            gauss_mix_options = GaussMixOptions(;       # use Gaussian mixture for the 0th iteration
                algorithm = BOBYQA(),
                multistart = 24,
                parallel,
                static_schedule = true, # issues with PRIMA.jl
                cluster_ϵs = nothing,
                rel_min_weight = 1e-8,
                rhoend = 1e-4,
            ),
        ),
        sample_count = 100,
        plot_callback = plots ? plt : BOLFI.NoCallback(),
    )

    ### options
    options = BolfiOptions(;
        # callback = plots ? plt : BOLFI.NoCallback(),
        callback = metric_cb,
        debug = true, # TODO
    )

    # RUN
    MakiePlots.init_plotting(plt)
    bolfi!(problem; model_fitter, acq_maximizer, term_cond, options)
    
    ### metrics
    score_history = options.callback.score_history
    if plots
        @show score_history
        lines(0:length(score_history)-1, score_history) |> display
    end

    # MakiePlots.plot_param_slices(plt, problem; options, samples=2_000)
    return problem
end

function generate_starts()
    run_count = 20
    init_data = 3

    experiment_data = [ToyProblem.get_init_data(init_data) for _ in 1:run_count]
    data = [(d.X, d.Y) for d in experiment_data]

    @save "examples/data/proxy/starts.jld2" data=data
end

function calc_stats(methodname)
    starts_file = "examples/data/proxy/starts.jld2"

    data = load(starts_file)["data"]
    experiment_data = [ExperimentData(d...) for d in data]
    run_count = length(experiment_data)
    
    problems = ToyProblem.bolfi_problem.(experiment_data)
    score_histories = Vector{Vector{Float64}}(undef, run_count)

    for i in 1:run_count
        _, score = main(problems[i]; plots=false)
        score_histories[i] = score
    end

    @save "examples/data/proxy/" * methodname * ".jld2" score=score_histories
end

function plot_stats()
    standard = load("examples/data/proxy/standard.jld2")["score"]
    absval = load("examples/data/proxy/absval.jld2")["score"]

    # Convert to matrix for easier computation
    standard_mat = hcat(standard...)
    absval_mat = hcat(absval...)

    # Compute median, 0.25, 0.75, 0.1, and 0.9 quantiles across runs for each iteration
    median_standard = mapslices(median, standard_mat; dims=2)[:]
    q25_standard = mapslices(x -> quantile(x, 0.25), standard_mat; dims=2)[:]
    q75_standard = mapslices(x -> quantile(x, 0.75), standard_mat; dims=2)[:]
    q10_standard = mapslices(x -> quantile(x, 0.1), standard_mat; dims=2)[:]
    q90_standard = mapslices(x -> quantile(x, 0.9), standard_mat; dims=2)[:]

    median_absval = mapslices(median, absval_mat; dims=2)[:]
    q25_absval = mapslices(x -> quantile(x, 0.25), absval_mat; dims=2)[:]
    q75_absval = mapslices(x -> quantile(x, 0.75), absval_mat; dims=2)[:]
    q10_absval = mapslices(x -> quantile(x, 0.1), absval_mat; dims=2)[:]
    q90_absval = mapslices(x -> quantile(x, 0.9), absval_mat; dims=2)[:]

    x = 0:length(median_standard)-1

    fig = Figure()
    ax = Axis(fig[1, 1])

    lines!(ax, x, median_standard, label="Standard", color=:blue)
    band!(ax, x, q25_standard, q75_standard, color=(:blue, 0.2), label="0.25 & 0.75 quantiles")
    lines!(ax, x, q10_standard, color=:blue, linestyle=:dot, label="0.1 & 0.9 quantiles")
    lines!(ax, x, q90_standard, color=:blue, linestyle=:dot)

    lines!(ax, x, median_absval, label="Absval", color=:red)
    band!(ax, x, q25_absval, q75_absval, color=(:red, 0.2), label="0.25 & 0.75 quantiles")
    lines!(ax, x, q10_absval, color=:red, linestyle=:dot, label="0.1 & 0.9 quantiles")
    lines!(ax, x, q90_absval, color=:red, linestyle=:dot)

    axislegend()
    display(fig)
end
