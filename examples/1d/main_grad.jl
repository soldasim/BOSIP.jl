using BOSS, BOSIP
using BOSS.KernelFunctions
using CairoMakie
using Distributions
using OptimizationPRIMA
using ForwardDiff

using Random
Random.seed!(555)

include("problem_grad.jl")
include("plot_grad.jl")

# TODO change the mode here
# mode = :divr
# mode = :dkg    # derivative Knowledge Gradient (mode-seeking, for reference)
mode = :bosip  # LogMaxVar acquisition
# mode = :grid


### Query initial data
if mode == :bosip || mode == :dkg || mode == :divr
    # 1 random point + run BOSIP
    X = [1.;;] # TODO
    results = f.(eachcol(X))
    Y  = hcat([r[1] for r in results]...)
    dY = cat([r[2] for r in results]...; dims=3)
    data = GradientData(X, Y, dY)

    term_cond = DataLimit(10) # TODO allow 10 BOSIP iters
elseif mode == :grid
    # grid of points + only visualize
    grid_size = 11
    x_grid = range(bounds[1][1], bounds[2][1]; length = grid_size)
    X = reshape(collect(x_grid), 1, :)
    results = f.(eachcol(X))
    Y  = hcat([r[1] for r in results]...)
    dY = hcat([r[2] for r in results]...)
    data = GradientData(X, Y, dY)

    term_cond = DataLimit(0) # don't query any new data points
else
    error("Unsupported mode: $mode")
end


### Initialize the `BosipProblem` structure
domain_size = bounds[2][1] - bounds[1][1]

bosip = BosipProblem(data;
    f,
    domain = Domain(; bounds),
    acquisition = mode == :dkg ? dKGAcquisition() :
                  mode == :divr ? dIVRAcquisition() : LogMaxVar(),
    model = GradientGaussianProcess(;
        kernel = Matern52Kernel(),
        lengthscale_priors = [product_distribution([calc_inverse_gamma(domain_size / 20, domain_size / 2)])],
        amplitude_priors = [calc_inverse_gamma(est_max_amplitude / 10, est_max_amplitude)],
        noise_std_priors = [Dirac(1e-6)],
        grad_noise_std_priors = [calc_inverse_gamma(grad_noise_std / 10, grad_noise_std * 2)],
    ),
    likelihood = NormalLikelihood(; z_obs, std_obs),
    x_prior = product_distribution([
        Uniform(bounds[1][1], bounds[2][1]),
    ]),
)

### Setup secondary algorithms, termination condition, and plotting
parallel = true
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
callback = BosipPlotCallback(;
    plot_dir="./examples/1d/plots_bosip",
    show_surrogate_in_simulator = true,
    credible_interval_level = 0.95,
)
options = BosipOptions(; callback)

### Run BOSIP
bosip!(bosip; model_fitter, acq_maximizer, term_cond, options)

### Plot and save legend
fig_legend = plot_legend()
display(fig_legend)
save("./examples/1d/plots_bosip_gradients/bosip_legend.pdf", fig_legend)

nothing
