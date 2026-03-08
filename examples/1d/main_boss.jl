using BOSS
using Distributions
using OptimizationPRIMA

using Random
Random.seed!(555)

include("problem.jl")
include("plot_boss.jl")

function mse_objective(x)
	y = f(x)
	mse = sum((y .- z_obs) .^ 2) / length(z_obs)
	return [mse]
end

# TODO change the mode here
mode = :boss
# mode = :grid


### Query initial data
if mode == :boss
    # 1 initial point + run BOSIP
    X = [1.;;]
    Y = hcat(mse_objective.(eachcol(X))...)
    data = ExperimentData(X, Y)

    term_cond = DataLimit(10) # allow 10 BOSIP iters
elseif mode == :grid
    # grid of points + only visualize
    grid_size = 11
    x_grid = range(bounds[1][1], bounds[2][1]; length = grid_size)
    X = reshape(collect(x_grid), 1, :)
    Y = hcat(mse_objective.(eachcol(X))...)
    data = ExperimentData(X, Y)

    term_cond = DataLimit(0) # don't query any new data points
else
    error("Unsupported mode: $mode")
end

### Construct the BOSS black-box optimization problem
domain_size = bounds[2][1] - bounds[1][1]

problem = BossProblem(
	f = mse_objective,
	domain = Domain(; bounds),
	acquisition = ExpectedImprovement(; fitness = LinFitness([-1.0])),
	model = GaussianProcess(
		kernel = BOSS.Matern52Kernel(),
		lengthscale_priors = [product_distribution([calc_inverse_gamma(domain_size / 20, domain_size / 2)])],
		amplitude_priors = [calc_inverse_gamma((est_max_amplitude / 10)^2, (est_max_amplitude)^2)],
		noise_std_priors = [Dirac(1e-6)],
	),
	data = data,
)

### Setup optimization subroutines and termination
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

callback = BossPlotCallback(
    f_true = f,
    z_obs = z_obs,
    display_plots = true,
    plot_dir = "./examples/1d/plots_boss",
)

options = BossOptions(callback = callback)

### Run Bayesian optimization
bo!(problem; model_fitter, acq_maximizer, term_cond, options)

### Report the best found point
best = result(problem)
if isnothing(best)
	@warn "No feasible point found."
else
	best_x, best_y = best
	@info "Best x found" best_x
	@info "Minimum MSE found" best_y[1]
	@info "f(best_x)" f(best_x)
end

nothing
