using BOSIP
using BOSS
using BOSS.KernelFunctions
using CairoMakie
using Distributions
using OptimizationPRIMA

using Random
Random.seed!(555)

include("toy_problem.jl")
include("plot_callback.jl")

println("="^60)
println("BOSIP.jl - Simple Example")
println("="^60)

# Sample some random initial data
n_init = 3
X = rand(x_prior, n_init)
Y = hcat(f.(eachcol(X))...)
data = ExperimentData(X, Y)

println("\nInitial data points: $n_init")
println("Observation: z_obs = $(z_obs)")
println("True parameter satisfies: x[1] * x[2] = $(z_obs[1])")

# Initialize the BosipProblem
bosip = BosipProblem(data;
    f,
    domain = Domain(; bounds),
    acquisition = LogMaxVar(),
    model = GaussianProcess(;
        kernel = Matern52Kernel(),
        lengthscale_priors,
        amplitude_priors,
        # we have no simulation noise here:
        noise_std_priors = fill(Dirac(0.), y_dim),
    ),
    likelihood,
    x_prior,
)

# Setup optimization algorithms
parallel = false
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

# Set termination condition
# (e.g., max 20 evaluations including initial data)
term_cond = DataLimit(20)

# Setup options with plotting callback
plot_callback = SimplePlotCallback(; 
    plot_dir = "./examples/simple/plots",
    plot_name = "bosip_state",
    step = 0.1,
    display = true,
    save_plots = true,
)
options = BosipOptions(; callback = plot_callback)

println("\n" * "="^60)
println("Running BOSIP...")
println("="^60)

# Run BOSIP
bosip!(bosip; model_fitter, acq_maximizer, term_cond, options)

println("\n" * "="^60)
println("BOSIP Complete!")
println("="^60)
println("\nTotal evaluations: $(size(bosip.problem.data.X, 2))")
println("\nComputing posterior marginals with plot_marginals_int...")
marg_fig = BOSIP.plot_marginals_int(
    bosip;
    func = BOSIP.posterior_mean,
    lhc_grid_size = 200,
    matrix_ops = false,
    display = true,
)
save("./examples/simple/plots/marginals.png", marg_fig)
println("Saved marginals plot to ./examples/simple/plots/marginals.png")
println("\nTo analyze the posterior, you can use:")
println("  post = BOSIP.posterior_mean(bosip)")
println("  post(x)")

nothing
