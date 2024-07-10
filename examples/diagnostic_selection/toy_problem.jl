module ToyProblem

using BOLFI
using BOSS
using Distributions

# - - - PROBLEM - - - - -

# TODO: CHANGE THE EXPERIMENT HERE
# Mode `:T1` is SBI problem.
# Mode `:T2` is SBFI problem.
# const mode = Val(:T1)
const mode = Val(:T2)

"""acquisition function"""
# `PostVarAcq` for SBI problems
# `MWMVAcq` for SBFI problems
acquisition(::Val{:T1}) = PostVarAcq()
acquisition(::Val{:T2}) = MWMVAcq()
acquisition() = acquisition(mode)

"""observation"""
y_obs(::Val{:T1}) = [1.]
y_dim(::Val{:T1}) = 1
y_obs(::Val{:T2}) = [1., 0.]
y_dim(::Val{:T2}) = 2
y_obs() = y_obs(mode)
y_dim() = y_dim(mode)

"""observation noise std"""
σe_true(::Val{:T1}) = [0.5]  # true noise
σe(::Val{:T1}) =      [0.5]  # hyperparameter
σe_true(::Val{:T2}) = [0.5, 0.5]  # true noise
σe(::Val{:T2}) =      [0.5, 0.5]  # hyperparameter
σe_true() = σe_true(mode)
σe() = σe(mode)
"""simulation noise std"""
const ω = [0.01 for _ in 1:y_dim(mode)]

# objective functions
f_(x) = x[1] * x[2]
g_(x) = (x[2] - x[1])

get_y_sets(::Val{:T1}) = nothing
get_y_sets(::Val{:T2}) = [true;false;; false;true;;]
get_y_sets() = get_y_sets(mode)

# The "real experiment". (for plotting only)
function experiment(m::Val{:T1}, x; noise_std=σe_true(m))
    y = [f_(x)] + rand(MvNormal(zeros(y_dim(m)), noise_std))
    return y
end
function experiment(m::Val{:T2}, x; noise_std=σe_true(m))
    y = [f_(x), g_(x)] + rand(MvNormal(zeros(y_dim(m)), noise_std))
    return y
end
experiment(x; noise_std=σe_true(mode)) = experiment(mode, x; noise_std)

# The "simulation". (approximates the "experiment")
function simulation(m::Val{:T1}, x; noise_std=ω)
    y = [f_(x)] + rand(MvNormal(zeros(y_dim(m)), noise_std))
    return y
end
function simulation(m::Val{:T2}, x; noise_std=ω)
    y = [f_(x), g_(x)] + rand(MvNormal(zeros(y_dim(m)), noise_std))
    return y
end
simulation(x; noise_std=ω) = simulation(mode, x; noise_std)

# The objective for the GP.
# TODO: Try adding/removing abs value of the simulation-experiment discrepancy.
obj(m, x) = simulation(m, x) .- y_obs(m)
# obj(m, x) = abs.(simulation(m, x) .- y_obs(m))

get_bounds() = ([-5., -5.], [5., 5.])


# - - - HYPERPARAMETERS - - - - -

get_kernel() = BOSS.Matern32Kernel()

const λ_MIN = 0.1
const λ_MAX = 10.
get_length_scale_priors(m) = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), 2)), y_dim(m))

function get_amplitude_priors(m)
    return fill(truncated(Normal(0., 5.); lower=0.), y_dim(m))
end

function get_noise_std_priors(m)
    return [truncated(Normal(0., 10 * ω[i]); lower=0.) for i in 1:y_dim(m)]
end

# get_x_prior() = Product(fill(Uniform(-5., 5.), 2))
get_x_prior() = MvNormal(zeros(2), fill(5/3, 2))


# - - - INITIALIZATION - - - - -

function get_init_data(m, count)
    X = reduce(hcat, (random_datapoint() for _ in 1:count))[:,:]
    Y = reduce(hcat, (obj(m, x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentDataPrior(X, Y)
end

bolfi_problem(init_data::Int) = bolfi_problem(get_init_data(mode, init_data))

function bolfi_problem(data::ExperimentData)
    m = mode
    return BolfiProblem(data;
        f = (x) -> obj(m, x),
        bounds = get_bounds(),
        kernel = get_kernel(),
        length_scale_priors = get_length_scale_priors(m),
        amp_priors = get_amplitude_priors(m),
        noise_std_priors = get_noise_std_priors(m),
        std_obs = σe(m),
        x_prior = get_x_prior(),
        y_sets = get_y_sets(m),
    )
end


# - - - UTILS - - - - -

function random_datapoint()
    x_prior = get_x_prior()
    bounds = get_bounds()

    x = rand(x_prior)
    while !BOSS.in_bounds(x, bounds)
        x = rand(x_prior)
    end
    return x
end

"""
Return an _approximate_ Inverse Gamma distribution
with 0.99 probability mass between `lb` and `ub.`
"""
function calc_inverse_gamma(lb, ub)
    μ = (ub + lb) / 2
    σ = (ub - lb) / 6
    a = (μ^2 / σ^2) + 2
    b = μ * ((μ^2 / σ^2) + 1)
    return InverseGamma(a, b)
end

end
