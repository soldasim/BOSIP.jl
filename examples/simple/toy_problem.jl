module ToyProblem

using BOLFI
using BOSS
using Distributions


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-5., x_dim()), fill(5., x_dim()))


# - - - OBSERVATION - - - - -

"""observation"""
const y_obs = [1.]
const y_dim = 1

"""observation noise std"""
const σe_true = [0.5]  # true noise
const σe =      [0.5]  # hyperparameter
"""simulation noise std"""
const ω = [0.001 for _ in 1:y_dim]


# - - - EXPERIMENT - - - - -

f_(x) = prod(x)

# The "real experiment". (for plotting only)
function experiment(x; noise_std=σe_true)
    y1 = f_(x) + rand(Normal(0., noise_std[1]))
    return [y1]
end

# The "simulation". (approximates the "experiment")
function simulation(x; noise_std=ω)
    y1 = f_(x) + rand(Normal(0., noise_std[1]))
    return [y1]
end

# The objective for the GP.
obj(x) = simulation(x) .- y_obs


# - - - HYPERPARAMETERS - - - - -

# get_x_prior() = Product(fill(Uniform(-5., 5.), x_dim()))
get_x_prior() = Product(fill(Normal(0., 5/3), x_dim()))

get_kernel() = BOSS.Matern32Kernel()

const λ_MIN = 0.01
const λ_MAX = 10.
get_length_scale_priors() = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), x_dim())), y_dim)

function get_amplitude_priors()
    return fill(truncated(Normal(0., 5.); lower=0.), y_dim)
end

function get_noise_std_priors()
    μ_std = ω
    max_std = 10 * ω
    return [truncated(Normal(μ_std[i], max_std[i] / 3); lower=0.) for i in 1:y_dim]
end


# - - - INITIALIZATION - - - - -

function get_init_data(count)
    X = reduce(hcat, (random_datapoint() for _ in 1:count))[:,:]
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]
    return BOSS.ExperimentDataPrior(X, Y)
end

bolfi_problem(init_data::Int) = bolfi_problem(get_init_data(init_data))

function bolfi_problem(data::ExperimentData)
    return BolfiProblem(data;
        f = obj,
        bounds = get_bounds(),
        kernel = get_kernel(),
        length_scale_priors = get_length_scale_priors(),
        amp_priors = get_amplitude_priors(),
        noise_std_priors = get_noise_std_priors(),
        std_obs = σe,
        x_prior = get_x_prior(),
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
