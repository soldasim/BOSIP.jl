module ToyProblem

using BOLFI
using BOSS
using Distributions

using Turing # to enable posterior sampling


# - - - PARAMETER DOMAIN - - - - -

x_dim() = 2
get_bounds() = (fill(-5., x_dim()), fill(5., x_dim()))


# - - - OBSERVATION - - - - -

"""observation"""
const z_obs = [1.]
const y_dim = 1

"""observation noise std"""
const σe = [0.2]
"""simulation noise std"""
const ω = fill(0., y_dim)


# - - - EXPERIMENT - - - - -

f_(x) = prod(x)

# The "simulation".
function simulation(x; noise_std=ω)
    y1 = f_(x) + rand(Normal(0., noise_std[1]))
    return [y1]
end

# The objective for the GP.
obj(x) = simulation(x)

get_likelihood() = NormalLikelihood(; z_obs=z_obs, std_obs=σe)

# truncate the prior to the bounds
function get_x_prior()
    prior = _get_x_prior()
    bounds = get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end
_get_x_prior() = Product(fill(Normal(0., 5/3), x_dim()))

function Distributions.truncated(d::Product; lower, upper)
    @assert length(d) == length(lower) == length(upper)
    return Product([truncated(d.v[i]; lower=lower[i], upper=upper[i]) for i in 1:length(d)])
end


# - - - HYPERPARAMETERS - - - - -

get_acquisition() = MaxVar()

get_kernel() = BOSS.Matern32Kernel()

const λ_MIN = 0.05
const λ_MAX = 10.
get_lengthscale_priors() = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), x_dim())), y_dim)

function get_amplitude_priors()
    # return fill(truncated(Normal(0., 5.); lower=0.), y_dim)
    return fill(calc_inverse_gamma(0.1, 20.), y_dim)
end

function get_noise_std_priors()
    # let the simulation noise be known
    return fill(Dirac(0.), y_dim)
end

get_model() = GaussianProcess(;
    kernel = get_kernel(),
    lengthscale_priors = get_lengthscale_priors(),
    amplitude_priors = get_amplitude_priors(),
    noise_std_priors = get_noise_std_priors(),
)


# - - - INITIAL DATA - - - - -

function get_init_data(count)
    X = rand(get_x_prior(), count)
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]
    return ExperimentData(X, Y)
end


# - - - INITIALIZATION - - - - -

bolfi_problem(init_data::Int) = bolfi_problem(get_init_data(init_data))

function bolfi_problem(data::ExperimentData)
    return BolfiProblem(data;
        f = obj,
        domain = Domain(; bounds=get_bounds()),
        acquisition = get_acquisition(),
        model = get_model(),
        likelihood = get_likelihood(),
        x_prior = get_x_prior(),
    )
end

function true_logpost(x)
    ll = true_loglike(x)
    lp = logpdf(ToyProblem.get_x_prior(), x)
    return ll + lp
end
function true_loglike(x)
    y = ToyProblem.simulation(x; noise_std=zeros(ToyProblem.y_dim))
    ll = loglike(get_likelihood(), y)
    return ll
end

end
