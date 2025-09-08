module ToyProblem

using BOSIP
using BOSS
using Distributions
using Bijectors
using LinearAlgebra

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
# TODO loglike
function simulation(x; noise_std=ω)
    y1 = f_(x) + rand(Normal(0., noise_std[1]))
    return [y1]
end
# function simulation(x; noise_std=ω)
#     y1 = f_(x)
#     ll = logpdf(Normal(y1, σe[1]), z_obs[1])
#     ll += rand(Normal(0, noise_std[1]))
#     return [ll]
# end
# function simulation(x; noise_std=ω)
#     y1 = f_(x)
#     ll = pdf(Normal(y1, σe[1]), z_obs[1])
#     ll += rand(Normal(0, noise_std[1]))
#     return [ll]
# end

# The objective for the GP.
obj(x) = simulation(x)

# TODO loglike
get_likelihood() = NormalLikelihood(; z_obs=z_obs, std_obs=σe)
# get_likelihood() = MvNormalLikelihood(; z_obs=z_obs, Σ_obs=Diagonal(σe.^2))
# get_likelihood() = ExpLikelihood()
# get_likelihood() = IdentityLikelihood()

# truncate the prior to the bounds
function get_x_prior()
    prior = _get_x_prior()
    bounds = get_bounds()
    return truncated(prior; lower=bounds[1], upper=bounds[2])
end
# _get_x_prior() = Product(fill(Uniform(-5., 5.), x_dim()))
_get_x_prior() = Product(fill(Normal(0., 5/3), x_dim()))

function Distributions.truncated(d::Product; lower, upper)
    @assert length(d) == length(lower) == length(upper)
    return Product([truncated(d.v[i]; lower=lower[i], upper=upper[i]) for i in 1:length(d)])
end


# - - - HYPERPARAMETERS - - - - -

# TODO loglike
get_acquisition() = MaxVar()
# get_acquisition() = LogMaxVar()
# get_acquisition() = EIMMD(;
#     y_samples = 20,    # 2 * 10^(y_dim)
#     x_samples = 200,   # 2 * 10^(x_dim)
#     x_proposal = get_x_prior(),
# )
# get_acquisition() = EIV(;
#     y_samples = 20,    # 2 * 10^(y_dim)
#     x_samples = 200,   # 2 * 10^(x_dim)
#     x_proposal = get_x_prior(),
# )
# get_acquisition() = IMIQR(;
#     p_u = 0.75,
#     x_samples = 200,    # 2 * 10^(x_dim)
#     x_proposal = get_x_prior(),
# )

get_kernel() = BOSS.Matern32Kernel()

const λ_MIN = 0.05
const λ_MAX = 10.
# get_lengthscale_priors() = fill(Product(fill(truncated(Normal(1., 10/3)), x_dim())), y_dim)
get_lengthscale_priors() = fill(Product(fill(calc_inverse_gamma(λ_MIN, λ_MAX), x_dim())), y_dim)

# TODO loglike
function get_amplitude_priors()
    # return fill(truncated(Normal(0., 5.); lower=0.), y_dim)
    return fill(calc_inverse_gamma(0.1, 20.), y_dim)
end
# function get_amplitude_priors()
#     return fill(calc_inverse_gamma(0., 1000.), y_dim)
# end
# function get_amplitude_priors()
#     M = pdf(Normal(0, σe[1]), 0)
#     return fill(calc_inverse_gamma(0., M), y_dim)
# end
# function get_amplitude_priors()
#     d = TDist(4)
#     d = truncated(d; lower=0.)
#     d = transformed(d, Bijectors.Scale(10.))
    
#     return fill(d, y_dim)
# end

function get_noise_std_priors()
    # μ_std = ω
    # max_std = 10 * ω
    # return [truncated(Normal(μ_std[i], max_std[i] / 3); lower=0.) for i in 1:y_dim]
    # return [calc_inverse_gamma(0.1, ω[i]*100) for i in 1:y_dim]
    
    # TODO loglike
    return fill(Dirac(0.), y_dim)
    # return fill(Dirac(1.), y_dim)
end

# TODO
get_model() = GaussianProcess(;
    kernel = get_kernel(),
    lengthscale_priors = get_lengthscale_priors(),
    amplitude_priors = get_amplitude_priors(),
    noise_std_priors = get_noise_std_priors(),
)
# get_model() = NonstationaryGP(;
#     lengthscale_model = BOSS.default_lengthscale_model(get_bounds(), y_dim),
#     amplitude_model = get_amplitude_priors(),
#     noise_std_model = get_noise_std_priors(),
# )


# - - - INITIAL DATA - - - - -

function get_init_data(count)
    # TODO
    X = rand(get_x_prior(), count)
    # X = BOSS.generate_LHC(get_bounds(), count) |> collect
    Y = reduce(hcat, (obj(x) for x in eachcol(X)))[:,:]

    # TODO
    return ExperimentData(X, Y)
    # return NormalizedData(X, Y; y_ub=[1.])
end


# - - - INITIALIZATION - - - - -

bosip_problem(init_data::Int) = bosip_problem(get_init_data(init_data))

function bosip_problem(data::ExperimentData)
    return BosipProblem(data;
        f = obj,
        domain = Domain(; bounds=get_bounds()),
        acquisition = get_acquisition(),
        model = get_model(),
        likelihood = get_likelihood(),
        x_prior = get_x_prior(),
    )
end

function true_logpost(x::AbstractVector)
    ll = true_loglike(x)
    lp = logpdf(ToyProblem.get_x_prior(), x)
    return ll + lp
end
function true_logpost(X::AbstractMatrix)
    ll = true_loglike(X)
    lp = logpdf.(Ref(ToyProblem.get_x_prior()), eachcol(X))
    return ll + lp
end

function true_loglike(x::AbstractVector)
    y = ToyProblem.simulation(x; noise_std=zeros(ToyProblem.y_dim))
    ll = loglike(get_likelihood(), y)
    return ll
end
function true_loglike(X::AbstractMatrix)
    ys = ToyProblem.simulation.(eachcol(X); noise_std=zeros(ToyProblem.y_dim))
    ll = loglike.(Ref(get_likelihood()), ys)
    return ll
end

end
