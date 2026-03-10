using BOSIP
using BOSS
using Distributions


# - - - Simulator - - - - -

# The UNKNOWN blackbox simulator
f(x) = [x[1] * x[2]]

x_dim = 2
y_dim = 1

# Input domain
bounds = ([-5., -5.], [5., 5.])
λ_min = (bounds[2] .- bounds[1]) ./ 20
λ_max = (bounds[2] .- bounds[1]) ./ 2
lengthscale_priors = fill(
    product_distribution(calc_inverse_gamma.(λ_min, λ_max)),
    y_dim,
)

# Output domain
α_est = 20. # we guess the amplitude of the simulator function
amplitude_priors = fill(calc_inverse_gamma(α_est / 5, α_est * 2), y_dim)


# - - - Parameter Prior - - - - -

# Prior beliefs about the parameter values
x_prior = product_distribution(truncated.(Ref(Normal(0., 5/3)), bounds...))


# - - - Observation & Likelihood - - - - -

# The real-world observation
z_obs = [1.]

# The likelihood relating the simulator output to the observation
likelihood = NormalLikelihood(; z_obs, std_obs=[0.2])
