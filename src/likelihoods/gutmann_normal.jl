
"""
    GutmannNormalLikelihood(; std_obs::Vector{Float64})

The observation is assumed to have been generated from a normal distribution
as `y_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `z = f(x)`.

This likelihood is an adaptation of the equations from Gutmann et al. [1,2]
for the deterministic simulator considered in BOLFI.jl.
The main difference to the original Gutmann's equations is that here
the predefined observation std `std_δ` is used instead of the noise std `σ_n`
used in the original paper, which is estimated as a GP hyperparameter.

It is defined in a slightly different way than the other `Likelihood`s.
Namely, this likelihood requires the simulator to return a _single_ non-negative scalar value,
describing the discrepancy `\\delta` of the simulator outcome from the real observation.

(See that the real observation `y_obs` is _not_ provided as a parameter to `GutmannNormalLikelihood`
as in the case of other `Likelihood`s. Instead, the observation `y_obs` should be used
to calculate the discrepancy `\\delta` in the simulator `f` provided to `BolfiProblem`.
The simulator should return the discrepancy as a vector `[\\delta]` of length 1.)

The likelihood is then defined as ``P[\\delta < \\epslion]`` instead of ``P[\\delta = 0]``,
which would correspond more closely to the other `Likelihood`s.
The parameter `\\epsilon` is the acceptance threshold.

# Kwargs
- `ϵ::Float64`: The threshold for the discrepancy from the real observation.
- `std_δ::Float64`: The standard deviation of the discrepancy `δ`
    caused by the observation noise on `y_obs`.

# References

[1] Gutmann, Michael U., and Jukka Cor. "Bayesian optimization for likelihood-free inference of simulator-based statistical models." Journal of Machine Learning Research 17.125 (2016): 1-47.

[2] Järvenpää, Marko, et al. "Efficient acquisition rules for model-based approximate Bayesian computation." (2019): 595-622.
"""
@kwdef struct GutmannNormalLikelihood <: Likelihood
    ϵ::Float64 = 0.
    std_δ::Float64
end

"""
    GutmannGaussianLikelihood(; y_obs, std_obs)

Alias for [`GutmannNormalLikelihood`](@ref).
"""
const GutmannGaussianLikelihood = GutmannNormalLikelihood

function approx_likelihood(like::GutmannNormalLikelihood, bolfi, gp_post)
    if (y_dim(bolfi) != 1) || any(bolfi.problem.data.Y .< 0.)
        throw(error("The simulator should return a positive scalar discrepancy for Gutmann's likelihood."))
    end

    ϵ = like.ϵ
    std_δ = like.std_δ

    function approx_like(x)
        μ_z, std_z = gp_post(x) .|> first
        z_stat = (ϵ - μ_z) / std_δ
        return normcdf(z_stat)
    end
end

function likelihood_mean(like::GutmannNormalLikelihood, bolfi, gp_post)
    if (y_dim(bolfi) != 1) || any(bolfi.problem.data.Y .< 0.)
        throw(error("The simulator should return a positive scalar discrepancy for Gutmann's likelihood."))
    end

    ϵ = like.ϵ
    std_δ = like.std_δ

    function like_mean(x)
        μ_z, std_z = gp_post(x) .|> first
        z_stat = (ϵ - μ_z) / sqrt(std_z^2 + std_δ^2)
        return normcdf(z_stat)
    end
end

# The derivation can be found in the supplementary material of the
# "Efficient Acquisition Rules for Model-Based Approximate Bayesian Computation" paper.
function sq_likelihood_mean(like::GutmannNormalLikelihood, bolfi, gp_post)
    if (y_dim(bolfi) != 1) || any(bolfi.problem.data.Y .< 0.)
        throw(error("The simulator should return a positive scalar discrepancy for Gutmann's likelihood."))
    end

    ϵ = like.ϵ
    std_δ = like.std_δ

    # taken from the equation (34) in the appendix of the Jarvenpaa's
    # "Efficient Acquisition Rules..." paper
    function sq_like_mean(x)
        μ_z, std_z = gp_post(x) .|> first
        z_stat = (ϵ - μ_z) / sqrt(std_z^2 + std_δ^2)
        ρ = std_z^2 / (std_δ^2 + std_z^2)
        return normcdf(z_stat) - 2 * owent(z_stat, (1 - ρ) / sqrt(1 - ρ^2))
    end
end

function get_subset(like::GutmannNormalLikelihood, y_set::AbstractVector{<:Bool})
    throw(error("Multiple y dimensions are not supported with Gutmann's likelihood."))
end
