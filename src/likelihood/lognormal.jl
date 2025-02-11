
"""
    LogNormalLikelihood(; y_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `y_o \\sim LogNormal(f(x), Diagonal(std_obs))`. We can use the simulator to query `y = f(x)`.

# Kwargs
- `y_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Vector{Float64}`: The standard deviations of the LogNormal
        observation noise.
"""
@kwdef struct LogNormalLikelihood <: Likelihood
    y_obs::Vector{Float64}
    std_obs::Vector{Float64}
end

function approx_likelihood(like::LogNormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function approx_like(x)
        μ_z, _ = gp_post(x)
        return pdf(MvLogNormal(μ_z, std_obs), y_obs)
    end
end

# Identical to `likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function likelihood_mean(like::LogNormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function like_mean(x)
        μ_z, std_z = gp_post(x)
        std = sqrt.(std_obs.^2 .+ std_z.^2)
        return pdf(MvLogNormal(μ_z, std), y_obs)
    end
end

# Identical to `sq_likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function sq_likelihood_mean(like::LogNormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function sq_like_mean(x)
        μ_z, std_z = gp_post(x)
        std = sqrt.((std_obs.^2 .+ (2 .* std_z.^2)) ./ 2)
        # C = 1 / prod(2 * sqrt(π) .* std_obs)
        C = exp((-1) * sum(log.(2 * sqrt(π) .* std_obs)))
        return C * pdf(MvLogNormal(μ_z, std), y_obs)
    end
end

function get_subset(like::LogNormalLikelihood, y_set::AbstractVector{<:Bool})
    return LogNormalLikelihood(
        like.y_obs[y_set],
        like.std_obs[y_set],
    )
end
