
"""
    GaussianLikelihood(; std_obs::Vector{Float64})

The observation is assumed to have been generated from a normal distribution
as `y_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `y = f(x)`.

# Kwargs
- `y_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct GaussianLikelihood <: Likelihood
    y_obs::Vector{Float64}
    std_obs::Vector{Float64}
end

function approx_likelihood(like::GaussianLikelihood, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function approx_like(x)
        μ_δ, _ = gp_post(x)
        return pdf(MvNormal(μ_δ, std_obs), y_obs)
    end
end

function likelihood_mean(like::GaussianLikelihood, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function like_mean(x)
        μ_δ, std_δ = gp_post(x)
        std = sqrt.(std_obs.^2 .+ std_δ.^2)
        return pdf(MvNormal(μ_δ, std), y_obs)
    end
end

function sq_likelihood_mean(like::GaussianLikelihood, gp_post)
    y_obs = like.y_obs
    std_obs = like.std_obs

    function sq_like_mean(x)
        μ_δ, std_δ = gp_post(x)
        std = sqrt.((std_obs.^2 .+ (2 .* std_δ.^2)) ./ 2)
        # C = 1 / prod(2 * sqrt(π) .* std_obs)
        C = exp((-1) * sum(log.(2 * sqrt(π) .* std_obs)))
        return C * pdf(MvNormal(μ_δ, std), y_obs)
    end
end

function get_subset(like::GaussianLikelihood, y_set::AbstractVector{<:Bool})
    return GaussianLikelihood(
        like.y_obs[y_set],
        like.std_obs[y_set],
    )
end
