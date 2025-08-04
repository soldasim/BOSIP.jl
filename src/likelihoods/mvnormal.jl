
"""
    MvNormalLikelihood(; z_obs, Σ_obs)

The observation is assumed to have been generated from a multivariate normal distribution
as `z_obs \\sim Normal(f(x), Σ_obs)`. We can use the simulator to query `y = f(x)`.

# Kwargs
- `z_obs::Vector{Float64}`: The observed values from the real experiment.
- `Σ_obs::Matrix{Float64}`: The covariance of the observation noise.
"""
@kwdef struct MvNormalLikelihood <: Likelihood
    z_obs::Vector{Float64}
    Σ_obs::Matrix{Float64}
end

function loglike(like::MvNormalLikelihood, y::AbstractVector{<:Real})
    return logpdf(MvNormal(y, like.Σ_obs), like.z_obs)
end

function log_likelihood_mean(like::MvNormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    Σ_obs = like.Σ_obs

    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        
        Σ = Σ_obs + Diagonal(std_y .^ 2)
        return logpdf(MvNormal(μ_y, Σ), z_obs)
    end
    return log_like_mean
end

function log_sq_likelihood_mean(like::MvNormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    Σ_obs = like.Σ_obs
    y_dim = length(z_obs)

    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        
        Σ = ((1/2) .* Σ_obs) + Diagonal(std_y .^ 2)
        # C = 1 / ( (4 * π)^(y_dim / 2) * det(Σ_obs)^(1/2) )
        log_C = (-1) * ( (y_dim / 2) * log(4 * π) + (1/2) * logdet(Σ_obs))
        return log_C + logpdf(MvNormal(μ_y, Σ), z_obs)
    end
    return log_like_mean
end
