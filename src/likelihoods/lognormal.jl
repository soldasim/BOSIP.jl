
"""
    LogNormalLikelihood(; z_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `z_o \\sim LogNormal(f(x), Diagonal(std_obs))`. We can use the simulator to query `y = f(x)`.

In many cases, one may want to take the logarithm of the output of the simulator.
Meaning, if one has simulator `sim(x)`, one would define `f` as `y = f(x) = log(sim(x))`.
This way, the `y` values with high likelihood will have similar values to the `z` values.

# Kwargs
- `z_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Vector{Float64}`: The standard deviations of the LogNormal observation noise.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct LogNormalLikelihood{
    S<:Union{Vector{Float64}, Nothing},
} <: Likelihood
    z_obs::Vector{Float64}
    std_obs::S
end

"""
    LogGaussianLikelihood(; z_obs, std_obs)

Alias for [`LogNormalLikelihood`](@ref).
"""
const LogGaussianLikelihood = LogNormalLikelihood

function loglike(like::LogNormalLikelihood, y::AbstractVector{<:Real})
    return logpdf(MvLogNormal(y, like.std_obs), like.z_obs)
end

function log_approx_likelihood(like::LogNormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_approx_like(x::AbstractVector{<:Real})
        μ_y = mean(model_post, x)
        return logpdf(MvLogNormal(μ_y, std_obs), z_obs)
    end
end

# Identical to `likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function log_likelihood_mean(like::LogNormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        std = sqrt.(std_obs.^2 .+ std_y.^2)
        return logpdf(MvLogNormal(μ_y, std), z_obs)
    end
end

# Identical to `sq_likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function log_sq_likelihood_mean(like::LogNormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        std = sqrt.((std_obs.^2 .+ (2 .* std_y.^2)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs .* z_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs .* z_obs))
        return log_C + logpdf(MvLogNormal(μ_y, std), z_obs)
    end
end

function _std_obs(like::LogNormalLikelihood{Nothing}, bolfi)
    @assert bolfi.problem.params isa UniFittedParams
    return bolfi.problem.params.σ
end
function _std_obs(like::LogNormalLikelihood, bolfi)
    return like.std_obs
end

function get_subset(like::LogNormalLikelihood{Nothing}, y_set::AbstractVector{<:Bool})
    return LogNormalLikelihood(
        like.z_obs[y_set],
        nothing,
    )
end
function get_subset(like::LogNormalLikelihood, y_set::AbstractVector{<:Bool})
    return LogNormalLikelihood(
        like.z_obs[y_set],
        like.std_obs[y_set],
    )
end
