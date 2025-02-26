
"""
    NormalLikelihood(; y_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `y_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `z = f(x)`.

# Kwargs
- `y_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct NormalLikelihood{
    S<:Union{Vector{Float64}, Nothing},
} <: Likelihood
    y_obs::Vector{Float64}
    std_obs::S
end

function loglike(like::NormalLikelihood, z::AbstractVector{<:Real})
    return logpdf(MvNormal(z, like.std_obs), like.y_obs)
end

function approx_likelihood(like::NormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = _std_obs(like, bolfi)

    function approx_like(x)
        μ_z, _ = gp_post(x)
        return pdf(MvNormal(μ_z, std_obs), y_obs)
    end
end

function likelihood_mean(like::NormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = _std_obs(like, bolfi)

    function like_mean(x)
        μ_z, std_z = gp_post(x)
        std = sqrt.(std_obs.^2 .+ std_z.^2)
        return pdf(MvNormal(μ_z, std), y_obs)
    end
end

function sq_likelihood_mean(like::NormalLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    std_obs = _std_obs(like, bolfi)

    function sq_like_mean(x)
        μ_z, std_z = gp_post(x)
        std = sqrt.((std_obs.^2 .+ (2 .* std_z.^2)) ./ 2)
        # C = 1 / prod(2 * sqrt(π) .* std_obs)
        C = exp((-1) * sum(log.(2 * sqrt(π) .* std_obs)))
        return C * pdf(MvNormal(μ_z, std), y_obs)
    end
end

function _std_obs(like::NormalLikelihood{Nothing}, bolfi)
    @assert bolfi.problem.data isa ExperimentDataMAP
    θ, λ, α, noise_std = bolfi.problem.data.params
    return noise_std
end
function _std_obs(like::NormalLikelihood, bolfi)
    return like.std_obs
end

function get_subset(like::NormalLikelihood{Nothing}, y_set::AbstractVector{<:Bool})
    return NormalLikelihood(
        like.y_obs[y_set],
        nothing,
    )
end
function get_subset(like::NormalLikelihood, y_set::AbstractVector{<:Bool})
    return NormalLikelihood(
        like.y_obs[y_set],
        like.std_obs[y_set],
    )
end
