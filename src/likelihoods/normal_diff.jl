
"""
    NormalDiffLikelihood(; z_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `z_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `abs(y - z_o) = f(x)`.

This likelihood is just for showcasing how discarding the sign can decrease performance of BOSIP.

(!) We ignore that the non-negative discrepancy δ is modeled by real-valued GP
    in order to maintain the analytical solution for the likelihood expectation and variance. 
    The difference should be negligible as long as `std_obs` is small.

# Kwargs
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct NormalDiffLikelihood{
    S<:Union{Vector{Float64}, Nothing},
} <: Likelihood
    std_obs::S
end

function loglike(like::NormalDiffLikelihood, Δ::AbstractVecOrMat{<:Real})
    # return logpdf(MvNormal(y, like.std_obs), like.z_obs)
    # return logpdf(MvNormal(like.z_obs, like.std_obs), Y)
    return logpdf(MvNormal(zero(like.std_obs), like.std_obs), Δ)
end

# (!) This is only an approximation. We ignore that the non-negative discrepancy δ is modeled by real-valued GP
#     in order to maintain the analytical solution. The difference should be negligible as long as `std_obs` is small.
function log_likelihood_mean(like::NormalDiffLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    std_obs = _std_obs(like, bosip)
    zero_diff = zero(std_obs)

    function log_like_mean(x::AbstractVector{<:Real})
        μ_δ, std_δ = mean_and_std(model_post, x)
        
        std = sqrt.(std_obs.^2 .+ std_δ.^2)
        return logpdf(MvNormal(μ_δ, std), zero_diff)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        μs_δ, stds_δ = mean_and_std(model_post, X)
        
        # return logpdf.(MvNormal.(eachrow(μs_δ), eachrow(stds_δ)), Ref(z_obs))
        std_obs_mat = repeat(std_obs', size(stds_δ, 1))
        std_mat = sqrt.(std_obs_mat.^2 .+ stds_δ.^2)
        y_mat = repeat(zero_diff', size(μs_δ, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_δ, std_mat, y_mat)
        return sum(lls; dims=2)
    end
    return log_like_mean
end

# (!) This is only an approximation. We ignore that the non-negative discrepancy δ is modeled by real-valued GP
#     in order to maintain the analytical solution. The difference should be negligible as long as std_obs is small.
function log_sq_likelihood_mean(like::NormalDiffLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    std_obs = _std_obs(like, bosip)
    zero_diff = zero(std_obs)

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_δ, std_δ = mean_and_std(model_post, x)
        
        std = sqrt.((std_obs.^2 .+ (2 .* std_δ.^2)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C + logpdf(MvNormal(μ_δ, std), zero_diff)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        μs_δ, stds_δ = mean_and_std(model_post, X)

        std_obs_mat = repeat(std_obs', size(stds_δ, 1))
        std_mat = sqrt.((std_obs_mat.^2 .+ (2 .* stds_δ.^2)) ./ 2)
        y_mat = repeat(zero_diff', size(μs_δ, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_δ, std_mat, y_mat)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C .+ sum(lls; dims=2)
    end
    return log_sq_like_mean
end

function _std_obs(like::NormalDiffLikelihood{Nothing}, bosip::BosipProblem)
    @assert bosip.problem.params isa UniFittedParams
    return bosip.problem.params.σ
end
function _std_obs(like::NormalDiffLikelihood, bosip)
    return like.std_obs
end

function get_subset(like::NormalDiffLikelihood{Nothing}, y_set::AbstractVector{<:Bool})
    return NormalDiffLikelihood(
        nothing,
    )
end
function get_subset(like::NormalDiffLikelihood, y_set::AbstractVector{<:Bool})
    return NormalDiffLikelihood(
        like.std_obs[y_set],
    )
end
