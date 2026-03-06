
"""
    NormalLikelihood(; z_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `z_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `y = f(x)`.

# Kwargs
- `z_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct NormalLikelihood <: Likelihood
    z_obs::Vector{Float64}
    std_obs::Vector{Float64}
end

"""
    GaussianLikelihood(; z_obs, std_obs)

Alias for [`NormalLikelihood`](@ref).
"""
const GaussianLikelihood = NormalLikelihood

function loglike_marginal(like::NormalLikelihood, y::AbstractVector{<:Real})
    return logpdf.(Normal.(y, like.std_obs), like.z_obs)
end

function log_marginal_likelihood_mean(like::NormalLikelihood, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = like.std_obs

    function log_ml_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)

        std = sqrt.(std_obs.^2 .+ std_y.^2)
        return logpdf.(Normal.(μ_y, std), z_obs)
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        μs_y, stds_y = mean_and_std(model_post, X)
        
        # return logpdf.(MvNormal.(eachrow(μs_y), eachrow(stds_y)), Ref(z_obs))
        std_obs_mat = repeat(std_obs, 1, size(stds_y, 2))
        std_mat = sqrt.(std_obs_mat.^2 .+ stds_y.^2)
        y_mat = repeat(z_obs, 1, size(μs_y, 2))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_y, std_mat, y_mat)
        return lls
    end
    return log_ml_mean
end

function log_sq_likelihood_mean(like::NormalLikelihood, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = like.std_obs

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        
        std = sqrt.((std_obs.^2 .+ (2 .* std_y.^2)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C + logpdf(MvNormal(μ_y, std), z_obs)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        μs_y, stds_y = mean_and_std(model_post, X)

        std_obs_mat = repeat(std_obs, 1, size(stds_y, 2))
        std_mat = sqrt.((std_obs_mat.^2 .+ (2 .* stds_y.^2)) ./ 2)
        y_mat = repeat(z_obs, 1, size(μs_y, 2))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_y, std_mat, y_mat)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C .+ sum.(eachcol(lls))
    end
    return log_sq_like_mean
end

function get_subset(like::NormalLikelihood, y_set::AbstractVector{<:Bool})
    return NormalLikelihood(
        like.z_obs[y_set],
        like.std_obs[y_set],
    )
end

# ### Non-marginal likelihood versions:
# ### These are unnecessary, as they are already defined by the marginal likelihood versions.
# ### Commented-out to simplify the code as the performance benefits should be minimal.

# function loglike(like::NormalLikelihood, Y::AbstractVecOrMat{<:Real})
#     # return logpdf(MvNormal(y, like.std_obs), like.z_obs)
#     return logpdf(MvNormal(like.z_obs, like.std_obs), Y)
# end

# function log_likelihood_mean(like::NormalLikelihood, model_post::ModelPosterior)
#     z_obs = like.z_obs
#     std_obs = like.std_obs

#     function log_like_mean(x::AbstractVector{<:Real})
#         μ_y, std_y = mean_and_std(model_post, x)
        
#         std = sqrt.(std_obs.^2 .+ std_y.^2)
#         return logpdf(MvNormal(μ_y, std), z_obs)
#     end
#     function log_like_mean(X::AbstractMatrix{<:Real})
#         μs_y, stds_y = mean_and_std(model_post, X)
        
#         # return logpdf.(MvNormal.(eachrow(μs_y), eachrow(stds_y)), Ref(z_obs))
#         std_obs_mat = repeat(std_obs, 1, size(stds_y, 2))
#         std_mat = sqrt.(std_obs_mat.^2 .+ stds_y.^2)
#         y_mat = repeat(z_obs, 1, size(μs_y, 2))
#         lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_y, std_mat, y_mat)
#         return sum.(eachcol(lls))
#     end
#     return log_like_mean
# end
