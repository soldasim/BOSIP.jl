
"""
    NormalLikelihood(; z_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `z_o \\sim Normal(f(x), Diagonal(std_obs))`. We can use the simulator to query `z = f(x)`.

# Kwargs
- `z_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct NormalLikelihood{
    S<:Union{Vector{Float64}, Nothing},
} <: Likelihood
    z_obs::Vector{Float64}
    std_obs::S
end

"""
    GaussianLikelihood(; z_obs, std_obs)

Alias for [`NormalLikelihood`](@ref).
"""
const GaussianLikelihood = NormalLikelihood

function loglike(like::NormalLikelihood, δ::AbstractVector{<:Real})
    return logpdf(MvNormal(δ, like.std_obs), like.z_obs)
end

function log_approx_likelihood(like::NormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_approx_like(x::AbstractVector{<:Real})
        μ_δ = mean(model_post, x)
        
        return logpdf(MvNormal(μ_δ, std_obs), z_obs)
    end
    function log_approx_like(X::AbstractMatrix{<:Real})
        μs_δ = mean(model_post, X)
        
        # return logpdf.(MvNormal.(eachrow(μs_δ), Ref(Diagonal(std_obs))), Ref(z_obs))
        σ_mat = repeat(std_obs', size(μs_δ, 1))
        y_mat = repeat(z_obs', size(μs_δ, 1))
        ll_mat = ((μ, σ, y) -> logpdf(Normal(μ, σ), y)).(μs_δ, σ_mat, y_mat)
        return sum(ll_mat; dims=2)
    end
    return log_approx_like
end

function log_likelihood_mean(like::NormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_like_mean(x::AbstractVector{<:Real})
        μ_δ, std_δ = mean_and_std(model_post, x)
        
        std = sqrt.(std_obs.^2 .+ std_δ.^2)
        return logpdf(MvNormal(μ_δ, std), z_obs)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        μs_δ, stds_δ = mean_and_std(model_post, X)
        
        # return logpdf.(MvNormal.(eachrow(μs_δ), eachrow(stds_δ)), Ref(z_obs))
        std_obs_mat = repeat(std_obs', size(stds_δ, 1))
        std_mat = sqrt.(std_obs_mat.^2 .+ stds_δ.^2)
        y_mat = repeat(z_obs', size(μs_δ, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_δ, std_mat, y_mat)
        return sum(lls; dims=2)
    end
    return log_like_mean
end

function log_sq_likelihood_mean(like::NormalLikelihood, bolfi::BolfiProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bolfi)

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_δ, std_δ = mean_and_std(model_post, x)
        
        std = sqrt.((std_obs.^2 .+ (2 .* std_δ.^2)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C + logpdf(MvNormal(μ_δ, std), z_obs)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        μs_δ, stds_δ = mean_and_std(model_post, X)

        std_obs_mat = repeat(std_obs', size(stds_δ, 1))
        std_mat = sqrt.((std_obs_mat.^2 .+ (2 .* stds_δ.^2)) ./ 2)
        y_mat = repeat(z_obs', size(μs_δ, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_δ, std_mat, y_mat)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C .+ sum(lls; dims=2)
    end
    return log_sq_like_mean
end

function _std_obs(like::NormalLikelihood{Nothing}, bolfi::BolfiProblem)
    @assert bolfi.problem.params isa UniFittedParams
    return bolfi.problem.params.σ
end
function _std_obs(like::NormalLikelihood, bolfi)
    return like.std_obs
end

function get_subset(like::NormalLikelihood{Nothing}, y_set::AbstractVector{<:Bool})
    return NormalLikelihood(
        like.z_obs[y_set],
        nothing,
    )
end
function get_subset(like::NormalLikelihood, y_set::AbstractVector{<:Bool})
    return NormalLikelihood(
        like.z_obs[y_set],
        like.std_obs[y_set],
    )
end
