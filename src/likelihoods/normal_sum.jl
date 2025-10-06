
"""
    NormalSumLikelihood(; sum_lengths, z_obs, std_obs)

The observation is assumed to have been generated from a normal distribution
as `z_o \\sim Normal(f(x), Diagonal(std_obs))`, where each element of `z_o`
is a sum of multiple variables modeled by the surrogate model.
(I.e. model resolution is higher than the observation resolution.)

We can use the simulator to query `y = f(x)`, and then sum the appropriate subsets of `y`
to obtain the observations `z_o`. The subsets are assumed to be contiguous.

# Kwargs
- `sum_lengths::Vector{Int}`: The lengths of each sum group. Assure that `length(sum_lengths) == length(z_obs)`.
- `z_obs::Vector{Float64}`: The observed values from the real experiment.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
"""
@kwdef struct NormalSumLikelihood{
    S<:Union{Vector{Float64}, Nothing},
} <: Likelihood
    sum_lengths::Vector{Int}
    z_obs::Vector{Float64}
    std_obs::S

    function NormalSumLikelihood(sum_lengths::Vector{Int}, z_obs::Vector{Float64}, std_obs::S) where {S}
        @assert length(sum_lengths) == length(z_obs) "sum_lengths and z_obs must have the same length"
        new{S}(sum_lengths, z_obs, std_obs)
    end
end

function _indexed_sum(y::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    z = similar(y, length(sum_lengths))
    return _indexed_sum!(z, y, sum_lengths)
end
function _indexed_sum!(z::AbstractVector{<:Real}, y::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    idx = 1
    for (i, len) in enumerate(sum_lengths)
        z[i] = sum(@view y[idx:(idx + len - 1)])
        idx += len
    end
    return z
end

function _mean_and_var_sum(μ_y::AbstractVector{<:Real}, var_y::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    μ_z = _indexed_sum(μ_y, sum_lengths)
    var_z = _indexed_sum(var_y, sum_lengths)
    return μ_z, var_z
end
function _mean_and_var_sum(μs_y::AbstractMatrix{<:Real}, vars_y::AbstractMatrix{<:Real}, sum_lengths::Vector{Int})
    # μs_z = mapslices(μ_y -> _indexed_sum(μ_y, like.sum_lengths), μs_y; dims=1) # by row
    # vars_z = mapslices(var_y -> _indexed_sum(var_y, like.sum_lengths), vars_y; dims=1) # by row
    
    # indexed by col
    μs_z = similar(μs_y, length(sum_lengths), size(μs_y, 1))
    vars_z = similar(vars_y, length(sum_lengths), size(vars_y, 1))
    
    # Threads.@threads for i in 1:size(μs_y, 1)
    Threads.@threads for i in 1:size(μs_y, 1)
        _indexed_sum!((@view μs_z[:, i]), (@view μs_y[i, :]), sum_lengths)
        _indexed_sum!((@view vars_z[:, i]), (@view vars_y[i, :]), sum_lengths)
    end

    # indexed by row
    return μs_z', vars_z'
end

function loglike(like::NormalSumLikelihood, y::AbstractVector{<:Real})
    # return logpdf(MvNormal(z, like.std_obs), like.z_obs)
    z = _indexed_sum(y, like.sum_lengths)
    return logpdf(MvNormal(z, like.std_obs), like.z_obs)
end
function loglike(like::NormalSumLikelihood, Y::AbstractMatrix{<:Real})
    # return logpdf(MvNormal(z, like.std_obs), like.z_obs)
    dist = MvNormal(like.z_obs, like.std_obs)
    return map(y -> logpdf(dist, _indexed_sum(y, like.sum_lengths)), eachcol(Y))
end

function log_likelihood_mean(like::NormalSumLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bosip)

    function log_like_mean(x::AbstractVector{<:Real})
        μ_y, var_y = mean_and_var(model_post, x)
        μ_sum, var_sum = _mean_and_var_sum(μ_y, var_y, like.sum_lengths)
        
        std = sqrt.(std_obs.^2 .+ var_sum)
        return logpdf(MvNormal(μ_sum, std), z_obs)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        μs_y, vars_y = mean_and_var(model_post, X)
        μs_sum, vars_sum = _mean_and_var_sum(μs_y, vars_y, like.sum_lengths)

        # return logpdf.(MvNormal.(eachrow(μs_z), eachrow(vars_z)), Ref(z_obs))
        std_obs_mat = repeat(std_obs', size(vars_sum, 1))
        std_mat = sqrt.(std_obs_mat.^2 .+ vars_sum)
        y_mat = repeat(z_obs', size(μs_sum, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_sum, std_mat, y_mat)
        return sum(lls; dims=2)
    end
    return log_like_mean
end

function log_sq_likelihood_mean(like::NormalSumLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = _std_obs(like, bosip)

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_y, var_y = mean_and_var(model_post, x)
        μ_sum, var_sum = _mean_and_var_sum(μ_y, var_y, like.sum_lengths)
        
        std = sqrt.((std_obs.^2 .+ (2 .* var_sum)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C + logpdf(MvNormal(μ_sum, std), z_obs)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        μs_y, vars_y = mean_and_var(model_post, X)
        μs_sum, vars_sum = _mean_and_var_sum(μs_y, vars_y, like.sum_lengths)
        
        std_obs_mat = repeat(std_obs', size(vars_sum, 1))
        std_mat = sqrt.((std_obs_mat.^2 .+ (2 .* vars_sum)) ./ 2)
        y_mat = repeat(z_obs', size(μs_sum, 1))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_sum, std_mat, y_mat)
        # log_C = log( 1 / prod(2 * sqrt(π) .* std_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* std_obs))
        return log_C .+ sum(lls; dims=2)
    end
    return log_sq_like_mean
end

function _std_obs(like::NormalSumLikelihood{Nothing}, bosip::BosipProblem)
    @assert bosip.problem.params isa UniFittedParams
    return bosip.problem.params.σ
end
function _std_obs(like::NormalSumLikelihood, bosip)
    return like.std_obs
end

function get_subset(like::NormalSumLikelihood{Nothing}, y_set::AbstractVector{<:Bool})
    return NormalSumLikelihood(
        like.z_obs[y_set],
        nothing,
    )
end
function get_subset(like::NormalSumLikelihood, y_set::AbstractVector{<:Bool})
    return NormalSumLikelihood(
        like.z_obs[y_set],
        like.std_obs[y_set],
    )
end
