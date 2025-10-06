
"""
    LogNormalSumLikelihood(; kwargs...)

Same as the [`LogNormalLikelihood`](@ref), but instead of each log-response `log(y)`,
multiple responses `log(y1),...,log(yn)` are modeled by the surrogate model.
The original response is finally obtained as `log(y) = log(sum(exp.(ys...)))`.
This allows to model with higher fidelity than the real-world observation `z_obs`.

## Kwargs
- `log_z_obs::Vector{Float64}`: Log of the observed values from the real experiment.
- `CV::Vector{Float64}`: The coefficients of variation of the observations
        describing the relative observation error.
        (If a measurement device is described to have precision "± 20%",
        this usually means that ~95% of the measurements fall within 20% of the true value,
        which corresponds to `CV = 0.2 / 2 = 0.1`.)
- `sum_lengths::Vector{Int}`: A vector specifying the number of individual modeled variables to be summed
        for each original response variable. (E.g. if `length(y) == 2` and each of the observations
        is split into 5 variables modeled by the surrogate model, use `sum_lengths = [5, 5]`.)
"""
struct LogNormalSumLikelihood <: Likelihood
    sum_lengths::Vector{Int}
    log_z_obs::Vector{Float64}
    CV::Vector{Float64}
    z_obs::Vector{Float64}
    σ_log::Vector{Float64}

    function LogNormalSumLikelihood(sum_lengths::Vector{Int}, log_z_obs::Vector{Float64}, CV::Vector{Float64})
        @assert length(sum_lengths) == length(log_z_obs) == length(CV)
        z_obs = exp.(log_z_obs)
        σ_log = _σ_log_z.(CV)
        new(sum_lengths, log_z_obs, CV, z_obs, σ_log)
    end
end
function LogNormalSumLikelihood(; sum_lengths, log_z_obs, CV)
    return LogNormalSumLikelihood(sum_lengths, log_z_obs, CV)
end

### from lognormal.jl
# _μ_log_z(log_y::Real, σ_log::Real) = log_y - (σ_log^2) / 2
# _σ_log_z(CV::Real) = sqrt(log(1 + CV^2))

function loglike(like::LogNormalSumLikelihood, log_ys::AbstractVector{<:Real})
    log_y = _indexed_logsumexp(log_ys, like.sum_lengths)
    
    μ_log = _μ_log_z.(log_y, like.σ_log)
    return logpdf(MvLogNormal(μ_log, like.σ_log), like.z_obs)
end
function loglike(like::LogNormalSumLikelihood, log_Y::AbstractMatrix{<:Real})
    return map(log_y -> loglike(like, log_y), eachcol(log_Y))
end

function log_likelihood_mean(like::LogNormalSumLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    σ_log = like.σ_log

    function log_like_mean(x::AbstractVector{<:Real})
        μ_log_ys, var_log_ys = mean_and_var(model_post, x)
        μ_log_y, var_log_y = _approx_mean_and_var_logsum(μ_log_ys, var_log_ys, like.sum_lengths)
                
        μ_log = _μ_log_z.(μ_log_y, σ_log)
        std = sqrt.(σ_log.^2 .+ var_log_y)
        return logpdf(MvLogNormal(μ_log, std), z_obs)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end

function log_sq_likelihood_mean(like::LogNormalSumLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    σ_log = like.σ_log

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_log_ys, var_log_ys = mean_and_var(model_post, x)
        μ_log_y, var_log_y = _approx_mean_and_var_logsum(μ_log_ys, var_log_ys, like.sum_lengths)
        
        μ_log = _μ_log_z.(μ_log_y, σ_log)
        std = sqrt.((σ_log.^2 .+ (2 .* var_log_y)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* σ_log .* z_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* σ_log .* z_obs))
        return log_C + logpdf(MvLogNormal(μ_log, std), z_obs)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end

function _approx_mean_and_var_logsum(μ_log_ys::AbstractVector{<:Real}, var_log_ys::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    μ_log_y = similar(μ_log_ys, length(sum_lengths))
    var_log_y = similar(var_log_ys, length(sum_lengths))
    
    idx = 1
    for (i, len) in enumerate(sum_lengths)
        μs_group = @view μ_log_ys[idx:(idx + len - 1)]
        vars_group = @view var_log_ys[idx:(idx + len - 1)]
        
        μ_log_y[i], var_log_y[i] = _fenton_wilkinson_approximation(μs_group, vars_group)
        
        idx += len
    end
    
    return μ_log_y, var_log_y
end

# Fenton-Wilkinson approximation of a log-normal distribution of a sum of log-normal variables
function _fenton_wilkinson_approximation(log_μs::AbstractVector{<:Real}, log_vars::AbstractVector{<:Real})
    # Convert log-normal parameters to normal space moments
    # If X ~ LogNormal(μ_log, σ_log), then:
    # E[X] = exp(μ_log + σ_log²/2)
    # Var[X] = exp(2*μ_log + σ_log²) * (exp(σ_log²) - 1)
    
    # Compute mean and variance of each log-normal variable in normal space
    means = exp.(log_μs .+ log_vars ./ 2)
    variances = exp.(2 .* log_μs .+ log_vars) .* (exp.(log_vars) .- 1)
    
    # Sum moments (sum of independent log-normals)
    sum_mean = sum(means)
    sum_var = sum(variances)
    
    # Convert back to log-normal parameters using Fenton-Wilkinson
    # If S ~ LogNormal(μ_S, σ_S), then:
    # σ_S² = log(1 + sum_var / sum_mean²)
    # μ_S = log(sum_mean) - σ_S² / 2
    
    if sum_mean <= 0 || sum_var <= 0
        # Fallback for degenerate cases
        @warn "Numerical issues in Fenton-Wilkinson approximation of log-normal sum. Using a fallback formula."
        return logsumexp(log_μs), sum(log_vars)
    end
    
    log_var = log(1 + sum_var / (sum_mean^2))
    log_μ = log(sum_mean) - log_var / 2
    
    return log_μ, log_var
end

function _indexed_logsumexp(log_y::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    log_z = similar(log_y, length(sum_lengths))
    return _indexed_logsumexp!(log_z, log_y, sum_lengths)
end
function _indexed_logsumexp!(log_z::AbstractVector{<:Real}, log_y::AbstractVector{<:Real}, sum_lengths::Vector{Int})
    idx = 1
    for (i, len) in enumerate(sum_lengths)
        log_z[i] = logsumexp(@view log_y[idx:(idx + len - 1)])
        idx += len
    end
    return log_z
end

function logsumexp(x::AbstractVector{<:Real})
    x_max = maximum(x)
    isinf(x_max) && return x_max
    return x_max + log(sum(exp.(x .- x_max)))
end

function get_subset(like::LogNormalSumLikelihood, y_set::AbstractVector{<:Bool})
    return LogNormalSumLikelihood(
        like.sum_lengths[y_set],
        like.log_z_obs[y_set],
        like.CV[y_set],
    )
end
