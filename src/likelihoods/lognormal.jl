
"""
    LogNormalLikelihood(; kwargs...)

The observation `z` is assumed to follow a log-normal distribution
with the expected value `\\mathbf{E}[y] = z_obs` and the fixed coefficient of variation `CV`,
where `y` is the true response variable (without observation noise).

We assume that the surrogate model approximates the log-response `log(y) = log(f(x))`.
Modeling the log-response is more suitable as `y` is strictly positive.
Accordingly, the observation is provided in the log-space as `log(z_obs)` to avoid confusion.
(This way, the simulator `log(y) = log(f(x))` should return similar values to `log(z_obs)`.)

Multiple dimensions of the observation `z` are assumed to be independent.

This likelihood model corresponds to many physical applications with measurement diagnostics
with a relative error (e.g. "± 20%") rather than an absolute error (e.g. "± 0.1").

## Kwargs
- `log_z_obs::Vector{Float64}`: Log of the observed values from the real experiment.
- `CV::Vector{Float64}`: The coefficients of variation of the observations
        describing the relative observation error.
        (If a measurement device is described to have precision "± 20%",
        this usually means that ~95% of the measurements fall within 20% of the true value,
        which corresponds to `CV = 0.2 / 2 = 0.1`.)
"""
struct LogNormalLikelihood <: Likelihood
    log_z_obs::Vector{Float64}
    CV::Vector{Float64}
    z_obs::Vector{Float64}
    σ_log::Vector{Float64}

    function LogNormalLikelihood(log_z_obs::Vector{Float64}, CV::Vector{Float64})
        @assert length(log_z_obs) == length(CV)
        z_obs = exp.(log_z_obs)
        σ_log = _σ_log_z.(CV)
        new(log_z_obs, CV, z_obs, σ_log)
    end
end
function LogNormalLikelihood(; log_z_obs, CV)
    return LogNormalLikelihood(log_z_obs, CV)
end

# Transformations for the LogNormal distribution parameters
_μ_log_z(log_y::Real, σ_log::Real) = log_y - (σ_log^2) / 2
_σ_log_z(CV::Real) = sqrt(log(1 + CV^2))

function loglike(like::LogNormalLikelihood, log_y::AbstractVector{<:Real})
    μ_log = _μ_log_z.(log_y, like.σ_log)
    return logpdf(MvLogNormal(μ_log, like.σ_log), like.z_obs)
end
function loglike(like::LogNormalLikelihood, log_Y::AbstractMatrix{<:Real})
    return loglike.(Ref(like), eachcol(log_Y))
end

# Almost identical to `likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function log_likelihood_mean(like::LogNormalLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    σ_log = like.σ_log

    function log_like_mean(x::AbstractVector{<:Real})
        μ_log_y, std_log_y = mean_and_std(model_post, x)
        μ_log = _μ_log_z.(μ_log_y, σ_log)
        std = sqrt.(σ_log.^2 .+ std_log_y.^2)
        return logpdf(MvLogNormal(μ_log, std), z_obs)
    end
    function log_like_mean(X::AbstractMatrix{<:Real})
        return log_like_mean.(eachcol(X))
    end
    return log_like_mean
end

# Almost identical to `sq_likelihood_mean(::GaussianLikelihood)`, just swapped `MvNormal` for `MvLogNormal`
function log_sq_likelihood_mean(like::LogNormalLikelihood, bosip::BosipProblem, model_post::ModelPosterior)
    z_obs = like.z_obs
    σ_log = like.σ_log

    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_log_y, std_log_y = mean_and_std(model_post, x)
        μ_log = _μ_log_z.(μ_log_y, σ_log)
        std = sqrt.((σ_log.^2 .+ (2 .* std_log_y.^2)) ./ 2)
        # log_C = log( 1 / prod(2 * sqrt(π) .* σ_log .* z_obs) )
        log_C = (-1) * sum(log.(2 * sqrt(π) .* σ_log .* z_obs))
        return log_C + logpdf(MvLogNormal(μ_log, std), z_obs)
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end

function get_subset(like::LogNormalLikelihood, y_set::AbstractVector{<:Bool})
    return LogNormalLikelihood(
        like.z_obs[y_set],
        like.CV[y_set],
    )
end
