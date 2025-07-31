
"""
    IMIQR(; kwargs...)

An acquisition function based on minimizing the interquantile range (IQR).

The intractable median integrated IQR is approximated by the integrated median IQR instead.
This acquisition is only implemented for problems with the `ExpLikelihood`.
See the "IMIQR" section in [1] for more information.

# Kwargs
- `p_u::Float64`: The quantile level used to calculate the IQR.
        Should be within ``(0.5, 1.0)``. Defaults to `0.75`.
        This corresponds to calculating the interquantile variance between
        the ``0.25`` and ``0.75`` quantiles.
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `x_proposal::MultivariateDistribution`: This distribution is used to sample
        parameter samples used to numerically approximate the integral over the parameter domain.

# References
[1] Järvenpää, Marko, et al. "Parallel Gaussian process surrogate Bayesian inference with noisy likelihood evaluations." (2021): 147-178.
"""
@kwdef struct IMIQR <: BolfiAcquisition
    p_u::Float64 = 0.75
    x_samples::Int64
    x_proposal::MultivariateDistribution
end

struct IMIQRFunc{
    P<:BolfiProblem,
    M<:ModelPosterior,
    X<:AbstractMatrix{<:Real},
    W<:AbstractVector{<:Real},
}
    # some fields may not be used
    p_u::Float64
    bolfi::P
    model_post::M
    xs::X
    ws::W
end

function (acq::IMIQR)(::Type{<:UniFittedParams}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)    
    # Sample parameter values.
    xs = rand(acq.x_proposal, acq.x_samples)
    
    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.( log_ws .- log(sum(exp.(log_ws))) ) # normalize to sum up to 1

    return IMIQRFunc(
        acq.p_u,
        bolfi,
        BOSS.model_posterior(bolfi.problem),
        xs,
        ws,
    )
end

function (f::IMIQRFunc)(x_::AbstractVector{<:Real})
    return _log_imiqr(
        f.bolfi.likelihood,
        f,
        x_,
    )
end

# Create a model posterior augmented by the median speculative datapoint at `x_`.
function _mean_augmented_model_posterior(
    problem::BossProblem,
    x_::AbstractVector{<:Real},
)
    # obtain the median prediction
    model_post = model_posterior(problem)
    y_ = mean(model_post, x_)

    # obtain the median-augmented model posterior
    problem_ = deepcopy(problem)
    augment_dataset!(problem_, x_, y_)
    model_post_ = model_posterior(problem_)

    return model_post_
end

# TODO
function _log_imiqr(
    like::Likelihood,
    f::IMIQRFunc,
    x_::AbstractVector{<:Real},
)
    @error "The IMIQR acquisition is only implemented for `ExpLikelihood`. Consider using the EIIQR acquisition instead."
    throw(MethodError(
        _log_imiqr,
        (typeof(like), typeof(f), typeof(x_)),
    ))
end


### General implementation for any `Likelihood` ###
# This is not implemented as the IMIQR acquisition cannot be generalized for any likelihood
# in a straight-forward way. (The issue is with calculating the quantiles in the code below.)

# # Calculate the log IMIQR.
# function _log_imiqr(
#     like::Likelihood,
#     f::IMIQRFunc,
#     x_::AbstractVector{<:Real},
# )
#     x_prior = f.bolfi.x_prior

#     # obtain the median-augmented model posterior
#     model_post_ = _mean_augmented_model_posterior(f.bolfi.problem, x_)

#     # obtain the lower and upper quantile model posteriors
#     # TODO: Such `quantile` functions are not implemented in BOSS.jl.
#     #       Also, this whole thing really only makes good sense if the model output is scalar
#     #       and the mapping from the model output to the likelihood is monotonic.
#     model_post_l = quantile(model_post_, 1 - f.p_u)
#     model_post_u = quantile(model_post_, f.p_u)

#     # obtain the lower and upper likelihood quantiles
#     like_l = approx_likelihood(like, f.bolfi, model_post_l)
#     like_u = approx_likelihood(like, f.bolfi, model_post_u)

#     # calculate the interquantile ranges
#     log_med_iqrs = [_log_iqr(x_prior, like_l, like_u, x) for x in eachcol(f.xs)]

#     # use the "logsumexp" trick for numerical stability
#     M = maximum(log_med_iqrs)
#     log_imiqr = M + log(sum(f.ws .* exp.(log_med_iqrs .- M)))

#     # the IMIQR is to be minimized
#     return (-1) * log_imiqr
# end

# # Calculate the log IQR for the given point `x`.
# function _log_iqr(
#     x_prior::MultivariateDistribution,
#     like_l::Function,
#     like_u::Function,
#     x::AbstractVector{<:Real},
# )
#     log_p = logpdf(x_prior, x)
#     l_l = like_l(x)
#     l_u = like_u(x)

#     log_iqr = log_p + log(l_u - l_l)
#     return log_iqr
# end


### Specialized analytical implementation for `ExpLikelihood` ###
# Taken from https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-1/Parallel-Gaussian-Process-Surrogate-Bayesian-Inference-with-Noisy-Likelihood-Evaluations/10.1214/20-BA1200.full

# Calculate the log IMIQR.
function _log_imiqr(
    like::ExpLikelihood,
    f::IMIQRFunc,
    x_::AbstractVector{<:Real},
)
    x_prior = f.bolfi.x_prior

    # obtain the median-augmented model posterior
    model_post_ = _mean_augmented_model_posterior(f.bolfi.problem, x_)

    # calculate the log IQR values
    u = quantile(Normal(), f.p_u)
    log_med_iqrs = [_log_iqr(like, u, x_prior, model_post_, x) for x in eachcol(f.xs)]

    # use the "logsumexp" trick for numerical stability
    M = maximum(log_med_iqrs)
    log_imiqr = M + log(sum(f.ws .* exp.(log_med_iqrs .- M)))
    log_imiqr *= 2 # unnecessary, but for consistency with the paper

    # the IMIQR is to be minimized
    return (-1) * log_imiqr
end

# Calculate the log IQR for the given point `x`.
function _log_iqr(
    ::ExpLikelihood,
    u::Real,
    x_prior::MultivariateDistribution,
    model_post::ModelPosterior,
    x::AbstractVector{<:Real},
)
    log_p = logpdf(x_prior, x)
    mean_ll, std_ll = mean_and_std(model_post, x) .|> first

    log_med_iqr = log_p + mean_ll + sinh(u * std_ll)
    return log_med_iqr
end
