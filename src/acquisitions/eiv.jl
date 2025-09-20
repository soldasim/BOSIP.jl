
"""
    EIV(; kwargs...)

The `EIV` acquisition selects the next evaluation point by minimizing the Expected Integrated Variance
of the posterior approximation after the speculative evaluation. The variance of the posterior approximation
is implicitly given by the variance of the predictive distribution of the surrogate model.

# Kwargs
- `y_samples::Int64`: The amount of samples drawn from the model predictive distribution
        to approximate the expected variance reduction.
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `x_proposal::MultivariateDistribution`: This distribution is used to sample
        parameter samples used to numerically approximate the integral over the parameter domain.
"""
@kwdef struct EIV <: BosipAcquisition
    y_samples::Int64
    x_samples::Int64
    x_proposal::MultivariateDistribution
end

struct EIVFunc{
    P<:BosipProblem,
    M<:ModelPosterior,
    X<:AbstractMatrix{<:Real},
    W<:AbstractVector{<:Real},
    E<:AbstractVector{<:AbstractVector{<:Real}},
}
    # some fields may not be used
    bosip::P
    model_post::M
    xs::X
    ws::W
    ϵs_y::E
end

function (acq::EIV)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, options::BosipOptions)
    y_dim = BOSS.y_dim(bosip.problem)
    
    # Sample parameter values.
    xs = rand(acq.x_proposal, acq.x_samples)

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.y_samples) # vector-vector
    
    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.( log_ws .- log(sum(exp.(log_ws))) ) # normalize to sum up to 1
    
    return EIVFunc(
        bosip,
        BOSS.model_posterior(bosip.problem),
        xs,
        ws,
        ϵs_y,
    )
end

function (f::EIVFunc)(x_::AbstractVector{<:Real})
    return _log_posterior_variance(
        f.bosip.likelihood,
        f,
        x_,
    )
end

# Calculate the updated posterior variance of ``y | x``
# given the dataset ``D ∪ {(x_,y_)}`` augmented by the speculative evaluation of ``y_ | x_``.
function _log_posterior_variance(
    ::Likelihood,
    f::EIVFunc,
    x_::AbstractVector{<:Real},
)
    # sample `N` y_ samples at the new x_
    μy, σy = mean_and_std(f.model_post, x_)
    ys_ = calc_y.(Ref(μy), Ref(σy), f.ϵs_y) # -> immd.jl

    # augment problems
    augmented_problems = [deepcopy(f.bosip) for _ in eachindex(ys_)]
    for (p, y_) in zip(augmented_problems, ys_)
        augment_dataset!(p.problem, x_, y_)
    end
    log_post_vars = log_posterior_variance.(augmented_problems)

    # calculate the expected integrated variance
    log_vars = _log_posterior_variance_pointwise.(Ref(log_post_vars), eachcol(f.xs))

    # special case for zero EIV
    if all(==(-Inf), log_vars)
        @warn "Estimated EIV for x_=$(x_) is equal to zero."
        log_eiv = -Inf
    
    else
        # use the "logsumexp" trick for numerical stability
        M = maximum(log_vars)
        log_eiv = M + log(sum(f.ws .* exp.(log_vars .- M)))
    end

    # the EIV is to be minimized
    return (-1) * log_eiv
end

# calculate the log of expected updated posterior variance of ``y | x``
# given the dataset ``D ∪ {(x_,y_)}`` augmented by the speculative evaluation of ``y_ | x_``.
function _log_posterior_variance_pointwise(
    log_post_vars::AbstractVector{<:Function},
    x::AbstractVector{<:Real},
)
    # calculate the expected integrated variance
    vals = map(fvar -> fvar(x), log_post_vars)

    # special case for zero expected variance
    all(==(-Inf), vals) && return -Inf

    # use the "logsumexp" trick for numerical stability
    M = maximum(vals)
    return M + log(mean(exp.(vals .- M)))
end


### Specialized analytical implementation for `ExpLikelihood` ###
# Taken from https://projecteuclid.org/journals/bayesian-analysis/volume-16/issue-1/Parallel-Gaussian-Process-Surrogate-Bayesian-Inference-with-Noisy-Likelihood-Evaluations/10.1214/20-BA1200.full

# For `ExpLikelihood` (i.e. modeling the log-likelihood by the surrogate model),
# the posterior variance update can be computed analytically without MC sampling of ``y_``
# thanks to the result 5.3 presented in the Jarvenpaa et al. paper (link above).
function _log_posterior_variance(
    like::ExpLikelihood,
    f::EIVFunc,
    x_::AbstractVector{<:Real},
)
    log_var_reds = [_log_posterior_variance_reduction_pointwise(like, f.bosip.x_prior, f.model_post, x, x_) for x in eachcol(f.xs)]
    M = maximum(log_var_reds)
    
    # only compute the second term (the variance reduction)
    # skip the first term (the current pre-update variance)
    # use the "logsumexp" trick for numerical stability
    
    # the calculated negative term is still in log
    # no need to exponentiate as we are not substracting it from the current variance
    log_eiv = 0 - ( M + log(sum(f.ws .* exp.(log_var_reds .- M))) )

    # the EIV is to be minimized
    return (-1) * log_eiv
end

# calculate the log of the reduction in the posterior variance at ``x``
# caused by observing the value at ``x_```
function _log_posterior_variance_reduction_pointwise(
    ::ExpLikelihood,
    x_prior::MultivariateDistribution,
    model_post::ModelPosterior,
    x::AbstractVector{<:Real},
    x_::AbstractVector{<:Real},
)
    log_p = logpdf(x_prior, x)
    m, s2 = mean_and_var(model_post, x)
    τ2 = _tau2(model_post, x, x_)

    m = m[1]
    s2 = s2[1]

    # post_var = p^2 * exp(2 * m + s2) * ( exp(s2) - exp(τ2) )
    # log_post_var = (2 * log_p) + (2 * m + s2) + log( exp(s2) - exp(τ2) )

    # only calculate the second negative term, which depends on `x_`
    return (2 * log_p) + (2 * m + s2 + τ2)
end

# The reduction in the predictive variance of ``y | x``
# caused by the speculative evaluation of ``y_ | x_``.
function _tau2(model_post::ModelPosterior, x::AbstractVector{<:Real}, x_::AbstractVector{<:Real})
    X = Hcat(x, x_) # lazy concatenation
    c = cov(model_post, X)[2,1] # equal to cov[1,2]
    return c # TODO check that this line is correct
end
