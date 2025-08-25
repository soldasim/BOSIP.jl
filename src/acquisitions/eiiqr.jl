
"""
    EIIQR(; kwargs...)

An acquisition function based on minimizing the interquantile range (IQR).

The EIIQR acquisition approximates the expected integrated IQR by Monte-Carlo sampling.
In comparison to the `IMIQR` acquisition, the EIIQR acquisition is implemented for any `Likelihood`.
The EIIQR acquisition is directly inspired by the the IMIQR acquisition from [1].

# Kwargs
- `p_u::Float64`: The quantile level used to calculate the IQR.
        Should be within ``(0.5, 1.0)``. Defaults to `0.75`.
        This corresponds to calculating the interquantile variance between
        the ``0.25`` and ``0.75`` quantiles.
- `y_samples::Int64`: The amount of samples drawn from the model predictive distribution
        to approximate the expected variance reduction.
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `x_proposal::MultivariateDistribution`: This distribution is used to sample
        parameter samples used to numerically approximate the integral over the parameter domain.

# References
[1] Järvenpää, Marko, et al. "Parallel Gaussian process surrogate Bayesian inference with noisy likelihood evaluations." (2021): 147-178.
"""
@kwdef struct EIIQR <: BolfiAcquisition
    p_u::Float64 = 0.75
    y_samples::Int64
    x_samples::Int64
    x_proposal::MultivariateDistribution
end

struct EIIQRFunc{
    P<:BolfiProblem,
    M<:ModelPosterior,
    X<:AbstractMatrix{<:Real},
    W<:AbstractVector{<:Real},
    E<:AbstractVector{<:AbstractVector{<:Real}},
}
    # some fields may not be used
    p_u::Float64
    bolfi::P
    model_post::M
    xs::X
    ws::W
    ϵs_y::E
end

function (acq::EIV)(::Type{<:UniFittedParams}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    y_dim = BOSS.y_dim(bolfi.problem)
    
    # Sample parameter values.
    xs = rand(acq.x_proposal, acq.x_samples)

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.y_samples) # vector-vector
    
    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.( log_ws .- log(sum(exp.(log_ws))) ) # normalize to sum up to 1

    return EIIQRFunc(
        acq.p_u,
        bolfi,
        BOSS.model_posterior(bolfi.problem),
        xs,
        ws,
        ϵs_y,
    )
end

function (f::EIIQRFunc)(x_::AbstractVector{<:Real})
    return _eiiqr(f, x_)
end

# Calculate the expected integrated IQR of the posterior
# given the speculative evaluation of ``y_ | x_``.
function _eiiqr(f::EIIQRFunc, x_::AbstractVector{<:Real})
    # sample `N` y_ samples at the new x_
    μy, σy = mean_and_std(f.model_post, x_)
    ys_ = calc_y.(Ref(μy), Ref(σy), f.ϵs_y) # -> eimmd.jl

    # augment problems
    augmented_problems = [deepcopy(f.bolfi) for _ in eachindex(ys_)]
    for (p, y_) in zip(augmented_problems, ys_)
        augment_dataset!(p.problem, x_, y_)
    end
    log_iqrs = log_interquantile_range.(augmented_problems, Ref(f.p_u))

    # calculate the expected integrated variance
    log_vars = _log_exp_iqr.(Ref(log_iqrs), eachcol(f.xs))
    M = maximum(log_vars)

    # use the "logsumexp" trick for numerical stability
    log_eiv = M + log(sum(f.ws .* exp.(log_vars .- M)))

    # the EIV is to be minimized
    return (-1) * log_eiv
end

# Calculate the log of expected IQR of ``y | x``
# given the dataset ``D ∪ {(x_,y_)}`` augmented by the speculative evaluation of ``y_ | x_``.
function _log_exp_iqr(
    log_iqrs::AbstractVector{<:Function},
    x::AbstractVector{<:Real},
)
    # calculate the expected integrated variance
    vals = map(fiqr -> fiqr(x), log_iqrs)
    M = maximum(vals)

    # use the "logsumexp" trick for numerical stability
    return M + log(mean(exp.(vals .- M)))
end

# TODO
# function log_interquantile_range(model_post::ModelPosterior, p_u::Real)
#     function log_iqr(x::AbstractVector{<:Real})
        
#         return log(q3 - q1)
#     end
# end
