
"""
    EIV(; kwargs...)

The `EIV` acquisition selects the next evaluation point by minimizing the Expected Integrated Variance
of the posterior approximation after the speculative evaluation. The variance of the posterior approximation
is implicitly given by the variance of the predictive distribution of the surrogate model.

The speculative posteriors are computed via the GP rank-1 conditioning update of the base model posterior.

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
    return _neg_log_eiv(
        f.bosip.likelihood,
        f,
        x_,
    )
end

# The negative log of the EIV resulting from the speculative evaluation of ``y_ | x_``.
function _neg_log_eiv(
    ::Likelihood,
    f::EIVFunc,
    x_::AbstractVector{<:Real},
)
    # sample `N` y_ samples at the new x_
    μy, σy = mean_and_std(f.model_post, x_)
    ys_ = calc_y.(Ref(μy), Ref(σy), f.ϵs_y) # -> immd.jl

    log_vars = _fantasy_log_posterior_variances(f, x_, ys_)

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

### Rank-1 conditioning update ###
# The posterior variance after conditioning on `(x_, y_)` does not depend on `y_`,
# so the `y_`-independent terms are computed once and shared across all `y_` samples.

# Log posterior variance at each `x` in `f.xs`, averaged over `ys_`.
function _fantasy_log_posterior_variances(f::EIVFunc, x_::AbstractVector{<:Real}, ys_::AbstractVector{<:AbstractVector{<:Real}})
    # type assertion: `params` is `Union{Nothing, FittedParams}`, which would infer as `Any`
    noise_vars = (BOSS.get_params(f.bosip.problem.params).σ .^ 2)::Vector{Float64}
    x_query = _fantasy_x_query(f.model_post, x_, noise_vars)
    return _fantasy_log_posterior_variances_at(f.bosip.likelihood, f.bosip.x_prior, f.model_post, f.xs, x_, x_query, ys_)
end

# Function barrier: `like` and `x_prior` have abstract types in `BosipProblem`,
# passing them as arguments lets the closure from `log_likelihood_variance` infer concretely.
function _fantasy_log_posterior_variances_at(
    like::Likelihood,
    x_prior::MultivariateDistribution,
    model_post::ModelPosterior,
    xs::AbstractMatrix{<:Real},
    x_::AbstractVector{<:Real},
    x_query,
    ys_::AbstractVector{<:AbstractVector{<:Real}},
)
    # `log_like_var` reads `post` at call time, so `post` is built once and mutated in place
    first_base_query = _fantasy_base_query(model_post, view(xs, :, 1), x_, x_query)
    post = _fantasy_point_posterior(model_post, first_base_query, first(ys_))
    log_like_var = log_likelihood_variance(like, post)

    return map(eachcol(xs)) do x
        base_query = _fantasy_base_query(model_post, x, x_, x_query)
        log_p = _log_prior(x_prior, x)

        vals = similar(ys_, Float64)
        for (i, y_) in enumerate(ys_)
            _fantasy_update_point_posterior!(post, model_post, base_query, y_)
            vals[i] = (2 * log_p) + log_like_var(x)
        end

        all(==(-Inf), vals) && return -Inf

        # logsumexp trick
        M = maximum(vals)
        return M + log(mean(exp.(vals .- M)))
    end
end

# A posterior slice pre-evaluated at one point; returns the stored `(mean, var)` for any query,
# so it must only be queried at the point it was built for.
struct _FantasyPointSlice{M<:SurrogateModel} <: BOSS.ModelPosteriorSlice{M}
    mean::Float64
    var::Float64
end

BOSS.mean(post::_FantasyPointSlice, ::AbstractVector{<:Real}) = post.mean
BOSS.var(post::_FantasyPointSlice, ::AbstractVector{<:Real}) = post.var
BOSS.mean_and_var(post::_FantasyPointSlice, ::AbstractVector{<:Real}) = (post.mean, post.var)

# The four `_fantasy_*` functions below are extension points: a `ModelPosterior` that wraps
# another `ModelPosterior` needs its own methods (otherwise a `MethodError` is thrown).

# Terms depending only on `x_`: `m_ = mean(x_)` and `denom = var(x_) + noise_var` per dimension.
function _fantasy_x_query(model_post::BOSS.DefaultModelPosterior, x_::AbstractVector{<:Real}, noise_vars::AbstractVector{<:Real})
    return map(model_post.slices, noise_vars) do base, noise_var
        m_, v_ = mean_and_var(base, x_)
        denom = v_ + noise_var
        (; m_, denom)
    end
end

# Terms depending on `x` and `x_` but not on `y_`.
function _fantasy_base_query(model_post::BOSS.DefaultModelPosterior, x::AbstractVector{<:Real}, x_::AbstractVector{<:Real}, x_query)
    return map(model_post.slices, x_query) do base, xq
        m0, v0 = mean_and_var(base, x)
        c = cov(base, Hcat(x, x_))[2, 1]
        (; m0, v0, c, m_=xq.m_, denom=xq.denom)
    end
end

# Build the updated posterior for a given `y_`.
function _fantasy_point_posterior(model_post::BOSS.DefaultModelPosterior{M}, base_query, y_::AbstractVector{<:Real}) where {M}
    post = BOSS.DefaultModelPosterior(Vector{_FantasyPointSlice{M}}(undef, length(base_query)))
    return _fantasy_update_point_posterior!(post, model_post, base_query, y_)
end

# Overwrite `post` in place with the update for a new `y_`.
function _fantasy_update_point_posterior!(post::BOSS.DefaultModelPosterior, model_post::BOSS.DefaultModelPosterior{M}, base_query, y_::AbstractVector{<:Real}) where {M}
    for (i, s) in enumerate(base_query)
        m = s.m0 + (s.c / s.denom) * (y_[i] - s.m_)
        v = BOSS._clip_var(s.v0 - s.c^2 / s.denom)
        post.slices[i] = _FantasyPointSlice{M}(m, v)
    end
    return post
end


### Specialized analytical implementation for `ExpLikelihood` ###
# Result 5.3 in Järvenpää et al. (2021), https://doi.org/10.1214/20-BA1200:
# the variance update is analytic, so no MC sampling of ``y_`` is needed.
function _neg_log_eiv(
    like::ExpLikelihood,
    f::EIVFunc,
    x_::AbstractVector{<:Real},
)
    noise_var = BOSS.get_params(f.bosip.problem.params).σ[1]^2

    log_var_reds = [_log_posterior_variance_reduction(like, f.bosip.x_prior, f.model_post, noise_var, x, x_) for x in eachcol(f.xs)]
    M = maximum(log_var_reds)

    # only the `x_`-dependent variance-reduction term is computed (logsumexp trick)
    log_eiv = 0 - ( M + log(sum(f.ws .* exp.(log_var_reds .- M))) )

    # the EIV is to be minimized
    return (-1) * log_eiv
end

# calculate the log of the reduction in the posterior variance at ``x``
# caused by observing the value at ``x_```
function _log_posterior_variance_reduction(
    ::ExpLikelihood,
    x_prior::MultivariateDistribution,
    model_post::ModelPosterior,
    noise_var::Real,
    x::AbstractVector{<:Real},
    x_::AbstractVector{<:Real},
)
    log_p = logpdf(x_prior, x)
    m, s2 = mean_and_var(model_post, x)
    τ2 = _tau2(model_post, noise_var, x, x_)

    m = m[1]
    s2 = s2[1]

    # post_var = p^2 * exp(2 * m + s2) * ( exp(s2) - exp(τ2) ); only the `x_`-dependent term is kept
    return (2 * log_p) + (2 * m + s2 + τ2)
end

# τ²(x, x_) = cov(x, x_)² / (var(x_) + noise_var), the variance reduction at ``x`` from observing ``x_``
# (eq. 20 in Järvenpää et al., 2019, https://doi.org/10.1214/18-BA1121).
# `noise_var` is the GP's fitted noise, not the likelihood's.
function _tau2(model_post::ModelPosterior, noise_var::Real, x::AbstractVector{<:Real}, x_::AbstractVector{<:Real})
    c = cov(model_post, Hcat(x, x_))[2, 1]
    v_ = var(model_post, x_)[1]
    return c^2 / (v_ + noise_var)
end
