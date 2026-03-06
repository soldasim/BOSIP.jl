
"""
    BinomialLikelihood(; z_obs, trials, kwargs...)

The observation is assumed to have been generated from a Binomial distribution
as `z_o \\sim Binomial(trials, f(x))`. We can use the simulator to query `y = f(x)`.

The simulator should only return values between 0 and 1. The GP estimates are clamped to this range.

# Kwargs
- `z_obs::Vector{Int64}`: The observed values from the real experiment.
- `trials::Vector{Int64}`: The number of trials for each observation dimension.
- `int_grid_size::Int64`: The number of samples used to approximate the expected likelihood.
"""
@kwdef struct BinomialLikelihood <: Likelihood
    z_obs::Vector{Int64}
    trials::Vector{Int64}
    int_grid_size::Int64 = 200

    function BinomialLikelihood(z_obs, trials, int_grid_size)
        @assert all(z_obs .>= 0)
        @assert all(trials .>= 1)
        @assert all(z_obs .<= trials)
        new(z_obs, trials, int_grid_size)
    end
end

function loglike_marginal(like::BinomialLikelihood, y::AbstractVector{<:Real})
    y_ = clamp.(y, 0., 1.)
    return logpdf.(Binomial.(like.trials, y_), like.z_obs)
end

function log_marginal_likelihood_mean(like::BinomialLikelihood, model_post::ModelPosterior;
    ϵs = nothing,
)
    z_obs = like.z_obs
    trials = like.trials

    if isnothing(ϵs)
        ϵs = rand(Uniform(0, 1), like.int_grid_size)
    end

    function log_ml_mean(x::AbstractVector{<:Real})
        ps_dists = truncated.(Normal.(mean_and_std(model_post, x)...); lower=0., upper=1.)
        return map(eachindex(z_obs)) do i
            zs = quantile.(Ref(ps_dists[i]), ϵs)
            vals = pdf.(Binomial.(Ref(trials[i]), zs), Ref(z_obs[i]))
            log(mean(vals))
        end
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        return hcat(log_ml_mean.(eachcol(X))...)
    end
    return log_ml_mean
end

# `log_likelihood_variance` shares `ϵs` with `log_sq_likelihood_mean` for noise cancellation.
function log_sq_likelihood_mean(like::BinomialLikelihood, model_post::ModelPosterior;
    ϵs = nothing,    
)
    z_obs = like.z_obs
    trials = like.trials

    if isnothing(ϵs)
        ϵs = rand(Uniform(0, 1), like.int_grid_size)
    end

    # TODO refactor
    function log_sq_like_mean(x::AbstractVector{<:Real})
        ps_dists = truncated.(Normal.(mean_and_std(model_post, x)...); lower=0., upper=1.)
        
        ll = 0.
        for i in eachindex(z_obs)
            zs = quantile.(Ref(ps_dists[i]), ϵs)
            vals = pdf.(Binomial.(Ref(trials[i]), zs), Ref(z_obs[i])) .^ 2
            ll += log(mean(vals))
        end
        return ll
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end

# share the noise samples `ϵs`
function log_likelihood_variance(like::BinomialLikelihood, model_post::ModelPosterior)
    ϵs = rand(Uniform(0, 1), like.int_grid_size)

    log_ml_mean = log_marginal_likelihood_mean(like, model_post; ϵs)
    log_sq_like_mean = log_sq_likelihood_mean(like, model_post; ϵs)

    function log_like_var(x::AbstractVector{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        log_lm = sum(log_ml_mean(x))
        log_sqlm = log_sq_like_mean(x)
        return log( exp(log_sqlm) - exp(2 * log_lm) )
    end
    function log_like_var(X::AbstractMatrix{<:Real})
        return log_like_var.(eachcol(X))
    end
    return log_like_var
end

function get_subset(like::BinomialLikelihood, y_set::AbstractVector{<:Bool})
    return BinomialLikelihood(
        like.z_obs[y_set],
        like.trials[y_set],
        like.int_grid_size,
    )
end
