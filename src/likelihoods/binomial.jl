
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

likelihood_kind(::BinomialLikelihood) = Marginalizable()

function loglike_marginal(like::BinomialLikelihood, y::AbstractVector{<:Real})
    y_ = clamp.(y, 0., 1.)
    return logpdf.(Binomial.(like.trials, y_), like.z_obs)
end

function log_marginal_likelihood_mean(::GaussianPredictive, like::BinomialLikelihood, model_post::ModelPosterior;
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
            log_vals = logpdf.(Binomial.(Ref(trials[i]), zs), Ref(z_obs[i]))
            logmeanexp(log_vals)
        end
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        return hcat(log_ml_mean.(eachcol(X))...)
    end
    return log_ml_mean
end
function log_marginal_likelihood_mean(::SampledPredictive, like::BinomialLikelihood, model_post::ModelPosterior;
    ϵs = nothing,
)
    z_obs = like.z_obs
    trials = like.trials

    function log_ml_mean(x::AbstractVector{<:Real})
        dims = per_dim_predictive_samples(model_post, x; lower=0., upper=1.)
        return [_binomial_atom_log_mean(ys, ws, trials[i], z_obs[i]) for (i, (ys, ws)) in enumerate(dims)]
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        dims = per_dim_predictive_samples(model_post, X; lower=0., upper=1.)
        rows = [[_binomial_atom_log_mean((@view ys[:, j]), (@view ws[:, j]), trials[i], z_obs[i])
                 for j in axes(X, 2)]
                for (i, (ys, ws)) in enumerate(dims)]
        return reduce(vcat, [row' for row in rows])
    end
    return log_ml_mean
end

function log_sq_likelihood_mean(::GaussianPredictive, like::BinomialLikelihood, model_post::ModelPosterior;
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
            log_vals = 2 .* logpdf.(Binomial.(Ref(trials[i]), zs), Ref(z_obs[i]))
            ll += logmeanexp(log_vals)
        end
        return ll
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        return log_sq_like_mean.(eachcol(X))
    end
    return log_sq_like_mean
end
function log_sq_likelihood_mean(::SampledPredictive, like::BinomialLikelihood, model_post::ModelPosterior;
    ϵs = nothing,
)
    z_obs = like.z_obs
    trials = like.trials

    function log_sq_like_mean(x::AbstractVector{<:Real})
        dims = per_dim_predictive_samples(model_post, x; lower=0., upper=1.)
        return sum(_binomial_atom_log_mean(ys, ws, trials[i], z_obs[i]; square=true) for (i, (ys, ws)) in enumerate(dims))
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        dims = per_dim_predictive_samples(model_post, X; lower=0., upper=1.)
        per_dim_logs = [[_binomial_atom_log_mean((@view ys[:, j]), (@view ws[:, j]), trials[i], z_obs[i]; square=true)
                          for j in axes(X, 2)]
                         for (i, (ys, ws)) in enumerate(dims)]
        return reduce(+, per_dim_logs)
    end
    return log_sq_like_mean
end

function _binomial_atom_log_mean(ys::AbstractVector{<:Real}, ws::AbstractVector{<:Real}, trials::Int, z::Int; square::Bool=false)
    in_range = 0. .<= ys .<= 1.
    if any(in_range)
        ys_in = ys[in_range]
        ws_in = ws[in_range]
        ws_in = ws_in ./ sum(ws_in)
    else
        ys_in = clamp.(ys, 0., 1.)
        ws_in = ws
    end
    log_vals = logpdf.(Binomial.(trials, ys_in), z)
    square && (log_vals = 2 .* log_vals)
    return logsumexp(log_vals .+ log.(ws_in))
end

# Shares `ϵs` between `log_marginal_likelihood_mean`/`log_sq_likelihood_mean` for noise cancellation.
function log_likelihood_variance(::GaussianPredictive, like::BinomialLikelihood, model_post::ModelPosterior;
    ϵs = nothing,
)
    if isnothing(ϵs)
        ϵs = rand(Uniform(0, 1), like.int_grid_size)
    end

    log_ml_mean = log_marginal_likelihood_mean(like, model_post; ϵs)
    log_sq_like_mean = log_sq_likelihood_mean(like, model_post; ϵs)

    function log_like_var(x::AbstractVector{<:Real})
        # return sq_like_mean(x) - like_mean(x)^2
        log_lm = sum(log_ml_mean(x))
        log_sqlm = log_sq_like_mean(x)

        # return log( exp(log_sqlm) - exp(2 * log_lm) )
        return log_sqlm + log1mexp(2 * log_lm - log_sqlm)
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
