
"""
    BinomialLikelihood(; y_obs, trials, kwargs...)

The observation is assumed to have been generated from a Binomial distribution
as `y_o \\sim Binomial(trials, f(x))`. We can use the simulator to query `z = f(x)`.

The simulator should only return values between 0 and 1. The GP estimates are clamped to this range.

# Kwargs
- `y_obs::Vector{Int64}`: The observed values from the real experiment.
- `trials::Vector{Int64}`: The number of trials for each observation dimension.
- `int_grid_size::Int64`: The number of samples used to approximate the expected likelihood.
"""
@kwdef struct BinomialLikelihood <: Likelihood
    y_obs::Vector{Int64}
    trials::Vector{Int64}
    int_grid_size::Int64 = 200

    function BinomialLikelihood(y_obs, trials, int_grid_size)
        @assert all(y_obs .>= 0)
        @assert all(trials .>= 1)
        @assert all(y_obs .<= trials)
        new(y_obs, trials, int_grid_size)
    end
end

function loglike(like::BinomialLikelihood, z::AbstractVector{<:Real})
    # if any(z .< 0.) || any(z .> 1.)
    #     @warn "Called `loglike(::BinomialLikelihood, z)`, where `z = $z` is outside of range `[0, 1]`."
    #     z .= clamp.(z, 0., 1.)
    # end
    z .= clamp.(z, 0., 1.)

    # return sum(logpdf.(Binomial.(like.trials, z), like.y_obs))
    return mapreduce((t, z, y) -> logpdf(Binomial(t, z), y), +, like.trials, z, like.y_obs)
end

function approx_likelihood(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials

    function approx_like(x::AbstractVector{<:Real})
        μ_ps, _ = gp_post(x)
        ps = clamp.(μ_ps, Ref(0.), Ref(1.))
        return logpdf.(Binomial.(trials, ps), y_obs) |> sum |> exp
    end
end

function likelihood_mean(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials

    ϵs = rand(Uniform(0, 1), like.int_grid_size)

    # TODO refactor
    function like_mean(x::AbstractVector{<:Real})
        ps_dists = truncated.(Normal.(gp_post(x)...); lower=0., upper=1.)
        
        ll = 0.
        for i in eachindex(y_obs)
            zs = quantile.(Ref(ps_dists[i]), ϵs)
            vals = pdf.(Binomial.(Ref(trials[i]), zs), Ref(y_obs[i]))
            ll += log(mean(vals))
        end
        return exp(ll)
    end
end

function sq_likelihood_mean(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials

    ϵs = rand(Uniform(0, 1), like.int_grid_size)

    # TODO refactor
    function like_mean(x::AbstractVector{<:Real})
        ps_dists = truncated.(Normal.(gp_post(x)...); lower=0., upper=1.)
        
        ll = 0.
        for i in eachindex(y_obs)
            zs = quantile.(Ref(ps_dists[i]), ϵs)
            vals = pdf.(Binomial.(Ref(trials[i]), zs), Ref(y_obs[i])) .^ 2
            ll += log(mean(vals))
        end
        return exp(ll)
    end
end

function get_subset(like::BinomialLikelihood, y_set::AbstractVector{<:Bool})
    return BinomialLikelihood(
        like.y_obs[y_set],
        like.trials[y_set],
        like.int_grid_size,
    )
end
