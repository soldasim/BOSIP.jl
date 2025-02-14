
"""
    BinomialLikelihood(; std_obs::Vector{Float64})

The observation is assumed to have been generated from a Binomial distribution
as `y_o \\sim Binomial(trials, f(x))`. We can use the simulator to query `z = f(x)`.

The simulator should only return values between 0 and 1. The GP estimates are clamped to this range.

# Kwargs
- `y_obs::Vector{Int64}`: The observed values from the real experiment.
- `trials::Vector{Int64}`: The number of trials for each observation dimension.
- `int_grid_size::Int64`: The number of points of the grid used for numerical integration.
"""
@kwdef struct BinomialLikelihood <: Likelihood
    y_obs::Vector{Int64}
    trials::Vector{Int64}
    int_grid_size::Int64 = 200
end

function approx_likelihood(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials

    function approx_like(x)
        μ_ps, _ = gp_post(x)
        ps = clamp.(μ_ps, Ref(0.), Ref(1.))
        return logpdf.(Binomial.(trials, ps), y_obs) |> sum |> exp
    end
end

function likelihood_mean(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials
    grid = range(0., 1.; length=like.int_grid_size)

    # TODO refactor
    function like_mean(x)
        ps_dists = Normal.(gp_post(x)...)
        
        ll = 0.
        for i in eachindex(y_obs)
            ws = pdf.(Ref(ps_dists[i]), grid)
            _normalize_weights!(ws)
            vals = pdf.(Binomial.(Ref(trials[i]), grid), Ref(y_obs[i]))
            ll += log(ws' * vals)
        end
        return exp(ll)
    end
end

function sq_likelihood_mean(like::BinomialLikelihood, bolfi, gp_post)
    y_obs = like.y_obs
    trials = like.trials
    grid = range(0., 1.; length=like.int_grid_size)

    # TODO refactor
    function like_mean(x)
        ps_dists = Normal.(gp_post(x)...)

        ll = 0.
        for i in eachindex(y_obs)
            ws = pdf.(Ref(ps_dists[i]), grid)
            _normalize_weights!(ws)
            vals = pdf.(Binomial.(Ref(trials[i]), grid), Ref(y_obs[i])) .^ 2
            ll += log(ws' * vals)
        end
        return exp(ll)
    end
end

function _normalize_weights!(ws)
    s = sum(ws)
    if s == 0.
        @warn "BinomialLikelihood: All weights are zero during numerical integration!"
        ws .= 1 / length(ws)
    else
        ws ./= s
    end
    return ws
end

function get_subset(like::BinomialLikelihood, y_set::AbstractVector{<:Bool})
    return BinomialLikelihood(
        like.y_obs[y_set],
        like.trials[y_set],
        like.int_grid_size,
    )
end
