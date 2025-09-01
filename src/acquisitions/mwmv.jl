
"""
    MWMV(; kwargs...)

The Mass-Weighted Mean Variance acquisition function.

Selects the next evaluation point by maximizing a weighted average of the variances
of the individual posterior approximations given by different sensor sets.
The weights are determined as the total probability mass of the current data
w.r.t. each approximate posterior.

# Keywords

- `samples::Int`: The number of samples used to estimate the evidence.
"""
@kwdef struct MWMV <: BosipAcquisition
    samples::Int = 10_000
end

function (acq::MWMV)(::Type{<:UniFittedParams}, bosip::BosipProblem{Matrix{Bool}}, options::BosipOptions)
    sets = size(bosip.y_sets)[2]
    xs = rand(bosip.x_prior, acq.samples)  # shared samples
    
    set_vars = Vector{Function}(undef, sets)
    ws = Vector{Float64}(undef, sets)

    for i in 1:sets
        bosip_ = get_subset(bosip, i)
        post_mean = posterior_mean(bosip_; normalize=true, xs)
        ws[i] = 1. / sum(post_mean.(eachcol(bosip_.problem.data.X)))
        set_vars[i] = posterior_variance(bosip_; normalize=true, xs)
    end
    
    function mwmv_acq(x)
        val = 0.
        for i in 1:sets
            val += ws[i] * set_vars[i](x)
        end
        return val / sets
    end
end

# TODO: Think about MWMV for BI
function (acq::MWMV)(::Type{<:MultiFittedParams}, bosip::BosipProblem{Matrix{Bool}}, options::BosipOptions)
    throw(ArgumentError("""
        The support for Bayesian inference is not implemented yet for the MWMV acquisition.
        Use a MAP model fitter or another acquisition function.
    """))
end

function combine_gp_posts(gp_posts)
    function post(x)
        preds = [post(x) for post in gp_posts]
        first.(preds), last.(preds)
    end
end
