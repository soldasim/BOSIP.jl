
"""
    MWMVAcq(; kwargs...)

The Mass-Weighted Mean Variance acquisition function.

Selects the next evaluation point by maximizing a weighted average of the variances
of the individual posterior approximations given by different sensor sets.
The weights are determined as the total probability mass of the current data
w.r.t. each approximate posterior.

# Keywords

- `samples::Int`: The number of samples used to estimate the evidence.
"""
@kwdef struct MWMVAcq <: BolfiAcquisition
    samples::Int = 10_000
end

function (acq::MWMVAcq)(::Type{<:UniFittedParams}, bolfi::BolfiProblem{Matrix{Bool}}, options::BolfiOptions)
    sets = size(bolfi.y_sets)[2]
    xs = rand(bolfi.x_prior, acq.samples)  # shared samples
    
    set_vars = Vector{Function}(undef, sets)
    ws = Vector{Float64}(undef, sets)

    for i in 1:sets
        bolfi_ = get_subset(bolfi, i)
        post_mean = posterior_mean(bolfi_; normalize=true, xs)
        ws[i] = 1. / sum(post_mean.(eachcol(bolfi_.problem.data.X)))
        set_vars[i] = posterior_variance(bolfi_; normalize=true, xs)
    end
    
    function mwmv_acq(x)
        val = 0.
        for i in 1:sets
            val += ws[i] * set_vars[i](x)
        end
        return val / sets
    end
end

# TODO: Think about MWMVAcq for BI
function (acq::MWMVAcq)(::Type{<:MultiFittedParams}, bolfi::BolfiProblem{Matrix{Bool}}, options::BolfiOptions)
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
