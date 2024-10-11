
# - - - Posterior Variance - - - - -

"""
    PostVarAcq()

Selects the new evaluation point by maximizing the variance of the posterior approximation.
"""
struct PostVarAcq <: BolfiAcquisition end

function (acq::PostVarAcq)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return posterior_variance(bolfi; normalize=false)
end


# - - - Posterior Variance for Observation Sets - - - - -

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
struct MWMVAcq <: BolfiAcquisition
    samples::Int
end
MWMVAcq(;
    samples = 10_000,
) = MWMVAcq(samples)

function (acq::MWMVAcq)(bolfi::BolfiProblem{Matrix{Bool}}, options::BolfiOptions)
    problem = bolfi.problem
    @assert problem.data isa BOSS.ExperimentDataMAP

    gp_posts = BOSS.model_posterior_slice.(Ref(problem), 1:BOSS.y_dim(problem))
    xs = rand(bolfi.x_prior, acq.samples)  # shared samples
    
    set_vars = Vector{Function}(undef, size(bolfi.y_sets)[2])
    ws = Vector{Float64}(undef, size(bolfi.y_sets)[2])
    for i in eachindex(set_vars)
        set = bolfi.y_sets[:,i]
        post = combine_gp_posts(gp_posts[set])
        std_obs = bolfi.std_obs[set]
        μ = posterior_mean(post, bolfi.x_prior, std_obs; normalize=true, xs)
        ws[i]  = 1. / sum(μ.(eachcol(bolfi.problem.data.X)))
        set_vars[i] = posterior_variance(post, bolfi.x_prior, std_obs; normalize=true, xs)
    end
    
    return (x) -> mean((w * v(x) for (w, v) in zip(ws, set_vars)))
end

function combine_gp_posts(gp_posts)
    function post(x)
        preds = [post(x) for post in gp_posts]
        first.(preds), last.(preds)
    end
end
