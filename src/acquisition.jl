
"""
An abstract type for BOLFI acquisition functions.

# Creating custom acquisition function for BOLFI:
- Create type `CustomAcq <: BolfiAcquisition`
- Create type `CustomAcqBoss <: AcquisitionFunction`
- Implement method `(::CustomAcq)(::BolfiProblem) -> ::CustomAcqBoss`
- Implement method `(::CustomAcqBoss)(::BossProblem, ::BossOptions) -> ::Function` (see `BOSS.AcquisitionFunction`)
"""
abstract type BolfiAcquisition end


# - - - Posterior Variance - - - - -

struct PostVariance <: BolfiAcquisition end

(::PostVariance)(problem::BolfiProblem{Nothing}) = PostVarianceBoss(
    problem.var_e,
    problem.x_prior,
)

struct PostVarianceBoss <: AcquisitionFunction
    var_e::Vector{Float64}
    x_prior::MultivariateDistribution
end

function (acq::PostVarianceBoss)(problem::BossProblem, options::BossOptions)
    @assert problem.data isa BOSS.ExperimentDataMLE
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(acq.x_prior, gp_post; var_e=acq.var_e)
end


# - - - Posterior Variance for Observation Sets - - - - -

struct SetsPostVariance <: BolfiAcquisition
    samples::Int
end
SetsPostVariance(;
    samples = 10_000,
) = SetsPostVariance(samples)

(acq::SetsPostVariance)(problem::BolfiProblem{Matrix{Bool}}) = SetsPostVarianceBoss(
    acq.samples,
    problem.var_e,
    problem.x_prior,
    problem.y_sets,
)

struct SetsPostVarianceBoss <: AcquisitionFunction
    samples::Int
    var_e::Vector{Float64}
    x_prior::MultivariateDistribution
    y_sets::Matrix{Bool}
end

function (acq::SetsPostVarianceBoss)(problem::BossProblem, options::BossOptions)
    @assert problem.data isa BOSS.ExperimentDataMLE

    gp_posts = BOSS.model_posterior(problem.model, problem.data; split=true)
    xs = rand(acq.x_prior, acq.samples)  # shared samples
    
    set_vars = Vector{Function}(undef, size(acq.y_sets)[2])
    for i in 1:size(acq.y_sets)[2]
        set = acq.y_sets[:,i]
        post = combine_gp_posts(gp_posts[set])
        var_e = acq.var_e[set]
        py = evidence(acq.x_prior, post; var_e, xs)
        set_vars[i] = posterior_variance(acq.x_prior, post; var_e, py)
    end

    return (x) -> mean((v(x) for v in set_vars))
end

function combine_gp_posts(gp_posts)
    function post(x)
        preds = [post(x) for post in gp_posts]
        first.(preds), last.(preds)
    end
end
