
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

(::PostVariance)(problem::BolfiProblem) = PostVarianceBoss(problem.var_e, problem.x_prior)

struct PostVarianceBoss <: AcquisitionFunction
    var_e::Vector{Float64}
    x_prior::MultivariateDistribution
end

function (acq::PostVarianceBoss)(problem::BossProblem, options::BossOptions)
    @assert problem.data isa BOSS.ExperimentDataMLE
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(acq.x_prior, gp_post; var_e=acq.var_e)
end
