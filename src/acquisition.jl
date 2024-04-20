
struct PDFVariance <: AcquisitionFunction
    var_e::Vector{Float64}
end
PDFVariance(; var_e) = PDFVariance(var_e)

function (acq::PDFVariance)(problem::BossProblem, options::BossOptions)
    @assert problem.data isa BOSS.ExperimentDataMLE
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    x_prior = get_x_prior()
    return posterior_variance(problem.model, x_prior, gp_post; var_e=acq.var_e)
end
