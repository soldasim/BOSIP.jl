
@kwdef mutable struct MetricCallback <: BolfiCallback
    reference::Any #::Union{Function, Matrix{Float64}} true logpost or reference samples
    logpost_estimator::Function = log_posterior_mean
    metric::DistributionMetric
    sampler::DistributionSampler
    sample_count::Int
    score_history::Vector{Float64} = Float64[]
    true_samples::Union{Nothing, Matrix{Float64}} = nothing
    approx_samples::Union{Nothing, Matrix{Float64}} = nothing
end

function (cb::MetricCallback)(problem::BolfiProblem; kwargs...)
    domain = problem.problem.domain

    ### sample posterior
    if cb.reference isa Function
        true_samples = pure_sample_posterior(cb.sampler, cb.reference, domain, cb.sample_count)
    else
        true_samples = cb.reference
    end

    est_logpost = cb.logpost_estimator(problem)
    approx_samples = pure_sample_posterior(cb.sampler, est_logpost, domain, cb.sample_count)

    cb.true_samples = true_samples
    cb.approx_samples = approx_samples

    ### calculate metric
    score = calculate_metric(cb.metric, true_samples, approx_samples)
    
    @show score
    push!(cb.score_history, score)

    nothing
end
