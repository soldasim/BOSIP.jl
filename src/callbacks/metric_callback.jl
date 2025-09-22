
@kwdef mutable struct MetricCallback <: BosipCallback
    reference::Any #::Union{Function, Matrix{Float64}} true logpost or reference samples
    logpost_estimator::Function = log_posterior_mean
    metric::DistributionMetric
    sampler::DistributionSampler
    sample_count::Int
    score_history::Vector{Float64} = Float64[]
    true_samples::Union{Nothing, Matrix{Float64}} = nothing
    approx_samples::Union{Nothing, Matrix{Float64}} = nothing
end

function (cb::MetricCallback)(problem::BosipProblem; kwargs...)
    score = _calc_score(cb.metric, cb, problem)
    @show score
    push!(cb.score_history, score)
    nothing
end

function _calc_score(metric::SampleMetric, cb::MetricCallback, problem::BosipProblem)
    domain = problem.problem.domain

    ### sample posterior
    if cb.reference isa Function
        true_samples = sample_posterior_pure(cb.sampler, cb.reference, domain, cb.sample_count)
    else
        true_samples = cb.reference
    end

    est_logpost = cb.logpost_estimator(problem)
    approx_samples = sample_posterior_pure(cb.sampler, est_logpost, domain, cb.sample_count)

    cb.true_samples = true_samples
    cb.approx_samples = approx_samples

    ### calculate metric
    score = calculate_metric(metric, true_samples, approx_samples)
    return score
end
function _calc_score(metric::PDFMetric, cb::MetricCallback, problem::BosipProblem)
    ### retrieve the true and approx logpdf
    @assert cb.reference isa Function
    true_logpdf = cb.reference
    approx_logpdf = cb.logpost_estimator(problem)

    ### calculate metric
    score = calculate_metric(metric, true_logpdf, approx_logpdf)
    return score
end
