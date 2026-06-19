
"""
    MetricCallback(; kwargs...)

A [`BosipCallback`](@ref) that evaluates a [`DistributionMetric`](@ref) between the learned posterior
and a reference once before the run and after every iteration, recording the scores in `score_history`.

## Keywords
- `reference`: The ground-truth log-posterior function, or a matrix of reference samples.
- `logpost_estimator::Function`: Maps the `BosipProblem` to the log-posterior estimate to be scored.
- `metric::DistributionMetric`: The metric used to compare the posteriors.
- `sampler::DistributionSampler`: The sampler used to draw samples from the approximate posterior.
- `sample_count::Int`: The number of samples drawn to estimate the metric.
- `score_history::Vector{Float64}`: The recorded metric scores (populated during the run).
"""
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

function (cb::MetricCallback)(problem::BosipProblem; first::Bool, options::BossOptions, kwargs...)
    if first && !isempty(cb.score_history)
        options.info && @warn "A continued run detected. Not calculating the first metric score to avoid duplicates."
        return
    end

    score = _calc_score(cb.metric, cb, problem)
    options.info && @show score
    push!(cb.score_history, score)
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
