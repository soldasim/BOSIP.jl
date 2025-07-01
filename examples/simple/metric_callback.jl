
@kwdef mutable struct MetricCallback <: BolfiCallback
    metric::DistributionMetric
    sampler::DistributionSampler
    sample_count::Int
    score_history::Vector{Float64} = Float64[]
    true_samples::Union{Nothing, Matrix{Float64}} = nothing
    approx_samples::Union{Nothing, Matrix{Float64}} = nothing
    plot_callback::BolfiCallback = BOLFI.NoCallback()
end

function (cb::MetricCallback)(problem::BolfiProblem; kwargs...)
    @info "Calculating the score metric ..."

    # TODO

    prior = problem.x_prior
    domain = problem.problem.domain

    if cb.metric isa SampleMetric
        ### approximate likelihood
        approx_like = likelihood_mean(problem)

        ### sample posteriors
        true_samples = sample_posterior(cb.sampler, ToyProblem.true_like, prior, domain, cb.sample_count)
        approx_samples = sample_posterior(cb.sampler, approx_like, prior, domain, cb.sample_count)
        
        cb.true_samples = true_samples
        cb.approx_samples = approx_samples

        ### calculate metric
        score = calculate_metric(cb.metric, true_samples, approx_samples)

    elseif cb.metric isa PDFMetric
        ### approximate posterior
        approx_post = approx_posterior(problem)

        e_true = 

        ### calculate metric
        score = calculate_metric(cb.metric, ToyProblem.true_post, approx_post)

    else
        @assert false
    end
    

    @show score
    push!(cb.score_history, score)

    ### plot
    if !isnothing(cb.plot_callback)
        cb.plot_callback(problem;
            # TODO
            # sample_reference = cb.true_samples,
            # sample_posterior = cb.approx_samples,
            sample_reference = false,
            sample_posterior = false,
            kwargs...,
        )
    end
end
