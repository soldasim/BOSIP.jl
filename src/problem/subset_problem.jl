
get_subset(bolfi::BolfiProblem{Matrix{Bool}}, idx::Int) =
    get_subset(bolfi, bolfi.y_sets[:, idx])

function get_subset(bolfi::BolfiProblem, y_set::AbstractVector{<:Bool})
    return BolfiProblem(
        get_subset(bolfi.problem, y_set),
        get_subset(bolfi.likelihood, y_set),
        bolfi.x_prior,
        nothing,
    )
end

function get_subset(prob::BossProblem, y_set::AbstractVector{<:Bool})
    @assert prob.fitness isa NoFitness
    return BossProblem(
        prob.fitness,
        (x) -> prob.f(x)[y_set],
        prob.domain,
        prob.y_max[y_set],
        get_subset(prob.model, y_set),
        get_subset(prob.params, y_set),
        get_subset(prob.data, y_set),
        prob.consistent,
    )
end

function get_subset(model::GaussianProcess, y_set::AbstractVector{<:Bool})
    return GaussianProcess(
        isnothing(model.mean) ? nothing : (x) -> model.mean(x)[y_set],
        model.kernel,
        model.amp_priors[y_set],
        model.length_scale_priors[y_set],
        model.noise_std_priors[y_set],
    )
end

function get_subset(params::GaussianProcessParams, y_set::AbstractVector{<:Bool})
    return GaussianProcessParams(
        params.λ[:,y_set],
        params.α[y_set],
        params.σ[y_set],
    )
end

function get_subset(data::ExperimentData, y_set::AbstractVector{<:Bool})
    return ExperimentData(
        data.X,
        data.Y[y_set, :],
    )
end
