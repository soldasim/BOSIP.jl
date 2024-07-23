
get_subset(bolfi::BolfiProblem{Matrix{Bool}}, idx::Int) =
    get_subset(bolfi, bolfi.y_sets[:, idx])

function get_subset(bolfi::BolfiProblem, y_set::AbstractVector{<:Bool})
    return BolfiProblem(
        get_subset(bolfi.problem, y_set),
        bolfi.std_obs[y_set],
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
        prob.noise_std_priors[y_set],
        get_subset(prob.data, y_set),
    )
end

function get_subset(model::GaussianProcess, y_set::AbstractVector{<:Bool})
    return GaussianProcess(
        isnothing(model.mean) ? nothing : (x) -> model.mean(x)[y_set],
        model.kernel,
        model.amp_priors[y_set],
        model.length_scale_priors[y_set],
    )
end

function get_subset(data::ExperimentDataPrior, y_set::AbstractVector{<:Bool})
    return ExperimentDataPrior(
        data.X,
        data.Y[y_set, :],
    )
end
function get_subset(data::ExperimentDataMAP, y_set::AbstractVector{<:Bool})
    @assert isempty(data.θ)
    return ExperimentDataMAP(
        data.X,
        data.Y[y_set, :],
        data.θ,
        data.length_scales[:, y_set],
        data.amplitudes[y_set],
        data.noise_std[y_set],
        data.consistent,
    )
end
function get_subset(data::ExperimentDataBI, y_set::AbstractVector{<:Bool})
    @assert isempty(data.θ)
    return ExperimentDataBI(
        data.X,
        data.Y[y_set, :],
        data.θ,
        [λ[:, y_set] for λ in data.length_scales],
        [α[y_set] for α in data.amplitudes],
        [d[y_set] for d in data.noise_std],
        data.consistent,
    )
end
