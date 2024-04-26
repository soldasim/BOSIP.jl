# Contains useful methods for the user. They are not used within the package.

get_subset(bolfi::BolfiProblem{Matrix{Bool}}, idx::Int) =
    get_subset(bolfi, bolfi.y_sets[:, idx])

function get_subset(bolfi::BolfiProblem, y_set::AbstractVector{<:Bool})
    return BolfiProblem(
        get_subset(bolfi.problem, y_set),
        bolfi.var_e[y_set],
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
        prob.noise_var_priors[y_set],
        get_subset(prob.data, y_set),
    )
end

function get_subset(model::GaussianProcess, y_set::AbstractVector{<:Bool})
    return GaussianProcess(
        isnothing(model.mean) ? nothing : (x) -> model.mean(x)[y_set],
        model.kernel,
        model.length_scale_priors[y_set],
    )
end

function get_subset(data::ExperimentDataPrior, y_set::AbstractVector{<:Bool})
    return ExperimentDataPrior(
        data.X,
        data.Y[y_set, :],
    )
end
function get_subset(data::ExperimentDataMLE, y_set::AbstractVector{<:Bool})
    return ExperimentDataMLE(
        data.X,
        data.Y[y_set, :],
        data.θ,
        data.length_scales[:, y_set],
        data.noise_vars[y_set],
    )
end
function get_subset(data::ExperimentDataBI, y_set::AbstractVector{<:Bool})
    return ExperimentDataBI(
        data.X,
        data.Y[y_set, :],
        data.θ,
        [ls[:, y_set] for ls in data.length_scales],
        data.noise_vars[y_set, :],
    )
end
