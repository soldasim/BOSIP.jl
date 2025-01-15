
get_subset(bolfi::BolfiProblem{<:Any, Matrix{Bool}}, idx::Int) =
    get_subset(bolfi, bolfi.y_sets[:, idx])

function get_subset(bolfi::BolfiProblem, y_set::AbstractVector{<:Bool})
    return BolfiProblem(
        get_subset(bolfi.problem, y_set),
        isnothing(bolfi.std_obs) ? nothing : bolfi.std_obs[y_set],
        bolfi.x_prior,
        nothing,
    )
end

function get_subset(prob::BossProblem, y_set::AbstractVector{<:Bool})
    @assert prob.fitness isa NoFitness
    return BossProblem(
        prob.fitness,
        (x) -> prob.f(x)[y_set],
        get_subset(prob.model, y_set),
        get_subset(prob.data, y_set),
        prob.domain,
        prob.y_max[y_set],
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

function get_subset(data::ExperimentDataPrior, y_set::AbstractVector{<:Bool})
    return ExperimentDataPrior(
        data.X,
        data.Y[y_set, :],
    )
end
function get_subset(data::ExperimentDataMAP, y_set::AbstractVector{<:Bool})
    return ExperimentDataMAP(
        data.X,
        data.Y[y_set, :],
        get_subset(data.params, y_set),
        data.consistent,
    )
end
function get_subset(data::ExperimentDataBI, y_set::AbstractVector{<:Bool})
    return ExperimentDataBI(
        data.X,
        data.Y[y_set, :],
        get_subset.(data.params, Ref(y_set)),
        data.consistent,
    )
end

function get_subset(params::BOSS.ModelParams, y_set::AbstractVector{<:Bool})
    θ, λ, α, noise = params
    return (
        θ,
        λ[:, y_set],
        α[y_set],
        noise[y_set],
    )
end
