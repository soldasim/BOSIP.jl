
"""
    BolfiProblem(X, Y; kwargs...)
    BolfiProblem(::ExperimentData; kwargs...)

Defines the likelihood-free inference problem and stores all data.

# Args

The initial data are provided either as two column-wise matrices `X` and `Y`
with inputs and outputs of the simulator respectively, or as an instance of `BOSS.ExperimentData`.

Currently, at least one datapoint has to be provided (purely for implementation reasons).

# Kwargs

- `f::Any`: The simulation to be queried for data.
- `domain::Domain`: The parameter domain of the problem.
- `acquisition::BolfiAcquisition`: Defines the acquisition function.
- `model::SurrogateModel`: The surrogate model to be used to model the proxy `δ`.
- `likelihood::Likelihood`: The likelihood of the experiment observation `z_o`.
- `x_prior::MultivariateDistribution`: The prior `p(x)` on the input parameters.
- `y_sets::Union{Nothing, Matrix{Bool}}`: Optional parameter intended for advanced usage.
        The binary columns define subsets `y_1, ..., y_m` of the observation dimensions within `y`.
        The algorithm then trains multiple posteriors `p(θ|y_1), ..., p(θ|y_m)` simultaneously.
        The posteriors can be compared after the run is completed to see which observation subsets are most informative.
"""
mutable struct BolfiProblem{
    S<:Union{Nothing, Matrix{Bool}},
}
    problem::BossProblem
    likelihood::Likelihood
    x_prior::MultivariateDistribution
    y_sets::S
end

BolfiProblem(problem, likelihood, x_prior) =
    BolfiProblem(problem, likelihood, x_prior, nothing)

BolfiProblem(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; kwargs...) =
    BolfiProblem(ExperimentData(X, Y); kwargs...)

"""
    DummyAcq()

Dummy acquisition function used only in the `BolfiProblem` constructor
as a temporary placeholder.

It does _not_ implement the `BOSS.AcquisitionFunction` API.
"""
struct DummyAcq <: AcquisitionFunction end

function BolfiProblem(data::ExperimentData;
    f,
    domain,
    acquisition = PostVarAcq(),
    model,
    likelihood,
    x_prior,
    y_sets = nothing,
)
    problem = BossProblem(;
        f,
        domain,
        acquisition = DummyAcq(), # is filled in below
        model,
        data,
    )

    bolfi = BolfiProblem(
        problem,
        likelihood,
        x_prior,
        y_sets,
    )

    # fill in the acquisition function
    bolfi.problem.acquisition = AcqWrapper(
        acquisition,
        bolfi,
        # leave default options for now
        # they are updated later in `_init_problem!`
        BolfiOptions(),
    )

    return bolfi
end

x_dim(bolfi::BolfiProblem) = x_dim(bolfi.problem)
y_dim(bolfi::BolfiProblem) = y_dim(bolfi.problem)
