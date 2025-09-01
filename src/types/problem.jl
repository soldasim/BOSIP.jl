
"""
    BosipProblem(X, Y; kwargs...)
    BosipProblem(::ExperimentData; kwargs...)

Defines the likelihood-free inference problem and stores all data.

# Args

The initial data are provided either as two column-wise matrices `X` and `Y`
with inputs and outputs of the simulator respectively, or as an instance of `BOSS.ExperimentData`.

Currently, at least one datapoint has to be provided (purely for implementation reasons).

# Kwargs

- `f::Any`: The simulation to be queried for data.
- `domain::Domain`: The parameter domain of the problem.
- `acquisition::BosipAcquisition`: Defines the acquisition function.
- `model::SurrogateModel`: The surrogate model to be used to model the proxy `δ`.
- `likelihood::Likelihood`: The likelihood of the experiment observation `z_o`.
- `x_prior::MultivariateDistribution`: The prior `p(x)` on the input parameters.
- `y_sets::Union{Nothing, Matrix{Bool}}`: Optional parameter intended for advanced usage.
        The binary columns define subsets `y_1, ..., y_m` of the observation dimensions within `y`.
        The algorithm then trains multiple posteriors `p(θ|y_1), ..., p(θ|y_m)` simultaneously.
        The posteriors can be compared after the run is completed to see which observation subsets are most informative.
"""
mutable struct BosipProblem{
    S<:Union{Nothing, Matrix{Bool}},
}
    problem::BossProblem
    likelihood::Likelihood
    x_prior::MultivariateDistribution
    y_sets::S
end

BosipProblem(problem, likelihood, x_prior) =
    BosipProblem(problem, likelihood, x_prior, nothing)

BosipProblem(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}; kwargs...) =
    BosipProblem(ExperimentData(X, Y); kwargs...)

"""
    DummyAcq()

Dummy acquisition function used only in the `BosipProblem` constructor
as a temporary placeholder.

It does _not_ implement the `BOSS.AcquisitionFunction` API.
"""
struct DummyAcq <: AcquisitionFunction end

function BosipProblem(data::ExperimentData;
    f,
    domain,
    acquisition = MaxVar(),
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

    bosip = BosipProblem(
        problem,
        likelihood,
        x_prior,
        y_sets,
    )

    # fill in the acquisition function
    bosip.problem.acquisition = AcqWrapper(
        acquisition,
        bosip,
        # leave default options for now
        # they are updated later in `_init_problem!`
        BosipOptions(),
    )

    return bosip
end

x_dim(bosip::BosipProblem) = x_dim(bosip.problem)
y_dim(bosip::BosipProblem) = y_dim(bosip.problem)
