
"""
    BolfiProblem(X, Y; kwargs...)
    BolfiProblem(data::ExperimentData; kwargs...)
    BolfiProblem(::BossProblem, ::Likelihood, x_prior::MultivariateDistribution)
    BolfiProblem(::BossProblem, ::Likelihood, x_prior::MultivariateDistribution, y_sets::Matrix{Bool})

Defines the likelihood-free inference problem and stores all data.

# Args

The initial data are provided either as two column-wise matrices `X` and `Y`
with inputs and outputs of the simulator respectively, or as an instance of `BOSS.ExperimentData`.

Currently, at least one datapoint has to be provided (purely for implementation reasons).

# Kwargs

- `f::Any`: The simulation to be queried for data.
- `bounds::AbstractBounds`: The basic box-constraints on `x`. This field is mandatory.
- `discrete::AbstractVector{<:Bool}`: Can be used to designate some dimensions
        of the domain as discrete.
- `cons::Union{Nothing, Function}`: Used to define arbitrary nonlinear constraints on `x`.
        Feasible points `x` must satisfy `all(cons(x) .> 0.)`. An appropriate acquisition
        maximizer which can handle nonlinear constraints must be used if `cons` is provided.
        (See `BOSS.AcquisitionMaximizer`.)
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `lengthscale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `lengthscale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
- `amplitude_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        for the amplitude hyperparameters of the GP. The `amplitude_priors` should be a vector
        of `y_dim` univariate distributions.
- `noise_std_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        of the standard deviations the Gaussian simulator noise on each dimension of the output `y`.
- `likelihood::Likelihood`: The likelihood of the experiment observation `y_o`.
- `x_prior::MultivariateDistribution`: The prior `p(x)` on the input parameters.
- `y_sets::Matrix{Bool}`: Optional parameter intended for advanced usage.
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

function BolfiProblem(data::ExperimentData;
    f,
    bounds,
    discrete = fill(false, length(first(bounds))),
    cons = nothing,
    kernel = BOSS.Matern32Kernel(),
    lengthscale_priors = BOSS.NoVal, length_scale_priors = BOSS.NoVal, # deprecated
    amplitude_priors = BOSS.NoVal, amp_priors = BOSS.NoVal, # deprecated
    noise_std_priors,
    likelihood,
    x_prior,
    y_sets = nothing,
)
    domain = Domain(;
        bounds,
        discrete,
        cons,
    )

    model = GaussianProcess(;
        kernel,
        amplitude_priors, amp_priors,
        lengthscale_priors, length_scale_priors,
        noise_std_priors,
    )

    problem = BossProblem(;
        f,
        domain,
        model,
        data,
    )

    return BolfiProblem(
        problem,
        likelihood,
        x_prior,
        y_sets,
    )
end

BOSS.x_dim(bolfi::BolfiProblem) = BOSS.x_dim(bolfi.problem)
BOSS.y_dim(bolfi::BolfiProblem) = BOSS.y_dim(bolfi.problem)
