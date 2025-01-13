
"""
    BolfiProblem(X, Y; kwargs...)
    BolfiProblem(data::ExperimentData; kwargs...)

Defines the LFI problem together with most hyperparameters for the BOLFI procedure.

# Args

The initial data are provided either as two column-wise matrices `X` and `Y`
with inputs and outputs of the simulator respectively, or as an instance of `BOSS.ExperimentData`.

Currently, at least one datapoint has to be provided (purely for implementation reasons).

# Kwargs

- `f::Any`: The simulation to be queried for data. Must follow the signature `f(x) = sim(x) - y_obs`.
- `bounds::AbstractBounds`: The basic box-constraints on `x`. This field is mandatory.
- `discrete::AbstractVector{<:Bool}`: Can be used to designate some dimensions
        of the domain as discrete.
- `cons::Union{Nothing, Function}`: Used to define arbitrary nonlinear constraints on `x`.
        Feasible points `x` must satisfy `all(cons(x) .> 0.)`. An appropriate acquisition
        maximizer which can handle nonlinear constraints must be used if `cons` is provided.
        (See `BOSS.AcquisitionMaximizer`.)
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `length_scale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
- `amp_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        for the amplitude hyperparameters of the GP. The `amp_priors` should be a vector
        of `y_dim` univariate distributions.
- `noise_std_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        of the standard deviations the Gaussian simulator noise on each dimension of the output `y`.
- `std_obs::Union{Vector{Float64}, Nothing}`: The standard deviations of the Gaussian
        observation noise on each dimension of the "ground truth" observation.
        (If the observation is considered to be generated from the simulator and not some "real" experiment,
        provide `std_obs = nothing`` and the adaptively trained simulation noise deviation will be used
        in place of the experiment noise deviation as well. This may be the case for some toy problems or benchmarks.)
- `x_prior::MultivariateDistribution`: The prior `p(x)` on the input parameters.
- `y_sets::Matrix{Bool}`: Optional parameter intended for advanced usage.
        The binary columns define subsets `y_1, ..., y_m` of the observation dimensions within `y`.
        The algorithm then trains multiple posteriors `p(θ|y_1), ..., p(θ|y_m)` simultaneously.
        The posteriors can be compared after the run is completed to see which observation subsets are most informative.
"""
struct BolfiProblem{
    N<:Union{Vector{Float64}, Nothing},
    S<:Union{Nothing, Matrix{Bool}},
}
    problem::BossProblem
    std_obs::N
    x_prior::MultivariateDistribution
    y_sets::S
end

BolfiProblem(X, Y; kwargs...) =
    BolfiProblem(ExperimentData(X, Y); kwargs...)

function BolfiProblem(data;
    f,
    bounds,
    discrete = fill(false, length(first(bounds))),
    cons = nothing,
    kernel = BOSS.Matern32Kernel(),
    length_scale_priors,
    amp_priors,
    noise_std_priors,
    std_obs,
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
        amp_priors,
        length_scale_priors,
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
        std_obs,
        x_prior,
        y_sets,
    )
end

x_dim(bolfi::BolfiProblem) = BOSS.x_dim(bolfi.problem)
y_dim(bolfi::BolfiProblem) = BOSS.y_dim(bolfi.problem)

function std_obs(bolfi::BolfiProblem)
    return bolfi.std_obs
end
function std_obs(bolfi::BolfiProblem{Nothing})
    θ, λ, α, noise_std = bolfi.problem.data.params
    return noise_std
end
