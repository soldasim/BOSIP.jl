
"""
    BolfiProblem(data::ExperimentData; kwargs...)

Defines the LFI problem together with most hyperparameters for the BOLFI procedure.

# Kwargs
- `f::Any`: The simulation to be queried for data. Must follow the signature `f(x) = sim(y) - y_obs`.
- `bounds::AbstractBounds`: The basic box-constraints on `x`. This field is mandatory.
- `discrete::AbstractVector{<:Bool}`: Can be used to designate some dimensions
        of the domain as discrete.
- `cons::Union{Nothing, Function}`: Used to define arbitrary nonlinear constraints on `x`.
        Feasible points `x` must satisfy `all(cons(x) .> 0.)`. An appropriate acquisition
        maximizer which can handle nonlinear constraints must be used if `cons` is provided.
        (See [`BOSS.AcquisitionMaximizer`](@ref).)
- `kernel::Kernel`: The kernel used in the GP. Defaults to the `Matern32Kernel()`.
- `length_scale_priors::AbstractVector{<:MultivariateDistribution}`: The prior distributions
        for the length scales of the GP. The `length_scale_priors` should be a vector
        of `y_dim` `x_dim`-variate distributions where `x_dim` and `y_dim` are
        the dimensions of the input and output of the model respectively.
- `noise_var_priors::AbstractVector{<:UnivariateDistribution}`: The prior distributions
        of the noise variances of each `y` dimension.
- `var_e::Vector{Float64}`: The variances of the Gaussian noise of the observation `y_obs`
        in individual dimensions.
- `x_prior::MultivariateDistribution`: The prior `p(x)` on the input parameters.
"""
struct BolfiProblem
    problem::BossProblem
    var_e::Vector{Float64}
    x_prior::MultivariateDistribution
end

function BolfiProblem(data;
    f,
    bounds,
    discrete=fill(false, length(first(bounds))),
    cons=nothing,
    kernel=BOSS.Matern32Kernel(),
    length_scale_priors,
    noise_var_priors,
    var_e,
    x_prior,
)
    domain = Domain(;
        bounds,
        discrete,
        cons,
    )

    model = GaussianProcess(;
        kernel,
        length_scale_priors,
    )

    problem = BossProblem(;
        f,
        domain,
        model,
        noise_var_priors,
        data,
    )

    return BolfiProblem(
        problem,
        var_e,
        x_prior,
    )
end
