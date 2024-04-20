
struct BolfiProblem
    problem::BossProblem
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
        x_prior,
    )
end
