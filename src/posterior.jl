
# - - - APPROX. POSTERIOR - - - - -

function approx_posterior(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return approx_posterior(gp_post, bolfi.x_prior, bolfi.var_e; normalize, xs, samples)
end

function approx_posterior(gp_post, x_prior, var_e; normalize=false, xs=nothing, samples=10_000)
    gp_μ = gp_mean(gp_post)
    return posterior_mean(gp_μ, x_prior, var_e; normalize, xs, samples)
end


# - - - POSTERIOR MEAN - - - - -

# E[p(y_obs|x) p(x)] <or> E[p(y_obs|x) p(x) / p(y_obs)]
function posterior_mean(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_mean(gp_post, bolfi.x_prior, bolfi.var_e; normalize, xs, samples)
end

function posterior_mean(gp_post, x_prior, var_e; normalize=false, xs=nothing, samples=10_000)
    function mean(x)
        pred = gp_post(x)
        μ_gp, var_gp = pred[1], pred[2]
        y_dim = length(μ_gp)
        ll = pdf(MvNormal(μ_gp, sqrt.(var_e .+ var_gp)), zeros(y_dim))
        px = pdf(x_prior, x)
        return ll * px # / py
    end

    if normalize
        py = evidence(mean, x_prior; xs, samples)
        return (x) -> mean(x) / py
    else
        return mean
    end
end


# - - - POSTERIOR VARIANCE - - - - -

# V[p(y_obs|x) p(x)] <or> V[p(y_obs|x) p(x) / p(y_obs)]
function posterior_variance(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(gp_post, bolfi.x_prior, bolfi.var_e; normalize, xs, samples)
end

function posterior_variance(gp_post, x_prior, var_e; normalize=false, xs=nothing, samples=10_000)
    if normalize
        isnothing(xs) && (xs = rand(x_prior, samples))
        mean = posterior_mean(gp_post, x_prior, var_e; normalize=false, xs)
        py = evidence(mean, x_prior; xs)
    else
        py = 1.
    end

    function var(x)
        pred = gp_post(x)
        μ_y, var_y = pred[1], pred[2]
        prodA = log.(A_.(var_e, μ_y, var_y)) |> sum |> exp
        prodB = log.(B_.(var_e, μ_y, var_y)) |> sum |> exp
        px = pdf(x_prior, x)
        return (px^2 / py^2) * (prodA - prodB)
    end
end
function A_(var_e, μ_y, var_y)
    varA = var_e + 2*var_y
    return sqrt(varA / var_e) * pdf(Normal(0, sqrt(varA)), μ_y)^2
end
function B_(var_e, μ_y, var_y)
    varB = var_e + var_y
    return pdf(Normal(0, sqrt(varB)), μ_y)^2
end


# - - - EVIDENCE ESTIMATION - - - - -

"""
Estimate p(y_obs).

`samples` have to be drawn from `x_prior`.
"""
function evidence(post, x_prior; xs=nothing, samples=10_000)
    ll(x) = post(x) / pdf(x_prior, x)
    isnothing(xs) && (xs = rand(x_prior, samples))
    py = mean((ll(x) for x in eachcol(xs)))
    return py
end
