
# - - - POSTERIOR MEAN - - - - -

# E[p(y_obs|x) p(x)] <or> E[p(y_obs|x) p(x) / p(y_obs)]
function posterior_mean(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_mean(bolfi.x_prior, gp_post, bolfi.var_e; normalize, xs, samples)
end

function posterior_mean(x_prior, gp_post, var_e; normalize=false, xs=nothing, samples=10_000)
    if normalize
        py = evidence(x_prior, gp_post, var_e; xs, samples)
    else
        py = 1.
    end
    
    function mean(x)
        pred = gp_post(x)
        μ_gp, var_gp = pred[1], pred[2]
        y_dim = length(μ_gp)
        ll = pdf(MvNormal(μ_gp, sqrt.(var_e .+ var_gp)), zeros(y_dim))
        px = pdf(x_prior, x)
        return (px / py) * ll
    end
end


# - - - POSTERIOR VARIANCE - - - - -

# V[p(y_obs|x) p(x)] <or> V[p(y_obs|x) p(x) / p(y_obs)]
function posterior_variance(bolfi::BolfiProblem; normalize=false, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return posterior_variance(bolfi.x_prior, gp_post, bolfi.var_e; normalize, xs, samples)
end

function posterior_variance(x_prior, gp_post, var_e; normalize=false, xs=nothing, samples=10_000)
    if normalize
        py = evidence(x_prior, gp_post, var_e; xs, samples)
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

# Estimate p(y_obs)
function evidence(bolfi::BolfiProblem; xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return evidence(bolfi.x_prior, gp_post, bolfi.var_e; xs, samples)
end

function evidence(x_prior, gp_post, var_e; xs=nothing, samples=10_000)
    p = posterior_mean(x_prior, gp_post, var_e; normalize=false)
    isnothing(xs) && (xs = rand(x_prior, samples))
    py = mean((p(x) for x in eachcol(xs)))
    return py
end
