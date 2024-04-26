
"""
Returns the (approximate) normalized posterior `post(x) = p(y_obs|x) p(x) / p(y_obs)`
together with cutoff `c` s.t. the ratio `q` of probability mass
lies within the area given by `{x | post(x) > c}`
and area `V` relative to the whole support of `post(x)`.
"""
function find_cutoff(bolfi::BolfiProblem, q; xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return find_cutoff(gp_post, bolfi.x_prior, bolfi.var_e, q; xs, samples)
end
function find_cutoff(gp_post, x_prior, var_e, q; xs=nothing, samples=10_000)    
    μ = posterior_mean(x_prior, gp_post; var_e)  # μ(x) ≈ p(y_obs|x) p(x)
    ll(x) = μ(x) / pdf(x_prior, x)  # ll(x) ≈ p(y_obs|x)

    if isnothing(xs)
        xs = rand(x_prior, samples)
    end
    # py = mean((ll(x) for x in eachcol(xs)))
    # post(x) = μ(x) / py  # post(x) ≈ p(x|y_obs)
    
    ws = ll.(eachcol(xs))
    vals = μ.(eachcol(xs))
    c = quantile(vals, Distributions.weights(ws), 1. - q)
    
    V = approx_cutoff_area(x_prior, μ, c; xs)

    return μ, c, V
end

"""
Returns a function `p_max` (not a proper pdf function) and cutoff `c`
which together define a subset `S` of the domain of the relative size `V`
s.t. `S` contains all points `x` which belong to the `q` confidence interval
of _any_ of the GP realization within the `conf_int` confidence interval of the GP posterior.
"""
function find_cutoff_confint(bolfi::BolfiProblem, q; conf_int=0.9, xs=nothing, samples=10_000)
    problem = bolfi.problem
    gp_post = BOSS.model_posterior(problem.model, problem.data)
    return find_cutoff_confint(gp_post, bolfi.x_prior, bolfi.var_e, q; conf_int, xs, samples)
end
function find_cutoff_confint(gp_post, x_prior, var_e, q; conf_int=0.9, xs=nothing, samples=10_000)
    gp_lb = gp_quantile(gp_post, 0.5 - (conf_int / 2))
    gp_ub = gp_quantile(gp_post, 0.5 + (conf_int / 2))

    # Posterior predictive function of `x -> (y - y_obs)` which maximizes
    # the likelihood `p(y_obs|x)` within the `conf_int` confidence interval
    # of the GP posterior.
    function gp_max(x)
        m_lb, _ = gp_lb(x)
        m_ub, _ = gp_ub(x)

        m = ifelse.(abs.(m_lb) .< abs.(m_ub), m_lb, m_ub)
        m[sign.(m_lb) .!= sign.(m_ub)] .= 0.
        return m, zeros(length(m))
    end

    return find_cutoff(gp_max, x_prior, var_e, q; xs, samples)
end

"""
Return predictive function `(x) -> (m, 0.)`, where `m` is the `q`th quantile
of the posterior predictive distribution of the GP posterior.
"""
function gp_quantile(gp_post, q)
    return function post(x)
        m, var = gp_post(x)
        d = Normal.(m, sqrt.(var))
        m_ = quantile.(d, Ref(q))
        var_ = fill(0., length(m))
        return m_, var_
    end
end

"""
Approximate the area where `p(x) > c` relative to the whole support of `p(x)`.
(The prior `x_prior` must support the whole support of `p(x)`.)
"""
function approx_cutoff_area(x_prior, p, c; xs=nothing, samples=10_000)
    if isnothing(xs)
        xs = rand(x_prior, samples)
    end
    ps = p.(eachcol(xs))
    ws = 1 ./ pdf.(Ref(x_prior), eachcol(xs))
    ws ./= sum(ws)
    V = sum(ws[ps .> c])
    return V
end
