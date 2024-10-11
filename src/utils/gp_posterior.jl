
"""
    gp_post_ = gp_mean(gp_post)

Get the posterior predictive distribution of the GPs `(x) -> (μ, σ)`
and return a function `(x) -> (μ, 0)`.

# See Also

[`gp_bound`](@ref)
[`gp_quantile`](@ref)
"""
function gp_mean(gp_post)
    return function post(x)
        m, std = gp_post(x)
        std_ = zero(m)
        return m, std_
    end
end

"""
    gp_post_ = gp_bound(gp_post, n)

Get the posterior predictive distribution of the GPs `(x) -> (μ, σ)`
and return a function `(x) -> (μ .+ n * σ, 0)`.

# See Also

[`gp_mean`](@ref)
[`gp_quantile`](@ref)
"""
function gp_bound(gp_post, n)
    return function post(x)
        m, std = gp_post(x)
        m_ = m .+ (n * std)
        std_ = fill(0., length(m))
        return m_, std_
    end
end

"""
    gp_post_ = gp_quantile(gp_post, q)

Get the posterior predictive distribution of the GPs `(x) -> (μ, σ)`
and return a function `(x) -> (a, 0)`,
where `a` is a vector of `q`-th quantiles of `Normal.(μ, σ)`.

# See Also

[`gp_mean`](@ref)
[`gp_bound`](@ref)
"""
function gp_quantile(gp_post, q)
    return function post(x)
        m, std = gp_post(x)
        d = Normal.(m, std)
        m_ = quantile.(d, Ref(q))
        std_ = fill(0., length(m))
        return m_, std_
    end
end
