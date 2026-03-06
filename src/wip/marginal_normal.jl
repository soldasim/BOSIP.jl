### Methods for BOSIP api

export approx_marginal_likelihood, marginal_likelihood_mean
export log_approx_marginal_likelihood, log_marginal_likelihood_mean
export loglike_marginal

"""
    approx_marginal_likelihood(bosip::BosipProblem) -> ::Function

## Example
```julia
bosip::BosipProblem
ml = approx_marginal_likelihood(bosip)
x = [1,1]
X = [1;1;; 2;2;; 3;3;;]
ml(x) # -> vector
ml(X) # -> matrix
```
"""
function approx_marginal_likelihood(bosip::BosipProblem)
    log_approx_ml = log_approx_marginal_likelihood(bosip)
    return x -> exp.(log_approx_ml(x))
end

"""
    marginal_likelihood_mean(bosip::BosipProblem) -> ::Function

## Example
```julia
bosip::BosipProblem
ml = marginal_likelihood_mean(bosip)
x = [1,1]
X = [1;1;; 2;2;; 3;3;;]
ml(x) # -> vector
ml(X) # -> matrix
```
"""
function marginal_likelihood_mean(bosip::BosipProblem)
    log_ml_mean = log_marginal_likelihood_mean(bosip)
    return x -> exp.(log_ml_mean(x))
end

"""
    log_approx_marginal_likelihood(bosip::BosipProblem) -> ::Function
    log_approx_marginal_likelihood(like::Likelihood, model_post::ModelPosterior) -> ::Function
"""
function log_approx_marginal_likelihood(bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    return log_approx_marginal_likelihood(bosip.likelihood, model_post)
end

"""
    log_marginal_likelihood_mean(bosip::BosipProblem) -> ::Function
    log_marginal_likelihood_mean(like::Likelihood, model_post::ModelPosterior) -> ::Function
"""
function log_marginal_likelihood_mean(bosip::BosipProblem)
    model_post = BOSS.model_posterior(bosip.problem)
    return log_marginal_likelihood_mean(bosip.likelihood, model_post)
end


### Methods for any Likelihood

function log_approx_marginal_likelihood(like::Likelihood, model_post::ModelPosterior)
    function log_approx_ml(x::AbstractVector{<:Real})
        μy = mean(model_post, x)

        return loglike_marginal(like, μy, x)
    end
    function log_approx_ml(X::AbstractMatrix{<:Real})
        μY = mean(model_post, X)

        lls = loglike_marginal.(Ref(like), eachcol(μY), eachcol(X))
        return hcat(lls...)
    end
    return log_approx_ml
end

"""
    loglike_marginal(like::Likelihood, y::AbstractVector{<:Real}, [x::AbstractVector{<:Real}]) -> ::Real
"""
function loglike_marginal(like::Likelihood, y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    return loglike_marginal(like, y)
end


### Methods for NormalLikelihood

function log_marginal_likelihood_mean(like::NormalLikelihood, model_post::ModelPosterior)
    z_obs = like.z_obs
    std_obs = like.std_obs

    function log_ml_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)

        std = sqrt.(std_obs.^2 .+ std_y.^2)
        return logpdf.(Normal.(μ_y, std), z_obs)
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        μs_y, stds_y = mean_and_std(model_post, X)
        
        # return logpdf.(MvNormal.(eachrow(μs_y), eachrow(stds_y)), Ref(z_obs))
        std_obs_mat = repeat(std_obs, 1, size(stds_y, 2))
        std_mat = sqrt.(std_obs_mat.^2 .+ stds_y.^2)
        y_mat = repeat(z_obs, 1, size(μs_y, 2))
        lls = ((μ, std, y) -> logpdf(Normal(μ, std), y)).(μs_y, std_mat, y_mat)
        return lls
    end
    return log_ml_mean
end

function loglike_marginal(like::NormalLikelihood, y::AbstractVector{<:Real})
    return logpdf.(Normal.(y, like.std_obs), like.z_obs)
end


### A few tests

# p = BosipProblem(...)
# x_dim = ...

# x = rand(x_dim)
# xs = hcat(zeros(x_dim), ones(x_dim), rand(x_dim))

# approx_log_ml = log_approx_marginal_likelihood(p)
# approx_log_l = log_approx_likelihood(p)

# @show approx_log_ml(xs)
# @show approx_log_l(xs)
# @assert sum(approx_log_ml(x)) ≈ approx_log_l(x)
# @assert all(sum.(eachcol(approx_log_ml(xs))) .≈ approx_log_l(xs))

# log_ml_mean = log_marginal_likelihood_mean(p)
# log_l_mean = log_likelihood_mean(p)

# @show log_ml_mean(xs)
# @show log_l_mean(xs)
# @assert sum(log_ml_mean(x)) ≈ log_l_mean(x)
# @assert all(sum.(eachcol(log_ml_mean(xs))) .≈ log_l_mean(xs))
