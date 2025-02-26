
"""
Represents the assumed likelihood of the experiment observation ``y_o``.

# Defining a Custom Likelihood

To define a custom likelihood, create a new struct `CustomLike <: Likelihood` and implement the following methods;

- `loglike(::CustomLike, z::AbstractVector{<:Real})` where `z` is the simulator output
- `approx_likelihood(::CustomLike, bolfi, gp_post)`: Necessary to be able to use `approx_posterior`.
- `likelihood_mean(::CustomLike, bolfi, gp_post)`: Necessary to be able to use `posterior_mean`.
- `sq_likelihood_mean(::CustomLike, bolfi, gp_post)`: Necessary to be able to use `posterior_variance`.
- `get_subset(::CustomLike, y_set::AsbtractVector{<:Bool})`: Necessary for `BolfiProblem`s where `!isnothing(problem.y_sets)`.
"""
abstract type Likelihood end

"""
    loglike(::Likelihood, z::AbstractVector{<:Real})

Return the log-likelihood of the observation given the simulator output `z`.
"""
function loglike(::Likelihood, z::AbstractVector{<:Real}) end

"""
    pdf(::Likelihood, z::AbstractVector{<:Real})

Return the likelihood of the observation given the simulator output `z`.
"""
function like(l::Likelihood, z::AbstractVector{<:Real})
    return exp(loglike(l, z))
end

"""
    approx_likelihood(::Likelihood, bolfi, gp_post)

Returns a function mapping ``x`` to ``\\hat{p}(y_o|x)``.
"""
function approx_likelihood(::Likelihood, bolfi, gp_post) end

"""
    likelihood_mean(::Likelihood, bolfi, gp_post)

Returns a function mapping ``x`` to ``\\mathbb{E}[ \\hat{p}(y_o|x) | GP ]``.
"""
function likelihood_mean(::Likelihood, bolfi, gp_post) end

"""
    sq_likelihood_mean(::Likelihood, bolfi, gp_post)

Returns a function mapping ``x`` to ``\\mathbb{E}[ \\hat{p}(y_o|x)^2 | GP ]``.
"""
function sq_likelihood_mean(::Likelihood, bolfi, gp_post) end

function get_subset(::Likelihood, y_set::AbstractVector{<:Bool}) end
