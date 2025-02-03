
"""
Represents the assumed likelihood of the experiment observation ``y_o``.

# Defining a Custom Likelihood

To define a custom likelihood, create a new struct `CustomLike <: Likelihood` and implement the following methods;

- `approx_likelihood(::CustomLike, gp_post)`: Necessary to be able to use `approx_posterior`.
- `likelihood_mean(::CustomLike, gp_post)`: Necessary to be able to use `posterior_mean`.
- `sq_likelihood_mean(::CustomLike, gp_post)`: Necessary to be able to use `posterior_variance`.
- `get_subset(::CustomLike, y_set::AsbtractVector{<:Bool})`: Necessary for `BolfiProblem`s where `!isnothing(problem.y_sets)`.
"""
abstract type Likelihood end

"""
    approx_likelihood(::Likelihood, gp_post)

Returns a function mapping ``x`` to ``\\hat{p}(y_o|x)``.
"""
function approx_likelihood(::Likelihood, gp_post) end

"""
    likelihood_mean(::Likelihood, gp_post)

Returns a function mapping ``x`` to ``\\mathbb{E}[ \\hat{p}(y_o|x) | GP ]``.
"""
function likelihood_mean(::Likelihood, gp_post) end

"""
    sq_likelihood_mean(::Likelihood, gp_post)

Returns a function mapping ``x`` to ``\\mathbb{E}[ \\hat{p}(y_o|x)^2 | GP ]``.
"""
function sq_likelihood_mean(::Likelihood, gp_post) end

function get_subset(::Likelihood, y_set::AbstractVector{<:Bool}) end
