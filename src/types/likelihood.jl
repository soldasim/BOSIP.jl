
"""
Represents the assumed likelihood of the experiment observation ``z_o``.

# Defining a Custom Likelihood

To define a custom likelihood, create a new subtype of `Likelihood`
and implement the following API;

Each subtype of `Likelihood` *should* implement:
- `loglike(::Likelihood, y::AbstractVector{<:Real})` where `y` is the simulator output
- `log_approx_likelihood(::Likelihood, ::BolfiProblem, ::ModelPosterior)`
- `log_likelihood_mean(::Likelihood, ::BolfiProblem, ::ModelPosterior)`

Each subtype of `Likelihood` *should* implement *at least one* of:
- `log_sq_likelihood_mean(::Likelihood, ::BolfiProblem, ::ModelPosterior)`
- `log_likelihood_variance(::Likelihood, ::BolfiProblem, ::ModelPosterior)`

Additionally, the following method is also necessary to implement
if `BolfiProblem` where `!isnothing(problem.y_sets)` is used:
- `get_subset(::Likelihood, y_set::AsbtractVector{<:Bool})`:

The following additional methods are provided by default and *need not be implemented*:
- `like(::Likelihood, y:AbstractVector{<:Real})`
"""
abstract type Likelihood end

"""
    loglike(::Likelihood, y::AbstractVector{<:Real})

Return the log-likelihood of the observation given the simulator output `δ`.
"""
function loglike end

"""
    like(::Likelihood, y::AbstractVector{<:Real})

Return the likelihood of the observation given the model output `δ`.
"""
function like(l::Likelihood, y::AbstractVector{<:Real})
    return exp(loglike(l, y))
end

"""
    log_approx_likelihood(::Likelihood, ::BolfiProblem, ::ModelPosterior)

Returns a function mapping ``x`` to ``log \\hat{p}(z_o|x)``.
"""
function log_approx_likelihood end

"""
    log_likelihood_mean(::Likelihood, ::BolfiProblem, ::ModelPosterior)

Returns a function mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x) | GP ]``.
"""
function log_likelihood_mean end

"""
    log_sq_likelihood_mean(::Likelihood, ::BolfiProblem, ::ModelPosterior)

Returns a function mapping ``x`` to ``log \\mathbb{E}[ \\hat{p}(z_o|x)^2 | GP ]``.
"""
function log_sq_likelihood_mean end

"""
    log_likelihood_variance(::Likelihood, ::BolfiProblem, ::ModelPosterior)

Return a function mapping ``x`` to ``log \\mathbb{V}[ \\hat{p}(z_o|x) | GP ]``.
"""
function log_likelihood_variance end

"""
    get_subset(::Likelihood, y_set::AbstractVector{<:Bool}) -> ::Likelihood

Construct the likelihood of the observation dimensions specified by `y_set`.
"""
function get_subset end
