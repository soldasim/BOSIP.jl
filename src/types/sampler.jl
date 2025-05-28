
"""
Subtypes of `DistributionSampler` are used to sample from a probability distribution.

Each subtype of `DistributionSampler` *should* implement:
- `sample_posterior(::DistributionSampler, posterior::Function, domain::Domain, count::Int; kwargs...) -> AbstractMatrix{<:Real}`

Each subtype of `DistributionSampler` *may* additionally implement:
- `sample_posterior(::DistributionSampler, likelihood::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...) -> AbstractMatrix{<:Real}`
"""
abstract type DistributionSampler end

"""
    sample_posterior(::DistributionSampler, posterior::Function, domain::Domain, count::Int; kwargs...) -> AbstractMatrix{<:Real}
    sample_posterior(::DistributionSampler, likelihood::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...) -> AbstractMatrix{<:Real}

Sample `count` samples from the given posterior density function.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.
"""
function sample_posterior end

# default implementation for `DistributionSampler`s not implementing the second method
function sample_posterior(sampler::DistributionSampler, likelihood::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...)
    function posterior(x)
        loglike = log(likelihood(x))
        logprior = logpdf(prior, x)
        return exp(loglike + logprior)
    end

    return sample_posterior(sampler, posterior, domain, count; kwargs...)
end
