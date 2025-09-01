
"""
    DistributionSampler

Subtypes of `DistributionSampler` are used to sample from a probability distribution.

Each subtype of `DistributionSampler` *should* implement:
- `sample_posterior(::DistributionSampler, logpost::Function, domain::Domain, count::Int; kwargs...) -> (X, ws)`

Each subtype of `DistributionSampler` *may* additionally implement:
- `sample_posterior(::DistributionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...) -> (X, ws)`

See also: [`PureSampler`](@ref), [`WeightedSampler`](@ref)
"""
abstract type DistributionSampler end

"""
    PureSampler <: DistributionSampler

A `DistributionSampler` which samples directly from the provided pdf,
and always returns samples with uniform weights.
"""
abstract type PureSampler <: DistributionSampler end

"""
    WeightedSampler <: DistributionSampler

A `DistributionSampler` which does not sample directly from the pdf,
but instead returns samples with non-uniform weights correcting for the sampling bias.
"""
abstract type WeightedSampler <: DistributionSampler end

"""
    sample_posterior(::DistributionSampler, logpost::Function, domain::Domain, count::Int; kwargs...)
    sample_posterior(::DistributionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...)

Sample `count` samples from the given posterior log-density function.

# Keywords
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.
"""
function sample_posterior end

# default implementation for `DistributionSampler`s not implementing the second method
function sample_posterior(sampler::DistributionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...)
    function logpost(x)
        lp = logpdf(prior, x)
        ll = loglike(x)
        return ll + lp
    end

    return sample_posterior(sampler, logpost, domain, count; kwargs...)
end
