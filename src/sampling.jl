
"""
    xs, ws = sample_approx_posterior(bosip::BosipProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the approximate posterior of the `BosipProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.

# See Also

[`sample_expected_posterior`](@ref),
`sample_posterior` and `resample` (see DistributionSamplers.jl)
"""
function sample_approx_posterior(bosip::BosipProblem, sampler::DistributionSampler, count::Int;
    options::BosipOptions = BosipOptions(),
)
    loglike = log_approx_likelihood(bosip)
    return sample_posterior(sampler, loglike, bosip.x_prior, bosip.problem.domain, count; options)
end

"""
    xs, ws = sample_approx_posterior(bosip::BosipProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the expected posterior (i.e. the posterior mean) of the `BosipProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.

# See Also

[`sample_approx_posterior`](@ref),
`sample_posterior` and `resample` (see DistributionSamplers.jl)
"""
function sample_expected_posterior(bosip::BosipProblem, sampler::DistributionSampler, count::Int;
    options::BosipOptions = BosipOptions(),
)
    loglike = log_likelihood_mean(bosip)
    return sample_posterior(sampler, loglike, bosip.x_prior, bosip.problem.domain, count; options)
end

"""
Return the box bounds of the `Domain`. The samplers support only box-constrained domains.
"""
function _box_bounds(domain::Domain)
    @assert !any(domain.discrete) "Sampling from domains with discrete variables is not supported."
    @assert isnothing(domain.cons) "Sampling from domains with nonlinear constraints is not supported."
    return domain.bounds
end

"""
    xs, ws = sample_posterior(sampler::DistributionSampler, logpost::Function, domain::Domain, count::Int; kwargs...)
    xs, ws = sample_posterior(sampler::DistributionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...)

Sample `count` samples (columns of `xs`) with weights `ws` from the given (possibly unnormalized) log-posterior,
or from `loglike + logpdf(prior, x)`.

Wraps `DistributionSamplers.sample_posterior` for BOSIP's `Domain`.
Only box-constrained domains are supported.

# Keywords
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.
"""
function sample_posterior(sampler::DistributionSampler, logpost::Function, domain::Domain, count::Int;
    options::BosipOptions = BosipOptions(),
)
    return DistributionSamplers.sample_posterior(sampler, logpost, _box_bounds(domain), count; info=options.info, debug=options.debug)
end
function sample_posterior(sampler::DistributionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int;
    options::BosipOptions = BosipOptions(),
)
    return DistributionSamplers.sample_posterior(sampler, loglike, prior, _box_bounds(domain), count; info=options.info, debug=options.debug)
end

"""
    xs = sample_posterior_pure(sampler::DistributionSampler, logpost::Function, domain::Domain, count::Int; kwargs...)

Sample `count` samples from the posterior distribution defined by the `logpost` log-density.
Assures that the returned samples are "pure" (unweighted).

In case of a `WeightedSampler`, `supersample_ratio` × `count` samples are drawn,
and subsequently down-sampled to `count` samples according to their weights.

# Keywords
- `supersample_ratio::Int = 20`
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.
"""
function sample_posterior_pure(sampler::DistributionSampler, logpost::Function, domain::Domain, count::Int;
    supersample_ratio = 20,
    options::BosipOptions = BosipOptions(),
)
    return DistributionSamplers.sample_posterior_pure(sampler, logpost, _box_bounds(domain), count; supersample_ratio, info=options.info, debug=options.debug)
end

"""
    gauss_mix = approx_by_gauss_mix(logpost, domain::Domain, opt::GaussMixOptions; kwargs...)

Approximate the given (possibly unnormalized) log-density by a Gaussian mixture.
Wraps `DistributionSamplers.approx_by_gauss_mix` for BOSIP's `Domain`.

# Keywords
- `options::BosipOptions`: Miscellaneous preferences. Defaults to `BosipOptions()`.
"""
function approx_by_gauss_mix(logpost, domain::Domain, opt::GaussMixOptions;
    options::BosipOptions = BosipOptions(),
)
    return DistributionSamplers.approx_by_gauss_mix(logpost, _box_bounds(domain), opt; info=options.info, debug=options.debug)
end
