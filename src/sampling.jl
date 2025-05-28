
"""
    xs = sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the approximate posterior of the `BolfiProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.

# See Also

[`sample_expected_posterior`](@ref),
[`sample_posterior`](@ref)
"""
function sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    like = approx_likelihood(bolfi)
    return sample_posterior(sampler, like, bolfi.x_prior, count; options)
end

"""
    xs = sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the expected posterior (i.e. the posterior mean) of the `BolfiProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.

# See Also

[`sample_approx_posterior`](@ref),
[`sample_posterior`](@ref)
"""
function sample_expected_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    like = likelihood_mean(bolfi)
    return sample_posterior(sampler, like, bolfi.x_prior, count; options)
end

