
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
    # TODO log
    loglike = approx_likelihood(bolfi)
    like = loglike
    # like(x) = exp(loglike(x))

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
    loglike = likelihood_mean(bolfi)
    # TODO log
    like = loglike
    # like(x) = exp(loglike(x))
    return sample_posterior(sampler, like, bolfi.x_prior, count; options)
end

"""
    xs = resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)

Resample `count` samples from the given data set `xs` weighted by the given weights `ws`
with replacement to obtain a new un-weighted data set.

Some data points may repeat in the resampled data set. Increasing the sample size
of the initial data set may help to reduce the number of repetitions.
"""
function resample(xs::AbstractMatrix{<:Real}, ws::AbstractVector{<:Real}, count::Int)
    return hcat(wsample(eachcol(xs), ws, count; replace=true)...)
end
