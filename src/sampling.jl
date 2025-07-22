
"""
    xs, ws = sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the approximate posterior of the `BolfiProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.

# See Also

[`sample_expected_posterior`](@ref),
[`sample_posterior`](@ref),
[`resample`](@ref)
"""
function sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    like = approx_likelihood(bolfi)
    return sample_posterior(sampler, like, bolfi.x_prior, count; options)
end

"""
    xs, ws = sample_approx_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int; kwargs...)

Sample `count` samples from the expected posterior (i.e. the posterior mean) of the `BolfiProblem`
using the specified `sampler`. Return a column-wise matrix of the drawn samples.

# Keywords
- `options::BolfiOptions`: Miscellaneous preferences. Defaults to `BolfiOptions()`.

# See Also

[`sample_approx_posterior`](@ref),
[`sample_posterior`](@ref),
[`resample`](@ref)
"""
function sample_expected_posterior(bolfi::BolfiProblem, sampler::DistributionSampler, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    like = likelihood_mean(bolfi)
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

"""
    xs = pure_sample_posterior(sampler::PureSampler, posterior::Function, domain::Domain, count::Int;
        supersample_ratio = 20,
    )

Sample `count` samples from the posterior distribution defined by the `posterior` pdf.
Assures that the returned samples are "pure" (unweighted).

In case of a `WeightedSampler`, `supersample_ratio` × `count` samples are drawn,
and subsequently down-sampled to `count` samples according to their weights.
"""
function pure_sample_posterior(sampler::PureSampler, posterior::Function, domain::Domain, count::Int;
    supersample_ratio = 20,
)
    xs, ws = sample_posterior(sampler, posterior, domain, count)
    @assert allequal(ws)
    return xs
end
function pure_sample_posterior(sampler::WeightedSampler, posterior::Function, domain::Domain, count::Int;
    supersample_ratio = 20,
)
    xs, ws = sample_posterior(sampler, posterior, domain, supersample_ratio * count)
    xs = resample(xs, ws, count)
    return xs
end
