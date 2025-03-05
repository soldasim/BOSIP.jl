
"""
    TuringOptions(; kwargs...)

Aggregates settings for the `sample_posterior` function, which uses the Turing.jl package.

# Keywords
- `sampler::Any`: The sampling algorithm used to draw the samples.
- `warmup::Int`: The amount of initial unused 'warmup' samples in each chain.
- `samples_in_chain::Int`: The amount of samples used from each chain.
- `chain_count::Int`: The amount of independent chains sampled.
- `leap_size`: Every `leap_size`-th sample is used from each chain. (To avoid correlated samples.)
- `parallel`: If `parallel=true` then the chains are sampled in parallel.

# Sampling Process

In each sampled chain;
  - The first `warmup` samples are discarded.
  - From the following `leap_size * samples_in_chain` samples each `leap_size`-th is kept.
Then the samples from all chains are concatenated and returned.

Total drawn samples:    'chain_count * (warmup + leap_size * samples_in_chain)'
Total returned samples: 'chain_count * samples_in_chain'
"""
abstract type TuringOptions end

"""
    using Turing
    xs = sample_approx_posterior(::BolfiProblem)
    xs = sample_approx_posterior(::BolfiProblem, ::TuringOptions)

Sample from the approximate posterior (`see approx_posterior`).

The `TuringOptions` argument controls the hyperparameters of the sampling.
It is an optional argument and defaults to `TuringOptions()` if not specified.

# See Also

[`sample_posterior_mean`](@ref),
[`sample_posterior`](@ref)
"""
function sample_approx_posterior end

"""
    using Turing
    xs = sample_posterior_mean(::BolfiProblem)
    xs = sample_posterior_mean(::BolfiProblem, ::TuringOptions)

Sample from the expected posterior (see `posterior_mean`).

The `TuringOptions` argument controls the hyperparameters of the sampling.
It is an optional argument and defaults to `TuringOptions()` if not specified.

# See Also

[`sample_approx_posterior`](@ref),
[`sample_posterior`](@ref)
"""
function sample_posterior_mean end

"""
    using Turing
    xs = sample_posterior(logpost, bounds::AbstractBounds, options::TuringOptions)
    xs = sample_posterior(loglike, prior::MultivariateDistribution, options::TuringOptions)

Sample from the learned posterior stored in `problem`.

Either provide the log-posterior (as a function) and the domain bounds.
Or provide the log-likelihood (as a function) and the prior distribution.

The last `options` argument controls the hyperparameters of the sampling.
It is an optional argument and defaults to `TuringOptions()` if not specified.

# See Also

[`sample_approx_posterior`](@ref),
[`sample_posterior_mean`](@ref)
"""
function sample_posterior end

# The sampling is implemented in the `TuringExt` extension.
