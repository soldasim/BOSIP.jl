module TuringExt

using BOLFI
using BOSS
using Turing

"""
Implementation of the abstract `BOLFI.TuringOptions`. See the docs `? BOLFI.TuringOptions`.
"""
@kwdef struct TuringOptions{S} <: BOLFI.TuringOptions
    sampler::S = NUTS(0, 0.65)
    warmup::Int = 1000
    samples_in_chain::Int = 200
    chain_count::Int = 6
    leap_size::Int = 5
    parallel::Bool = true
end

BOLFI.TuringOptions(args...; kwargs...) = TuringOptions(args...; kwargs...)

function BOLFI.sample_approx_posterior(bolfi::BolfiProblem, options::TuringOptions = TuringOptions())
    like = approx_likelihood(bolfi)
    loglike_ = x -> log(like(x))
    return BOLFI.sample_posterior(loglike_, bolfi.x_prior, options)
end
function BOLFI.sample_posterior_mean(bolfi::BolfiProblem, options::TuringOptions = TuringOptions())
    like = likelihood_mean(bolfi)
    loglike_ = x -> log(like(x))
    return BOLFI.sample_posterior(loglike_, bolfi.x_prior, options)
end

function BOLFI.sample_posterior(logpost, bounds::AbstractBounds, options::TuringOptions = TuringOptions())
    model = turing_model(logpost, bounds)
    return sample_posterior_(model, options)
end
function BOLFI.sample_posterior(loglike, prior::MultivariateDistribution, options::TuringOptions = TuringOptions())
    model = turing_model(loglike, prior)
    return sample_posterior_(model, options)
end

@model function turing_model(logpost, bounds::AbstractBounds)
    x ~ product_distribution(Uniform.(bounds...))
    Turing.@addlogprob! logpost(x)
end
@model function turing_model(loglike, x_prior::MultivariateDistribution)
    x ~ x_prior
    Turing.@addlogprob! loglike(x)
end

function sample_posterior_(model, options::TuringOptions = TuringOptions())
    samples_in_chain = options.warmup + (options.leap_size * options.samples_in_chain)
    if options.parallel
        chains = Turing.sample(model, options.sampler, MCMCThreads(), samples_in_chain, options.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(model, options.sampler, samples_in_chain; progress=false), chainscat, 1:options.chain_count)
    end

    # `samples_in_chain` × `x_dim` × `options.chain_count` matrix
    samples = group(chains, :x).value.data
    # skip warmup samples and leaps
    samples = samples[options.warmup+options.leap_size:options.leap_size:end, :, :]
    # concatenate chains
    samples = reduce(vcat, eachslice(samples; dims=3))
    # transpose
    samples = samples' |> collect

    return samples
end

end # module TuringExt
