module TuringExt

using BOSIP
using BOSS
using Turing
using Random

"""
Implementation of the abstract `BOSIP.TuringSampler`. See the docs `? BOSIP.TuringSampler`.
"""
@kwdef struct TuringSampler{S} <: BOSIP.TuringSampler
    sampler::S = NUTS(0, 0.65)
    warmup::Int = 1000
    chain_count::Int = 6
    leap_size::Int = 5
    parallel::Bool = true
end

BOSIP.TuringSampler(args...; kwargs...) = TuringSampler(args...; kwargs...)

function BOSIP.sample_posterior(sampler::TuringSampler, logpost::Function, domain::Domain, count::Int; kwargs...)
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)
    
    model = turing_model(logpost, domain.bounds)
    return _sample_posterior_turing(model, sampler, count)
end
function BOSIP.sample_posterior(sampler::TuringSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int; kwargs...)
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)
    @assert extrema(prior) == domain.bounds
    
    model = turing_model(loglike, prior)
    return _sample_posterior_turing(model, sampler, count)
end

@model function turing_model(logpost, bounds::AbstractBounds)
    x ~ product_distribution(Uniform.(bounds...))
    Turing.@addlogprob! logpost(x)
end
@model function turing_model(loglike, x_prior::MultivariateDistribution)
    x ~ x_prior
    Turing.@addlogprob! loglike(x)
end

function _sample_posterior_turing(model, sampler::TuringSampler, count::Int)
    # This count will possibly result in a few extra samples. They are discarded later.
    count_per_chain = (count / sampler.chain_count) |> ceil |> Int
    samples_in_chain = sampler.warmup + (sampler.leap_size * count_per_chain)

    if sampler.parallel
        chains = Turing.sample(model, sampler.sampler, MCMCThreads(), samples_in_chain, sampler.chain_count; progress=false)
    else
        chains = mapreduce(_ -> Turing.sample(model, sampler.sampler, samples_in_chain; progress=false), chainscat, 1:sampler.chain_count)
    end

    # `samples_in_chain` × `x_dim` × `options.chain_count` matrix
    samples = group(chains, :x).value.data
    # skip warmup samples and leaps
    samples = samples[sampler.warmup+sampler.leap_size:sampler.leap_size:end, :, :]
    # concatenate chains
    samples = reduce(vcat, eachslice(samples; dims=3))
    # transpose
    samples = samples'

    # shuffle & discard extra samples
    keep = randperm(size(samples, 2))[1:count]
    samples = samples[:, keep]

    ws = fill(1 / size(samples, 2), size(samples, 2))
    return samples, ws
end

end # module TuringExt
