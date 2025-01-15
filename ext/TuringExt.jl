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

function BOLFI.sample_posterior(bolfi::BolfiProblem, options::TuringOptions = TuringOptions())
    (bolfi.problem.data isa ExperimentDataBI) && @warn """
        Calling `sample_posterior` with BI model fitter. Sampling from the averaged posterior.
        You may want to fit the model via some MAP model fitter and call then call `sample_posterior` again.
    """

    approx_like = approx_likelihood(bolfi)
    model = turing_model(bolfi.x_prior, approx_like)

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

@model function turing_model(x_prior, approx_likelihood)
    x ~ x_prior
    Turing.@addlogprob! log(approx_likelihood(x))
end

end # module TuringExt
