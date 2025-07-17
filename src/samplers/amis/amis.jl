
include("proposal_distributions/abstract_distribution.jl")
include("proposal_distributions/normal.jl")

include("distribution_fitters/abstract_fitter.jl")
include("distribution_fitters/analytical.jl")
include("distribution_fitters/optimization.jl")

include("amis_method.jl")

"""
    AMIS(; kwargs...)

Adaptive Metropolis Importance Sampling (AMIS) sampler for posterior distributions.

The sampler first aproximates the posterior distribution by a Laplace approximation
centered on the maximum of the posterior, or with a Gaussian mixture model,
and draws samples from it in the 0th iteration.

Afterwards, the AMIS algorithm is run for `iters` iterations with a simple Gaussian proposal distribution
re-fitted in each iteration.

# Keywords
- `iters::Int`: Number of iterations of the AMIS algorithm.
- `proposal_fitter::DistributionFitter`: The algorithm used to re-fit the proposal distribution
        in each iteration. Defaults to the `AnalyticalFitter`.
- `gauss_mix_options::Union{Nothing, GaussMixOptions}`: Options for the Gaussian mixture approximation
        used for the 0th iteration. Defaults to `nothing`, which means the Laplace approximation is used instead.
"""
@kwdef struct AMISSampler <: WeightedSampler
    iters::Int
    proposal_fitter::DistributionFitter = AnalyticalFitter()
    gauss_mix_options::GaussMixOptions
end

function sample_posterior(sampler::AMISSampler, post::Function, domain::Domain, count::Int;
    options::BolfiOptions = BolfiOptions(),
)
    iters = sampler.iters
    samples_total = iters * count # more samples than needed to allow for efficient down-sampling
    samples_per_iter = samples_total / iters |> ceil |> Int
    (samples_per_iter < 50) && @warn "AMIS: Low sample count ($samples_per_iter) per iteration!"

    logpost = x -> log(post(x))

    # Initialize the proposal distribution
    x_dim_ = x_dim(domain)
    q = NormalProposal(MvNormal(zeros(x_dim_), ones(x_dim_)))
    init_q = approx_by_gauss_mix(logpost, domain, sampler.gauss_mix_options)

    amis = AMIS(;
        T = iters,
        N = samples_per_iter,
        init_q,
    )
    xs, ws = amis(logpost, q, sampler.proposal_fitter; options)
   
    return xs, ws
end
