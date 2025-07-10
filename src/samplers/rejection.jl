
"""
    LikelihoodMaximizer(; kwargs...)

A helper struct for the [`RejectionSampler`](@ref) that defines the optimization algorithm
used to maximize the likelihood function. The `RejectionSampler` uses needs to know the maximum
likelihood value for the rejection process.

# Keywords
- `algorithm::Any`: The optimization algorithm used to maximize the likelihood. 
- `multistart::Int64`: The number of random starting points used in the optimization.
- `parallel::Bool`: If `true`, the optimization is performed in parallel. Defaults to `true`.
- `static_schedule::Bool`: If `static_schedule=true` then the `:static` schedule is used for parallelization.
        This is makes the parallel tasks sticky (non-migrating), but can decrease performance.
- `kwargs::Base.Pairs{Symbol, <:Any}`: Additional keyword arguments passed to the optimization algorithm.
"""
struct LikelihoodMaximizer
    algorithm::Any
    multistart::Int64
    parallel::Bool
    static_schedule::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function LikelihoodMaximizer(;
    algorithm,
    multistart,
    parallel = true,
    static_schedule = false,
    kwargs...
)
    return LikelihoodMaximizer(algorithm, multistart, parallel, static_schedule, kwargs)
end

"""
    RejectionSampler(; kwargs...)

A sampler that uses trivial rejection sampling to draw samples from the posterior distribution.

# Keywords
- `likelihood_maximizer::LikelihoodMaximizer`: The optimizer used to find the maximum likelihood value.
"""
@kwdef struct RejectionSampler <: DistributionSampler
    likelihood_maximizer::LikelihoodMaximizer
end

function sample_posterior(::RejectionSampler, posterior::Function, domain::Domain, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    # TODO: complex domains not supported for now
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)

    # uniform prior
    prior = Uniform.(domain.bounds...)

    # likelihood := posterior
    max_post = opt_likelihood(sampler.likelihood_maximizer, posterior, domain)

    return sample_posterior_rej(prior, posterior, max_post, count)
end

function sample_posterior(sampler::RejectionSampler, likelihood::Function, prior::MultivariateDistribution, domain::Domain, count::Int;
    options::BolfiOptions = BolfiOptions(),    
)
    # TODO: complex domains not supported for now
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)
    @assert extrema(prior) == domain.bounds

    # max_like = _max_like(bolfi.likelihood, bolfi)
    max_like = opt_likelihood(sampler.likelihood_maximizer, likelihood, domain)

    return sample_posterior_rej(prior, likelihood, max_like, count)
end

function sample_posterior_rej(prior, likelihood, max_like, count)
    x_dim = length(prior)
    
    prog = Progress(count; desc="Sampling the posterior: ")
    xs = zeros(x_dim, count)
    drawn = 0
    rejected = 0
    
    while drawn < count
        x = rand(prior)
        p = likelihood(x)
        if max_like * rand() < p
            next!(prog)
            drawn += 1
            xs[:, drawn] .= x
        else
            rejected += 1
        end
    end

    @info "Rejection rate: $(rejected / (drawn + rejected))"
    ws = fill(1 / size(xs, 2), size(xs, 2))
    return xs, ws
end

# function _max_like(like::NormalLikelihood, bolfi)
#     std_obs = BOLFI._std_obs(like, bolfi)
#     z_obs = like.z_obs

#     return pdf(MvNormal(z_obs, std_obs), z_obs)
# end
# function _max_like(like::LogNormalLikelihood, bolfi)
#     std_obs = like.std_obs
#     z_obs = like.z_obs

#     return pdf(MvLogNormal(z_obs, std_obs), z_obs)
# end
# function _max_like(like::BinomialLikelihood, bolfi)
#     trials = like.trials
#     z_obs = like.z_obs

#     ps = z_obs ./ trials
#     return logpdf.(Binomial.(trials, ps), z_obs) |> sum |> exp
# end

function opt_likelihood(opt::LikelihoodMaximizer, likelihood::Function, domain::Domain)
    objective = OptimizationFunction((x, _) -> -likelihood(x), AutoForwardDiff())
    problem(start) = OptimizationProblem(objective, start, nothing;
        lb = domain.bounds[1],
        ub = domain.bounds[2],
        int = domain.discrete,
    )

    function optim(start)
        x = Optimization.solve(problem(start), opt.algorithm; opt.kwargs...).u
        val = likelihood(x)
        return x, val
    end

    starts = BOSS.generate_LHC(domain.bounds, opt.multistart)
    best_x, best_val = BOSS.optimize_multistart(
        optim,
        starts;
        opt.parallel,
        opt.static_schedule,
    )

    return best_val
end
