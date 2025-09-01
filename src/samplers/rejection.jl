
"""
    LogpdfMaximizer(; kwargs...)

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
struct LogpdfMaximizer
    algorithm::Any
    multistart::Int64
    parallel::Bool
    static_schedule::Bool
    kwargs::Base.Pairs{Symbol, <:Any}
end
function LogpdfMaximizer(;
    algorithm,
    multistart,
    parallel = true,
    static_schedule = false,
    kwargs...
)
    return LogpdfMaximizer(algorithm, multistart, parallel, static_schedule, kwargs)
end

"""
    RejectionSampler(; kwargs...)

A sampler that uses trivial rejection sampling to draw samples from the posterior distribution.

# Keywords
- `logpdf_maximizer::LogpdfMaximizer`: The optimizer used to find the maximum logpdf value.
"""
@kwdef struct RejectionSampler <: PureSampler
    logpdf_maximizer::LogpdfMaximizer
end

function sample_posterior(sampler::RejectionSampler, logpost::Function, domain::Domain, count::Int;
    options::BosipOptions = BosipOptions(),    
)
    # TODO: complex domains not supported for now
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)

    # uniform prior
    prior = product_distribution(Uniform.(domain.bounds...))

    # likelihood := posterior
    max_logpost = opt_logpdf(sampler.logpdf_maximizer, logpost, domain)

    return sample_posterior_rej(prior, logpost, max_logpost, count)
end

function sample_posterior(sampler::RejectionSampler, loglike::Function, prior::MultivariateDistribution, domain::Domain, count::Int;
    options::BosipOptions = BosipOptions(),    
)
    # TODO: complex domains not supported for now
    @assert !any(domain.discrete)
    @assert isnothing(domain.cons)
    @assert extrema(prior) == domain.bounds

    max_loglike = opt_logpdf(sampler.logpdf_maximizer, loglike, domain)

    return sample_posterior_rej(prior, loglike, max_loglike, count)
end

function sample_posterior_rej(prior, loglike, max_loglike, count)
    x_dim = length(prior)
    
    prog = Progress(count; desc="Sampling the posterior: ")
    xs = zeros(x_dim, count)
    drawn = 0
    rejected = 0
    
    while drawn < count
        x = rand(prior)
        log_p = loglike(x)
        
        # if max_like * rand() < p
        if max_loglike + log(rand()) < log_p
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

# function _max_like(like::NormalLikelihood, bosip)
#     std_obs = BOSIP._std_obs(like, bosip)
#     z_obs = like.z_obs

#     return pdf(MvNormal(z_obs, std_obs), z_obs)
# end
# function _max_like(like::LogNormalLikelihood, bosip)
#     std_obs = like.std_obs
#     z_obs = like.z_obs

#     return pdf(MvLogNormal(z_obs, std_obs), z_obs)
# end
# function _max_like(like::BinomialLikelihood, bosip)
#     trials = like.trials
#     z_obs = like.z_obs

#     ps = z_obs ./ trials
#     return logpdf.(Binomial.(trials, ps), z_obs) |> sum |> exp
# end

function opt_logpdf(opt::LogpdfMaximizer, loglike::Function, domain::Domain)
    objective = OptimizationFunction((x, _) -> -loglike(x), AutoForwardDiff())
    problem(start) = OptimizationProblem(objective, start, nothing;
        lb = domain.bounds[1],
        ub = domain.bounds[2],
        int = domain.discrete,
    )

    function optim(start)
        x = Optimization.solve(problem(start), opt.algorithm; opt.kwargs...).u
        val = loglike(x)
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
