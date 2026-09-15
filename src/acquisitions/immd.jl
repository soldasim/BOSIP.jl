
"""
    IMMD(; kwargs...)

Selects the next evaluation point by maximizing the Integrated MMD (IMMD),
where MMD stands for maximum mean discrepancy.

The acquisition is (loosely) based on information gain. Ideally, we would select the point
that maximizes the mutual information between the *new data point* (a vector-valued random
variable given by the GP predictive distributions) and the *posterior approximation*
(a function-valued random variable from an infinite-dimensional distribution). This ideal
quantity is intractable, so it is replaced by a sequence of approximations:

1. **Function ``\\to`` point, integrated over the domain.** Mutual information of a
    function-valued variable is infeasible, so we instead consider the mutual information
    between the new data point and the posterior density value ``s(x)`` at a single point `x`,
    and integrate it over `x`.
2. **Monte Carlo integration.** The integral over the parameter domain is still infeasible,
    so it is approximated by (importance-weighted) Monte Carlo integration, using samples
    drawn from `x_proposal`.
3. **MMD instead of KLD.** Mutual information equals the Kullback-Leibler divergence between
    the joint and the product of marginals of the two variables. We replace the KLD by the MMD,
    as the latter can be readily estimated from samples.
4. **HSIC instead of MMD.** The MMD between the joint and the product of marginals is exactly
    the Hilbert-Schmidt independence criterion (HSIC) of the two variables, which we estimate
    directly from paired samples.

In conclusion, instead of the mutual information of the new data point and the posterior pdf,
we compute the HSIC between the new data point and the posterior density value ``s(x)``,
integrated over `x`.

The kernel bandwidths are set so that each kernel discriminates at the relevant scale:
- The `y_kernel` bandwidth is the weighted median of the GP predictive standard deviation over the
    domain samples (per output dimension / ARD) — the typical spread of the new data point.
- The `p_kernel` bandwidth is the maximum unnormalized posterior value over the domain, found by a
    multistart local optimization of `log_posterior_mean` (via `posterior_maximizer`). The same
    `log_posterior_mean` underlies the posterior-density samples used in the HSIC, and its maximum is
    reused as the numerical shift applied to those samples, so the scaling is consistent.

# Kwargs
- `y_samples::Int64`: The amount of samples drawn from the joint and marginal distributions
        to estimate the HSIC value.
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `x_proposal::MultivariateDistribution`: This distribution is used to sample
        parameter samples used to numerically approximate the integral over the parameter domain.
- `posterior_maximizer::OptimizationAM`: The optimizer used to find the maximum (log) mean posterior
        value over the domain (the `p_kernel` bandwidth reference) via multistart local optimization.
        The domain samples are added to its starting points. Reusing the same `OptimizationAM`
        passed to `bosip!` as the `acq_maximizer` is a reasonable default. (Note that `log_posterior_mean`
        is much cheaper to evaluate than the acquisition itself, so a lighter optimizer may suffice.)
- `y_kernel::Kernel`: The kernel used for the samples of the new data point.
- `p_kernel::Kernel`: The kernel used for the posterior function value samples.
"""
@kwdef struct IMMD <: BosipAcquisition
    y_samples::Int64
    x_samples::Int64
    x_proposal::MultivariateDistribution
    posterior_maximizer::OptimizationAM
    y_kernel::BOSS.Kernel = BOSS.GaussianKernel()
    p_kernel::BOSS.Kernel = BOSS.GaussianKernel()
end

# info gain on the posterior approximation
function (acq::IMMD)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, options::BosipOptions)
    y_dim = BOSS.y_dim(bosip.problem)

    # Sample parameter values (integration points for the MC integral over the domain).
    xs = rand(acq.x_proposal, acq.x_samples)

    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.(log_ws .- logsumexp(log_ws)) # normalize to sum up to 1

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.y_samples) # vector-vector
    Es_s = [sample_ϵs(y_dim, acq.y_samples) for _ in 1:acq.x_samples] # vector-vector-vector

    # precalculate model posterior
    model_post = BOSS.model_posterior(bosip.problem)

    # `y_kernel` bandwidth: weighted median of the GP predictive std over the domain samples (per-dimension/ARD).
    σy = estimate_y_bandwidth(model_post, xs, ws; info=options.info)

    # `p_kernel` bandwidth: the maximum (log) mean posterior value over the domain, found by multistart
    # local optimization. Reused as `s_shift` (the numerical shift for the HSIC posterior-density samples),
    # so the samples are measured relative to their domain maximum -- a consistent scaling.
    s_shift = maximize_log_posterior_mean(bosip, acq.posterior_maximizer, xs, options)

    return IMMDFunc(acq, bosip, model_post, xs, ws, ϵs_y, Es_s, σy, s_shift)
end

function sample_ϵs(y_dim, y_samples)
    d = MvNormal(zeros(y_dim), ones(y_dim))
    ϵs = [rand(d) for _ in 1:y_samples]
    return ϵs
end

struct IMMDFunc{
    B<:BosipProblem,
    M<:ModelPosterior,
    X<:AbstractMatrix{<:Real},
    W<:AbstractVector{<:Real},
    E1<:AbstractVector{<:AbstractVector{<:Real}},
    E2<:AbstractVector{<:AbstractVector{<:AbstractVector{<:Real}}},
    LY<:AbstractVector{<:Real},
    SH<:Real,
}
    acq::IMMD
    bosip::B
    model_post::M
    xs::X
    ws::W
    ϵs_y::E1
    Es_s::E2
    σy::LY
    s_shift::SH
end

function (f::IMMDFunc)(x_::AbstractVector{<:Real})
    # sample `N` y_ samples at the new x_
    μy, std_y = mean_and_std(f.model_post, x_)
    ys_ = calc_y.(Ref(μy), Ref(std_y), f.ϵs_y)

    # augment problems with the speculative observations
    problem_copies = [deepcopy(f.bosip.problem) for _ in 1:f.acq.y_samples]
    for (p, y_) in zip(problem_copies, ys_)
        augment_dataset!(p, x_, y_)
    end
    aug_posts_samples = model_posterior.(problem_copies)

    # sample `K x N` y_eval (and s_eval) samples (N for each x_eval sample)
    Y_evals = get_ys_eval.(Ref(aug_posts_samples), eachcol(f.xs), f.Es_s) # vector-vector-vector
    log_S_evals = get_log_ss_eval.(Y_evals, eachcol(f.xs), Ref(f.bosip.likelihood), Ref(f.bosip.x_prior)) # vector-vector

    # S_evals are UNNORMALIZED! (This is intentional, as there is no straight-forward way to normalize them.)

    # shift by `s_shift` (the domain maximum) and exponentiate -> S measured relative to its domain maximum
    shift_log_S_evals!(log_S_evals; shift=f.s_shift)
    S_evals = exponentiate_S_evals!(log_S_evals) # in-place exponentiation

    # calculate `K` HSICs between `y_` and `s_1,...,s_K`, integrated over the domain
    y_kernel = BOSS.with_lengthscale(f.acq.y_kernel, f.σy)
    p_kernel = f.acq.p_kernel
    vals = hsic.(Ref(y_kernel), Ref(p_kernel), Ref(ys_), S_evals)

    return sum(f.ws .* vals)
end

function calc_y(μ, σ, ϵ)
    return μ .+ (σ .* ϵ)
end

function get_ys_eval(aug_posts, x_eval, ϵs)
    pred_distrs = [mean_and_std(aug_posts[i], x_eval) for i in eachindex(aug_posts)]
    ys_eval = [calc_y(pred_distrs[i]..., ϵs[i]) for i in eachindex(pred_distrs)]
    return ys_eval
end

function get_log_ss_eval(ys_eval, x_eval, likelihood, x_prior)
    log_ls = BOSIP.loglike.(Ref(likelihood), ys_eval, Ref(x_eval))
    log_px = logpdf(x_prior, x_eval)
    ss = log_ls .+ log_px # TODO unnormalized
    return ss
end

# (Biased) HSIC estimator: (1/n²) tr(C Kx C Ky) = (1/n²) ⟨K̃x, K̃y⟩_F with the centered Gram matrices.
function hsic(kernel_X, kernel_Y, X::AbstractVector, Y::AbstractVector)
    n = length(X)
    K̃x = _double_center(BOSS.kernelmatrix(kernel_X, X))
    K̃y = _double_center(BOSS.kernelmatrix(kernel_Y, Y))
    return sum(K̃x .* K̃y) / n^2
end

# Double-center a Gram matrix: K̃ = C K C with C = I - (1/n) 11ᵀ, computed in O(n²).
function _double_center(K::AbstractMatrix)
    return K .- mean(K; dims=1) .- mean(K; dims=2) .+ mean(K)
end

# Shift the log posterior-density values by `shift` (for numerical stability) before exponentiation.
function shift_log_S_evals!(log_S_evals; shift)
    for i in eachindex(log_S_evals)
        for j in eachindex(log_S_evals[i])
            log_S_evals[i][j] -= shift
        end
    end
    return log_S_evals
end

function exponentiate_S_evals!(log_S_evals)
    for ix in eachindex(log_S_evals)
        log_S_evals[ix] = exp.(log_S_evals[ix])
    end
    return log_S_evals # S_evals
end

# Estimate the per-dimension (ARD) `y_kernel` bandwidth as the weighted median of the GP predictive std
# over the domain samples. The weights `ws` correct for the non-uniform sampling of `xs`; the median
# (rather than the mean) gives a robust central scale, unaffected by the high-variance unexplored tails.
function estimate_y_bandwidth(model_post, xs, ws; info::Bool)
    stds = reduce(hcat, std.(Ref(model_post), eachcol(xs))) # y_dim × n_x: rows are output dimensions
    return _safe_bandwidth.(weighted_median.(eachrow(stds), Ref(ws)); info)
end

# Lower weighted median: smallest value whose cumulative (sorted) weight reaches half the total weight.
function weighted_median(values::AbstractVector{<:Real}, weights::AbstractVector{<:Real})
    perm = sortperm(values)
    cw = cumsum(weights[perm])
    i = findfirst(>=(cw[end] / 2), cw)
    return values[perm[i]]
end

# Guard against a degenerate (zero / non-finite) bandwidth that would break the kernel.
function _safe_bandwidth(bw; info::Bool)
    (isfinite(bw) && (bw > 0)) && return bw
    info && @warn "Degenerate HSIC kernel bandwidth ($bw); falling back to a unit bandwidth."
    return one(bw)
end

# Maximum (log) mean posterior value over the domain, via multistart local optimization.
# The domain samples `xs` are appended to the optimizer's starting points.
function maximize_log_posterior_mean(bosip, maximizer::OptimizationAM, xs, options::BosipOptions)
    domain = bosip.problem.domain
    log_post_mean = log_posterior_mean(bosip)
    obj(x) = sum(log_post_mean(x)) # ensure a scalar objective

    cons_func = isnothing(domain.cons) ? nothing : (res, x, _) -> (res .= domain.cons(x))
    starts = hcat(BOSS.get_starts(maximizer.multistart, domain), xs)

    _, max_log_post = BOSS.optimize(
        maximizer,
        obj,
        cons_func,
        domain.bounds[1],
        domain.bounds[2],
        domain.discrete,
        BOSS.cons_dim(domain),
        starts,
        create_boss_options(options, bosip),
    )
    return max_log_post
end
