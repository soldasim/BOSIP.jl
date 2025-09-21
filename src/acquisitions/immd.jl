
"""
    IMMD(; kwargs...)

Selects new data point by maximizing the Integrated MMD (IMMD),
where MMD stands for maximum mean discrepancy.
    
This acquisition function is (loosely) based on information gain.
Ideally, we would like to calculate the mutual information between
the new data point (a vector-valued random variable from a multivariate distribution given by the GPs)
and the posterior approximation (a "random function" from a infinite-dimensional distribution).

Calculating mutual information of an infinite-dimensional variable is infeasible.
Thus, we calculate the mutual information of the new data point and the posterior
probability value at a single point `x`, integrated over `x`.
This integral is still infeasible, but can be approximated by Monte Carlo integration.

Mutual information is calculated as the Kullback-Leibler divergence (KLD)
of the joint and marginal distributions of the two variables.
Instead of the KLD distance, we use the MMD distance,
as it can be readily estimated from samples.
Finally, instead of the MMD between the joing and marginal distributions,
we can calculate the HSIC (Hilbert-Schmidt independence criterion) of the two variables.

In conclusion, instead of the mutual information of the new data point
(vector-valued random variable) and the posterior pdf (a function-valued random variable),
we calculate the HSIC between the new data point and some point `x` on the domain,
integrated over `x`.

# Kwargs
- `y_samples::Int64`: The amount of samples drawn from the joint and marginal distributions
        to estimate the HSIC value.
- `x_samples::Int64`: The amount of samples used to approximate the integral
        over the parameter domain.
- `x_proposal::MultivariateDistribution`: This distribution is used to sample
        parameter samples used to numerically approximate the integral over the parameter domain.
- `y_kernel::Kernel`: The kernel used for the samples of the new data point.
- `p_kernel::Kernel`: The kernel used for the posterior function value samples.
"""
@kwdef struct IMMD <: BosipAcquisition
    y_samples::Int64
    x_samples::Int64
    x_proposal::MultivariateDistribution
    y_kernel::BOSS.Kernel = BOSS.GaussianKernel()
    p_kernel::BOSS.Kernel = BOSS.GaussianKernel()
end

# info gain on the posterior approximation
function (acq::IMMD)(bosip::BosipProblem{Nothing}, options::BosipOptions)
    problem = bosip.problem
    y_dim = BOSS.y_dim(problem)

    # Sample parameter values.
    xs = rand(acq.x_proposal, acq.x_samples)
    
    # w_i = 1 / pdf(x_proposal, x_i)
    log_ws = 0 .- logpdf.(Ref(acq.x_proposal), eachcol(xs))
    ws = exp.( log_ws .- log(sum(exp.(log_ws))) ) # normalize to sum up to 1

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs_y = sample_ϵs(y_dim, acq.y_samples) # vector-vector
    Es_s = [sample_ϵs(y_dim, acq.y_samples) for _ in 1:acq.x_samples] # vector-vector-vector

    # precalculate model posterior
    model_post = BOSS.model_posterior(bosip.problem)

    @warn "Using experimental length-scales for the HSIC kernels."
    # calculate `σy` used for the y lengthscale
    σs_ = std.(Ref(model_post), eachcol(xs))
    σy = sum(ws .* σs_)  # sum(ws) == 1
    # calculate `M` used for the s lengthscale
    post_ = approx_posterior(bosip)
    ss_ = post_.(eachcol(xs))
    M = maximum(ss_)

    return IMMDFunc(acq, bosip, model_post, xs, ws, ϵs_y, Es_s, σy, M)
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
    LS<:Real,
}
    acq::IMMD
    bosip::B
    model_post::M
    xs::X
    ws::W
    ϵs_y::E1
    Es_s::E2
    σy::LY
    M::LS
end

function (f::IMMDFunc)(x_::AbstractVector{<:Real})
    # sample `N` y_ samples at the new x_
    μy, σy = mean_and_std(f.model_post, x_)
    ys_ = calc_y.(Ref(μy), Ref(σy), f.ϵs_y)

    # augment problems
    problem_copies = [deepcopy(f.bosip.problem) for _ in 1:f.acq.y_samples]
    for (p, y_) in zip(problem_copies, ys_)
        augment_dataset!(p, x_, y_)
    end
    aug_posts_samples = model_posterior.(problem_copies)

    # sample `K x N` y_eval (and s_eval) samples (N for each x_eval sample)
    Y_evals = get_ys_eval.(Ref(aug_posts_samples), eachcol(f.xs), f.Es_s) # vector-vector-vector
    # S_evals = get_ss_eval.(Y_evals, eachcol(f.xs), Ref(f.bosip.likelihood), Ref(f.bosip.x_prior))
    log_S_evals = get_log_ss_eval.(Y_evals, eachcol(f.xs), Ref(f.bosip.likelihood), Ref(f.bosip.x_prior)) # vector-vector

    # S_evals are UNNORMALIZED! (This is intentional, as there is no straight-forward way to normalize them.)
    
    # calculate `K` HSICs between `y_` and `s_1,...,s_K`
    
    # y lengthscale
    y_kernel = BOSS.with_lengthscale(f.acq.y_kernel, f.σy)

    # s lengthscales
    p_kernel = f.acq.p_kernel
    normalize_log_S_evals!(log_S_evals; f.M) # (normalization instead of kernel lengthscale)
    S_evals = exponentiate_S_evals!(log_S_evals) # in-place exponentiation

    vals = hsic.(Ref(y_kernel), Ref(p_kernel), Ref(ys_), S_evals, Ref(f.acq.y_samples))
    
    # return sum(f.ws .* vals) |> log
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

# function get_ss_eval(ys_eval, x_eval, likelihood, x_prior)
#     log_ls = BOSIP.loglike.(Ref(likelihood), ys_eval)
#     log_px = logpdf(x_prior, x_eval)
#     ss = exp.(log_px .+ log_ls) # TODO unnormalized
#     return ss
# end
function get_log_ss_eval(ys_eval, x_eval, likelihood, x_prior)
    log_ls = BOSIP.loglike.(Ref(likelihood), ys_eval)
    log_px = logpdf(x_prior, x_eval)
    ss = log_ls .+ log_px # TODO unnormalized
    return ss
end

# function mmd(kernel, X::AbstractVector, Y::AbstractVector)
#     val_X = mean(BOSS.kernelmatrix(kernel, X))
#     val_Y = mean(BOSS.kernelmatrix(kernel, Y))
#     val_XY = mean(BOSS.kernelmatrix(kernel, X, Y))
#     return val_X + val_Y - 2*val_XY
# end
function hsic(kernel_X, kernel_Y, X::AbstractVector, Y::AbstractVector, n::Int)
    Kx = BOSS.kernelmatrix(kernel_X, X)
    Ky = BOSS.kernelmatrix(kernel_Y, Y)
    C = Diagonal(ones(n)) - (1/n) * ones(n,n)
    return (1 / (n^2)) * tr( (C * Kx) * (C * Ky) )
end

function normalize_log_S_evals!(log_S_evals; M)
    for i in eachindex(log_S_evals)
        for j in eachindex(log_S_evals[i])
            log_S_evals[i][j] -= M
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
