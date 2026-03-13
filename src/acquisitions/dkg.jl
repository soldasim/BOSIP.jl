
"""
    dKGAcquisition(; kwargs...)

The BOSIP-adapted derivative Knowledge Gradient (dKG) acquisition function,
extending Wu et al. (2017) "Bayesian Optimization with Gradients" to the
likelihood-based inference setting of BOSIP.

For each candidate simulator evaluation point `x_new`, dKG estimates the
one-step-ahead expected improvement in the best achievable **expected log-likelihood**
`𝔼[log p(z_obs | f(x'))]` over the parameter space, when both `f(x_new)` and
`∇f(x_new)` are observed jointly.

The value is approximated via Monte Carlo over a random fantasy grid `{x'_m}`:

    dKG(x_new) ≈ (1/S) Σₛ max_m [ ℓ(μₙ(x'_m) + bₘᵀεₛ, σ²ₙ₊₁(x'_m)) + log p(x'_m) ]
                 − max_m [ ℓ(μₙ(x'_m), σ²ₙ(x'_m)) + log p(x'_m) ]

where `ℓ(μ, σ²) = E_y[log p(z_obs | y)]` (expected log-likelihood under GP predictive),
`bₘ = L⁻¹ cₘ` is the influence vector, and the posterior variance reduces deterministically
as `σ²ₙ₊₁ = σ²ₙ − ‖bₘ‖²`.

See `docs/dgp-bosip.md` for the full theoretical derivation.

Requires the surrogate to be a `GradientGaussianProcess` with `y_dim == 1`.

## Keywords

- `n_fantasy::Int`: Number of fantasy grid points for the inner maximisation. Default: 64.
- `n_mc::Int`: Number of Monte Carlo samples for the expectation. Default: 512.
- `temperature::Float64`: Softmax temperature for aggregating fantasy values (default: 0.0 = hard max).
    Positive values replace `max_m` with `τ log Σ_m exp(aₘ/τ)`, spreading credit across all
    high-likelihood fantasy points and preventing collapse near the posterior mode.
    Typical range: 0.5–5.0 (larger = more exploratory).
"""
@kwdef struct dKGAcquisition <: BosipAcquisition
    n_fantasy::Int = 64
    n_mc::Int = 512
    """
    Softmax temperature for aggregating fantasy-grid values (default: 0.0 = hard max).
    Positive values smooth the max into a log-sum-exp, encouraging exploration of
    all high-likelihood regions rather than just the absolute best fantasy point.
    Try values in [0.5, 5.0] when points cluster near the posterior mode.
    """
    temperature::Float64 = 0.0
end


# ── Expected log-likelihood ℓ(μ, σ²) per likelihood type ──────────────────────

"""
    _ell(likelihood, μ, σ²) -> scalar

Expected log-likelihood `𝔼_{y ~ N(μ, σ²)}[log p(z_obs | y)]` for a scalar GP output.

Specific formulas:
- `NormalLikelihood`: analytic Gaussian convolution (z_obs | x ~ N(μ, σ²+std_obs²))
- `ExpLikelihood`:    lognormal mean in log space (= μ + σ²/2)
- Fallback:           plug-in `log p(z_obs | μ)`, ignoring GP variance
"""
function _ell(like::NormalLikelihood, μ::Real, σ²::Real)
    # y ~ N(μ, σ²), z_obs ~ N(y, std_obs²) → z_obs|x ~ N(μ, σ²+std_obs²)
    std_total = sqrt(like.std_obs[1]^2 + max(0.0, σ²))
    return logpdf(Normal(μ, std_total), like.z_obs[1])
end

function _ell(::ExpLikelihood, μ::Real, σ²::Real)
    # GP models log-likelihood directly; E[exp(y)] = exp(μ + σ²/2) → log = μ + σ²/2
    return μ + 0.5 * max(0.0, σ²)
end

function _ell(like::Likelihood, μ::Real, ::Real)
    # Generic fallback: plug-in point estimate
    return loglike(like, [μ])
end


# ── Likelihood variance ℓ_var(μ, σ²) per likelihood type ─────────────────────

"""
    _like_var(likelihood, μ, σ²) -> scalar ≥ 0

Variance of the likelihood `Var_{y ~ N(μ,σ²)}[ p(z_obs | y) ]` for a scalar GP output.

This is the quantity reduced by BOSIP acquisitions (MaxVar, LogMaxVar, EIV all work
in likelihood-variance space, not in GP-output-variance space).

Specific formulas:
- `NormalLikelihood`: `E[L²] - E[L]²` via analytic Gaussian moments
- `ExpLikelihood`:    `exp(2μ + σ²)(exp(σ²) - 1)` (lognormal variance)
- Fallback:           `0.0` (plug-in gives no variance; dIVR falls back to GP variance)
"""
function _like_var(like::NormalLikelihood, μ::Real, σ²::Real)
    σ_obs = like.std_obs[1]
    z     = like.z_obs[1]
    σ²_c  = max(0.0, σ²)
    # E[L]  = N(z; μ, σ²+σ_obs²)
    # E[L²] = 1/(2π σ_obs √(2σ²+σ_obs²)) · exp(-(z-μ)²/(2σ²+σ_obs²))
    log_EL  = logpdf(Normal(μ, sqrt(σ²_c + σ_obs^2)), z)
    log_EL2 = -log(2π) - log(σ_obs) - 0.5*log(2σ²_c + σ_obs^2) - (z - μ)^2 / (2σ²_c + σ_obs^2)
    return max(0.0, exp(log_EL2) - exp(2*log_EL))
end

function _like_var(::ExpLikelihood, μ::Real, σ²::Real)
    # V[exp(y)] = exp(2μ + σ²)(exp(σ²) - 1)
    σ²_c = max(0.0, σ²)
    return exp(2μ + σ²_c) * expm1(σ²_c)
end

function _like_var(::Likelihood, ::Real, ::Real)
    # Generic fallback: no closed form; dIVR caller should fall back to GP variance
    return 0.0
end


# ── dKG helper functions ───────────────────────────────────────────────────────

"""
Posterior predictive covariance matrix `(1+d) × (1+d)` of the augmented observation
`(f(x_new), ∂f/∂x₁(x_new), …, ∂f/∂x_d(x_new))` under the current GP posterior.
"""
function _posterior_aug_cov(post::GradientGPPosteriorSlice, x_new::AbstractVector)
    d = length(x_new)
    k_val, _, _, d2k = _kernel_and_derivs(post.k_fn, x_new, x_new)

    K_prior = zeros(1 + d, 1 + d)
    K_prior[1, 1] = k_val + post.σ^2 + BOSS.MIN_PARAM_VALUE
    for l in 1:d, m in 1:d
        K_prior[1+l, 1+m] = d2k[l, m] + (l == m ? post.σ_∂^2 + BOSS.MIN_PARAM_VALUE : 0.0)
    end

    K_cross = _build_cross_cov_matrix(post.k_fn, x_new, post.X_train)  # (N, 1+d)
    V = post.chol.L \ K_cross                                           # (N, 1+d)
    return Symmetric(K_prior - V' * V)
end

"""
Posterior cross-covariance `(1+d)` vector between `f(x_star)` and the augmented
observation `(f(x_new), ∂f/∂x₁(x_new), …)`.

Pass precomputed `K_cross_new` and `alpha_star` for efficiency.
"""
function _posterior_cross_cov(
    post::GradientGPPosteriorSlice,
    x_star::AbstractVector,
    x_new::AbstractVector,
    K_cross_new::AbstractMatrix,
    alpha_star::AbstractVector,
)
    k_val, _, dk_dxnew, _ = _kernel_and_derivs(post.k_fn, x_star, x_new)
    k_prior = vcat(k_val, dk_dxnew)
    return k_prior - K_cross_new' * alpha_star
end


# ── Aggregation helper ────────────────────────────────────────────────────────

"""
    _softmax_agg(vals, τ) -> scalar

Aggregate a vector of values:
- `τ == 0`:  hard maximum  (`maximum(vals)`)
- `τ  > 0`:  log-sum-exp smoothed maximum  (`τ * log Σ exp(vᵢ/τ)`)

The log-sum-exp form distributes credit across *all* high-value fantasy points
rather than concentrating on the single winner, encouraging broader exploration.
"""
function _softmax_agg(vals::AbstractVector, τ::Real)
    τ == 0.0 && return maximum(vals)
    v_max = maximum(vals)
    return v_max + τ * log(sum(exp((v - v_max) / τ) for v in vals))
end


# ── Acquisition construction ──────────────────────────────────────────────────

function (acq::dKGAcquisition)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, ::BosipOptions)
    post = model_posterior(bosip.problem)
    like = bosip.likelihood

    @assert post isa BOSS.DefaultModelPosterior && length(post.slices) == 1 """
    dKGAcquisition requires a single-output (y_dim == 1) GradientGaussianProcess model.
    """
    ps = post.slices[1]   # GradientGPPosteriorSlice

    lb, ub = bosip.problem.domain.bounds
    d = length(lb)
    rng = Random.default_rng()

    # Fantasy grid: uniform random over the domain (resampled each BO step)
    grid = lb .+ (ub .- lb) .* rand(rng, d, acq.n_fantasy)  # d × n_fantasy

    # Current GP posterior statistics at each fantasy point
    μ_grid  = [BOSS.mean(ps, grid[:, m]) for m in 1:acq.n_fantasy]   # scalar
    σ²_grid = [BOSS.var( ps, grid[:, m]) for m in 1:acq.n_fantasy]   # scalar

    # Current log-prior weights (constant across updates)
    log_prior = [logpdf(bosip.x_prior, grid[:, m]) for m in 1:acq.n_fantasy]

    # Current expected log-posterior: ℓ(μ_m, σ²_m) + log p(x'_m)
    a      = [_ell(like, μ_grid[m], σ²_grid[m]) + log_prior[m] for m in 1:acq.n_fantasy]
    best_a = _softmax_agg(a, acq.temperature)

    # Precompute K_aug⁻¹ k_cross(x′ₘ) for each fantasy point (fixed across candidates)
    k_crosses   = [_build_cross_cov(ps.k_fn, grid[:, m], ps.X_train) for m in 1:acq.n_fantasy]
    alpha_stars = [ps.chol \ kc for kc in k_crosses]

    function dkg(x_new::AbstractVector)
        x_new = vec(x_new)

        # (1) Posterior predictive covariance Σₙ(x_new) and Cholesky L
        Σ_new = _posterior_aug_cov(ps, x_new)
        L_new = cholesky(Symmetric(Σ_new + BOSS.MIN_PARAM_VALUE * I)).L

        # (2) Cross-cov between training obs and augmented obs at x_new
        K_cross_new = _build_cross_cov_matrix(ps.k_fn, x_new, ps.X_train)

        # (3) Influence matrix B and deterministic variance reductions
        #     GP mean update:     μₙ₊₁(x'_m) = μₙ(x'_m) + B[:,m]ᵀ ε
        #     GP variance update: σ²ₙ₊₁(x'_m) = σ²ₙ(x'_m) − ‖B[:,m]‖²
        B          = Matrix{Float64}(undef, 1 + d, acq.n_fantasy)
        σ²_updated = Vector{Float64}(undef, acq.n_fantasy)
        for m in 1:acq.n_fantasy
            c_m = _posterior_cross_cov(ps, grid[:, m], x_new, K_cross_new, alpha_stars[m])
            b_m = L_new \ c_m
            # Mathematically ‖b_m‖² ≤ σ²_grid[m] (variance reduction ≤ current variance).
            # Near training points, catastrophic cancellation in c_m inflates b_m numerically.
            # Clipping enforces the invariant and prevents dKG from peaking artificially
            # at already-observed locations.
            bsq = b_m ⋅ b_m
            if bsq > σ²_grid[m]
                b_m .*= sqrt(σ²_grid[m] / bsq)
                bsq   = σ²_grid[m]
            end
            B[:, m]       = b_m
            σ²_updated[m] = max(0.0, σ²_grid[m] - bsq)
        end

        # (4) MC estimate of E_ε[ agg_m ( ℓ(μₙ(x'_m)+δₘ, σ²ₙ₊₁(x'_m)) + log p(x'_m) ) ]
        #     agg = hard max (temperature==0) or log-sum-exp (temperature>0)
        expected_agg = 0.0
        for _ in 1:acq.n_mc
            ε = randn(rng, 1 + d)
            δ = B' * ε
            vals = [_ell(like, μ_grid[m] + δ[m], σ²_updated[m]) + log_prior[m]
                    for m in 1:acq.n_fantasy]
            expected_agg += _softmax_agg(vals, acq.temperature)
        end

        return expected_agg / acq.n_mc - best_a
    end

    return dkg
end
