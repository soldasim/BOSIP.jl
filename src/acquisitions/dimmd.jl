
"""
    dIMMDAcquisition(; kwargs...)

Derivative-GP Integrated Mean Maximum Discrepancy (dIMMD) acquisition for BOSIP.

For each candidate `x_new`, dIMMD estimates the expected absolute shift in the
expected log-likelihood over the parameter space when `(f(x_new), ∇f(x_new))` are
observed jointly:

    dIMMD(x_new) = √(2/π) · Σₘ w̃ₘ · |∂ℓ/∂μ(μₙ(x'_m), σ²ₙ(x'_m))| · ‖bₘ‖

where:

- `{x'_m}` is a fantasy grid over the domain,
- `w̃_m ∝ exp(ℓ(μₙ(x'_m), σ²ₙ(x'_m)) + log p(x'_m))` are normalised posterior weights,
- `bₘ = L_new⁻¹ cₘ` is the influence vector (GP mean-shift magnitude `‖bₘ‖`),
- `∂ℓ/∂μ` is the **likelihood sensitivity**: how much the expected log-likelihood
  changes per unit shift of the GP mean at `x'_m`.

**BOSIP adaptation vs plain IMMD**

The plain IMMD uses `‖bₘ‖` (GP mean shift in output space).  dIMMD converts this
to the expected absolute *likelihood* shift:

    𝔼[|Δℓ|] ≈ |∂ℓ/∂μ| · ‖bₘ‖ · √(2/π)

The likelihood sensitivity `∂ℓ/∂μ` depends on the likelihood type:
- `NormalLikelihood`: `(z_obs - μ) / (σ² + std_obs²)` — zero at the posterior mode,
  large where the likelihood slope is steep (away from the observation)
- `ExpLikelihood`:    `1.0` — constant sensitivity, identical to plain IMMD
- Fallback:           `1.0`

For Normal likelihoods the sensitivity weight naturally suppresses sampling near the
posterior mode (where the likelihood is flat) and promotes sampling in regions where
a GP mean shift produces a meaningful likelihood change.

**Fully deterministic** — no MC sampling required.

See `docs/dgp-bosip.md §6.3` for the full derivation.

Requires the surrogate to be a `GradientGaussianProcess` with `y_dim == 1`.

## Keywords

- `n_grid::Int`: Number of fantasy grid points for the weighted integral. Default: 256.
"""
@kwdef struct dIMMDAcquisition <: BosipAcquisition
    n_grid::Int = 256
end


# ── Acquisition construction ──────────────────────────────────────────────────

function (acq::dIMMDAcquisition)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, ::BosipOptions)
    post = model_posterior(bosip.problem)
    like = bosip.likelihood

    @assert post isa BOSS.DefaultModelPosterior && length(post.slices) == 1 """
    dIMMDAcquisition requires a single-output (y_dim == 1) GradientGaussianProcess model.
    """
    ps = post.slices[1]   # GradientGPPosteriorSlice

    lb, ub = bosip.problem.domain.bounds
    d = length(lb)
    rng = Random.default_rng()

    # Fantasy grid: uniform random over the domain (resampled each BO step)
    grid = lb .+ (ub .- lb) .* rand(rng, d, acq.n_grid)   # d × n_grid

    # GP posterior statistics at each grid point
    μ_grid  = [BOSS.mean(ps, grid[:, m]) for m in 1:acq.n_grid]
    σ²_grid = [BOSS.var( ps, grid[:, m]) for m in 1:acq.n_grid]

    # Posterior weights:  w̃_m ∝ exp(ℓ(μ_m, σ²_m) + log p(x'_m))
    log_prior = [logpdf(bosip.x_prior, grid[:, m]) for m in 1:acq.n_grid]
    a = [_ell(like, μ_grid[m], σ²_grid[m]) + log_prior[m] for m in 1:acq.n_grid]
    a_max = maximum(a)
    w = exp.(a .- a_max)
    w ./= sum(w)   # normalise (log-sum-exp stable)

    # Likelihood sensitivity weights |∂ℓ/∂μ(μ_m, σ²_m)| — fixed across candidates
    sens = [abs(_dell_dmu(like, μ_grid[m], σ²_grid[m])) for m in 1:acq.n_grid]

    # Precompute K_aug⁻¹ k_cross(x'_m) for each grid point (fixed across candidates)
    k_crosses   = [_build_cross_cov(ps.k_fn, grid[:, m], ps.X_train) for m in 1:acq.n_grid]
    alpha_stars = [ps.chol \ kc for kc in k_crosses]

    sqrt2π = sqrt(2.0 / π)

    function dimmd(x_new::AbstractVector)
        x_new = vec(x_new)

        # (1) Posterior predictive covariance Σₙ(x_new) and Cholesky L
        Σ_new = _posterior_aug_cov(ps, x_new)
        L_new = cholesky(Symmetric(Σ_new + BOSS.MIN_PARAM_VALUE * I)).L

        # (2) Cross-cov between training obs and augmented obs at x_new
        K_cross_new = _build_cross_cov_matrix(ps.k_fn, x_new, ps.X_train)

        # (3) Weighted sum of expected absolute likelihood shifts
        #     𝔼[|Δℓ_m|] ≈ |∂ℓ/∂μ|_m · ‖b_m‖ · √(2/π)
        #     (first-order Taylor expansion of ℓ in the stochastic mean update b_m^T ε)
        total = 0.0
        for m in 1:acq.n_grid
            c_m = _posterior_cross_cov(ps, grid[:, m], x_new, K_cross_new, alpha_stars[m])
            b_m = L_new \ c_m
            total += w[m] * sens[m] * norm(b_m)
        end

        return sqrt2π * total
    end

    return dimmd
end
