
"""
    dIVRAcquisition(; kwargs...)

Derivative-GP Integrated Variance Reduction (dIVR) acquisition function for BOSIP.

Selects the next simulator evaluation point by maximising the expected reduction in
GP posterior variance, integrated over the parameter space and weighted by the current
approximate posterior:

    dIVR(x_new) = Σₘ ΔV_like(x'_m) · w̃ₘ

where:

- `{x'_m}` is a fantasy grid over the domain,
- `w̃_m ∝ exp(ℓ(μₙ(x'_m), σ²ₙ(x'_m)) + log p(x'_m))` are normalised posterior weights,
- `bₘ = L_new⁻¹ cₘ` is the influence vector,
- `‖bₘ‖² = σ²ₙ(x'_m) − σ²ₙ₊₁(x'_m)` is the deterministic GP variance reduction at x'_m,
- `ΔV_like(x'_m) = V[p(z_o|f(x'_m)) | σ²ₙ] − V[p(z_o|f(x'_m)) | σ²ₙ₊₁]` is the
  resulting **likelihood variance** reduction (the quantity BOSIP reduces).

**How the gradient GP changes IVR**

For a plain GP the variance reduction at x' is a scalar ratio:

    Δσ²(x') = Cov(f(x'), f(x_new))² / Var(f(x_new))

For a derivative GP, observing the augmented (f, ∇f) at x_new gives (1+d) virtual
observations. The variance reduction generalises to:

    Δσ²(x') = c(x', x_new)ᵀ Σₙ(x_new)⁻¹ c(x', x_new) = ‖bₘ‖²

where c ∈ ℝ^{1+d} contains the posterior cross-covariances between f(x') and each
component of the augmented observation, and Σₙ(x_new) is the (1+d)×(1+d) posterior
predictive covariance of the augmented observation.  This is always ≥ the plain-GP
reduction, with the inequality strict whenever gradient observations carry additional
information beyond f(x_new).

**Why this avoids mode-seeking**

Unlike dKG, dIVR uses a sum (not max) over the fantasy grid.  Every region of the
posterior contributes proportionally to its posterior weight w̃_m.  The acquisition
naturally seeks x_new that reduces uncertainty across the full posterior mass, not
only at its peak.

See `docs/divr-bosip.md` for the full theoretical derivation.

Requires the surrogate to be a `GradientGaussianProcess` with `y_dim == 1`.

## Keywords

- `n_grid::Int`: Number of fantasy grid points for the weighted integral. Default: 256.
"""
@kwdef struct dIVRAcquisition <: BosipAcquisition
    n_grid::Int = 256
end


# ── Acquisition construction ──────────────────────────────────────────────────

function (acq::dIVRAcquisition)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, ::BosipOptions)
    post = model_posterior(bosip.problem)
    like = bosip.likelihood

    @assert post isa BOSS.DefaultModelPosterior && length(post.slices) == 1 """
    dIVRAcquisition requires a single-output (y_dim == 1) GradientGaussianProcess model.
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

    # Precompute K_aug⁻¹ k_cross(x'_m) for each grid point (fixed across candidates)
    k_crosses   = [_build_cross_cov(ps.k_fn, grid[:, m], ps.X_train) for m in 1:acq.n_grid]
    alpha_stars = [ps.chol \ kc for kc in k_crosses]

    function divr(x_new::AbstractVector)
        x_new = vec(x_new)

        # (1) Posterior predictive covariance Σₙ(x_new) and Cholesky L
        Σ_new = _posterior_aug_cov(ps, x_new)
        L_new = cholesky(Symmetric(Σ_new + BOSS.MIN_PARAM_VALUE * I)).L

        # (2) Cross-cov between training obs and augmented obs at x_new
        K_cross_new = _build_cross_cov_matrix(ps.k_fn, x_new, ps.X_train)

        # (3) Weighted sum of likelihood-variance reductions ΔV_like
        #     GP variance update: σ²_{n+1}(x'_m) = σ²_n(x'_m) - ‖b_m‖²  (deterministic)
        #     Likelihood variance reduction: ΔV_like = V_like(σ²_n) - V_like(σ²_n - ‖b_m‖²)
        #     This is what BOSIP actually reduces: Var[p(z_o|f(x'))] not Var[f(x')].
        total = 0.0
        for m in 1:acq.n_grid
            c_m = _posterior_cross_cov(ps, grid[:, m], x_new, K_cross_new, alpha_stars[m])
            b_m = L_new \ c_m
            # Clip to enforce the mathematical invariant ‖b_m‖² ≤ σ²_grid[m].
            # Near training points, catastrophic cancellation in c_m inflates b_m
            # numerically; the clip prevents the acquisition from peaking there.
            bsq = min(b_m ⋅ b_m, σ²_grid[m])
            v_current = _like_var(like, μ_grid[m], σ²_grid[m])
            v_updated = _like_var(like, μ_grid[m], σ²_grid[m] - bsq)
            total += w[m] * max(0.0, v_current - v_updated)
        end

        return total
    end

    return divr
end
