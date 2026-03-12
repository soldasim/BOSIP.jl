
"""
    dKGAcquisition(; kwargs...)

The derivative Knowledge Gradient (dKG) acquisition function from Wu et al. (2017),
"Bayesian Optimization with Gradients".

For each candidate point `x`, dKG computes the one-step-ahead expected improvement in
the optimal posterior mean when *both* `f(x)` and `∇f(x)` are observed jointly.
Compared to variance-based acquisitions, dKG accounts for global information gain
about the optimum rather than local uncertainty reduction.

Requires the surrogate to be a `GradientGaussianProcess` with `y_dim == 1`.

The value is approximated via Monte Carlo:

    dKG(x) ≈ (1/S) Σₛ max_m [ μₙ(x′ₘ) + bₘᵀ εₛ ] − max_m μₙ(x′ₘ)

where `{x′ₘ}` is a random fantasy grid, `B = L⁻¹ C` with `C` the posterior
cross-covariance matrix and `L = chol(Σₙ(x))`, and `εₛ ~ N(0, I_{1+d})`.

## Keywords

- `n_fantasy::Int`: Number of fantasy grid points for the inner maximisation. Default: 64.
- `n_mc::Int`: Number of Monte Carlo samples for the expectation. Default: 512.
"""
@kwdef struct dKGAcquisition <: BosipAcquisition
    n_fantasy::Int = 64
    n_mc::Int = 512
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

    c[1]   = Cov_post(f(x_star), f(x_new))
    c[1+l] = Cov_post(f(x_star), ∂f(x_new)/∂xₗ)

Pass precomputed `K_cross_new = _build_cross_cov_matrix(k_fn, x_new, X_train)` and
`alpha_star = chol \\ _build_cross_cov(k_fn, x_star, X_train)` for efficiency when
iterating over many fantasy points with the same `x_new`.
"""
function _posterior_cross_cov(
    post::GradientGPPosteriorSlice,
    x_star::AbstractVector,
    x_new::AbstractVector,
    K_cross_new::AbstractMatrix,   # (N, 1+d): _build_cross_cov_matrix(k_fn, x_new, X_train)
    alpha_star::AbstractVector,    # (N,): chol \ _build_cross_cov(k_fn, x_star, X_train)
)
    k_val, _, dk_dxnew, _ = _kernel_and_derivs(post.k_fn, x_star, x_new)
    k_prior = vcat(k_val, dk_dxnew)              # (1+d,): prior cross-cov
    return k_prior - K_cross_new' * alpha_star   # (1+d,): posterior cross-cov
end


# ── Acquisition construction ──────────────────────────────────────────────────

function (acq::dKGAcquisition)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, ::BosipOptions)
    post = model_posterior(bosip.problem)

    @assert post isa BOSS.DefaultModelPosterior && length(post.slices) == 1 """
    dKGAcquisition requires a single-output (y_dim == 1) GradientGaussianProcess model.
    """
    ps = post.slices[1]   # GradientGPPosteriorSlice

    lb, ub = bosip.problem.domain.bounds
    d = length(lb)
    rng = Random.default_rng()

    # Fantasy grid: uniform random over the domain (resampled each BO step)
    grid = lb .+ (ub .- lb) .* rand(rng, d, acq.n_fantasy)  # d × n_fantasy

    # Current posterior mean on the grid and the best value
    a = [BOSS.mean(ps, grid[:, m]) for m in 1:acq.n_fantasy]
    best_a = maximum(a)

    # Precompute K_aug⁻¹ k_cross(x′ₘ) for each fantasy point (fixed across candidates)
    k_crosses   = [_build_cross_cov(ps.k_fn, grid[:, m], ps.X_train) for m in 1:acq.n_fantasy]
    alpha_stars = [ps.chol \ kc for kc in k_crosses]

    function dkg(x_new::AbstractVector)
        x_new = vec(x_new)

        # (1) Posterior predictive covariance Σₙ(x_new) and Cholesky factor L
        Σ_new = _posterior_aug_cov(ps, x_new)
        L_new = cholesky(Symmetric(Σ_new + BOSS.MIN_PARAM_VALUE * I)).L   # (1+d) × (1+d)

        # (2) Cross-cov between training obs and augmented obs at x_new (shared across fantasy pts)
        K_cross_new = _build_cross_cov_matrix(ps.k_fn, x_new, ps.X_train)  # (N, 1+d)

        # (3) Influence matrix B: column m = L⁻¹ cₘ
        #     Update rule: δμ(x′ₘ) = B[:, m]ᵀ ε,  ε ~ N(0, I_{1+d})
        B = Matrix{Float64}(undef, 1 + d, acq.n_fantasy)
        for m in 1:acq.n_fantasy
            c_m = _posterior_cross_cov(ps, grid[:, m], x_new, K_cross_new, alpha_stars[m])
            B[:, m] = L_new \ c_m
        end

        # (4) MC estimate of E_ε[ max_m (aₘ + (Bᵀε)ₘ) ] − max_m aₘ
        expected_max = 0.0
        for _ in 1:acq.n_mc
            ε = randn(rng, 1 + d)
            expected_max += maximum(a .+ B' * ε)
        end

        return expected_max / acq.n_mc - best_a
    end

    return dkg
end
