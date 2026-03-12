
# Type aliases matching BOSS.jl's internal definitions (not exported from BOSS)
const _LengthscalePriors = AbstractVector{<:MultivariateDistribution}
const _AmplitudePriors   = AbstractVector{<:UnivariateDistribution}
const _NoiseStdPriors    = AbstractVector{<:UnivariateDistribution}

# Explicitly extend BOSS.jl functions so BOSS's internal dispatch finds our methods.
# Non-exported functions (prefixed _) must use the qualified form below.
import BOSS: sliceable, join_slices, slice, model_posterior_slice,
             mean, var, cov, mean_and_var,
             data_loglike, params_loglike, vectorizer, bijector,
             param_priors

"""
    GradientGaussianProcess(; kwargs...)

A Gaussian Process surrogate conditioned on both function values and their gradients,
implementing the derivative-enhanced GP from Wu et al. (2017),
"Bayesian Optimization with Gradients".

Each simulator evaluation `(x, y, ∇y)` contributes `1 + x_dim` observations instead
of 1, improving sample efficiency. Use with [`GradientExperimentData`](@ref).

The simulator `f(x)` must return `(y, ∇y)` where:
- `y::Vector` is the function output of length `y_dim`
- `∇y::Vector` is the stacked row-wise Jacobian of length `y_dim * x_dim`:
  `∇y = vec(ForwardDiff.jacobian(f_y, x)')`

## Keywords

Same as `GaussianProcess`, plus:
- `grad_noise_std_priors::NoiseStdPriors`: Priors on gradient observation noise σ_∂.
  Should be non-Dirac to allow the GP to learn gradient uncertainty from data.
"""
struct GradientGaussianProcess{
    M<:Union{Nothing, AbstractVector{<:Real}, Function},
} <: SurrogateModel
    mean::M
    kernel::Kernel
    lengthscale_priors::_LengthscalePriors
    amplitude_priors::_AmplitudePriors
    noise_std_priors::_NoiseStdPriors
    grad_noise_std_priors::_NoiseStdPriors
end

function GradientGaussianProcess(;
    mean = nothing,
    kernel = Matern52Kernel(),
    lengthscale_priors,
    amplitude_priors,
    noise_std_priors,
    grad_noise_std_priors,
)
    return GradientGaussianProcess(
        mean, kernel,
        lengthscale_priors, amplitude_priors, noise_std_priors, grad_noise_std_priors,
    )
end

"""
    GradientGaussianProcessParams(λ, α, σ, σ_∂)

Parameters of [`GradientGaussianProcess`](@ref).

- `λ`: Lengthscales, shape `x_dim × y_dim`.
- `α`: Amplitudes, length `y_dim`.
- `σ`: Function observation noise std, length `y_dim`.
- `σ_∂`: Gradient observation noise std, length `y_dim`.
"""
struct GradientGaussianProcessParams{
    L<:AbstractMatrix{<:Real},
    A<:AbstractVector{<:Real},
    N<:AbstractVector{<:Real},
    ND<:AbstractVector{<:Real},
} <: ModelParams{GradientGaussianProcess}
    λ::L
    α::A
    σ::N
    σ_∂::ND
end

"""
Posterior slice for `GradientGaussianProcess`, holding precomputed quantities
for efficient prediction.
"""
struct GradientGPPosteriorSlice <: ModelPosteriorSlice{GradientGaussianProcess}
    k_fn::Any                    # (x, xp) -> scalar: the amplitude/lengthscale-scaled kernel
    X_train::Matrix{Float64}     # x_dim × n
    alpha::Vector{Float64}       # K_aug⁻¹ỹ, length n*(1 + x_dim)
    chol::Cholesky{Float64, Matrix{Float64}}
    σ::Float64                   # function observation noise std (stored for dKG)
    σ_∂::Float64                 # gradient observation noise std (stored for dKG)
end


### Sliceable model interface ###

sliceable(::GradientGaussianProcess) = true

function slice(m::GradientGaussianProcess, idx::Int)
    # Inline the mean-slice logic to avoid depending on BOSS internals.
    mean_idx = if isnothing(m.mean)
        nothing
    elseif m.mean isa AbstractVector
        m.mean[idx:idx]
    else
        x -> @view m.mean(x)[idx:idx]
    end
    return GradientGaussianProcess(
        mean_idx,
        m.kernel,
        m.lengthscale_priors[idx:idx],
        m.amplitude_priors[idx:idx],
        m.noise_std_priors[idx:idx],
        m.grad_noise_std_priors[idx:idx],
    )
end

function slice(p::GradientGaussianProcessParams, idx::Int)
    return GradientGaussianProcessParams(
        p.λ[:, idx:idx],
        p.α[idx:idx],
        p.σ[idx:idx],
        p.σ_∂[idx:idx],
    )
end

function join_slices(ps::AbstractVector{<:GradientGaussianProcessParams})
    return GradientGaussianProcessParams(
        hcat(getfield.(ps, Ref(:λ))...),
        vcat(getfield.(ps, Ref(:α))...),
        vcat(getfield.(ps, Ref(:σ))...),
        vcat(getfield.(ps, Ref(:σ_∂))...),
    )
end

param_lengths(p::GradientGaussianProcessParams) =
    (length(p.λ), length(p.α), length(p.σ), length(p.σ_∂))


### Kernel helpers ###

"""
Build the scaled kernel function `(x, xp) -> k(x, xp)` for a given output slice.
"""
function _make_kernel_fn(kernel::Kernel, λ::AbstractVector, α::Real)
    min_val = BOSS.MIN_PARAM_VALUE
    scaled_k = (α + min_val)^2 * with_lengthscale(kernel, λ .+ min_val)
    return (x, xp) -> scaled_k(x, xp)
end

"""
Compute kernel value and all derivatives needed for the augmented GP system at `(xi, xj)`.

Returns `(k_val, dk_dxi, dk_dxj, d2k)` where:
- `k_val`    = k(xi, xj)                          (scalar)
- `dk_dxi`   = ∂k(xi,xj)/∂(xi)_l                 (length d)
- `dk_dxj`   = ∂k(xi,xj)/∂(xj)_l                 (length d)
- `d2k[l,m]` = ∂²k(xi,xj) / (∂(xi)_l ∂(xj)_m)   (d × d)

Uses ForwardDiff on the concatenated input `z = [xi; xj]` for a single Hessian call.
"""
function _kernel_and_derivs(k_fn, xi::AbstractVector, xj::AbstractVector)
    d = length(xi)
    g(z) = k_fn(z[1:d], z[d+1:2d])
    z = vcat(xi, xj)
    # Many kernels (e.g. Matérn) have a cusp at r=0, causing ForwardDiff to return NaN
    # when xi==xj.  Perturbing xj by ε avoids the singularity; the error is O(ε²) in
    # the Hessian and O(ε) in the gradient (which is ≈ 0 at the diagonal anyway).
    z_ad = xi ≈ xj ? vcat(xi, xj .+ 1e-6) : z
    G = ForwardDiff.gradient(g, z_ad)
    H = ForwardDiff.hessian(g, z_ad)
    return g(z), G[1:d], G[d+1:2d], H[1:d, d+1:2d]
end

"""
Build the `N × N` augmented kernel matrix, N = n*(1 + x_dim).

Augmented observation ordering (consistent with `_build_obs_vector`):
  [f(x₁),...,f(xₙ),  ∂f/∂x₁(x₁),...,∂f/∂x₁(xₙ),  ...,  ∂f/∂x_d(x₁),...,∂f/∂x_d(xₙ)]

Block structure:
  K[i, j]                     = k(xᵢ, xⱼ)                   (function-function)
  K[i, n+(l-1)n+j]            = ∂k(xᵢ,xⱼ)/∂(xⱼ)_l           (function-gradient)
  K[n+(l-1)n+i, j]            = ∂k(xᵢ,xⱼ)/∂(xᵢ)_l           (gradient-function)
  K[n+(l-1)n+i, n+(m-1)n+j]  = ∂²k(xᵢ,xⱼ)/(∂(xᵢ)_l∂(xⱼ)_m) (gradient-gradient)

Noise terms: σ² on function block diagonal, σ_∂² on gradient block diagonal.
"""
function _build_augmented_kernel(k_fn, X::AbstractMatrix, σ::Real, σ_∂::Real)
    n = size(X, 2)
    d = size(X, 1)
    N = n * (1 + d)
    K = zeros(N, N)

    for i in 1:n, j in 1:n
        k_val, dk_dxi, dk_dxj, d2k = _kernel_and_derivs(k_fn, X[:, i], X[:, j])
        K[i, j] = k_val
        for l in 1:d
            K[i,             n + (l-1)*n + j] = dk_dxj[l]
            K[n + (l-1)*n + i,             j] = dk_dxi[l]
        end
        for l in 1:d, m in 1:d
            K[n + (l-1)*n + i, n + (m-1)*n + j] = d2k[l, m]
        end
    end

    noise_diag = vcat(fill(σ^2 + BOSS.MIN_PARAM_VALUE, n), fill(σ_∂^2 + BOSS.MIN_PARAM_VALUE, n * d))
    K[diagind(K)] .+= noise_diag

    return Symmetric(K)
end

"""
Build the augmented cross-covariance vector between test point `x_star`
and all training observations (function values + gradients), length n*(1+d).

  k_cross[j]            = Cov[f(x*), f(xⱼ)]             = k(x*, xⱼ)
  k_cross[n+(l-1)n+j]  = Cov[f(x*), ∂f(xⱼ)/∂(xⱼ)_l]  = ∂k(x*,xⱼ)/∂(xⱼ)_l
"""
function _build_cross_cov(k_fn, x_star::AbstractVector, X_train::AbstractMatrix)
    n = size(X_train, 2)
    d = size(X_train, 1)
    k_cross = Vector{Float64}(undef, n * (1 + d))
    for j in 1:n
        xj = X_train[:, j]
        g(xp) = k_fn(x_star, xp)
        k_cross[j] = g(xj)
        xj_ad = x_star ≈ xj ? xj .+ 1e-6 : xj
        dk_dxj = ForwardDiff.gradient(g, xj_ad)
        for l in 1:d
            k_cross[n + (l-1)*n + j] = dk_dxj[l]
        end
    end
    return k_cross
end

"""
Build the `N × (1+d)` matrix of prior covariances between all training observations
and the augmented new observation `(f(x_new), ∂f/∂x₁(x_new), ..., ∂f/∂x_d(x_new))`.

Row layout matches `_build_obs_vector`: [f obs₁…fobsₙ, ∂/∂x₁ obs₁…, …, ∂/∂x_d obs₁…].
Column layout: [f(x_new), ∂f(x_new)/∂x₁, …, ∂f(x_new)/∂x_d].
"""
function _build_cross_cov_matrix(k_fn, x_new::AbstractVector, X_train::AbstractMatrix)
    n = size(X_train, 2)
    d = length(x_new)
    N = n * (1 + d)
    K = zeros(N, 1 + d)
    for j in 1:n
        # xi = x_new, xj = training point
        k_val, dk_dxnew, dk_dxtrain, d2k = _kernel_and_derivs(k_fn, x_new, X_train[:, j])
        K[j, 1] = k_val
        for l in 1:d
            K[j,               1+l] = dk_dxnew[l]     # Cov(f(xⱼ),        ∂f(x_new)/∂xₗ)
            K[n+(l-1)*n+j,       1] = dk_dxtrain[l]   # Cov(∂f(xⱼ)/∂xⱼₗ, f(x_new))
            for m in 1:d
                K[n+(l-1)*n+j, 1+m] = d2k[m, l]       # Cov(∂f(xⱼ)/∂xⱼₗ, ∂f(x_new)/∂xₘ)
            end
        end
    end
    return K
end

"""
Build the augmented observation vector from function values and gradient matrix.

  ỹ = [y₁,...,yₙ,  ∂y₁/∂x₁,...,∂yₙ/∂x₁,  ...,  ∂y₁/∂x_d,...,∂yₙ/∂x_d]

`dY` has shape `x_dim × n` (the gradient data for this output slice after slicing).
"""
function _build_obs_vector(y::AbstractVector, dY::AbstractMatrix)
    d = size(dY, 1)
    return vcat(y, [dY[l, :] for l in 1:d]...)
end


### Posterior construction ###

function model_posterior_slice(
    model::GradientGaussianProcess,
    params::GradientGaussianProcessParams,
    data::GradientExperimentData,
    slice::Int,
)
    k_fn = _make_kernel_fn(model.kernel, params.λ[:, slice], params.α[slice])
    σ    = params.σ[slice]
    σ_∂  = params.σ_∂[slice]

    X  = data.X          # x_dim × n  (data is already sliced by BOSS.jl's sliceable machinery)
    y  = data.Y[1, :]    # function values for this output slice
    dY = data.dY         # x_dim × n  gradients for this output slice

    ỹ     = _build_obs_vector(y, dY)
    K_aug = _build_augmented_kernel(k_fn, X, σ, σ_∂)
    C     = cholesky(K_aug)
    alpha = C \ ỹ

    return GradientGPPosteriorSlice(k_fn, Matrix(X), alpha, C, σ, σ_∂)
end


### Posterior prediction ###

function mean(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    return _build_cross_cov(post.k_fn, x, post.X_train) ⋅ post.alpha
end

function mean(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    return [mean(post, X[:, j]) for j in axes(X, 2)]
end

function var(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    k_cross = _build_cross_cov(post.k_fn, x, post.X_train)
    k_self  = post.k_fn(x, x)
    v = post.chol.L \ k_cross
    return max(0.0, k_self - v ⋅ v)
end

function var(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    return [var(post, X[:, j]) for j in axes(X, 2)]
end

function mean_and_var(post::GradientGPPosteriorSlice, x::AbstractVector{<:Real})
    k_cross = _build_cross_cov(post.k_fn, x, post.X_train)
    k_self  = post.k_fn(x, x)
    μ  = k_cross ⋅ post.alpha
    v  = post.chol.L \ k_cross
    σ² = max(0.0, k_self - v ⋅ v)
    return μ, σ²
end

function mean_and_var(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    results = [mean_and_var(post, X[:, j]) for j in axes(X, 2)]
    return [r[1] for r in results], [r[2] for r in results]
end

function cov(post::GradientGPPosteriorSlice, X::AbstractMatrix{<:Real})
    cols = axes(X, 2)
    ks = [_build_cross_cov(post.k_fn, X[:, j], post.X_train) for j in cols]
    vs = [post.chol.L \ k for k in ks]
    return [post.k_fn(X[:, i], X[:, j]) - vs[i] ⋅ vs[j] for i in cols, j in cols]
end


### Data log-likelihood (log marginal likelihood of augmented GP) ###

function data_loglike(model::GradientGaussianProcess, data::GradientExperimentData)
    # Called per output slice (y_dim = 1) by BOSS.jl's sliceable optimization machinery.
    function ll(params::GradientGaussianProcessParams)
        k_fn = _make_kernel_fn(model.kernel, params.λ[:, 1], params.α[1])
        σ    = params.σ[1]
        σ_∂  = params.σ_∂[1]

        ỹ     = _build_obs_vector(data.Y[1, :], data.dY)
        K_aug = _build_augmented_kernel(k_fn, data.X, σ, σ_∂)

        C     = cholesky(K_aug)
        alpha = C \ ỹ
        N     = length(ỹ)

        # Log marginal likelihood: -½(ỹᵀK⁻¹ỹ + log|K| + N log 2π)
        return -0.5 * (ỹ ⋅ alpha + 2 * sum(log.(diag(C.L))) + N * log(2π))
    end
    return ll
end


### Hyperparameter prior log-likelihood ###

function params_loglike(model::GradientGaussianProcess)
    function ll(params::GradientGaussianProcessParams)
        ll_λ  = sum(logpdf.(model.lengthscale_priors,    eachcol(params.λ)))
        ll_α  = sum(logpdf.(model.amplitude_priors,      params.α))
        ll_σ  = sum(logpdf.(model.noise_std_priors,      params.σ))
        ll_σ∂ = sum(logpdf.(model.grad_noise_std_priors, params.σ_∂))
        return ll_λ + ll_α + ll_σ + ll_σ∂
    end
end

function BOSS._params_sampler(model::GradientGaussianProcess)
    function sample(rng::AbstractRNG)
        λ   = hcat(rand.(Ref(rng), model.lengthscale_priors)...)
        α   = rand.(Ref(rng), model.amplitude_priors)
        σ   = rand.(Ref(rng), model.noise_std_priors)
        σ_∂ = rand.(Ref(rng), model.grad_noise_std_priors)
        return GradientGaussianProcessParams(λ, α, σ, σ_∂)
    end
end


### Vectorizer and bijector (for MAP optimization) ###

function vectorizer(model::GradientGaussianProcess)
    is_dirac, dirac_vals = BOSS.create_dirac_mask(param_priors(model))

    function vectorize(params::GradientGaussianProcessParams)
        ps = vcat(vec(params.λ), params.α, params.σ, params.σ_∂)
        return BOSS.filter_diracs(ps, is_dirac)
    end

    function devectorize(params::GradientGaussianProcessParams, ps::AbstractVector{<:Real})
        ps = BOSS.insert_diracs(ps, is_dirac, dirac_vals)
        λ_len, α_len, σ_len, _ = param_lengths(params)
        λ   = reshape(ps[1:λ_len], size(params.λ))
        α   = ps[λ_len+1:λ_len+α_len]
        σ   = ps[λ_len+α_len+1:λ_len+α_len+σ_len]
        σ_∂ = ps[λ_len+α_len+σ_len+1:end]
        return GradientGaussianProcessParams(λ, α, σ, σ_∂)
    end

    return vectorize, devectorize
end

function bijector(model::GradientGaussianProcess)
    return BOSS.default_bijector(param_priors(model))
end

function param_priors(model::GradientGaussianProcess)
    return vcat(
        model.lengthscale_priors,
        model.amplitude_priors,
        model.noise_std_priors,
        model.grad_noise_std_priors,
    )
end
