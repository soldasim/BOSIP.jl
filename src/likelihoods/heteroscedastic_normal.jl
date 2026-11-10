
"""
    HeteroscedasticNormalLikelihood(; z_obs, std_fn, quad_nodes=31)

The observation is assumed to have been generated as `z_o[i] ~ Normal(y[i], std_fn(y)[i])`,
where the noise deviation depends on the simulator output `y = f(x)`.

## Kwargs
- `z_obs::Vector{Float64}`: The observed values.
- `std_fn::Function`: Maps the output vector `y` to the vector of noise deviations.
        It must act elementwise (`std_fn(y)[i]` depends on `y[i]` only),
        which is what makes the likelihood [`Marginalizable`](@ref). This is checked on construction.
- `quad_nodes::Int = 31`: The number of Gauss-Hermite nodes used for the expectations over the model predictive.

The log-likelihood is `-Inf` in the dimensions where `std_fn(y)[i] <= 0`.
Accordingly, the likelihood contribution is zero at the quadrature nodes where `std_fn(y)[i] <= 0`.
Requires `std_fn(z_obs) > 0`.

## Expectations
For `y[i] ~ Normal(μ, s²)` the expectations `E[p(z_i|y_i)^q]`, `q ∈ {1, 2}`, are computed by a Gauss-Hermite
quadrature around the Gaussian proposal `∝ Normal(y; μ, s²) Normal(z_i; y, σ₀²/q)`, where `σ₀` is a reference deviation found by a few fixed-point steps started from `std_fn(z_obs)[i]`.
The quadrature is thus exact for a constant `std_fn` and accurate even for `s ≫ std_fn(y)`.
"""
struct HeteroscedasticNormalLikelihood{F} <: Likelihood
    z_obs::Vector{Float64}
    std_fn::F
    std_ref::Vector{Float64}
    nodes::Vector{Float64}
    log_weights::Vector{Float64}

    function HeteroscedasticNormalLikelihood(z_obs::AbstractVector{<:Real}, std_fn::F; quad_nodes::Int=31) where {F}
        z_obs = Vector{Float64}(z_obs)
        _check_elementwise(std_fn, z_obs)
        std_ref = Vector{Float64}(std_fn(z_obs))
        @assert length(std_ref) == length(z_obs) "`std_fn` must return one deviation per observation dimension."
        @assert all(>(0), std_ref) "`std_fn(z_obs)` must be positive."
        nodes, weights = _gauss_hermite(quad_nodes)
        return new{F}(z_obs, std_fn, std_ref, nodes, log.(weights ./ sqrt(π)))
    end
end
function HeteroscedasticNormalLikelihood(; z_obs, std_fn, quad_nodes=31)
    return HeteroscedasticNormalLikelihood(z_obs, std_fn; quad_nodes)
end

likelihood_kind(::HeteroscedasticNormalLikelihood) = Marginalizable()

function _check_elementwise(std_fn, z_obs::Vector{Float64})
    length(z_obs) > 1 || return
    odd = isodd.(eachindex(z_obs))
    σ0 = std_fn(z_obs)
    for mask in (odd, .!odd)
        σ = std_fn(z_obs .+ mask)
        isequal(σ[.!mask], σ0[.!mask]) || throw(ArgumentError("`std_fn` must act elementwise."))
    end
end

"Gauss-Hermite nodes and weights (physicists' convention) by the Golub-Welsch algorithm."
function _gauss_hermite(n::Int)
    E = eigen(SymTridiagonal(zeros(n), sqrt.((1:n-1) ./ 2)))
    return E.values, sqrt(π) .* (E.vectors[1, :] .^ 2)
end

"Number of fixed-point steps moving the reference deviation `σ₀` towards the bulk of the integrand."
const PROPOSAL_ITERS = 3

_lognormpdf(z::Real, y::Real, σ::Real) = σ > 0 ? -(((z - y) / σ)^2 + log(2π)) / 2 - log(σ) : -Inf

function loglike_marginal(like::HeteroscedasticNormalLikelihood, y::AbstractVector{<:Real})
    return _lognormpdf.(like.z_obs, y, like.std_fn(y))
end

"Mean `m` and variance `v` of the Gaussian proportional to `Normal(y; μ, s²) Normal(z; y, τ²)`."
function _gauss_product(z, μ, s², τ²)
    return (μ .* τ² .+ z .* s²) ./ (s² .+ τ²), s² .* τ² ./ (s² .+ τ²)
end

"Returns `log E[p(z_i|y_i)^q]` for each dimension `i`, where `y_i ~ Normal(μ_i, std_i²)`."
function _log_lik_moment(like::HeteroscedasticNormalLikelihood, μ::AbstractVector{<:Real}, std::AbstractVector{<:Real}, q::Int)
    z = like.z_obs
    s² = std .^ 2

    σ0 = like.std_ref
    for _ in 1:PROPOSAL_ITERS
        m, _ = _gauss_product(z, μ, s², σ0 .^ 2 ./ q)
        σm = like.std_fn(m)
        σ0 = ifelse.(σm .> 0, σm, σ0)
    end
    τ² = σ0 .^ 2 ./ q
    m, v = _gauss_product(z, μ, s², τ²)

    Y = m .+ sqrt.(2 .* v) .* like.nodes'
    σ = reduce(hcat, like.std_fn.(eachcol(Y)))
    log_g = q .* _lognormpdf.(z, Y, σ)
    log_ref = logpdf.(Normal.(Y, sqrt.(τ²)), z)
    log_Z = logpdf.(Normal.(μ, sqrt.(s² .+ τ²)), z)
    return log_Z .+ logsumexp.(eachrow(like.log_weights' .+ log_g .- log_ref))
end

function log_marginal_likelihood_mean(::GaussianPredictive, like::HeteroscedasticNormalLikelihood, model_post::ModelPosterior)
    function log_ml_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        return _log_lik_moment(like, μ_y, std_y, 1)
    end
    function log_ml_mean(X::AbstractMatrix{<:Real})
        μs_y, stds_y = mean_and_std(model_post, X)
        return reduce(hcat, _log_lik_moment.(Ref(like), eachcol(μs_y), eachcol(stds_y), 1))
    end
    return log_ml_mean
end

function log_sq_likelihood_mean(::GaussianPredictive, like::HeteroscedasticNormalLikelihood, model_post::ModelPosterior)
    function log_sq_like_mean(x::AbstractVector{<:Real})
        μ_y, std_y = mean_and_std(model_post, x)
        return sum(_log_lik_moment(like, μ_y, std_y, 2))
    end
    function log_sq_like_mean(X::AbstractMatrix{<:Real})
        μs_y, stds_y = mean_and_std(model_post, X)
        return sum.(_log_lik_moment.(Ref(like), eachcol(μs_y), eachcol(stds_y), 2))
    end
    return log_sq_like_mean
end

struct _SubsetStd{F}
    std_fn::F
    z_obs::Vector{Float64}
    y_set::Vector{Bool}
end

function (s::_SubsetStd)(y::AbstractVector{<:Real})
    y_full = copy(s.z_obs)
    y_full[s.y_set] .= y
    return s.std_fn(y_full)[s.y_set]
end

function get_subset(like::HeteroscedasticNormalLikelihood, y_set::AbstractVector{<:Bool})
    return HeteroscedasticNormalLikelihood(
        like.z_obs[y_set],
        _SubsetStd(like.std_fn, like.z_obs, collect(y_set));
        quad_nodes = length(like.nodes),
    )
end
