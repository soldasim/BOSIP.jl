
"""
    GradientExperimentData(X, Y, dY)

Stores experiment data along with gradient observations for use with
[`GradientGaussianProcess`](@ref).

## Fields

- `X::AbstractMatrix{<:Real}`: Input data, shape `x_dim × n`.
- `Y::AbstractMatrix{<:Real}`: Function value data, shape `y_dim × n`.
- `dY::AbstractMatrix{<:Real}`: Gradient data, shape `(y_dim * x_dim) × n`.
  Each column `dY[:, j]` is the stacked row-wise Jacobian at `X[:, j]`:
  `dY[:, j] = [∇y₁ᵀ; ∇y₂ᵀ; ...; ∇y_{y_dim}ᵀ]`
  where `∇yᵢ = [∂yᵢ/∂x₁, ..., ∂yᵢ/∂x_d] ∈ Rᵈ`.

The simulator `f(x)` should return `(y, ∇y)` where `∇y = vec(J')` and
`J = ForwardDiff.jacobian(f_y, x)` is the `y_dim × x_dim` Jacobian.

## See Also

[`GradientGaussianProcess`](@ref)
"""
struct GradientExperimentData{
    XT<:AbstractMatrix{<:Real},
    YT<:AbstractMatrix{<:Real},
    DYT<:AbstractMatrix{<:Real},
} <: ExperimentData
    X::XT
    Y::YT
    dY::DYT

    function GradientExperimentData(X::XT, Y::YT, dY::DYT) where {XT, YT, DYT}
        @assert size(X, 2) == size(Y, 2) == size(dY, 2)
        return new{XT, YT, DYT}(X, Y, dY)
    end
end

"""
    augment_dataset(::GradientExperimentData, x, (y, ∇y))

Called automatically by BOSS.jl when `f(x)` returns a `(y, ∇y)` tuple.
Appends the new function value `y` and vectorized Jacobian `∇y` to the dataset.
"""
function BOSS.augment_dataset(
    data::GradientExperimentData,
    x::AbstractVecOrMat{<:Real},
    y_and_grad::Tuple,
)
    y, ∇y = y_and_grad
    return GradientExperimentData(
        hcat(data.X, x),
        hcat(data.Y, y),
        hcat(data.dY, ∇y),
    )
end

"""
    augment_dataset!(::BossProblem, x, (y, ∇y))

Extends BOSS.augment_dataset! to handle the `(y, ∇y)` tuple returned by simulators
used with GradientGaussianProcess. Called automatically by BOSS.jl's eval_objective!.
"""
function BOSS.augment_dataset!(problem::BossProblem, x::AbstractVector{<:Real}, y_and_grad::Tuple)
    problem.data = BOSS.augment_dataset(problem.data, x, y_and_grad)
    return nothing
end

function BOSS.update_dataset(
    ::GradientExperimentData,
    X::AbstractMatrix{<:Real},
    Y_and_dY::Tuple,
)
    Y, dY = Y_and_dY
    return GradientExperimentData(X, Y, dY)
end

"""
    slice(::GradientExperimentData, idx)

Extract data for output dimension `idx`. Called by BOSS.jl's sliceable model machinery.
After slicing, `data.dY` has shape `x_dim × n` and contains the gradient of output `idx`.
"""
function BOSS.slice(data::GradientExperimentData, idx::Int)
    d = BOSS.x_dim(data)
    return GradientExperimentData(
        data.X,
        data.Y[idx:idx, :],
        data.dY[((idx-1)*d + 1):(idx*d), :],
    )
end
