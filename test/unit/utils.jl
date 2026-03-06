struct MockSurrogate <: BOSS.SurrogateModel end

struct MockModelPosterior <: BOSS.ModelPosterior{MockSurrogate}
    mean_val::Vector{Float64}
    var_val::Vector{Float64}
end

BOSS.mean(post::MockModelPosterior, ::AbstractVector{<:Real}) = post.mean_val
BOSS.mean(post::MockModelPosterior, X::AbstractMatrix{<:Real}) = repeat(post.mean_val, 1, size(X, 2))
BOSS.var(post::MockModelPosterior, ::AbstractVector{<:Real}) = post.var_val
BOSS.var(post::MockModelPosterior, X::AbstractMatrix{<:Real}) = repeat(post.var_val, 1, size(X, 2))
