
"""
    AEConfidence(; kwargs...)

Calculates the `q`-confidence region of the expected and the approximate posteriors.
Terminates after the IoU of the two confidence regions surpasses `r`.
"""
struct AEConfidence{
    I<:Union{IterLimit, NoLimit},
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BolfiTermCond
    iter_limit::I
    samples::Int
    xs::X
    q::Float64
    r::Float64
end
function AEConfidence(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    q = 0.95,
    r = 0.95,
)
    iter_limit = isnothing(max_iters) ? NoLimit() : IterLimit(max_iters)
    return AEConfidence(iter_limit, samples, xs, q, r)
end

function (cond::AEConfidence)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratio = calculate(cond, bolfi)
    return ratio < cond.r
end

function (cond::AEConfidence)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::AEConfidence, bolfi::BolfiProblem)
    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end

    gp_post = BOSS.model_posterior(bolfi.problem)

    f_approx = approx_posterior(gp_post, bolfi.x_prior, bolfi.std_obs; xs,)
    f_expect = posterior_mean(gp_post, bolfi.x_prior, bolfi.std_obs; xs,)
    f_approx, c_approx = find_cutoff(f_approx, bolfi.x_prior, cond.q; xs)
    f_expect, c_expect = find_cutoff(f_expect, bolfi.x_prior, cond.q; xs)

    in_approx = (f_approx.(eachcol(xs)) .> c_approx)
    in_expect = (f_expect.(eachcol(xs)) .> c_expect)
    return set_iou(in_approx, in_expect, bolfi.x_prior, xs)
end
