
abstract type BolfiTermCond end

struct TermCondWrapper{
    T<:BolfiTermCond
} <: TermCond
    term_cond::T
    bolfi::BolfiProblem
end

TermCondWrapper(term_cond::TermCond, ::BolfiProblem) = term_cond

(wrap::TermCondWrapper)(::BossProblem) = wrap.term_cond(wrap.bolfi)


# - - - Common Utils - - - - -

struct NoLimit <: TermCond end
BOSS.IterLimit(::Nothing) = NoLimit()
(::NoLimit)(::BossProblem) = true


# - - - Approximation - Expectation Confidence - - - - -

"""
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
AEConfidence(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    q = 0.95,
    r = 0.95,
) = AEConfidence(IterLimit(max_iters), samples, xs, q, r)

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

    f_approx = approx_posterior(gp_post, bolfi.x_prior, bolfi.var_e; xs,)
    f_expect = posterior_mean(gp_post, bolfi.x_prior, bolfi.var_e; xs,)
    f_approx, c_approx = find_cutoff(f_approx, bolfi.x_prior, cond.q; xs)
    f_expect, c_expect = find_cutoff(f_expect, bolfi.x_prior, cond.q; xs)

    in_approx = (f_approx.(eachcol(xs)) .> c_approx)
    in_expect = (f_expect.(eachcol(xs)) .> c_expect)
    return set_iou(in_approx, in_expect, bolfi.x_prior, xs)
end


# - - - UB-LB Confidence - - - - -

"""
Calculates the `q`-confidence region of the UB and LB approximate posterior.
Terminates after the IoU of the two confidence intervals surpasses `r`.
The UB and LB confidence intervals are calculated using the GP mean +- `n` GP stds.
"""
struct UBLBConfidence{
    I<:Union{IterLimit, NoLimit},
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BolfiTermCond
    iter_limit::I
    samples::Int
    xs::X
    n::Float64
    q::Float64
    r::Float64
end
UBLBConfidence(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    n = 1.,
    q = 0.8,
    r = 0.8,
) = UBLBConfidence(IterLimit(max_iters), samples, xs, n, q, r)

function (cond::UBLBConfidence)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratio = calculate(cond, bolfi) 
    return ratio < cond.r
end

function (cond::UBLBConfidence)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::UBLBConfidence, bolfi::BolfiProblem)
    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end

    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_lb = gp_bound(gp_post, -cond.n)
    gp_ub = gp_bound(gp_post, +cond.n)

    f_lb = approx_posterior(gp_lb, bolfi.x_prior, bolfi.var_e; xs)
    f_ub = approx_posterior(gp_ub, bolfi.x_prior, bolfi.var_e; xs)
    f_lb, c_lb = find_cutoff(f_lb, bolfi.x_prior, cond.q; xs)
    f_ub, c_ub = find_cutoff(f_ub, bolfi.x_prior, cond.q; xs)

    in_lb = (f_lb.(eachcol(xs)) .> c_lb)
    in_ub = (f_ub.(eachcol(xs)) .> c_ub)
    return set_iou(in_lb, in_ub, bolfi.x_prior, xs)
end
