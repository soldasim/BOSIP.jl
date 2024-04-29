
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


# - - - Confidence Intervals - - - - -

struct ConfidenceTermCond{
    I<:Union{IterLimit, NoLimit},
    X<:Union{Nothing, <:AbstractMatrix{<:Real}},
} <: BolfiTermCond
    iter_limit::I
    samples::Int
    xs::X
    q::Float64
    r::Float64
end
ConfidenceTermCond(;
    max_iters = nothing,
    samples = 10_000,
    xs = nothing,
    q = 0.95,
    r = 0.95,
) = ConfidenceTermCond(IterLimit(max_iters), samples, xs, q, r)

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Nothing})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratio = calculate(cond, bolfi)
    return ratio < cond.r
end

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Matrix{Bool}})
    cond.iter_limit(bolfi.problem) || return false
    (bolfi.problem.data isa ExperimentDataPrior) && return true
    ratios = calculate.(Ref(cond), get_subset.(Ref(bolfi), eachcol(bolfi.y_sets)))
    return any(ratios .< cond.r)
end

function calculate(cond::ConfidenceTermCond, bolfi::BolfiProblem)
    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_med = gp_mean(gp_post)

    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end
    _, _, V_mean = find_cutoff(gp_post, bolfi.var_e, bolfi.x_prior, cond.q; xs)  # with GP uncertainty
    _, _, V_med = find_cutoff(gp_med, bolfi.var_e, bolfi.x_prior, cond.q; xs)  # without GP uncertainty

    return V_med / V_mean
end


# - - - UB-LB Confidence - - - - -

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
    q = 0.95,
    r = 0.95,
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
    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_lb = gp_bound(gp_post, -cond.n)
    gp_ub = gp_bound(gp_post, +cond.n)

    if isnothing(cond.xs)
        xs = rand(bolfi.x_prior, cond.samples)
    else
        xs = cond.xs
    end
    f_lb, c_lb, V_lb = find_cutoff(gp_lb, bolfi.var_e, bolfi.x_prior, cond.q; xs)
    f_ub, c_ub, V_ub = find_cutoff(gp_ub, bolfi.var_e, bolfi.x_prior, cond.q; xs)

    in_lb = (f_lb.(eachcol(xs)) .> c_lb)
    in_ub = (f_ub.(eachcol(xs)) .> c_ub)
    return set_overlap(in_lb, in_ub, bolfi.x_prior, xs)
end
