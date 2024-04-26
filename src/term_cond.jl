
abstract type BolfiTermCond <: TermCond end

struct TermCondWrapper{
    T<:BolfiTermCond
} <: TermCond
    term_cond::T
    bolfi::BolfiProblem
end

TermCondWrapper(term_cond::TermCond, ::BolfiProblem) = term_cond

(wrap::TermCondWrapper)(::BossProblem) = wrap.term_cond(wrap.bolfi)


# - - - Confidence Intervals - - - - -

struct ConfidenceTermCond <: BolfiTermCond
    samples::Int
    q::Float64
    r::Float64
end
ConfidenceTermCond(;
    samples = 10_000,
    q = 0.95,
    r = 0.95,
) = ConfidenceTermCond(samples, q, r)

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Nothing})
    (bolfi.problem.data isa ExperimentDataPrior) && return true

    gp_post = BOSS.model_posterior(bolfi.problem)
    gp_med = gp_quantile(gp_post, 0.5)

    xs = rand(bolfi.x_prior, cond.samples)
    _, _, V_mean = find_cutoff(gp_post, bolfi.x_prior, bolfi.var_e, cond.q; xs)
    _, _, V_med = find_cutoff(gp_med, bolfi.x_prior, bolfi.var_e, cond.q; xs)

    ratio = V_med / V_mean
    return ratio < cond.r
end

function (cond::ConfidenceTermCond)(bolfi::BolfiProblem{Matrix{Bool}})
    return any(
        cond.(
            (get_subset(bolfi, set) for set in eachcol(bolfi.y_sets))
        )
    )
end
