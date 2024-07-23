
"""
An abstract type for BOLFI termination conditions.

# Implementing custom termination condition:
- Create struct `CustomTermCond <: BolfiTermCond`
- Implement method `(::CustomTermCond)(::BolfiProblem) -> ::Bool`
"""
abstract type BolfiTermCond end

struct TermCondWrapper{
    T<:BolfiTermCond
} <: TermCond
    term_cond::T
    bolfi::BolfiProblem
end

TermCondWrapper(term_cond::TermCond, ::BolfiProblem) = term_cond

(wrap::TermCondWrapper)(::BossProblem) = wrap.term_cond(wrap.bolfi)

"""
    NoLimit()

Termination conditions which never terminates.
"""
struct NoLimit <: TermCond end
(::NoLimit)(::BossProblem) = true
