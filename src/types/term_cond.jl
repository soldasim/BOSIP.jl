
"""
An abstract type for BOSIP termination conditions.

# Implementing custom termination condition:
- Create struct `CustomTermCond <: BosipTermCond`
- Implement method `(::CustomTermCond)(::BosipProblem) -> ::Bool`
"""
abstract type BosipTermCond end

struct TermCondWrapper{
    T<:BosipTermCond
} <: TermCond
    term_cond::T
    bosip::BosipProblem
end

TermCondWrapper(term_cond::TermCond, ::BosipProblem) = term_cond

(wrap::TermCondWrapper)(::BossProblem) = wrap.term_cond(wrap.bosip)
