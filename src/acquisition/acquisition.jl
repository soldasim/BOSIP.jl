
"""
An abstract type for BOLFI acquisition functions.

# Implementing custom acquisition function for BOLFI:
- Create struct `CustomAcq <: BolfiAcquisition`
- Implement method `(::CustomAcq)(::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`
"""
abstract type BolfiAcquisition end

"""
A wrapper around any `BolfiAcquisition` function converting it to the BOSS.jl `AcquisitionFunction`.
"""
struct AcqWrapper{
    A<:BolfiAcquisition
} <: AcquisitionFunction
    acq::A
    bolfi::BolfiProblem
    options::BolfiOptions
end

(wrap::AcqWrapper)(::BossProblem, ::BossOptions) = wrap.acq(wrap.bolfi, wrap.options)
