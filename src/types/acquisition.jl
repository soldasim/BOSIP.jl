
"""
An abstract type for BOSIP acquisition functions.

### **Required** API for subtypes of `BosipAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:UniFittedParams}, ::BosipProblem, ::BosipOptions) -> (x -> ::Real)`.

### **Optional** API for subtypes of `BosipAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:MultiFittedParams}, ::BosipProblem, ::BosipOptions) -> (x -> ::Real)`.
    A default fallback is provided for `MultiFittedParams`, which averages individual acquisition functions for each sample.
"""
abstract type BosipAcquisition end

# Broadcast between MAP and BI parameters.
function (acq::BosipAcquisition)(bosip::BosipProblem{<:Any}, options::BosipOptions)
    return acq(typeof(bosip.problem.params), bosip, options)
end

"""
A wrapper around any `BosipAcquisition` function converting it to the BOSS.jl `AcquisitionFunction`.
"""
mutable struct AcqWrapper{
    A<:BosipAcquisition
} <: AcquisitionFunction
    acq::A
    bosip::BosipProblem
    options::BosipOptions
end

BOSS.construct_acquisition(wrap::AcqWrapper, ::BossProblem, ::BossOptions) =
    wrap.acq(wrap.bosip, wrap.options)
