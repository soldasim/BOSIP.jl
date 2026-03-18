
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

function construct_acquisition(wrap::AcqWrapper, boss::BossProblem, ::BossOptions)
    @assert wrap.bosip.problem === boss
    # This assert succeeds even if the BossProblem is deep-copied in BOSS.jl (e.g. in SequentialBatchAM),
    # since BossProblem.acquisition is this AcqWrapper and AcqWrapper.bosip is the BosipProblem,
    # so the BosipProblem is recursively copied as well.
    # This way, the whole structure remains consistent while not affecting the original BosipProblem.
    # This is a bit convoluted, but a wanted behavior.

    return wrap.acq(wrap.bosip, wrap.options)
end
