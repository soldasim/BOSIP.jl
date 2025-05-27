
"""
An abstract type for BOLFI acquisition functions.

### **Required** API for subtypes of `BolfiAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:UniFittedParams}, ::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`.

### **Optional** API for subtypes of `BolfiAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:MultiFittedParams}, ::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`.
    A default fallback is provided for `MultiFittedParams`, which averages individual acquisition functions for each sample.
"""
abstract type BolfiAcquisition end

# Broadcast between MAP and BI parameters.
function (acq::BolfiAcquisition)(bolfi::BolfiProblem{<:Any}, options::BolfiOptions)
    return acq(typeof(bolfi.problem.params), bolfi, options)
end

"""
A wrapper around any `BolfiAcquisition` function converting it to the BOSS.jl `AcquisitionFunction`.
"""
mutable struct AcqWrapper{
    A<:BolfiAcquisition
} <: AcquisitionFunction
    acq::A
    bolfi::BolfiProblem
    options::BolfiOptions
end

BOSS.construct_acquisition(wrap::AcqWrapper, ::BossProblem, ::BossOptions) =
    wrap.acq(wrap.bolfi, wrap.options)
