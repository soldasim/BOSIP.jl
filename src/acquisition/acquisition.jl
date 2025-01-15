
"""
An abstract type for BOLFI acquisition functions.

### **Required** API for subtypes of `BolfiAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:ExperimentDataMAP}, ::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`.

### **Optional** API for subtypes of `BolfiAcquisition`:

- Implement method `(::CustomAcq)(::Type{<:ExperimentDataBI}, ::BolfiProblem, ::BolfiOptions) -> (x -> ::Real)`.
    A default fallback is provided for `ExperimentDataBI`, which averages individual acquisition functions for each sample.
"""
abstract type BolfiAcquisition end

# Broadcast between MAP and BI parameters.
function (acq::BolfiAcquisition)(bolfi::BolfiProblem{<:Any, <:Any}, options::BolfiOptions)
    return acq(typeof(bolfi.problem.data), bolfi, options)
end

# Default fallback for `ExperimentDataBI`.
# Constructs a separate acquisition function for each sample and averages them.
function (acq::BolfiAcquisition)(::Type{<:ExperimentDataBI}, bolfi::BolfiProblem{<:Any, <:Any}, options::BolfiOptions)
    data = bolfi.problem.data
    sample_count = BOSS.sample_count(data)
    
    # create shallow copies of the problem
    bolfis = [shallow_copy(bolfi) for _ in 1:sample_count]
    problems = [shallow_copy(bolfi.problem) for _ in 1:sample_count]

    # change pointers to data
    for i in 1:sample_count
        bolfis[i].problem = problems[i]
        problems[i].data = BOSS.get_sample(data, i)
    end

    # average posterior variance over the samples
    acqs = acq.(Ref(ExperimentDataMAP), bolfis, Ref(options))
    function exp_acq(x)
        return mapreduce(a -> a(x), +, acqs) / sample_count
    end
end

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
