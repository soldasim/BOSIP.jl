
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

# Default fallback for `MultiFittedParams`.
# Constructs a separate acquisition function for each sample and averages them.
function (acq::BolfiAcquisition)(::Type{<:MultiFittedParams}, bolfi::BolfiProblem{<:Any}, options::BolfiOptions)
    params = bolfi.problem.params
    sample_count = length(params)
    
    # create shallow copies of the problem
    bolfis = [shallow_copy(bolfi) for _ in 1:sample_count]
    problems = [shallow_copy(bolfi.problem) for _ in 1:sample_count]

    # change pointers to data
    for i in 1:sample_count
        bolfis[i].problem = problems[i]
        problems[i].params = params[i]
    end

    # average posterior variance over the samples
    acqs = acq.(Ref(UniFittedParams), bolfis, Ref(options))
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
