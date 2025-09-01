
# Default fallback for `MultiFittedParams` that occur when Bayesian inference is used.
# Constructs a separate acquisition function for each sample and averages them.
function (acq::BosipAcquisition)(::Type{<:MultiFittedParams}, bosip::BosipProblem{<:Any}, options::BosipOptions)
    params = bosip.problem.params
    sample_count = length(params)
    
    # create shallow copies of the problem
    bosips = [shallow_copy(bosip) for _ in 1:sample_count]
    problems = [shallow_copy(bosip.problem) for _ in 1:sample_count]

    # change pointers to data
    for i in 1:sample_count
        bosips[i].problem = problems[i]
        problems[i].params = params[i]
    end

    # average posterior variance over the samples
    acqs = acq.(Ref(UniFittedParams), bosips, Ref(options))
    function exp_acq(x)
        return mapreduce(a -> a(x), +, acqs) / sample_count
    end
end
