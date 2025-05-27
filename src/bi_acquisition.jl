
# Default fallback for `MultiFittedParams` that occur when Bayesian inference is used.
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
