
"""
    PostVarAcq()

Selects the new evaluation point by maximizing the variance of the posterior approximation.
"""
struct PostVarAcq <: BolfiAcquisition end

function (acq::PostVarAcq)(::Type{<:ExperimentDataMAP}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return posterior_variance(bolfi; normalize=false)
end

# Use the specialized `posterior_variance` computation defined for Bayesian inference.
# This is NOT equivalent to the fallback behavior defined in
#   `(acq::BolfiAcquisition)(::Type{<:ExperimentDataBI}, bolfi::BolfiProblem, options::BolfiOptions)`.
function (acq::PostVarAcq)(::Type{<:ExperimentDataBI}, bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    return posterior_variance(bolfi; normalize=false)
end
