
"""
    VarDiff()

Selects the next evaluation point by maxmizing the reduction in the variance
caused by adding this point to the dataset.

The reduction is calculated by adding a speculative data point to the dataset
with the output value equal to the current mean prediction.
"""
struct VarDiff <: BosipAcquisition end

function (acq::VarDiff)(::Type{<:UniFittedParams}, bosip::BosipProblem{Nothing}, options::BosipOptions)
    @warn """
        `VarDiff` acquisition cannot be evaluated in parallel!
        Make sure to set `parallel=false` in the settings of your `AcquisitionMaximizer`.
    """
    # current data
    post_var = posterior_variance(bosip)
    model_post = model_posterior(bosip.problem)
    data = bosip.problem.data
    
    # augmented data
    bosip_ = shallow_copy(bosip)
    problem_ = shallow_copy(bosip.problem)
    bosip_.problem = problem_
    
    function var_diff(x_::AbstractVector{<:Real})
        y_ = mean(model_post, x_)
        bosip_.problem.data = ExperimentData(hcat(data.X, x_), hcat(data.Y, y_))
        post_var_ = posterior_variance(bosip_)

        return post_var(x_) - post_var_(x_)
    end
end
