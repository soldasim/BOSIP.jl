
"""
    bosip!(::BosipProblem; kwargs...)

Run the BOSIP method on the given `BosipProblem`.

The `bosip!` function is a wrapper for `BOSS.bo!`,
which implements the underlying Bayesian optimization procedure.

## Arguments

- `problem::BosipProblem`: Defines the inference problem,
        together with all model hyperparameters.

## Keywords

- `model_fitter::BOSS.ModelFitter`: Defines the algorithm
        used to estimate the model hyperparameters.
- `acq_maximizer::BOSS.AcquisitionMaximizer`: Defines the algorithm
        used to maximize the acquisition function in order to
        select the next evaluation point in each iteration.
- `term_cond::Union{<:BOSS.TermCond, <:BosipTermCond}`: Defines
        the termination condition of the whole procedure.
- `options::BosipOptions`: Can be used to specify additional
        miscellaneous options.

## References

`BOSS.bo!`,
[`BosipProblem`](@ref),
[`BosipAcquisition`](@ref),
`BOSS.ModelFitter`,
`BOSS.AcquisitionMaximizer`,
`BOSS.TermCond`,
[`BosipTermCond`](@ref),
[`BosipOptions`](@ref)

## Examples

See 'https://soldasim.github.io/BOSIP.jl/stable/example_lfi' for example usage.

"""
function bosip!(bosip::BosipProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::Union{<:TermCond, <:BosipTermCond} = IterLimit(1),
    options::BosipOptions = BosipOptions(),    
)
    # TODO the wrapper system is unnecessarily convoluted
    # unify the API with BOSS to simplify the code

    _init_problem!(bosip, options)

    boss_term_cond = TermCondWrapper(term_cond, bosip)
    boss_options = create_boss_options(options, bosip)

    bo!(bosip.problem;
        model_fitter,
        acq_maximizer,
        term_cond = boss_term_cond,
        options = boss_options,
    )
    return bosip
end

function _init_problem!(bosip::BosipProblem, options::BosipOptions)
    # replace the default options
    bosip.problem.acquisition.options = options

    return bosip
end

"""
    estimate_parameters!(::BosipProblem, ::ModelFitter)

Estimate the hyperparameters of the model.
Uses the provided `ModelFitter` to fit the hyperparameters of the model according to the data stored in the `BosipProblem`.

## Keywords
- `options::BosipOptions`: Defines miscellaneous settings.

"""
function estimate_parameters!(bosip::BosipProblem, model_fitter::ModelFitter; options::BosipOptions=BosipOptions())
    boss_options = create_boss_options(options, bosip)
    estimate_parameters!(bosip.problem, model_fitter; options=boss_options)
end

"""
    x = maximize_acquisition(::BosipProblem, ::AcquisitionMaximizer)

Select parameters for the next simulation.
Uses the provided `AcquisitionMaximizer` to maximize the acquisition function and find the optimal candidate parameters.

## Keywords

- `options::BosipOptions`: Defines miscellaneous settings.
"""
function maximize_acquisition(bosip::BosipProblem, acq_maximizer::AcquisitionMaximizer; options::BosipOptions=BosipOptions())
    boss_options = create_boss_options(options, bosip)
    return maximize_acquisition(bosip.problem, acq_maximizer; options=boss_options)
end

"""
    eval_objective!(::BosipProblem, x::AbstractVector{<:Real})

Evaluate the blackbox simulation for the given parameters `x`.

# Keywords

- `options::BosipOptions`: Defines miscellaneous settings.
"""
function eval_objective!(bosip::BosipProblem, x::AbstractVector{<:Real}; options::BosipOptions=BosipOptions())
    boss_options = create_boss_options(options, bosip)
    eval_objective!(bosip.problem, x; options=boss_options)
end
