
"""
    bolfi!(::BolfiProblem; kwargs...)

Run the BOLFI method on the given `BolfiProblem`.

The `bolfi!` function is a wrapper for `BOSS.bo!`,
which implements the underlying Bayesian optimization procedure.

# Arguments

- `problem::BolfiProblem`: Defines the inference problem,
        together with all model hyperparameters.

# Keywords

- `model_fitter::BOSS.ModelFitter`: Defines the algorithm
        used to estimate the model hyperparameters.
- `acq_maximizer::BOSS.AcquisitionMaximizer`: Defines the algorithm
        used to maximize the acquisition function in order to
        select the next evaluation point in each iteration.
- `term_cond::Union{<:BOSS.TermCond, <:BolfiTermCond}`: Defines
        the termination condition of the whole procedure.
- `options::BolfiOptions`: Can be used to specify additional
        miscellaneous options.

# References

`BOSS.bo!`,
[`BolfiProblem`](@ref),
[`BolfiAcquisition`](@ref),
`BOSS.ModelFitter`,
`BOSS.AcquisitionMaximizer`,
`BOSS.TermCond`,
[`BolfiTermCond`](@ref),
[`BolfiOptions`](@ref)

# Examples

See 'https://soldasim.github.io/BOLFI.jl/stable/example_lfi' for example usage.

"""
function bolfi!(bolfi::BolfiProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::Union{<:TermCond, <:BolfiTermCond} = IterLimit(1),
    options::BolfiOptions = BolfiOptions(),    
)
    # TODO the wrapper system is unnecessarily convoluted
    # unify the API with BOSS to simplify the code

    _init_problem!(bolfi, options)

    boss_term_cond = TermCondWrapper(term_cond, bolfi)
    boss_options = create_boss_options(options, bolfi)

    bo!(bolfi.problem;
        model_fitter,
        acq_maximizer,
        term_cond = boss_term_cond,
        options = boss_options,
    )
    return bolfi
end

function _init_problem!(bolfi::BolfiProblem, options::BolfiOptions)
    # replace the default options
    bolfi.problem.acquisition.options = options

    return bolfi
end

"""
    estimate_parameters!(::BolfiProblem, ::ModelFitter)

Estimate the hyperparameters of the model.
Uses the provided `ModelFitter` to fit the hyperparameters of the model according to the data stored in the `BolfiProblem`.

# Keywords

- `options::BolfiOptions`: Defines miscellaneous settings.

"""
function estimate_parameters!(bolfi::BolfiProblem, model_fitter::ModelFitter; options::BolfiOptions=BolfiOptions())
    boss_options = create_boss_options(options, bolfi)
    estimate_parameters!(bolfi.problem, model_fitter; options=boss_options)
end

"""
    x = maximize_acquisition(::BolfiProblem, ::AcquisitionMaximizer)

Select parameters for the next simulation.
Uses the provided `AcquisitionMaximizer` to maximize the acquisition function and find the optimal candidate parameters.

# Keywords

- `options::BolfiOptions`: Defines miscellaneous settings.
"""
function maximize_acquisition(bolfi::BolfiProblem, acq_maximizer::AcquisitionMaximizer; options::BolfiOptions=BolfiOptions())
    boss_options = create_boss_options(options, bolfi)
    return maximize_acquisition(bolfi.problem, acq_maximizer; options=boss_options)
end

"""
    eval_objective!(::BolfiProblem, x::AbstractVector{<:Real})

Evaluate the blackbox simulation for the given parameters `x`.

# Keywords

- `options::BolfiOptions`: Defines miscellaneous settings.
"""
function eval_objective!(bolfi::BolfiProblem, x::AbstractVector{<:Real}; options::BolfiOptions=BolfiOptions())
    boss_options = create_boss_options(options, bolfi)
    eval_objective!(bolfi.problem, x; options=boss_options)
end
