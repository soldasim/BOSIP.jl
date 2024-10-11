
"""
    bolfi!(::BolfiProblem; kwargs...)

Run the BOLFI method on the given `BolfiProblem`.

The `bolfi!` function is a wrapper for `BOSS.bo!`,
which implements the underlying Bayesian optimization procedure.

# Arguments

- `problem::BolfiProblem`: Defines the inference problem,
        together with all model hyperparameters.

# Keywords

- `acquisition::BolfiAcquisition`: Defines the acquisition function.
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
    acquisition::BolfiAcquisition = PostVarAcq(),
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    term_cond::Union{<:TermCond, <:BolfiTermCond} = IterLimit(1),
    options::BolfiOptions = BolfiOptions(),    
)
    boss_acq = AcqWrapper(acquisition, bolfi, options)
    boss_term_cond = TermCondWrapper(term_cond, bolfi)
    boss_options = create_boss_options(options, bolfi)

    bo!(bolfi.problem;
        acquisition = boss_acq,
        model_fitter,
        acq_maximizer,
        term_cond = boss_term_cond,
        options = boss_options,
    )
    return bolfi
end
