
# - - - Callbacks - - - - -

"""
If a callback `cb` of type `BolfiCallback` is defined in `BolfiOptions`,
the method `cb(::BolfiProblem; kwargs...)` will be called in every iteration.

```
cb(problem::BolfiProblem;
    model_fitter::ModelFitter,
    acq_maximizer::AcquisitionMaximizer,
    acquisition::AcqWrapper,                # `BolfiAcquisition` wrapped into `AcqWrapper`
    term_cond::TermCond,                    # either `TermCond` or a `BolfiTermCond` wrapped into `TermCondWrapper`
    options::BossOptions,
    first::Bool,
)
```
"""
abstract type BolfiCallback end

struct CallbackWrapper{
    CB<:BolfiCallback
} <: BossCallback
    callback::CB
    bolfi::BolfiProblem
end

CallbackWrapper(callback::BossCallback, ::BolfiProblem) = callback

(wrap::CallbackWrapper)(::BossProblem; kwargs...) = wrap.callback(wrap.bolfi; kwargs...)


# - - - Bolfi Options - - - - -

"""
Stores miscellaneous settings.

# Keywords
- `info::Bool`: Setting `info=false` silences the algorithm.
- `debug::Bool`: Set `debug=true` to print stactraces of caught optimization errors.
- `parallel_evals::Symbol`: Possible values: `:serial`, `:parallel`, `:distributed`. Defaults to `:parallel`.
        Determines whether to run multiple objective function evaluations
        within one batch in serial, parallel, or distributed fashion.
        (Only has an effect if batching AM is used.)
- `callback::Union{<:BossCallback, <:BolfiCallback}`: If provided,
        the callback will be called before the BO procedure starts and after every iteration.
"""
struct BolfiOptions{
    CB<:Union{<:BossCallback, <:BolfiCallback},
}
    info::Bool
    debug::Bool
    parallel_evals::Symbol
    callback::CB
end
BolfiOptions(;
    info = true,
    debug = false,
    parallel_evals = :parallel,
    callback = NoCallback(),
) = BolfiOptions(info, debug, parallel_evals, callback)

create_boss_options(opt::BolfiOptions, bolfi::BolfiProblem) = BOSS.BossOptions(
    opt.info,
    opt.debug,
    opt.parallel_evals,
    CallbackWrapper(opt.callback, bolfi),
)
