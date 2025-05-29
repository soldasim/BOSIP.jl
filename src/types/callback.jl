
"""
If a callback `cb` of type `BolfiCallback` is defined in `BolfiOptions`,
the method `cb(::BolfiProblem; kwargs...)` will be called in every iteration.

```
cb(problem::BolfiProblem;
    model_fitter::BOSS.ModelFitter,
    acq_maximizer::BOSS.AcquisitionMaximizer,
    term_cond::TermCond,                        # either `BOSS.TermCond` or a `BolfiTermCond` wrapped into `TermCondWrapper`
    options::BossOptions,
    first::Bool,
)
```
"""
abstract type BolfiCallback end

"""
Combines multiple `BolfiCallback`s.
"""
struct CombinedCallback <: BolfiCallback
    callbacks::Vector{BolfiCallback}
end
CombinedCallback(cbs...) = CombinedCallback([cbs...])
CombinedCallback(cb::BolfiCallback) = CombinedCallback([cb])

function (comb::CombinedCallback)(bolfi::BolfiProblem; kwargs...)
    for cb in comb.callbacks
        cb(bolfi; kwargs...)
    end
end

"""
Wrapper for BOSS around `BolfiCallback`.
"""
struct CallbackWrapper{
    CB<:BolfiCallback
} <: BossCallback
    callback::CB
    bolfi::BolfiProblem
end
CallbackWrapper(cb::BossCallback, ::BolfiProblem) = cb

(wrap::CallbackWrapper)(::BossProblem; kwargs...) = wrap.callback(wrap.bolfi; kwargs...)

"""
    NoCallback()

A dummy `BolfiCallback` which does nothing.
"""
struct NoCallback <: BolfiCallback end

function (::NoCallback)(::BolfiProblem; kwargs...) end
