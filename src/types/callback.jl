
"""
If a callback `cb` of type `BosipCallback` is defined in `BosipOptions`,
the method `cb(::BosipProblem; kwargs...)` will be called in every iteration.

```
cb(problem::BosipProblem;
    model_fitter::BOSS.ModelFitter,
    acq_maximizer::BOSS.AcquisitionMaximizer,
    term_cond::TermCond,                        # either `BOSS.TermCond` or a `BosipTermCond` wrapped into `TermCondWrapper`
    options::BossOptions,
    first::Bool,
)
```
"""
abstract type BosipCallback end

"""
Combines multiple `BosipCallback`s.
"""
struct CombinedCallback <: BosipCallback
    callbacks::Vector{BosipCallback}
end
CombinedCallback(cbs...) = CombinedCallback([cbs...])
CombinedCallback(cb::BosipCallback) = CombinedCallback([cb])

function (comb::CombinedCallback)(bosip::BosipProblem; kwargs...)
    for cb in comb.callbacks
        cb(bosip; kwargs...)
    end
end

"""
Wrapper for BOSS around `BosipCallback`.
"""
struct CallbackWrapper{
    CB<:BosipCallback
} <: BossCallback
    callback::CB
    bosip::BosipProblem
end
CallbackWrapper(cb::BossCallback, ::BosipProblem) = cb

(wrap::CallbackWrapper)(::BossProblem; kwargs...) = wrap.callback(wrap.bosip; kwargs...)

"""
    NoCallback()

A dummy `BosipCallback` which does nothing.
"""
struct NoCallback <: BosipCallback end

function (::NoCallback)(::BosipProblem; kwargs...) end
