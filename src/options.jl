
# - - - Callbacks - - - - -

"""
If a callback `cb` of type `BolfiCallback` is defined in `BolfiOptions`,
the method `cb(::BolfiProblem; kwargs...)` will be called after every iteration.
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
Miscellaneous options. See `BOSS.BossOptions` for more info.
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

BOSS.BossOptions(opt::BolfiOptions, bolfi::BolfiProblem) = BOSS.BossOptions(
    opt.info,
    opt.debug,
    opt.parallel_evals,
    CallbackWrapper(opt.callback, bolfi),
)
