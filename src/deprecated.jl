
# PostVarAcq, LogPostVarAcq

export PostVarAcq, LogPostVarAcq

function PostVarAcq(args...; kwargs...)
    Base.depwarn("`PostVarAcq` is deprecated. Use `MaxVar` instead.", :PostVarAcq; force=true)
    return MaxVar(args...; kwargs...)
end

function LogPostVarAcq(args...; kwargs...)
    Base.depwarn("`LogPostVarAcq` is deprecated. Use `LogMaxVar` instead.", :LogPostVarAcq; force=true)
    return LogMaxVar(args...; kwargs...)
end
