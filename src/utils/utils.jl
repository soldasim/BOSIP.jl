
function shallow_copy(x::T) where {T}
    @assert isstructtype(T)
    return T.name.wrapper(getfield.(Ref(x), fieldnames(T))...)
end

function logsumexp(x::AbstractVector{<:Real})
    x_max = maximum(x)
    isinf(x_max) && return x_max
    return x_max + log(sum(exp.(x .- x_max)))
end

function logmeanexp(x::AbstractVector{<:Real})
    return logsumexp(x) - log(length(x))
end
