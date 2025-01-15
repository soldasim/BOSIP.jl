
function shallow_copy(x::T) where {T}
    @assert isstructtype(T)
    return T(getfield.(Ref(x), fieldnames(T))...)
end
