
"""
    amis = AMIS(; kwargs...)
    xs, ws = amis(log_π, proposal, fitter; kwargs...)

Performs the AMIS (adaptive multiple importance sampling) algorithm.
Returns `(T * N)` importance samples with weights.

# Keywords

- `T::Int64`: Number of iterations.
- `N::Int64`: Number of samples in each iteration.
- `init_q::Union{Nothing, <:MultivariateDistribution}`: Initial proposal distribution
        used only for the 0th iteration. If `init_q = nothing`, then the provided
        `ProposalDistribution` (with the current parameters) is used for the 0th iteration instead.
        Defaults to `nothing`.
"""
@kwdef struct AMIS{
    Q<:Union{Nothing, <:MultivariateDistribution}
}
    T::Int64 = 10
    N::Int64 = 20
    init_q::Q = nothing
end

function (amis::AMIS)(log_π, q::ProposalDistribution, fitter::DistributionFitter;
    options::BosipOptions = BosipOptions(),
)
    x_dim_ = x_dim(q)
    N, T = amis.N, amis.T

    if isnothing(amis.init_q)
        qs = [deepcopy(q) for _ in 1:T+1]
    else
        qs = vcat(amis.init_q, [deepcopy(q) for _ in 1:T])
    end
    
    xs = zeros(x_dim_, N, T+1)
    log_P = zeros(N, T+1)
    Δ = zeros(N, T+1)
    log_Ω = zeros(N, T+1)

    options.info && (prog = Progress(T+1; desc="AMIS: "))

    # t = 0
    xs[:, :, 1] = rand(qs[1], N)

    # log_P[:, 1] = log_π.(eachcol(xs[:, :, 1]))
    log_P[:, 1] = log_π(xs[:, :, 1])
    Δ[:, 1] .= pdf.(Ref(qs[1]), eachcol(xs[:, :, 1]))
    log_Ω[:, 1] .= log_P[:, 1] .- log.(Δ[:, 1])

    options.info && next!(prog)

    # t = 1,...,T
    for t in 2:T+1
        # fit the next proposal distribution based on all the samples drawn so far
        fit_distribution!(fitter, qs[t], collect_samples(xs, 1:t-1), get_weights(log_Ω, 1:t-1); options)

        # draw new samples
        xs[:, :, t] = rand(qs[t], N)

        # calculate new weights
        # log_P[:, t] = log_π.(eachcol(xs[:, :, t]))
        log_P[:, t] = log_π(xs[:, :, t])
        for i in 1:t
            Δ[:, t] .+= pdf.(Ref(qs[i]), eachcol(xs[:, :, t]))
        end
        log_Ω[:, t] .= log_P[:, t] .- log.(Δ[:, t] ./ t)

        # update old weights
        for i in 1:t-1
            Δ[:, i] .+= pdf.(Ref(qs[t]), eachcol(xs[:, :, i]))
            log_Ω[:, i] .= log_P[:, i] .- log.(Δ[:, i] ./ t)
        end

        options.info && next!(prog)
    end

    xs_ = collect_samples(xs, 2:T+1)
    ws_ = get_weights(log_Ω, 2:T+1)
    return xs_, ws_
end

"""
Return all samples from the given iterations `ts`.
"""
function collect_samples(xs::AbstractArray{<:Real, 3}, ts::UnitRange)
    x_dim_, N, _ = size(xs)
    return reshape(xs[:, :, ts], x_dim_, length(ts) * N)
end

"""
Return weights of the samples from the iterations `ts`.
"""
function collect_weights(Ω::AbstractArray{<:Real, 2}, ts::UnitRange)
    N, _ = size(Ω)
    return reshape(Ω[:, ts], length(ts) * N)
end

"""
Get weights of the samples from the iteration `ts`.

Fall back to uniform sampling if all weights are zero.
(That is, return the inverse proposal probabilities of the samples as weights.)
"""
function get_weights(log_Ω::AbstractArray{<:Real, 2}, ts::UnitRange)
    log_ws = collect_weights(log_Ω, ts)

    if any(isnan.(log_ws)) || any(log_ws .== Inf) || all(log_ws .== -Inf)
        throw(ErrorException("AMIS: Numerical issues with sample weights."))
    end
    # if all(log_ws .== -Inf)
    #     @warn "All weights are zero. Falling back to uniform sampling."
    #     @assert false # TODO rem
    #     log_ws = 0. .- log.(collect_weights(Δ, ts))
    # end

    ws = exp_weights(log_ws)
    return ws
end

"""
Exponentiate the given weights in a numerically stable way.
"""
function exp_weights(log_ws)
    M = maximum(log_ws)
    (M == -Inf) && throw(ErrorException("Sampling failed due to numerical issues with sample weights."))

    log_ws .-= M
    ws = exp.(log_ws)
    ws ./= sum(ws)
    return ws
end
