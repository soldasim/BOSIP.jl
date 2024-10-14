
function true_post(x)
    y = ToyProblem.experiment(x; noise_std=zeros(ToyProblem.y_dim))

    ll = pdf(MvNormal(y, ToyProblem.σe_true), ToyProblem.y_obs)
    pθ = pdf(ToyProblem.get_x_prior(), x)
    return pθ * ll
end

function normalize_values!(vals::AbstractArray{<:Real})
    min, max = minimum(vals), maximum(vals)
    vals .-= min
    vals ./= (max - min)
end

function calculate_values(func, grid)
    grid = collect(grid)
    vals = zeros(size(grid))
    Threads.@threads for i in eachindex(grid)
        vals[i] = func(grid[i])
    end
    return vals
end

unwrap(acquisition::BolfiAcquisition) = acquisition
unwrap(acquisition::BOLFI.AcqWrapper) = acquisition.acq
