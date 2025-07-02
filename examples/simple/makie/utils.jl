
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
