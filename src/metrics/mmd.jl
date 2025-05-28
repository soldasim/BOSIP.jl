
"""
    MMDMetric(; kwargs...)

Measures the quality of the posterior approximation by sampling from the true posterior
and the approximate posterior and calculating the Maximum Mean Discrepancy (MMD)
between the two sample sets.

# Keywords
- `kernel::Kernel`: The kernel used to calculate the MMD.
        It is important to choose appropriate lengthscales for the kernel.
"""
@kwdef struct MMDMetric <: DistributionMetric
    kernel::Kernel
end

function calculate_metric(mmd::MMDMetric, true_samples::AbstractMatrix{<:Real}, approx_samples::AbstractMatrix{<:Real};
    options::BolfiOptions = BolfiOptions(),    
)
    return calc_mmd(mmd.kernel, eachcol(true_samples), eachcol(approx_samples))
end

function calc_mmd(kernel::Kernel, X::AbstractVector, Y::AbstractVector)
    val_X = mean(BOSS.kernelmatrix(kernel, X))
    val_Y = mean(BOSS.kernelmatrix(kernel, Y))
    val_XY = mean(BOSS.kernelmatrix(kernel, X, Y))
    return val_X + val_Y - 2*val_XY
end
