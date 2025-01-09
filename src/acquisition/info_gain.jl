
"""
Selects new data point by maximizing the information gain in the posterior.

The information gain is calculated as the mutual information between
the new data point (a vector-valued random variable from a multivariate distribution given by the GPs)
and the posterior approximation (a "random function" from a infinite-dimensional distribution).

The infinite-dimensional distribution over the possible posteriors
is approximated by a multivariate distribution over a finite set of functional values
at predefined grid points `θ_grid`.

The resulting mutual information between the new data point
and the functional values of the approximate posterior evaluated at `θ_grid`
is estimated by calculating the maximum mean discrepancy (MMD)
between their joint and marginal distributions.
This is also known as  the Hilbert-Schmidt independence criterion (HSIC).

# Kwargs
- `samples::Int64`: The amount of samples drawn from the joint distribution
        to estimate the HSIC value.
- `θ_grid::Matrix{Float64}`: The evaluation grid (a finite set of points
        from the parameter domain). The mutual information is calculated
        between the potential new data point and this grid.
- `δ_kernel::Kernel`: The kernel used for the samples of the new data point.
- `p_kernel::Kernel`: The kernel used for the posterior function value samples.
"""
@kwdef struct InfoGain <: BolfiAcquisition
    samples::Int64
    θ_grid::Matrix{Float64}
    δ_kernel::BOSS.Kernel = BOSS.GaussianKernel()
    p_kernel::BOSS.Kernel = BOSS.GaussianKernel()
end

# info gain on the posterior approximation
function (acq::InfoGain)(bolfi::BolfiProblem{Nothing}, options::BolfiOptions)
    problem = bolfi.problem
    y_dim = BOSS.y_dim(problem)

    # Sample noise variables (makes the resulting acquisition function deterministic)
    ϵs = sample_ϵs(y_dim, acq.samples)
    Es = sample_Es(y_dim, size(acq.θ_grid)[2], acq.samples)

    # Compute HSIC
    return construct_hsic_acquisition(acq, bolfi, ϵs, Es)
end

function sample_ϵs(y_dim, samples)
    d = MvNormal(zeros(y_dim), ones(y_dim))
    ϵs = [rand(d) for _ in 1:samples]
    return ϵs
end

function sample_Es(y_dim, grid_size, samples)
    d = MvNormal(zeros(grid_size), ones(grid_size))
    Es = [rand(d, y_dim) for _ in 1:samples]
    return Es
end

gp_posts(model, data) = BOSS.posterior_gp.(Ref(model), Ref(data), 1:BOSS.y_dim(data))

function construct_hsic_acquisition(
    acq::InfoGain,
    bolfi::BolfiProblem,
    ϵs::AbstractVector{<:AbstractVector{<:Real}},
    Es::AbstractVector{<:AbstractMatrix{<:Real}},
)
    boss = bolfi.problem
    post = BOSS.model_posterior(boss)
    
    log_pθ = logpdf.(Ref(bolfi.x_prior), eachcol(acq.θ_grid))

    function acq_(θ_)
        # sample δs at the new θ_
        μ, σ = post(θ_)
        δs = [μ .+ (σ .* ϵ) for ϵ in ϵs]

        # augment problems
        aug_posts_samples = gp_posts.(
            Ref(boss.model),
            BOSS.augment_dataset!.(
                [deepcopy(boss.data) for _ in 1:acq.samples],
                Ref(θ_),
                δs,
            ),
        )

        # sample Δs at the grid conditioned on the new data point
        # (one Δ sample for each δ_ sample)
        μs_samples = get_means.(aug_posts_samples, Ref(acq.θ_grid))
        ΣLs = get_covs(aug_posts_samples[1], acq.θ_grid) # same for all samples
        Δs = sample_Δ.(μs_samples, Ref(ΣLs), Es)
        
        # calculate Ss (functional values of the posterior) from the Δ samples
        Ss = get_S.(Δs, Ref(bolfi.std_obs), Ref(log_pθ))

        # # calculate MMD
        # δs_shuffled = δs[rand_perm]
        # joint_samples = vcat.(δs, Ss)
        # marginal_samples = vcat.(δs_shuffled, Ss) 
        # kernel = BOSS.GaussianKernel() 
        # return mmd(kernel, joint_samples, marginal_samples)

        # calculate HSIC
        val = hsic(acq.δ_kernel, acq.p_kernel, δs, Ss, acq.samples)
        return val
    end
end

function get_S(
    Δ::AbstractMatrix{<:Real},
    std_obs::AbstractVector{<:Real},
    log_pθ::AbstractVector{<:Real},
)
    y_dim = length(std_obs)
    # Z = inv( sqrt(2π)^y_dim * prod(std_obs) )
    log_Z = (-1) * ( (y_dim * log(sqrt(2π))) + sum(log.(std_obs)) )
    
    # d_k = sum( (δ_k .^ 2) ./ (std_obs .^ 2) )
    Dt = sum.(eachcol((Δ .^ 2) ./ (std_obs .^ 2)))
    # l_k = Z * exp( -(1/2) * d_k )
    log_Lt = log_Z .+ ((-1/2) .* Dt)
    
    w_norm = sum(exp.(log_pθ))
    py = sum(exp.(log_pθ .- log(w_norm) .+ log_Lt))

    # post = (prior / evidence) * likelihood
    St = exp.((log_pθ .- log(py)) .+ log_Lt)
    return St
end

function sample_Δ(
    μs::AbstractVector{<:AbstractVector{<:Real}},
    ΣLs::AbstractVector{<:AbstractMatrix{<:Real}},
    E::AbstractMatrix{<:Real}
)
    Δ = mapreduce((μ, L, ϵ) -> (μ + L * ϵ)', vcat, μs, ΣLs, eachcol(E))
    return Δ
end

function get_means(post_gps, θ_grid::AbstractMatrix{<:Real})
    μs = BOSS.mean.(post_gps, Ref(eachcol(θ_grid)))
    return μs
end

function get_covs(post_gps, θ_grid::AbstractMatrix{<:Real})
    Σs_ = BOSS.cov.(post_gps, Ref(eachcol(θ_grid)))
    ΣLs = map(Σ -> cholesky(Σ).L, Σs_)
    return ΣLs
end

# function mmd(kernel, X::AbstractVector, Y::AbstractVector)
#     val_X = mean(BOSS.kernelmatrix(kernel, X))
#     val_Y = mean(BOSS.kernelmatrix(kernel, Y))
#     val_XY = mean(BOSS.kernelmatrix(kernel, X, Y))
#     return val_X + val_Y - 2*val_XY
# end

function hsic(kernel_X, kernel_Y, X::AbstractVector, Y::AbstractVector, n::Int)
    Kx = BOSS.kernelmatrix(kernel_X, X)
    Ky = BOSS.kernelmatrix(kernel_Y, Y)
    C = Diagonal(ones(n)) - (1/n) * ones(n,n)
    return (1 / (n^2)) * tr( (C * Kx) * (C * Ky) )
end
