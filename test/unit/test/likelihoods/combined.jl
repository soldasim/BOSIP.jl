
# Combine two NormalLikelihoods: first over δ[1:2], second over δ[3:3]
_cl_like1 = NormalLikelihood(; z_obs=[1.0, 2.0], std_obs=[0.5, 1.0])
_cl_like2 = NormalLikelihood(; z_obs=[3.0],       std_obs=[2.0])
like_cl   = CombinedLikelihood([_cl_like1, _cl_like2], [1:2, 3:3])

δ_cl  = [1.1, 2.2, 3.3]
Δ_cl  = [1.1 0.5; 2.2 1.5; 3.3 2.5]  # δ_dim × n_points
x_cl  = [0.0]
X_cl  = [0.0 1.0]  # same n_points as Δ_cl

_cl_expected_lm(δi) = vcat(
    loglike_marginal(_cl_like1, δi[1:2]),
    loglike_marginal(_cl_like2, δi[3:3]),
)
_cl_expected_l(δi) = loglike(_cl_like1, δi[1:2]) + loglike(_cl_like2, δi[3:3])

@testset "loglike_marginal" begin
    result = loglike_marginal(like_cl, δ_cl, x_cl)
    @test result isa AbstractVector{<:Real}
    @test length(result) == 3
    @test result ≈ _cl_expected_lm(δ_cl)

    result_mat = loglike_marginal(like_cl, Δ_cl, X_cl)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (3, size(Δ_cl, 2))
    for i in axes(Δ_cl, 2)
        @test result_mat[:, i] ≈ loglike_marginal(like_cl, Δ_cl[:, i], X_cl[:, i])
    end
end

@testset "loglike" begin
    result = loglike(like_cl, δ_cl, x_cl)
    @test result isa Real
    @test result ≈ _cl_expected_l(δ_cl)

    result_vec = loglike(like_cl, Δ_cl, X_cl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Δ_cl, 2)
    for i in axes(Δ_cl, 2)
        @test result_vec[i] ≈ loglike(like_cl, Δ_cl[:, i], X_cl[:, i])
    end
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_cl, δ_cl, x_cl)) ≈ loglike(like_cl, δ_cl, x_cl)
    lm_mat = loglike_marginal(like_cl, Δ_cl, X_cl)
    @test vec(sum(lm_mat, dims=1)) ≈ loglike(like_cl, Δ_cl, X_cl)
end

@testset "log_likelihood_mean errors for generic ModelPosterior" begin
    post = MockModelPosterior([1.0, 2.0, 3.0], [0.1, 0.1, 0.1])
    @test_throws MethodError log_likelihood_mean(like_cl, post)
    @test_throws MethodError log_marginal_likelihood_mean(like_cl, post)
end
