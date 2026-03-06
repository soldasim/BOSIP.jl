z_obs  = [1.0, 2.0, 3.0]
std_obs = [0.5, 1.0, 2.0]
like = NormalLikelihood(; z_obs, std_obs)

y = [1.1, 2.2, 3.3]
Y = [1.1 0.5; 2.2 1.5; 3.3 2.5]  # obs_dim × n_points

_nl_expected_lm(yi) = logpdf.(Normal.(yi, std_obs), z_obs)
_nl_expected_l(yi)  = sum(_nl_expected_lm(yi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like, y)

    @test result isa AbstractVector{<:Real}
    @test length(result) == length(z_obs)
    @test result ≈ _nl_expected_lm(y)

    result_mat = loglike_marginal(like, Y)

    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(z_obs), size(Y, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like, Y[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like, Y[:, 2])
end

@testset "loglike" begin
    result = loglike(like, y)

    @test result isa Real
    @test result ≈ _nl_expected_l(y)

    result_mat = loglike(like, Y)

    @test result_mat isa AbstractVector{<:Real}
    @test length(result_mat) == size(Y, 2)
    @test result_mat[1] ≈ loglike(like, Y[:, 1])
    @test result_mat[2] ≈ loglike(like, Y[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like, y)) ≈ loglike(like, y)
    @test vec(sum(loglike_marginal(like, Y), dims=1)) ≈ loglike(like, Y)
end
