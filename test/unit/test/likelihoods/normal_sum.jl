
# sum_lengths=[2,2]: model outputs y[1:2] sum to z_obs[1], y[3:4] to z_obs[2]
sum_lengths_nsl = [2, 2]
z_obs_nsl       = [1.0, 2.0]
std_obs_nsl     = [0.5, 1.0]
like_nsl        = NormalSumLikelihood(; sum_lengths=sum_lengths_nsl, z_obs=z_obs_nsl, std_obs=std_obs_nsl)

y_nsl  = [0.4, 0.7, 0.8, 1.3]              # y_dim=4; sums: [1.1, 2.1]
Y_nsl  = [0.4 0.3; 0.7 0.6; 0.8 0.9; 1.3 1.2]  # y_dim × n_points; sums: [1.1,2.1] and [0.9,2.1]

_nsl_z(yi) = [sum(yi[1:2]), sum(yi[3:4])]
_nsl_expected_lm(yi) = logpdf.(Normal.(_nsl_z(yi), std_obs_nsl), z_obs_nsl)
_nsl_expected_l(yi)  = sum(_nsl_expected_lm(yi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like_nsl, y_nsl)
    @test result isa AbstractVector{<:Real}
    @test length(result) == length(z_obs_nsl)
    @test result ≈ _nsl_expected_lm(y_nsl)

    result_mat = loglike_marginal(like_nsl, Y_nsl)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(z_obs_nsl), size(Y_nsl, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like_nsl, Y_nsl[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like_nsl, Y_nsl[:, 2])
end

@testset "loglike" begin
    result = loglike(like_nsl, y_nsl)
    @test result isa Real
    @test result ≈ _nsl_expected_l(y_nsl)

    result_vec = loglike(like_nsl, Y_nsl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Y_nsl, 2)
    @test result_vec[1] ≈ loglike(like_nsl, Y_nsl[:, 1])
    @test result_vec[2] ≈ loglike(like_nsl, Y_nsl[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_nsl, y_nsl)) ≈ loglike(like_nsl, y_nsl)
    @test vec(sum(loglike_marginal(like_nsl, Y_nsl), dims=1)) ≈ loglike(like_nsl, Y_nsl)
end

let
    y_dim   = sum(sum_lengths_nsl)
    obs_dim = length(z_obs_nsl)
    post_zero    = MockModelPosterior(y_nsl, zeros(y_dim))
    post_nonzero = MockModelPosterior(y_nsl, fill(0.5^2, y_dim))
    x = [0.0]
    X = [0.0 1.0 2.0]

    # With zero GP variance, sums have zero variance too
    _nsl_expected_ml_mean_zero(μ) = logpdf.(Normal.(_nsl_z(μ), std_obs_nsl), z_obs_nsl)

    @testset "log_marginal_likelihood_mean" begin
        f_zero    = log_marginal_likelihood_mean(like_nsl, post_zero)
        f_nonzero = log_marginal_likelihood_mean(like_nsl, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa AbstractVector{<:Real}
        @test length(result_zero) == obs_dim
        @test result_zero ≈ _nsl_expected_ml_mean_zero(y_nsl)

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa AbstractVector{<:Real}
        @test !isapprox(result_nonzero, result_zero)

        result_mat = f_zero(X)
        @test result_mat isa AbstractMatrix{<:Real}
        @test size(result_mat) == (obs_dim, size(X, 2))
        for i in axes(X, 2)
            @test result_mat[:, i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_mean" begin
        f_zero = log_likelihood_mean(like_nsl, post_zero)
        @test f_zero isa Function

        result = f_zero(x)
        @test result isa Real
        @test result ≈ _nsl_expected_l(y_nsl)

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_marginal_likelihood_mean/log_likelihood_mean consistency" begin
        f_ml = log_marginal_likelihood_mean(like_nsl, post_nonzero)
        f_l  = log_likelihood_mean(like_nsl, post_nonzero)
        @test sum(f_ml(x)) ≈ f_l(x)
        @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_nsl, post_zero)
        @test f isa Function
        @test f(x) isa Real
        result_vec = f(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f(X[:, i])
        end
    end
end
