
z_obs_mvnl = [1.0, 2.0]
Σ_obs_mvnl = [1.0 0.3; 0.3 1.0]
like_mvnl  = MvNormalLikelihood(; z_obs=z_obs_mvnl, Σ_obs=Σ_obs_mvnl)

y_mvnl  = [1.1, 2.2]
Y_mvnl  = [1.1 0.8; 2.2 1.8]  # obs_dim × n_points

_mvnl_expected_l(yi) = logpdf(MvNormal(z_obs_mvnl, Σ_obs_mvnl), yi)

@testset "loglike" begin
    result = loglike(like_mvnl, y_mvnl)
    @test result isa Real
    @test result ≈ _mvnl_expected_l(y_mvnl)

    result_vec = loglike(like_mvnl, Y_mvnl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Y_mvnl, 2)
    @test result_vec[1] ≈ loglike(like_mvnl, Y_mvnl[:, 1])
    @test result_vec[2] ≈ loglike(like_mvnl, Y_mvnl[:, 2])
end

let
    obs_dim = length(z_obs_mvnl)
    # std_obs = sqrt.(diag(Σ_obs_mvnl)) = [1.0, 1.0] since Σ_obs_mvnl has unit diagonal
    std_obs = [1.0, 1.0]
    post_zero    = MockModelPosterior(y_mvnl, zeros(obs_dim))
    post_nonzero = MockModelPosterior(y_mvnl, std_obs .^ 2)
    x = [0.0]
    X = [0.0 1.0 2.0]

    # Σ_obs_mvnl + diag(σ^2) with σ = std_obs = [1,1] → [2.0 0.3; 0.3 2.0]
    _mvnl_Σ_inflated = [2.0 0.3; 0.3 2.0]
    _mvnl_expected_l_mean_inflated(μ) = logpdf(MvNormal(μ, _mvnl_Σ_inflated), z_obs_mvnl)

    @testset "log_likelihood_mean" begin
        f_zero    = log_likelihood_mean(like_mvnl, post_zero)
        f_nonzero = log_likelihood_mean(like_mvnl, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa Real
        # zero GP variance → matches loglike at the mean
        @test result_zero ≈ loglike(like_mvnl, y_mvnl)

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa Real
        @test result_nonzero ≈ _mvnl_expected_l_mean_inflated(y_mvnl)
        @test !isapprox(result_nonzero, result_zero)

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_mvnl, post_zero)
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
