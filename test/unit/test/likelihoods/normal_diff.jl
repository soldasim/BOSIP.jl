
std_obs_ndl = [0.5, 1.0, 2.0]
like_ndl    = NormalDiffLikelihood(; std_obs=std_obs_ndl)

δ_ndl  = [0.1, -0.2, 0.3]
Δ_ndl  = [0.1 0.4; -0.2 0.1; 0.3 -0.1]  # obs_dim × n_points

_ndl_expected_lm(δi) = logpdf.(Normal.(zero(std_obs_ndl), std_obs_ndl), δi)
_ndl_expected_l(δi)  = sum(_ndl_expected_lm(δi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like_ndl, δ_ndl)
    @test result isa AbstractVector{<:Real}
    @test length(result) == length(std_obs_ndl)
    @test result ≈ _ndl_expected_lm(δ_ndl)

    result_mat = loglike_marginal(like_ndl, Δ_ndl)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(std_obs_ndl), size(Δ_ndl, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like_ndl, Δ_ndl[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like_ndl, Δ_ndl[:, 2])
end

@testset "loglike" begin
    result = loglike(like_ndl, δ_ndl)
    @test result isa Real
    @test result ≈ _ndl_expected_l(δ_ndl)

    result_vec = loglike(like_ndl, Δ_ndl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Δ_ndl, 2)
    @test result_vec[1] ≈ loglike(like_ndl, Δ_ndl[:, 1])
    @test result_vec[2] ≈ loglike(like_ndl, Δ_ndl[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_ndl, δ_ndl)) ≈ loglike(like_ndl, δ_ndl)
    @test vec(sum(loglike_marginal(like_ndl, Δ_ndl), dims=1)) ≈ loglike(like_ndl, Δ_ndl)
end

let
    obs_dim = length(std_obs_ndl)
    mean_val = δ_ndl
    post_zero    = MockModelPosterior(mean_val, zeros(obs_dim))
    post_nonzero = MockModelPosterior(mean_val, std_obs_ndl .^ 2)
    x = [0.0]
    X = [0.0 1.0 2.0]

    _ndl_expected_ml_mean(μ, σ) = logpdf.(Normal.(μ, sqrt.(std_obs_ndl .^ 2 .+ σ .^ 2)), zero(std_obs_ndl))

    @testset "log_marginal_likelihood_mean" begin
        f_zero    = log_marginal_likelihood_mean(like_ndl, post_zero)
        f_nonzero = log_marginal_likelihood_mean(like_ndl, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa AbstractVector{<:Real}
        @test length(result_zero) == obs_dim
        @test result_zero ≈ _ndl_expected_lm(mean_val)

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa AbstractVector{<:Real}
        @test result_nonzero ≈ _ndl_expected_ml_mean(mean_val, std_obs_ndl)
        @test !isapprox(result_nonzero, result_zero)

        result_mat = f_zero(X)
        @test result_mat isa AbstractMatrix{<:Real}
        @test size(result_mat) == (obs_dim, size(X, 2))
        for i in axes(X, 2)
            @test result_mat[:, i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_mean" begin
        f_zero = log_likelihood_mean(like_ndl, post_zero)
        @test f_zero isa Function

        result = f_zero(x)
        @test result isa Real
        @test result ≈ _ndl_expected_l(mean_val)

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_marginal_likelihood_mean/log_likelihood_mean consistency" begin
        f_ml = log_marginal_likelihood_mean(like_ndl, post_nonzero)
        f_l  = log_likelihood_mean(like_ndl, post_nonzero)
        @test sum(f_ml(x)) ≈ f_l(x)
        @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_ndl, post_zero)
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

@testset "get_subset" begin
    y_set = [true, false, true]
    sub = get_subset(like_ndl, y_set)
    @test sub isa NormalDiffLikelihood
    @test sub.std_obs ≈ std_obs_ndl[y_set]
end
