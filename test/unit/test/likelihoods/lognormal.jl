
log_z_obs_ln = [0.0, 0.5, 1.0]
CV_ln        = [0.1, 0.2, 0.1]
like_ln      = LogNormalLikelihood(; log_z_obs=log_z_obs_ln, CV=CV_ln)

_lnl_z_obs  = exp.(log_z_obs_ln)
_lnl_σ_log  = sqrt.(log.(1 .+ CV_ln .^ 2))
_lnl_μ_log(log_yi) = log_yi .- _lnl_σ_log .^ 2 ./ 2

log_y_ln = [0.1, 0.6, 1.1]
log_Y_ln = [0.1 -0.1; 0.6 0.4; 1.1 0.9]  # obs_dim × n_points

_lnl_expected_lm(log_yi) = logpdf.(LogNormal.(_lnl_μ_log(log_yi), _lnl_σ_log), _lnl_z_obs)
_lnl_expected_l(log_yi)  = sum(_lnl_expected_lm(log_yi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like_ln, log_y_ln)
    @test result isa AbstractVector{<:Real}
    @test length(result) == length(log_z_obs_ln)
    @test result ≈ _lnl_expected_lm(log_y_ln)

    result_mat = loglike_marginal(like_ln, log_Y_ln)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(log_z_obs_ln), size(log_Y_ln, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like_ln, log_Y_ln[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like_ln, log_Y_ln[:, 2])
end

@testset "loglike" begin
    result = loglike(like_ln, log_y_ln)
    @test result isa Real
    @test result ≈ _lnl_expected_l(log_y_ln)

    result_vec = loglike(like_ln, log_Y_ln)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(log_Y_ln, 2)
    @test result_vec[1] ≈ loglike(like_ln, log_Y_ln[:, 1])
    @test result_vec[2] ≈ loglike(like_ln, log_Y_ln[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_ln, log_y_ln)) ≈ loglike(like_ln, log_y_ln)
    @test vec(sum(loglike_marginal(like_ln, log_Y_ln), dims=1)) ≈ loglike(like_ln, log_Y_ln)
end

let
    obs_dim = length(log_z_obs_ln)
    post_zero    = MockModelPosterior(log_y_ln, zeros(obs_dim))
    post_nonzero = MockModelPosterior(log_y_ln, _lnl_σ_log .^ 2)
    x = [0.0]
    X = [0.0 1.0 2.0]

    _lnl_expected_ml_mean(σ) = logpdf.(LogNormal.(_lnl_μ_log(log_y_ln), sqrt.(_lnl_σ_log .^ 2 .+ σ .^ 2)), _lnl_z_obs)

    @testset "log_marginal_likelihood_mean" begin
        f_zero    = log_marginal_likelihood_mean(like_ln, post_zero)
        f_nonzero = log_marginal_likelihood_mean(like_ln, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa AbstractVector{<:Real}
        @test length(result_zero) == obs_dim
        @test result_zero ≈ _lnl_expected_lm(log_y_ln)

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa AbstractVector{<:Real}
        @test result_nonzero ≈ _lnl_expected_ml_mean(_lnl_σ_log)
        @test !isapprox(result_nonzero, result_zero)

        result_mat = f_zero(X)
        @test result_mat isa AbstractMatrix{<:Real}
        @test size(result_mat) == (obs_dim, size(X, 2))
        for i in axes(X, 2)
            @test result_mat[:, i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_mean" begin
        f_zero = log_likelihood_mean(like_ln, post_zero)
        @test f_zero isa Function

        result = f_zero(x)
        @test result isa Real
        @test result ≈ _lnl_expected_l(log_y_ln)

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_marginal_likelihood_mean/log_likelihood_mean consistency" begin
        f_ml = log_marginal_likelihood_mean(like_ln, post_nonzero)
        f_l  = log_likelihood_mean(like_ln, post_nonzero)
        @test sum(f_ml(x)) ≈ f_l(x)
        @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_ln, post_zero)
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
    sub = get_subset(like_ln, y_set)
    @test sub isa LogNormalLikelihood
    @test length(sub.log_z_obs) == sum(y_set)
    @test length(sub.CV) == sum(y_set)
end
