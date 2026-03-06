
# sum_lengths=[2,2]: model outputs log_y[1:2] logsumexp to log_z[1], log_y[3:4] to log_z[2]
sum_lengths_lnsl = [2, 2]
log_z_obs_lnsl   = [0.5, 1.0]
CV_lnsl          = [0.1, 0.2]
like_lnsl        = LogNormalSumLikelihood(; sum_lengths=sum_lengths_lnsl, log_z_obs=log_z_obs_lnsl, CV=CV_lnsl)

_lnsl_z_obs  = exp.(log_z_obs_lnsl)
_lnsl_σ_log  = sqrt.(log.(1 .+ CV_lnsl .^ 2))
_lnsl_μ_log(log_z, σl) = log_z .- σl .^ 2 ./ 2

log_ys_lnsl = [-0.7, 0.4, 0.6, 0.7]  # y_dim=4
log_Ys_lnsl = [-0.7 -0.5; 0.4 0.3; 0.6 0.5; 0.7 0.8]  # y_dim × n_points

_lnsl_logsumexp_groups(log_yi) = [
    log(exp(log_yi[1]) + exp(log_yi[2])),
    log(exp(log_yi[3]) + exp(log_yi[4])),
]

_lnsl_expected_lm(log_yi) = begin
    log_z = _lnsl_logsumexp_groups(log_yi)
    μ_log = _lnsl_μ_log(log_z, _lnsl_σ_log)
    logpdf.(LogNormal.(μ_log, _lnsl_σ_log), _lnsl_z_obs)
end
_lnsl_expected_l(log_yi) = sum(_lnsl_expected_lm(log_yi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like_lnsl, log_ys_lnsl)
    @test result isa AbstractVector{<:Real}
    @test length(result) == length(log_z_obs_lnsl)
    @test result ≈ _lnsl_expected_lm(log_ys_lnsl)

    result_mat = loglike_marginal(like_lnsl, log_Ys_lnsl)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(log_z_obs_lnsl), size(log_Ys_lnsl, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like_lnsl, log_Ys_lnsl[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like_lnsl, log_Ys_lnsl[:, 2])
end

@testset "loglike" begin
    result = loglike(like_lnsl, log_ys_lnsl)
    @test result isa Real
    @test result ≈ _lnsl_expected_l(log_ys_lnsl)

    result_vec = loglike(like_lnsl, log_Ys_lnsl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(log_Ys_lnsl, 2)
    @test result_vec[1] ≈ loglike(like_lnsl, log_Ys_lnsl[:, 1])
    @test result_vec[2] ≈ loglike(like_lnsl, log_Ys_lnsl[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_lnsl, log_ys_lnsl)) ≈ loglike(like_lnsl, log_ys_lnsl)
    @test vec(sum(loglike_marginal(like_lnsl, log_Ys_lnsl), dims=1)) ≈ loglike(like_lnsl, log_Ys_lnsl)
end

let
    y_dim   = sum(sum_lengths_lnsl)
    obs_dim = length(log_z_obs_lnsl)
    # With zero GP variance, Fenton-Wilkinson is exact → matches loglike_marginal at the mean
    post_zero    = MockModelPosterior(log_ys_lnsl, zeros(y_dim))
    post_nonzero = MockModelPosterior(log_ys_lnsl, fill(0.1^2, y_dim))
    x = [0.0]
    X = [0.0 1.0 2.0]

    @testset "log_marginal_likelihood_mean" begin
        f_zero    = log_marginal_likelihood_mean(like_lnsl, post_zero)
        f_nonzero = log_marginal_likelihood_mean(like_lnsl, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa AbstractVector{<:Real}
        @test length(result_zero) == obs_dim
        @test result_zero ≈ _lnsl_expected_lm(log_ys_lnsl)

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
        f_zero = log_likelihood_mean(like_lnsl, post_zero)
        @test f_zero isa Function

        result = f_zero(x)
        @test result isa Real
        @test result ≈ _lnsl_expected_l(log_ys_lnsl)

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_marginal_likelihood_mean/log_likelihood_mean consistency" begin
        f_ml = log_marginal_likelihood_mean(like_lnsl, post_nonzero)
        f_l  = log_likelihood_mean(like_lnsl, post_nonzero)
        @test sum(f_ml(x)) ≈ f_l(x)
        @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_lnsl, post_zero)
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
    y_set = [true, false]
    sub = get_subset(like_lnsl, y_set)
    @test sub isa LogNormalSumLikelihood
    @test sub.log_z_obs ≈ log_z_obs_lnsl[y_set]
    @test sub.CV ≈ CV_lnsl[y_set]
    @test sub.sum_lengths == sum_lengths_lnsl[y_set]
end
