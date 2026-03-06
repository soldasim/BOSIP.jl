z_obs   = [1.0, 2.0, 3.0]
std_obs = [0.5, 1.0, 2.0]
like = NormalLikelihood(; z_obs, std_obs)

mean_val = [1.1, 2.2, 3.3]
var_zero = [0.0, 0.0, 0.0]
var_nonzero = std_obs .^ 2  # var = [0.25, 1.0, 4.0], std = std_obs

post_zero    = MockModelPosterior(mean_val, var_zero)
post_nonzero = MockModelPosterior(mean_val, var_nonzero)

x = [0.0]           # arbitrary; mock ignores x
X = [0.0 1.0 2.0]   # 1 × 3; three points

obs_dim  = length(z_obs)
n_points = size(X, 2)

# Analytical expected values
expected_lm(yi)        = logpdf.(Normal.(yi, std_obs), z_obs)
expected_l(yi)         = sum(expected_lm(yi))
expected_ml_mean(μ, σ) = logpdf.(Normal.(μ, sqrt.(std_obs .^ 2 .+ σ .^ 2)), z_obs)
expected_l_mean(μ, σ)  = sum(expected_ml_mean(μ, σ))

@testset "log_approx_marginal_likelihood" begin
    f = log_approx_marginal_likelihood(like, post_zero)

    @test f isa Function

    result = f(x)
    @test result isa AbstractVector{<:Real}
    @test length(result) == obs_dim
    @test result ≈ expected_lm(mean_val)

    result_mat = f(X)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (obs_dim, n_points)
    for i in 1:n_points
        @test result_mat[:, i] ≈ f(X[:, i])
    end
end

@testset "log_approx_likelihood" begin
    f = log_approx_likelihood(like, post_zero)

    @test f isa Function

    result = f(x)
    @test result isa Real
    @test result ≈ expected_l(mean_val)

    result_vec = f(X)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == n_points
    for i in 1:n_points
        @test result_vec[i] ≈ f(X[:, i])
    end
end

@testset "log_approx_marginal_likelihood/log_approx_likelihood consistency" begin
    f_ml = log_approx_marginal_likelihood(like, post_zero)
    f_l  = log_approx_likelihood(like, post_zero)

    @test sum(f_ml(x)) ≈ f_l(x)
    @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
end

@testset "log_marginal_likelihood_mean" begin
    f_zero    = log_marginal_likelihood_mean(like, post_zero)
    f_nonzero = log_marginal_likelihood_mean(like, post_nonzero)

    @test f_zero isa Function
    @test f_nonzero isa Function

    # zero GP variance: matches the approx (plug-in mean)
    result_zero = f_zero(x)
    @test result_zero isa AbstractVector{<:Real}
    @test length(result_zero) == obs_dim
    @test result_zero ≈ expected_lm(mean_val)

    # nonzero GP variance: convolved std gives different (lower) values
    result_nonzero = f_nonzero(x)
    @test result_nonzero isa AbstractVector{<:Real}
    @test length(result_nonzero) == obs_dim
    @test result_nonzero ≈ expected_ml_mean(mean_val, std_obs)
    @test !isapprox(result_nonzero, result_zero)

    # matrix output
    result_mat = f_zero(X)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (obs_dim, n_points)
    for i in 1:n_points
        @test result_mat[:, i] ≈ f_zero(X[:, i])
    end
end

@testset "log_likelihood_mean" begin
    f_zero    = log_likelihood_mean(like, post_zero)
    f_nonzero = log_likelihood_mean(like, post_nonzero)

    @test f_zero isa Function
    @test f_nonzero isa Function

    result_zero = f_zero(x)
    @test result_zero isa Real
    @test result_zero ≈ expected_l_mean(mean_val, zeros(obs_dim))

    result_nonzero = f_nonzero(x)
    @test result_nonzero isa Real
    @test result_nonzero ≈ expected_l_mean(mean_val, std_obs)
    @test !isapprox(result_nonzero, result_zero)

    result_vec = f_zero(X)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == n_points
    for i in 1:n_points
        @test result_vec[i] ≈ f_zero(X[:, i])
    end
end

@testset "log_marginal_likelihood_mean/log_likelihood_mean consistency" begin
    f_ml = log_marginal_likelihood_mean(like, post_nonzero)
    f_l  = log_likelihood_mean(like, post_nonzero)

    @test sum(f_ml(x)) ≈ f_l(x)
    @test vec(sum(f_ml(X), dims=1)) ≈ f_l(X)
end

@testset "zero GP variance: approx equals mean" begin
    f_approx_ml = log_approx_marginal_likelihood(like, post_zero)
    f_mean_ml   = log_marginal_likelihood_mean(like, post_zero)
    f_approx_l  = log_approx_likelihood(like, post_zero)
    f_mean_l    = log_likelihood_mean(like, post_zero)

    @test f_approx_ml(x) ≈ f_mean_ml(x)
    @test f_approx_l(x)  ≈ f_mean_l(x)
end
