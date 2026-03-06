
z_obs_bl  = [3, 5, 2]
trials_bl = [10, 10, 10]
like_bl   = BinomialLikelihood(; z_obs=z_obs_bl, trials=trials_bl)

y_bl  = [0.3, 0.5, 0.2]
Y_bl  = [0.3 0.4; 0.5 0.6; 0.2 0.3]  # obs_dim × n_points

_bl_expected_lm(yi) = logpdf.(Binomial.(trials_bl, clamp.(yi, 0., 1.)), z_obs_bl)
_bl_expected_l(yi)  = sum(_bl_expected_lm(yi))

@testset "loglike_marginal" begin
    result = loglike_marginal(like_bl, y_bl)
    @test result isa AbstractVector{<:Real}
    @test length(result) == length(z_obs_bl)
    @test result ≈ _bl_expected_lm(y_bl)

    result_mat = loglike_marginal(like_bl, Y_bl)
    @test result_mat isa AbstractMatrix{<:Real}
    @test size(result_mat) == (length(z_obs_bl), size(Y_bl, 2))
    @test result_mat[:, 1] ≈ loglike_marginal(like_bl, Y_bl[:, 1])
    @test result_mat[:, 2] ≈ loglike_marginal(like_bl, Y_bl[:, 2])
end

@testset "loglike" begin
    result = loglike(like_bl, y_bl)
    @test result isa Real
    @test result ≈ _bl_expected_l(y_bl)

    result_vec = loglike(like_bl, Y_bl)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Y_bl, 2)
    @test result_vec[1] ≈ loglike(like_bl, Y_bl[:, 1])
    @test result_vec[2] ≈ loglike(like_bl, Y_bl[:, 2])
end

@testset "loglike_marginal/loglike consistency" begin
    @test sum(loglike_marginal(like_bl, y_bl)) ≈ loglike(like_bl, y_bl)
    @test vec(sum(loglike_marginal(like_bl, Y_bl), dims=1)) ≈ loglike(like_bl, Y_bl)
end

let
    obs_dim = length(z_obs_bl)
    # Small positive variance to avoid degenerate truncated normals
    post = MockModelPosterior(y_bl, fill(0.01^2, obs_dim))
    x = [0.0]
    X = [0.0 1.0 2.0]

    @testset "log_marginal_likelihood_mean" begin
        f = log_marginal_likelihood_mean(like_bl, post)
        @test f isa Function

        result = f(x)
        @test result isa AbstractVector{<:Real}
        @test length(result) == obs_dim

        result_mat = f(X)
        @test result_mat isa AbstractMatrix{<:Real}
        @test size(result_mat) == (obs_dim, size(X, 2))
        for i in axes(X, 2)
            @test result_mat[:, i] ≈ f(X[:, i])
        end
    end

    @testset "log_likelihood_mean" begin
        f = log_likelihood_mean(like_bl, post)
        @test f isa Function

        result = f(x)
        @test result isa Real

        result_vec = f(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f(X[:, i])
        end
    end

    @testset "log_likelihood_variance" begin
        f = log_likelihood_variance(like_bl, post)
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
    sub = get_subset(like_bl, y_set)
    @test sub isa BinomialLikelihood
    @test sub.z_obs == z_obs_bl[y_set]
    @test sub.trials == trials_bl[y_set]
end
