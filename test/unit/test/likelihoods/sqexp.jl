
like_sel = SqExpLikelihood()

y_sel  = [1.5]
Y_sel  = reshape([1.5, 0.5], 1, 2)  # 1 × n_points

@testset "loglike" begin
    result = loglike(like_sel, y_sel)
    @test result isa Real
    @test result ≈ y_sel[1]^2

    result_vec = loglike(like_sel, Y_sel)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Y_sel, 2)
    @test result_vec[1] ≈ loglike(like_sel, Y_sel[:, 1])
    @test result_vec[2] ≈ loglike(like_sel, Y_sel[:, 2])
end

let
    μ, σ = y_sel[1], 0.5
    post_zero    = MockModelPosterior(y_sel, [0.0])
    post_nonzero = MockModelPosterior(y_sel, [σ^2])
    x = [0.0]
    X = [0.0 1.0 2.0]

    @testset "log_likelihood_mean" begin
        f_zero    = log_likelihood_mean(like_sel, post_zero)
        f_nonzero = log_likelihood_mean(like_sel, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa Real
        # zero GP variance: (-(1/2)*log(1+0)) + (-(1/2)*(μ²/(1+0))) = -(1/2)*μ²
        @test result_zero ≈ -(1/2) * μ^2

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa Real
        @test result_nonzero ≈ -(1/2) * log(1 + σ^2) + -(1/2) * (μ^2 / (1 + σ^2))

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_variance" begin
        # Use μ=0 to avoid the log_sq_likelihood_mean formula's instability at large |μ|
        post_zeromean = MockModelPosterior([0.0], [σ^2])
        f = log_likelihood_variance(like_sel, post_zeromean)
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
