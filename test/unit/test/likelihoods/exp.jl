
like_el = ExpLikelihood()

y_el  = [2.5]
Y_el  = reshape([2.5, 1.5], 1, 2)  # 1 × n_points

@testset "loglike" begin
    result = loglike(like_el, y_el)
    @test result isa Real
    @test result ≈ y_el[1]

    result_vec = loglike(like_el, Y_el)
    @test result_vec isa AbstractVector{<:Real}
    @test length(result_vec) == size(Y_el, 2)
    @test result_vec[1] ≈ loglike(like_el, Y_el[:, 1])
    @test result_vec[2] ≈ loglike(like_el, Y_el[:, 2])
end

let
    μ, σ2 = y_el[1], 0.5^2
    post_zero    = MockModelPosterior(y_el, [0.0])
    post_nonzero = MockModelPosterior(y_el, [σ2])
    x = [0.0]
    X = [0.0 1.0 2.0]

    @testset "log_likelihood_mean" begin
        f_zero    = log_likelihood_mean(like_el, post_zero)
        f_nonzero = log_likelihood_mean(like_el, post_nonzero)
        @test f_zero isa Function

        result_zero = f_zero(x)
        @test result_zero isa Real
        # zero GP variance: log(E[exp(δ)]) = μ + 0.5 * 0 = μ = loglike at mean
        @test result_zero ≈ μ

        result_nonzero = f_nonzero(x)
        @test result_nonzero isa Real
        @test result_nonzero ≈ μ + 0.5 * σ2
        @test result_nonzero > result_zero

        result_vec = f_zero(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        for i in axes(X, 2)
            @test result_vec[i] ≈ f_zero(X[:, i])
        end
    end

    @testset "log_likelihood_variance" begin
        f_var  = log_likelihood_variance(like_el, post_nonzero)
        f_mean = log_likelihood_mean(like_el, post_nonzero)
        @test f_var isa Function
        @test f_var(x) isa Real
        result_vec = f_var(X)
        @test result_vec isa AbstractVector{<:Real}
        @test length(result_vec) == size(X, 2)
        # Var[exp(δ)] ≥ 0, and for σ2 > 0 it is strictly positive
        @test exp(f_var(x)) > 0
    end
end
