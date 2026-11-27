
_rb_inner = MvNormalLikelihood(; z_obs=[1.0, 2.0], Σ_obs=[1.0 0.3; 0.3 1.0])
_rb_α, _rb_A = 0.1, 10.0
like_rb = RobustLikelihood(; like=_rb_inner, α=_rb_α, A=_rb_A)

δ_rb = [1.1, 2.2]
Δ_rb = [1.1 0.8; 2.2 1.8]
X_rb = [0.0 1.0]

_rb_expected(ll) = log((1 - _rb_α) * exp(ll) + _rb_α / _rb_A)

@testset "likelihood_kind" begin
    @test likelihood_kind(like_rb) isa JointOnly
end

@testset "loglike" begin
    result = loglike(like_rb, δ_rb)
    @test result isa Real
    @test result ≈ _rb_expected(loglike(_rb_inner, δ_rb))
    @test loglike(like_rb, δ_rb, [0.0]) ≈ result

    result_vec = loglike(like_rb, Δ_rb)
    @test result_vec isa AbstractVector{<:Real}
    @test result_vec ≈ _rb_expected.(loglike(_rb_inner, Δ_rb))
    @test loglike(like_rb, Δ_rb, X_rb) ≈ result_vec
end

@testset "bounded below by the uniform fallback" begin
    @test loglike(like_rb, [1e3, -1e3]) ≈ log(_rb_α / _rb_A)
end

@testset "mean and variance" begin
    post = MockModelPosterior(δ_rb, [0.2, 0.3])
    x = [0.0]

    f_mean = log_likelihood_mean(like_rb, post)
    f_mean_inner = log_likelihood_mean(_rb_inner, post)
    @test f_mean(x) ≈ _rb_expected(f_mean_inner(x))
    @test f_mean(X_rb) ≈ _rb_expected.(f_mean_inner(X_rb))

    f_var = log_likelihood_variance(like_rb, post)
    f_var_inner = log_likelihood_variance(_rb_inner, post)
    @test f_var(x) ≈ 2 * log(1 - _rb_α) + f_var_inner(x)
end

@testset "inside CombinedLikelihood" begin
    like2 = NormalLikelihood(; z_obs=[3.0], std_obs=[2.0])
    like_cl = CombinedLikelihood([like_rb, like2], [1:2, 3:3])
    δ = [1.1, 2.2, 3.3]
    @test likelihood_kind(like_cl) isa JointOnly
    @test loglike(like_cl, δ) ≈ loglike(like_rb, δ[1:2]) + loglike(like2, δ[3:3])
end

@testset "additive parametrization" begin
    ε = 0.01
    like_eps = RobustLikelihood(; like=_rb_inner, ε)
    @test loglike(like_eps, δ_rb) ≈ log(exp(loglike(_rb_inner, δ_rb)) + ε)
    @test loglike(like_eps, [1e3, -1e3]) ≈ log(ε)

    post = MockModelPosterior(δ_rb, [0.2, 0.3])
    f_mean_inner = log_likelihood_mean(_rb_inner, post)
    @test log_likelihood_mean(like_eps, post)([0.0]) ≈ log(exp(f_mean_inner([0.0])) + ε)
    @test log_likelihood_variance(like_eps, post)([0.0]) ≈ log_likelihood_variance(_rb_inner, post)([0.0])

    @test_throws AssertionError RobustLikelihood(; like=_rb_inner)
    @test_throws AssertionError RobustLikelihood(; like=_rb_inner, α=0.1, A=1.0, ε=0.1)
end

@testset "make_robust" begin
    like1 = NormalLikelihood(; z_obs=[1.0, 2.0], std_obs=[0.5, 1.0])
    like2 = NormalLikelihood(; z_obs=[3.0], std_obs=[2.0])
    like_cl = CombinedLikelihood([like1, like2], [1:2, 3:3])
    δ = [1.1, 2.2, 3.3]

    like_r = make_robust(like_cl, 0.01)
    @test like_r isa CombinedLikelihood
    @test like_r.δ_ranges == like_cl.δ_ranges
    @test all(l -> l isa RobustLikelihood, like_r.likelihoods)
    @test loglike(like_r, δ) ≈ log(exp(loglike(like1, δ[1:2])) + 0.01) + log(exp(loglike(like2, δ[3:3])) + 0.01)

    like_rv = make_robust(like_cl, [0.01, 0.5])
    @test loglike(like_rv, δ) ≈ log(exp(loglike(like1, δ[1:2])) + 0.01) + log(exp(loglike(like2, δ[3:3])) + 0.5)

    @test_throws AssertionError make_robust(like_cl, [0.01])
end

@testset "make_robust errors on already robust likelihoods" begin
    like1 = RobustLikelihood(; like=NormalLikelihood(; z_obs=[1.0], std_obs=[0.5]), ε=0.01)
    like2 = NormalLikelihood(; z_obs=[3.0], std_obs=[2.0])
    like_cl = CombinedLikelihood([like1, like2], [1:1, 2:2])
    @test_throws ArgumentError make_robust(like_cl, 0.01)
end

@testset "CombinedLikelihood ε keyword" begin
    like1 = NormalLikelihood(; z_obs=[1.0, 2.0], std_obs=[0.5, 1.0])
    like2 = NormalLikelihood(; z_obs=[3.0], std_obs=[2.0])
    ls, rs = [like1, like2], [1:2, 3:3]
    δ = [1.1, 2.2, 3.3]

    plain = CombinedLikelihood(; likelihoods=ls, δ_ranges=rs)
    @test all(l -> !(l isa RobustLikelihood), plain.likelihoods)

    for ε in (0.01, [0.01, 0.5])
        like_kw = CombinedLikelihood(; likelihoods=ls, δ_ranges=rs, ε)
        like_mr = make_robust(CombinedLikelihood(ls, rs), ε)
        @test all(l -> l isa RobustLikelihood, like_kw.likelihoods)
        @test loglike(like_kw, δ) ≈ loglike(like_mr, δ)
    end
end
