z_obs = [1.0, 2.0, 3.0]
std_obs = [0.5, 1.0, 2.0]

# Reference value of `log E[p(z|y)^q]` by brute-force high-order quadrature on a fine grid.
function _hnl_brute(z, μ, s, σ_fn, q)
    ys = range(μ - 12s, μ + 12s; length=400_001)
    f = [pdf(Normal(μ, s), y) * (σ_fn(y) > 0 ? pdf(Normal(y, σ_fn(y)), z)^q : 0.0) for y in ys]
    return log(sum(f) * step(ys))
end

@testset "constant std_fn agrees with NormalLikelihood" begin
    like_h = HeteroscedasticNormalLikelihood(; z_obs, std_fn = y -> std_obs)
    like_n = NormalLikelihood(; z_obs, std_obs)
    y = [1.1, 2.2, 3.3]
    Y = [1.1 0.5; 2.2 1.5; 3.3 2.5]
    x = [0.0]
    X = [0.0 1.0 2.0]

    @test loglike_marginal(like_h, y) ≈ loglike_marginal(like_n, y)
    @test loglike_marginal(like_h, Y) ≈ loglike_marginal(like_n, Y)
    @test loglike(like_h, Y) ≈ loglike(like_n, Y)

    for std_y in ([0.0, 0.0, 0.0], [0.3, 0.8, 2.5], [5.0, 4.0, 10.0])
        post = MockModelPosterior(y, std_y .^ 2)
        @test log_marginal_likelihood_mean(like_h, post)(x) ≈ log_marginal_likelihood_mean(like_n, post)(x) atol=1e-6
        @test log_marginal_likelihood_mean(like_h, post)(X) ≈ log_marginal_likelihood_mean(like_n, post)(X) atol=1e-6
        @test log_likelihood_mean(like_h, post)(x) ≈ log_likelihood_mean(like_n, post)(x) atol=1e-6
        @test BOSIP.log_sq_likelihood_mean(like_h, post)(x) ≈ BOSIP.log_sq_likelihood_mean(like_n, post)(x) atol=1e-6
        @test BOSIP.log_sq_likelihood_mean(like_h, post)(X) ≈ BOSIP.log_sq_likelihood_mean(like_n, post)(X) atol=1e-6
    end
end

@testset "nonconstant std_fn" begin
    cases = [
        ("proportional", y -> 0.2 .* y, [1.0, 2.0, 3.0]),
        ("sqrt(a² + (by)²)", y -> sqrt.(0.1^2 .+ (0.15 .* y) .^ 2), [1.0, 2.0, 3.0]),
    ]
    for (name, std_fn, z) in cases
        @testset "$name" begin
            like = HeteroscedasticNormalLikelihood(; z_obs=z, std_fn)
            @test loglike_marginal(like, [1.1, 2.2, 3.3]) ≈ logpdf.(Normal.([1.1, 2.2, 3.3], std_fn([1.1, 2.2, 3.3])), z)

            for (μ, s) in ((1.0, 0.0), (1.0, 0.05), (1.2, 0.3), (0.8, 1.0), (2.0, 0.5))
                post = MockModelPosterior(fill(μ, 3), fill(s^2, 3))
                σ_fn(i) = y -> std_fn(fill(y, 3))[i]
                ml = log_marginal_likelihood_mean(like, post)([0.0])
                sq = BOSIP.log_sq_likelihood_mean(like, post)([0.0])
                s_eff = max(s, 1e-8)
                ml_ref = [_hnl_brute(z[i], μ, s_eff, σ_fn(i), 1) for i in 1:3]
                sq_ref = sum(_hnl_brute(z[i], μ, s_eff, σ_fn(i), 2) for i in 1:3)
                @test ml ≈ ml_ref rtol=1e-4
                @test sq ≈ sq_ref rtol=1e-4
            end
        end
    end
end

@testset "matches the BIPBM formula" begin
    a, b = [0.1, 0.2, 0.3], [0.15, 0.1, 0.2]
    std_fn(y) = sqrt.(a .^ 2 .+ (b .* y) .^ 2)
    like = HeteroscedasticNormalLikelihood(; z_obs, std_fn)
    y = [1.1, 2.2, 3.3]
    σ = std_fn(y)
    bipbm_loglike = -(sum(abs2, (y .- z_obs) ./ σ) + length(y) * log(2π)) / 2 - sum(log, σ)
    @test loglike(like, y) ≈ bipbm_loglike
    @test sum(loglike_marginal(like, y)) ≈ bipbm_loglike
end

@testset "nonpositive std" begin
    like = HeteroscedasticNormalLikelihood(; z_obs, std_fn = y -> 0.2 .* y)
    @test loglike_marginal(like, [-1.0, 2.0, 0.0]) == [-Inf, loglike_marginal(like, [1.0, 2.0, 3.0])[2], -Inf]
    @test loglike(like, [-1.0, 2.0, 3.0]) == -Inf
    post = MockModelPosterior([-5.0, -5.0, -5.0], fill(0.01, 3))
    @test log_marginal_likelihood_mean(like, post)([0.0]) == fill(-Inf, 3)
    @test BOSIP.log_sq_likelihood_mean(like, post)([0.0]) == -Inf
end

@testset "non-elementwise std_fn is rejected" begin
    @test_throws ArgumentError HeteroscedasticNormalLikelihood(; z_obs, std_fn = y -> fill(1.0 + sum(abs2, y), 3))
end

@testset "get_subset" begin
    a = [0.1, 0.2, 0.3]
    like = HeteroscedasticNormalLikelihood(; z_obs, std_fn = y -> sqrt.(a .^ 2 .+ (0.1 .* y) .^ 2))
    sub = get_subset(like, [true, false, true])
    @test sub.z_obs == z_obs[[1, 3]]
    @test loglike_marginal(sub, [1.1, 3.3]) ≈ loglike_marginal(like, [1.1, 2.2, 3.3])[[1, 3]]
    post = MockModelPosterior([1.0, 3.0], [0.1, 0.2])
    full = MockModelPosterior([1.0, 2.0, 3.0], [0.1, 0.1, 0.2])
    @test log_marginal_likelihood_mean(sub, post)([0.0]) ≈ log_marginal_likelihood_mean(like, full)([0.0])[[1, 3]]
end
