@testset "normal.jl" begin
    include("normal.jl")
end

@testset "lognormal.jl" begin
    include("lognormal.jl")
end

@testset "binomial.jl" begin
    include("binomial.jl")
end

@testset "normal_sum.jl" begin
    include("normal_sum.jl")
end

@testset "lognormal_sum.jl" begin
    include("lognormal_sum.jl")
end

@testset "normal_diff.jl" begin
    include("normal_diff.jl")
end

@testset "mvnormal.jl" begin
    include("mvnormal.jl")
end

@testset "exp.jl" begin
    include("exp.jl")
end

@testset "sqexp.jl" begin
    include("sqexp.jl")
end

@testset "combined.jl" begin
    include("combined.jl")
end
