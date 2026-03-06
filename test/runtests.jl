using BOSIP
using BOSS
using Test
using Aqua
using Distributions

@testset "BOSIP.jl" verbose=true begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BOSIP)
    end

    @testset "Unit Tests" verbose=true begin
        include("unit/include.jl")
    end
end
