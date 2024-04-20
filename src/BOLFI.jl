module BOLFI

export bolfi
export BolfiProblem

using BOSS
using Distributions

include("posterior.jl")
include("acquisition.jl")
include("problem.jl")
include("main.jl")

end
