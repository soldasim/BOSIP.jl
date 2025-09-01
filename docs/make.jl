using BOSIP
using Documenter

makedocs(sitename="BOSIP.jl";
    pages = [
        "index.md",
        "sip.md",
        "functions.md",
        "types.md",
        "hyperparams.md",
        "example.md",
    ],
)

deploydocs(;
    repo = "github.com/soldasim/BOSIP.jl",
    devbranch = "dev",
    devurl = "dev",
    versions = [
        "stable" => "v^",
        "dev" => "dev",
    ],
)
