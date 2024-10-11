using BOLFI
using Documenter

makedocs(sitename="BOLFI.jl";
    pages = [
        "index.md",
        "lfi.md",
        "lfss.md",
        "functions.md",
        "types.md",
        "hyperparams.md",
        "example_lfi.md",
        "example_lfss.md",
    ],
)

deploydocs(;
    repo = "github.com/soldasim/BOLFI.jl",
    devbranch = "dev",
    devurl = "dev",
    versions = [
        "stable" => "v^",
        "dev" => "dev",
    ],
)
