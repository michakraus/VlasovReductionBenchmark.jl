using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "ExplicitImports" include("quality/explicit_imports.jl")
    @safetestset "Dispersion relations" include("dispersion.jl")
    @safetestset "Full-order runs" include("full_order.jl")
    @safetestset "Parity split" include("parity.jl")
    @safetestset "Metrics" include("metrics.jl")
    @safetestset "POD and PSD reduction" include("reduction.jl")
    @safetestset "DLRA solver" include("dlra.jl")
end

if "slow" in GROUPS
    @safetestset "Full-order script, full mode" include("integration/full_order_script.jl")
    @safetestset "Galerkin oracles" include("verification/galerkin_oracles.jl")
    @safetestset "Linear baselines script, smoke mode" include("integration/linear_baselines_script.jl")
    @safetestset "DLRA script, smoke and full mode" include("integration/dlra_script.jl")
end
