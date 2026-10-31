using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "ExplicitImports" include("quality/explicit_imports.jl")
    @safetestset "Dispersion relations" include("dispersion.jl")
    @safetestset "Full-order runs" include("full_order.jl")
end

if "slow" in GROUPS
    @safetestset "Full-order script, full mode" include("integration/full_order_script.jl")
end
