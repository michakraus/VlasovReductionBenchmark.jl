using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua" include("quality/aqua.jl")
    @safetestset "ExplicitImports" include("quality/explicit_imports.jl")
    @safetestset "Dispersion relations" include("dispersion.jl")
    @safetestset "Full-order runs" include("full_order.jl")
    @safetestset "Particle half" include("particles.jl")
end

if "slow" in GROUPS
    @safetestset "Full-order script, full mode" include("integration/full_order_script.jl")
    @safetestset "Particle script, smoke mode" include("integration/particles_smoke.jl")
    @safetestset "Particle script, full mode" include("integration/particles_script.jl")
end
