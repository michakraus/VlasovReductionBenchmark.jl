using ExplicitImports
using VlasovReductionBenchmark
using Test

# `GeometricIntegratorsBase.solutionstep`, which steps a run one solution step at a time as
# VlasovMethods' own scripts do, is not public. Nor are `SimpleSolvers.status` and
# `SimpleSolvers.isconverged`, which tell a reduced run whether the nonlinear solve of a step
# converged.
test_explicit_imports(VlasovReductionBenchmark;
    all_explicit_imports_are_public = (; ignore = (:solutionstep, :status, :isconverged)))
