using ExplicitImports
using VlasovReductionBenchmark
using Test

# `GeometricIntegratorsBase.solutionstep`, which steps a run one solution step at a time as
# VlasovMethods' own scripts do, is not public; nor is `VlasovMethods.VlasovHamiltonian`, the
# energy of a particle run.
test_explicit_imports(VlasovReductionBenchmark;
    all_explicit_imports_are_public = (; ignore = (:solutionstep,)),
    all_qualified_accesses_are_public = (; ignore = (:VlasovHamiltonian,)))
