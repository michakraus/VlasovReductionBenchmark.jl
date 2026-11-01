using ExplicitImports
using VlasovReductionBenchmark
using Test

# `GeometricIntegratorsBase.solutionstep`, which steps a run one solution step at a time as
# VlasovMethods' own scripts do, is not public.
# `VlasovMethods.VlasovHamiltonian`, the energy of a particle run, is not public either.
test_explicit_imports(VlasovReductionBenchmark;
    all_qualified_accesses_are_public = (; ignore = (:VlasovHamiltonian,)),
    all_explicit_imports_are_public = (; ignore = (:solutionstep,)))
