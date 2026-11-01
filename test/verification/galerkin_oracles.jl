using GeometricBase: initialstate, integrate!, reset!
using GeometricIntegrators: ImplicitMidpoint
using GeometricIntegratorsBase: solutionstep
using LinearAlgebra: norm, qr
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: VlasovMethods
using VlasovReductionBenchmark: GridCoordinates, LinearReduction, ParityCoordinates
using Test

# With a full orthogonal basis the reduced model is the full-order model in other coordinates, so
# its `ImplicitMidpoint` trajectory, decoded, is the full-order `ImplicitMidpoint` trajectory.
const NX, NV, NSTEPS = 32, 64, 5
const CASE = VlasovCase("linear_landau", maxwellian, 0.01, 0.5, 0.25)

# Five full-order `ImplicitMidpoint` steps of VlasovMethods' own integrator, every state.
function full_order_midpoint()
    model = vlasov_model(CASE, NX, NV)
    int = VlasovMethods.GeometricIntegrator(model, (0.0, CASE.tend), 0.05, ImplicitMidpoint())
    solstep = solutionstep(int.integrator, initialstate(int.equation))
    states = [reshape(copy(solstep.q), NX, NV)]
    for s in 1:NSTEPS
        reset!(solstep, s * 0.05)
        integrate!(solstep, int.integrator)
        push!(states, reshape(copy(solstep.q), NX, NV))
    end
    return states
end

const FULL = full_order_midpoint()

function oracle_error(r)
    run = run_reduced(r, CASE, NX, NV, FULL[1])
    @test run.status == "ok"
    reduced = [decode(r, z) for z in run.states]
    return maximum(norm(reduced[s] - FULL[s]) / norm(FULL[s]) for s in 1:(NSTEPS + 1))
end

@testset "POD-Galerkin, full basis" begin
    f0 = background(FULL[1])
    V = Matrix(qr(randn(Xoshiro(42), NX * NV, NX * NV)).Q)
    @test size(V) == (2048, 2048)
    @test oracle_error(LinearReduction(GridCoordinates(), V, f0, NX, NV)) < 1e-10
end

@testset "PSD-Galerkin, full basis" begin
    f0 = background(FULL[1])
    Φ = Matrix(qr(randn(Xoshiro(42), NX * NV ÷ 2, NX * NV ÷ 2)).Q)
    @test size(Φ) == (1024, 1024)
    A = [Φ zero(Φ); zero(Φ) Φ]
    @test oracle_error(LinearReduction(ParityCoordinates(), A, f0, NX, NV)) < 1e-10
end
