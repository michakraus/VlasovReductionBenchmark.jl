"""
    VlasovReductionBenchmark

The 1D1V Vlasov–Poisson benchmark of reduced models: the test cases, the full-order runs of
VlasovMethods' grid scheme that produce their data, and the dispersion-relation oracle of their
linear rates.

The particle half: a PIC run of nonlinear Landau damping, the POD mode counts of its labelled
and sorted snapshots, and the particle frontends of GeometricMachineLearning's
`SymmetricSymplecticAutoencoder` trained on them.
"""
module VlasovReductionBenchmark

using AbstractNeuralNetworks: NeuralNetwork
using GeometricBase: initialstate, integrate!, reset!
using GeometricIntegrators: RK4
using GeometricIntegratorsBase: solutionstep
using GeometricMachineLearning: SymmetricSymplecticAutoencoder
using GeometricOptimizers: Adam, CompositeMethod, FactoredStiefelAdam, TrainingOptimizer,
                           optimization_step!
using HDF5: attrs, h5open
using LinearAlgebra: dot
using LinearAlgebra: norm, svdvals
using PoissonSolvers: FFTWBasis, Potential
using PoissonSolvers: PeriodicBasisSpline
using Printf: @printf, @sprintf
using QuadGK: quadgk
using Random: Xoshiro
using SpecialFunctions: erfcx
using TOML: TOML
using VlasovMethods: VlasovMethods, GridDistribution, VlasovPoisson
using VlasovMethods: ParticleDistribution, SplittingMethod, run!
using Zygote: Zygote

"The velocity interval ``[v_a, v_b]`` of every grid."
const VDOMAIN = (-6.0, 6.0)

include("initial_conditions.jl")
include("dispersion.jl")
include("full_order.jl")
include("particles.jl")

export VlasovCase, grid, initial_condition, linear_landau, maxwellian, nonlinear_landau,
       two_stream, two_stream_profile, vlasov_model
export dielectric_maxwellian, dielectric_quadrature, dielectric_two_stream, dispersion_root,
       landau_root, plasma_dispersion, two_stream_derivative, two_stream_root
export check_rate, growth_rate_fit, landau_rate_fit, run_case, run_full_order, write_run
export pod_error, pod_modes, run_particles, run_pic, sample_particles, sorted_snapshots,
       train_frontend, write_particle_run

end
