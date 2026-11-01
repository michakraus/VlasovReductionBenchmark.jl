"""
    VlasovReductionBenchmark

The 1D1V Vlasov–Poisson benchmark of reduced models: the test cases, the full-order runs of
VlasovMethods' grid scheme that produce their data, the dispersion-relation oracle of their
linear rates, and the POD and PSD Galerkin baselines with their metrics.
It also holds the two symplectic autoencoders and their manifold Galerkin models.

The particle half: a PIC run of nonlinear Landau damping, the POD mode counts of its labelled
and sorted snapshots, and the particle frontends of GeometricMachineLearning's
`SymmetricSymplecticAutoencoder` trained on them.
"""
module VlasovReductionBenchmark

using ForwardDiff: ForwardDiff
using GeometricBase: initialstate, integrate!, reset!
using GeometricEquations: ODEProblem, functions, parameters
using GeometricIntegrators: ImplicitMidpoint, RK4
using GeometricIntegratorsBase: GeometricIntegrator, solutionstep
using GeometricMachineLearning: GeometricMachineLearning, BilinearStiefelLayer, CPU,
                                CharbonnierRelativeLoss, MultiModalSymplecticAutoencoder,
                                NeuralNetwork, PSDArch, SymmetricSymplecticAutoencoder,
                                SymplecticAutoencoder, WeightTiedSymplecticAutoencoder
using GeometricOptimizers: Adam, CompositeMethod, FactoredStiefelAdam, TrainingOptimizer,
                           optimization_step!, solve!
using HDF5: attrs, h5open
using LinearAlgebra: Diagonal, dot, mul!, norm, opnorm, qr, svd, svdvals
using PoissonSolvers: FFTWBasis, Potential
using PoissonSolvers: PeriodicBasisSpline
using Printf: @printf, @sprintf
using QuadGK: quadgk
using Random: Xoshiro
using ReducedBasisMethods: CotangentLiftSVD, PODSVD, ReducedBasis
using ReducedComplexityModeling: Parameter, ParameterSpace
using Serialization: deserialize, serialize
using SimpleSolvers: NonlinearSolverException, isconverged, status
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
include("parity.jl")
include("metrics.jl")
include("reduction.jl")
include("linear_baselines.jl")
include("autoencoders.jl")
include("dlra.jl")
include("particles.jl")

export VlasovCase, grid, initial_condition, linear_landau, maxwellian, nonlinear_landau,
       two_stream, two_stream_profile, vlasov_model
export dielectric_maxwellian, dielectric_quadrature, dielectric_two_stream, dispersion_root,
       landau_root, plasma_dispersion, two_stream_derivative, two_stream_root
export check_rate, growth_rate_fit, landau_rate_fit, run_case, run_full_order, write_run
export background, parity_join, parity_snapshots, parity_split
export fitted_rate, l1_norm, max_drift, relative_l2_errors, trajectory_metrics
export check_pod_optimality, checked_projection_errors, decode, encode, full_order_field,
       galerkin_field, pod_reduction, psd_basis, psd_reduction, relative_projection_error,
       run_reduced
export run_linear_baselines
export AutoencoderReduction, ManifoldGalerkinField, canonical_flow_comparison,
       decoder_jacobian, run_autoencoders, train_untied, train_weight_tied
export bug_step, dlra_state_size, full_rank_agreement, initial_factors, run_dlra,
       run_dlra_benchmark, run_full_rank
export pod_error, pod_modes, run_particles, run_pic, sample_particles, sorted_snapshots,
       train_frontend, write_particle_run

end
