"""
    VlasovReductionBenchmark

The 1D1V Vlasov–Poisson benchmark of reduced models: the test cases, the full-order runs of
VlasovMethods' grid scheme that produce their data, the dispersion-relation oracle of their
linear rates, and the POD and PSD Galerkin baselines with their metrics.
"""
module VlasovReductionBenchmark

using GeometricBase: initialstate, integrate!, reset!
using GeometricEquations: ODEProblem, functions, parameters
using GeometricIntegrators: ImplicitMidpoint, RK4
using GeometricIntegratorsBase: GeometricIntegrator, solutionstep
using HDF5: attrs, h5open
using LinearAlgebra: dot, mul!, norm
using LinearAlgebra: Diagonal, opnorm, qr, svd
using PoissonSolvers: FFTWBasis, Potential
using Printf: @printf, @sprintf
using QuadGK: quadgk
using Random: Xoshiro
using ReducedBasisMethods: CotangentLiftSVD, PODSVD, ReducedBasis
using ReducedComplexityModeling: Parameter, ParameterSpace
using SimpleSolvers: NonlinearSolverException, isconverged, status
using SpecialFunctions: erfcx
using TOML: TOML
using VlasovMethods: VlasovMethods, GridDistribution, VlasovPoisson

"The velocity interval ``[v_a, v_b]`` of every grid."
const VDOMAIN = (-6.0, 6.0)

include("initial_conditions.jl")
include("dispersion.jl")
include("full_order.jl")
include("parity.jl")
include("metrics.jl")
include("reduction.jl")
include("linear_baselines.jl")
include("dlra.jl")

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
export bug_step, dlra_state_size, full_rank_agreement, initial_factors, run_dlra,
       run_dlra_benchmark, run_full_rank

end
