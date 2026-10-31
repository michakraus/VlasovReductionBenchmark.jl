"""
    VlasovReductionBenchmark

The 1D1V Vlasov–Poisson benchmark of reduced models: the test cases, the full-order runs of
VlasovMethods' grid scheme that produce their data, and the dispersion-relation oracle of their
linear rates.
"""
module VlasovReductionBenchmark

using GeometricBase: initialstate, integrate!, reset!
using GeometricIntegrators: RK4
using GeometricIntegratorsBase: solutionstep
using HDF5: attrs, h5open
using LinearAlgebra: dot
using PoissonSolvers: FFTWBasis, Potential
using Printf: @printf, @sprintf
using QuadGK: quadgk
using SpecialFunctions: erfcx
using TOML: TOML
using VlasovMethods: VlasovMethods, GridDistribution, VlasovPoisson

"The velocity interval ``[v_a, v_b]`` of every grid."
const VDOMAIN = (-6.0, 6.0)

include("initial_conditions.jl")
include("dispersion.jl")
include("full_order.jl")

export VlasovCase, grid, initial_condition, linear_landau, maxwellian, nonlinear_landau,
       two_stream, two_stream_profile, vlasov_model
export dielectric_maxwellian, dielectric_quadrature, dielectric_two_stream, dispersion_root,
       landau_root, plasma_dispersion, two_stream_derivative, two_stream_root
export check_rate, growth_rate_fit, landau_rate_fit, run_case, run_full_order, write_run

end
