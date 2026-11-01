# Changelog

All notable changes to VlasovReductionBenchmark.jl are recorded here. Nothing here is registered,
so `[Unreleased]` has no target version.

## [Unreleased]

### New Features

- The project, its tests (Aqua and ExplicitImports in the suite) and the shared git hooks and CI.
  VlasovMethods comes from its `main` through `[sources]`, because no registered release binds
  ReducedComplexityModeling 0.5.
- The three test cases on the grid $x \in [0, 4\pi)$, $v \in [-6, 6]$: linear Landau damping
  (`linear_landau`, $\alpha = 0.01$, $t \le 40$), nonlinear Landau damping (`nonlinear_landau`,
  $\alpha = 0.5$, $t \le 60$) and the two-stream instability (`two_stream`, $\alpha = 0.05$,
  $t \le 60$), with `vlasov_model` to build VlasovMethods' grid model of each on a given grid.
- The oracle of the linear rates: the roots of the dielectric function of the Maxwellian
  (`landau_root`, through the plasma dispersion function on the Landau contour) and of the
  two-stream profile (`two_stream_root`, through the plasma dispersion function or by quadrature
  over the real line).
- `scripts/full_order.jl --runs-dir <d> --results-dir <r> [--smoke]`: the full-order data on
  $64 \times 128$ (smoke: $32 \times 64$), one HDF5 file per run, with the snapshots every 0.5,
  the field energy and $|E_1|$ every step, the drifts of the mass, the energy, the $L^2$ norm and
  the entropy, and the train/test label of the amplitude sweep. The script asserts the mass to
  $10^{-12}$ and the energy to $10^{-4}$ relative in every run, checks the fitted rates of linear
  Landau damping and of the two-stream instability against the oracle, and writes the cost per
  step, measured in a fresh process, to `<r>/full_order_cost.toml`. The test suite runs its smoke
  mode, and its full mode in the `slow` group.
- `scripts/particles.jl --runs-dir <d> --results-dir <r> [--smoke]`: a PIC run of nonlinear Landau
  damping ($\alpha = 0.5$, $k = 0.5$) with VlasovMethods' `SplittingMethod` on a periodic cubic
  B-spline potential with 32 cells. It draws 4096 particles (smoke: 512) by inverse-CDF sampling
  with `Xoshiro(42)` and steps them with $\Delta t = 0.05$ over $[0, 40]$ (smoke: $[0, 10]$). It
  writes the wrapped positions and velocities every 0.5, the weights, $|E_1|$ at every step,
  and the drifts of the energy and the momentum to `<d>/particles_nonlinear_landau.h5`. It asserts
  the energy to $10^{-3}$ relative and prints the momentum drift without asserting it.
- `<r>/particle_pod_modes.csv` gives the POD modes reaching 99 % and 99.99 % of the snapshot
  energy, for the labelled and the position-sorted snapshots of all particles.
  `<r>/particle_frontends.csv` gives the relative reconstruction error of
  `SymmetricSymplecticAutoencoder` with the `:sort`, `:slot` and `:pairwise` frontends, beside
  POD with $n$ modes of the same particles, sorted in each snapshot (`pod`) and with their labels
  (`pod_labelled`). `--labelled-pod` writes the `pod_labelled` rows into an existing
  `particle_frontends.csv` from the stored run, with no training. The frontends train on the
  256 particles of smallest initial position (smoke: 64) at $n \in \{4, 8, 16, 32\}$
  (smoke: $\{4, 8\}$), for 2000 steps (smoke: 100) at seeds 42, 43 and 44 (smoke: 42), with
  a Charbonnier relative loss, `FactoredStiefelAdam`/`Adam` at step size $10^{-3}$. A run
  whose loss or gradient is not finite is marked `failed at step <s>`, and the script prints the
  count of such runs. `<r>/particle_cost.toml` gives the wall time per PIC step of a `run!` of
  100 steps in a fresh process, with the HDF5 write of each step included. The test suite runs
  the smoke mode of the script in the `slow` group; the full mode is a script run, outside the
  suite. The training takes GeometricMachineLearning from its `main` through `[sources]`.
