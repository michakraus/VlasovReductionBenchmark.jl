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
- The velocity-parity split (`parity_split`, `parity_join`): the fluctuation $f - f_0$ around the
  spatial average $f_0(v)$ (`background`) as the canonical vector $(q, p)$, the odd and the even
  part in $v$ on the nodes with $v > 0$.
- The linear baselines: POD-Galerkin in grid coordinates (`pod_reduction`, ReducedBasisMethods'
  `PODSVD`) and the cotangent-lift PSD-Galerkin in parity coordinates (`psd_reduction`,
  `CotangentLiftSVD` on the stacked parity snapshots), each reducing VlasovMethods' Arakawa
  vector field around $f_0$ and integrated with `ImplicitMidpoint` (`run_reduced`). A run whose
  Newton solve does not converge, or whose state is not finite, stops with its status. With a
  full orthogonal basis both reproduce a full-order `ImplicitMidpoint` run to $10^{-10}$; the
  `slow` group checks this on $32 \times 64$.
- The metrics of a reduced run (`trajectory_metrics`, `fitted_rate`): the relative $L^2$ error at
  each snapshot and its maximum, the fitted damping or growth rate, and the drifts of the
  discrete energy, of the $L^1$ and $L^2$ norms and of the entropy, with the count of nodes where
  $f \le 0$. Each error is relative to $f$ and also relative to the fluctuation $f - f_0$: where
  the fluctuation is small against $f$, only the second shows that a reduced model lost it.
- `scripts/linear_baselines.jl --runs-dir <d> --results-dir <r> [--smoke]`: POD and PSD at
  $n \in \{4, 8, 16, 32\}$ (smoke: $\{4, 8\}$) on the three data runs of `scripts/full_order.jl`
  and on the amplitude sweep, one row each in `<r>/linear_baselines.csv`, with every error both
  relative to $f$ and relative to the fluctuation (`max_l2_error_fluct`, `l2_error_fluct`). It
  asserts that the POD projection error of the fluctuation snapshots does not rise with $n$ and is
  at most the PSD error, and it records the
  online cost per reduced step, measured in a fresh process, as a ratio to the full-order cost.
  A Landau fit that finds too few maxima in a reduced run leaves the rate empty, with the reason
  in the row's `rate_status`. The test suite runs its smoke mode in the `slow` group.
- ReducedBasisMethods, ReducedComplexityModeling (for `ParameterSpace`), GeometricEquations (for
  the vector field of VlasovMethods' integrator) and SimpleSolvers (for the convergence of a
  step) are direct dependencies.
- The two symplectic autoencoders (`train_untied`, `train_weight_tied`): GeometricMachineLearning's
  untied `SymplecticAutoencoder(128, n)` on the 128 leading coordinates of the PSD basis, and a
  `MultiModalSymplecticAutoencoder` with one grid modality, `BilinearStiefelLayer(nx, nv, m_x,
  m_v)` with $m_x m_v = n$ and the weight-tied core
  `WeightTiedSymplecticAutoencoder(2n, n; n_ambient_layers = 0)`, started from the layer's
  `solve!` and from `PSDArch(2n, n)` `solve!` for the core's lift. Both map a grid state to $z$
  and back through `encode` and `decode` (`AutoencoderReduction`), so the metrics of the linear
  baselines apply unchanged. Each trains with `CompositeMethod(; manifold =
  FactoredStiefelAdam(), array = Adam())` at step size $10^{-3}$ on the Charbonnier relative
  reconstruction loss; a loss that becomes non-finite ends the training with its step.
- The manifold Galerkin projection $\dot z = (\nabla\Psi)^{+} F(f_0 + \Psi(z))$ of VlasovMethods'
  vector field through the decoder $\Psi$ (`ManifoldGalerkinField`), a least-squares solve
  against the decoder's Jacobian from ForwardDiff (`decoder_jacobian`), integrated with
  `ImplicitMidpoint`. A Jacobian of numerical rank below $n$ ends the run with the status
  `failed at step <s>: rank-deficient decoder`.
- `canonical_flow_comparison`: the cosine and the norm ratio between the canonical flow of the
  discrete energy in parity coordinates and VlasovMethods' vector field, the record that the
  parity split is not canonical for Vlasov–Poisson.
- `scripts/autoencoders.jl --runs-dir <d> --results-dir <r> [--smoke]`: both networks at
  $n \in \{4, 8, 16, 32\}$ with seeds 42, 43 and 44 and 2000 training steps (smoke: $\{4, 8\}$,
  seed 42, 100 steps) on the three data runs and on the amplitude sweep, one row each in
  `<r>/autoencoders.csv` with every metric of the linear baselines, the loss before and after
  training and the training time; the online cost per reduced step, measured in a fresh process,
  in `<r>/autoencoders_cost.toml`; and the energy comparison of the linear Landau run at $t = 0$ in
  `<r>/autoencoders_hamiltonian.toml`. It asserts that each training lowers its loss. The test
  suite runs its smoke mode in the `slow` group. GeometricMachineLearning comes from its `main`
  through `[sources]`; GeometricOptimizers, ForwardDiff and Zygote are direct dependencies.
