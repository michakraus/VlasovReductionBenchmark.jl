# VlasovReductionBenchmark

A benchmark of reduced models for the 1D1V Vlasov–Poisson system. It asks whether a reduced model
with a shared canonical latent space reproduces a full-order simulation better than the linear and
low-rank baselines.

This repository holds the code that produces the data and the results, not the data and the
results themselves: `runs/` and `results/` are not tracked, and every file in them comes from a
script in `scripts/`.

## The test cases

All three run with VlasovMethods' grid scheme, with `RK4()` and a time step of 0.05, on
$x \in [0, 4\pi)$ and $v \in [-6, 6]$, with the initial condition
$f(x, v) = (1 + \alpha \cos kx)\, g(v)$ and $k = 0.5$:

| case | $g(v)$ | $\alpha$ | $t$ |
|:--|:--|:--|:--|
| `linear_landau` | $e^{-v^2/2}/\sqrt{2\pi}$ | 0.01 | $[0, 40]$ |
| `nonlinear_landau` | $e^{-v^2/2}/\sqrt{2\pi}$ | 0.5 | $[0, 60]$ |
| `two_stream` | $v^2 e^{-v^2/2}/\sqrt{2\pi}$ | 0.05 | $[0, 60]$ |

Linear Landau damping also runs at the amplitudes of the generalisation sweep: $\alpha \in \{0.005,
0.01, 0.02\}$ for training and $\alpha \in \{0.015, 0.03\}$ for testing.

`landau_root(k)` and `two_stream_root(k)` give the complex root $\omega(k)$ of the dispersion
relation of each equilibrium, which the full-order runs are checked against.

## The full-order data

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project scripts/full_order.jl --runs-dir runs --results-dir results
```

The grid is $64 \times 128$; `--smoke` runs the same cases and checks on $32 \times 64$. The script
writes one HDF5 file per run to `runs/`, and the cost per step to
`results/full_order_cost.toml`. It stops with an error when a run breaks its invariants or when a
fitted rate misses the dispersion relation.

## The linear baselines

```sh
julia --project scripts/linear_baselines.jl --runs-dir runs --results-dir results
```

POD-Galerkin and the cotangent-lift PSD-Galerkin reduce the fluctuation around the spatial
average $f_0(v)$ of each run's initial condition at the latent dimensions
$n \in \{4, 8, 16, 32\}$ (`--smoke`: $\{4, 8\}$, on the smoke runs), and integrate the Galerkin
projection of VlasovMethods' vector field with `ImplicitMidpoint`. PSD works in the parity
coordinates $q$, the odd, and $p$, the even part of $f - f_0$ in $v$. The script writes one row per
method, case and $n$, and per test amplitude of the sweep, to `results/linear_baselines.csv`, with
each relative $L^2$ error both against $f$ and against the fluctuation $f - f_0$ (the columns that
end in `_fluct`), and the online cost per reduced step, measured in a fresh process, to
`results/linear_baselines_cost.toml`. It stops with an error when the POD projection error rises
with $n$ or exceeds the PSD error.

## The autoencoders

```sh
julia --project scripts/autoencoders.jl --runs-dir runs --results-dir results
```

Two symplectic autoencoders of GeometricMachineLearning reduce the same fluctuations at the same
latent dimensions: the untied `SymplecticAutoencoder` on the 128 leading PSD coordinates of each
run, and a multi-modal autoencoder with one grid modality, a `BilinearStiefelLayer` with
$m_x m_v = n$ modes per parity and a weight-tied core from $2n$ to $n$, whose bases and core lift
start from their SVD solutions. Each trains for 2000 steps with three seeds (`--smoke`: 100 steps,
one seed) on the Charbonnier relative reconstruction loss. The reduced model is the manifold
Galerkin projection $\dot z = (\nabla\Psi)^{+} F(f_0 + \Psi(z))$ of VlasovMethods' vector field
through the decoder $\Psi$, integrated with `ImplicitMidpoint`. The script writes one row per
network, case, $n$ and seed, and per test amplitude of the sweep, to `results/autoencoders.csv`,
with the losses before and after training, the online cost per reduced step to
`results/autoencoders_cost.toml`, and the cosine and the norm ratio between the canonical flow of
the energy in parity coordinates and VlasovMethods' vector field to
`results/autoencoders_hamiltonian.toml`. A training whose loss does not fall, or becomes
non-finite, is a row with a `failed` status, and the script goes on.

## The DLRA solver

```sh
julia --project scripts/dlra.jl --runs-dir runs --results-dir results
```

A dynamical low-rank solver integrates the factored state $f = f_0 + X S V^{\mathsf T}$ of rank
$r \in \{4, 8, 16, 32\}$ (`--smoke`: $\{4, 8\}$, on the smoke runs), with $X$ and $V$
orthonormal in the inner products weighted by $h_x$ and $h_v$. Each step is the BUG step with
augmented bases (K-step, L-step, then the Galerkin S-step on the bases of up to $2r$ columns),
truncated back to rank $r$; each substep is one RK4 step. The discretisation is spectral in $x$
and centred differences in $v$, periodic in $v$, with the same Poisson solve as the full-order
runs. The time step is 0.025, half the full-order step, because RK4 with the spectral derivative
is unstable at 0.05 on $64 \times 128$. The solver does not conserve the mass exactly; the drift
is written, not asserted.

Before it measures, the script checks the solver. At full rank in $x$ it agrees with a full-rank
solver of the same discretisation to $10^{-10}$ over 20 steps on $32 \times 64$, and the
full-rank solver and the run of rank 16 meet the rate check of linear Landau damping (2 % over
$[0, 40]$ on $64 \times 128$; `--smoke`: 10 % over $[0, 25]$ on $32 \times 64$). It writes one row
per case and rank, and per test amplitude of the sweep and rank, to `results/dlra.csv`, with the
metrics of the linear baselines against the full-order runs, the state size
$r(N_x + N_v) + r^2$, the error of the full-rank solver against the same full-order run (the
difference of the two discretisations), the error of the run of rank $r$ against the full-rank
solver relative to its fluctuation (`max_l2_error_fluct_full_rank`, the error of the rank
alone), and the online cost per DLRA step, measured in a fresh
process, to `results/dlra_cost.toml`. The column `online_ratio` is per step: the cost of one DLRA
step of 0.025 over the cost of one full-order step of 0.05, so the cost per unit of simulated time
is twice the ratio. The column `n` holds the rank $r$.

## The particle half

```sh
julia --project scripts/particles.jl --runs-dir runs --results-dir results
```

A PIC run of nonlinear Landau damping with 4096 particles over $[0, 40]$, the POD mode counts of
its labelled and sorted snapshots, and the training of the `:sort`, `:slot` and `:pairwise`
frontends of `SymmetricSymplecticAutoencoder` on 256 of its particles. The script writes the run
to `runs/particles_nonlinear_landau.h5`, and `particle_pod_modes.csv`, `particle_frontends.csv`
and `particle_cost.toml` to `results/`. It stops with an error when the energy of the run drifts
by more than $10^{-3}$ relative. `--smoke` runs 512 particles over $[0, 10]$ and trains on 64 of
them. `particle_frontends.csv` sets each frontend beside two POD baselines of the same particles:
`pod`, of the snapshots sorted by position at each time, and `pod_labelled`, of the snapshots with
their labels, which the frontends train on. `--labelled-pod` runs nothing and trains nothing: it
writes the `pod_labelled` rows into the `particle_frontends.csv` of `--results-dir` from the run
stored in `--runs-dir`, and leaves the other rows as they are.

## The verdict

```sh
julia --project scripts/verdict.jl --results-dir <r>... --out results/verdict.md
```

The script reads the result files of the full modes of the other scripts from the results
directories `<r>`, whose paths each name the commit that wrote them, as in
`VlasovReductionBenchmark-I2-760c7a5/results`, and writes one Markdown page. For each case and
criterion it sets POD, PSD, DLRA and the two autoencoders side by side at each $n$: the
autoencoders by the median over the seeds, with the range; DLRA by its error against its own
full-rank solver; the online cost per unit of simulated time. An autoencoder beats the best
baseline on a criterion when it is strictly better at three of the four $n$. A row whose status
is not `ok` enters no median and is listed. The last line of the page answers whether the
Vlasov half of the plan goes on. A missing input file stops the script with its name.

## The reconstruction probes

```sh
julia --project scripts/reconstruction.jl --runs-dir runs --results-dir results
```

The probes measure the reconstruction error alone, with no reduced dynamics, on five datasets of
$64 \times 128$: the passive advection $f = M(v)\, g(x - vt)$ of a Gaussian bump of width 1 in
$x$ times the Maxwellian, without an electric field, over half and over one recurrence time
$T_R = 2\pi / (k\, \Delta v)$ of the grid (`advection_half`, `advection_full`), and three runs to
$t = 100$: the two-stream instability with the profile of `two_stream` (`two_stream_long`) and
with two Maxwellians of thermal width 0.5 at $v = \pm 1.5$ (`two_stream_beams_long`), and
nonlinear Landau damping (`nonlinear_landau_long`). The script writes the three runs to `runs/`.
Each dataset is measured in sample, on every snapshot, and in extrapolation in time, trained on
$t \le T/2$ and tested on the rest. POD, PSD, the untied network (three seeds) and the weight-tied
network (one seed) are built or trained on the training snapshots, around the spatial average of
the first of them, at $n \in \{4, 8, 16, 32\}$; the truncated SVD of rank $r = n$ of each test
snapshot is the reference that a factored state of size $r(N_x + N_v) + r^2$ can reach. The
script writes one row per dataset, split, method, $n$ and seed, with the aggregate and the
per-snapshot relative errors of the test fluctuations, to `results/reconstruction.csv`, the
singular values of each dataset with the number of modes that reach 99 % and 99.99 % of
$\sum \sigma_i^2$ to `results/singular_values.csv`, and a page of both to
`results/reconstruction.md`. `--smoke` runs $32 \times 64$, $n \in \{4, 8\}$, the long runs to
$t = 40$, one seed and 100 training steps. A training whose loss does not fall, or becomes
non-finite, is a row with a `failed` status, and the script goes on.

## The input channels

```sh
julia --project scripts/channels.jl --runs-dir runs --results-dir results
```

The script measures the reconstruction error of the weight-tied multi-modal network through its
input channels, in sample, on the five datasets of *The reconstruction probes*, around the
spatial average of the first snapshot of each, at $n \in \{4, 8, 16, 32\}$, with seed 42 and 2000
training steps. It reads each of the three long runs from `runs/` where a file there holds it on
the same grid, at the same amplitude and to the same end time, and otherwise runs and writes it.
The grid channel, `BilinearStiefelLayer(64, 128, m_x, m_v)`, is measured at every split
$m_x m_v = n$ with $m_v \le N_v/2$ (18 splits): the error of the frontend and its inverse alone,
the Stage I error of the untrained network, and the error after training at the split of *The
autoencoders* (`PARITY_MODES`) and at the split of the smallest Stage I error. The DLRA channel,
`LowRankStiefelLayer` with the bases of the trained grid network at the split of
`PARITY_MODES`, encodes the truncated SVD $(X, S, V)$ of rank
$r_{\mathrm{in}} \in \{4, 8, 16\}$ of each snapshot into the same core; its error is measured
against the fluctuation and against its input.
A network with both channels, its DLRA frontend set from the dense states $X S V^{\mathsf T}$ of
rank 8, trains on the sum of the two losses; the script measures each channel and both
translations between them against the fluctuation. It writes `results/channel_splits.csv`,
`results/channel_dlra.csv`, `results/channel_joint.csv` and a page of the three,
`results/channels.md`. `--smoke` runs $32 \times 64$, $n \in \{4, 8\}$, the long runs to
$t = 40$, $r_{\mathrm{in}} = 4$ and 100 training steps. A training whose loss does not fall, or
becomes non-finite, gives its rows a `failed` status and no errors of the trained network, and
the script goes on.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The suite runs the smoke mode of `scripts/full_order.jl` in the `core` group, and its full mode,
the smoke mode of `scripts/linear_baselines.jl` and the full-basis checks of the Galerkin models in
the `slow` group; `Pkg.test()` runs both. The `slow` group also runs the smoke mode of
`scripts/autoencoders.jl` and the checks of the two networks; its full mode is a script run and
no test. The checks of the DLRA solver run in the `core` group, and the smoke and the full mode of
`scripts/dlra.jl` in the `slow` group. The `slow` group also runs the smoke mode of
`scripts/particles.jl`; its full mode trains for hours, so it is the script run of *The particle
half* and no test. The `core` group runs `scripts/verdict.jl` and the rules of its comparison on
fixture result files. The `core` group also runs the checks of the reconstruction probes, and the
`slow` group the smoke mode of `scripts/reconstruction.jl`; its full mode is the script run of
*The reconstruction probes* and no test. The checks of the input channels without training run in
the `core` group, and the checks of their rows and the smoke mode of `scripts/channels.jl` in the
`slow` group; its full mode is the script run of *The input channels* and no test.
