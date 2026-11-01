# Known issues

### K1 · The initial conditions do not vanish to `eps · max f` at `v = ±6`, as VlasovMethods asks of its periodic-in-`v` grid scheme.

- **location:** `src/initial_conditions.jl:72`
- **evidence:** the initial conditions are evaluated at the nodes as they stand, with no
  cut-off. At `v = ±6` the Maxwellian is `6.1e-9` and the two-stream profile `2.2e-7`, which is
  `6.9e7` and `3.4e9` times `eps() * maximum(f)` (VlasovMethods' docstring of `VlasovPoisson`,
  `src/models/vlasov_poisson.jl:21`, asks for at most `1`). The runs of
  `scripts/full_order.jl` show no effect on the invariants: on 64 × 128 the largest relative
  mass drift is `1.6e-14`, and the largest relative energy drift `3.5e-10` (linear Landau),
  `8.3e-7` (nonlinear Landau) and `6.3e-10` (two-stream). The effect on the snapshots
  themselves is not measured.
- **kind:** not verified
- **found:** 2026-10-31

### K2 · `landau_root` and `two_stream_root` accept a wave number for which they return a wrong root or do not converge.

- **location:** `src/dispersion.jl:96`
- **evidence:** the benchmark uses `k = 0.5` only. `landau_root(-0.5)` returns
  `1.4156618886045367 + 0.15335946690960475im`, the growing root, with no error, and
  `two_stream_root(0.1)` throws "Newton's iteration did not converge in 50 steps from
  ω = 0.0 + 0.25im" from its default guess.
- **kind:** found late
- **found:** 2026-10-31

### K3 · A run that throws leaves the file of the same name from an earlier invocation in `<d>`.

- **location:** `src/full_order.jl:133`
- **evidence:** `run_case` writes its file only when the run completes, and removes no file
  when it throws, so after a failed run `<d>` can hold a file of that case from an earlier
  invocation, which reads as current. The script itself stops with the run's error.
- **kind:** found late
- **found:** 2026-10-31

### K4 · `VlasovMethods.GeometricIntegrator(model, tspan, dt, RK4())` does not infer a concrete type, so the step loops that use its result dispatch at run time.

- **location:** `src/full_order.jl:88`
- **evidence:** the same call is at `src/full_order.jl:244`, in `measure_step_time`. VlasovMethods
  (`main` at `8937e91`, `src/gridbased/vlasov_poisson_1d1v.jl:125-131`) builds its `ODEProblem`
  from a parameter `NamedTuple`. `Base.return_types` of the call on the 64 × 128 linear Landau
  model gives a `Union` of two `UnionAll` types, whose `ODE` type parameter holds
  `@NamedTuple{arakawa::DataType, v::DataType, G::DataType, …}`; the object built at run time has
  concrete field types. On 64 × 128 one `RK4()` step takes 2.72–2.76e-4 s, and of the 10.7 kB
  that a step allocates, about 2 kB is boxing and dynamic dispatch. A function barrier around
  the step loop changes the time per step by a factor of 0.99–1.06, within the noise of the
  machine, so the time per step in `full_order_cost.toml` is not distorted. The fix belongs in
  VlasovMethods.
- **kind:** upstream
- **found:** 2026-10-31

### K5 · A relative error or a relative drift whose reference is zero is `NaN`.

- **location:** `src/metrics.jl:10`
- **evidence:** `relative_l2_errors(ones(2, 2, 1), ones(2, 2, 1), [1.0, 1.0])` returns `[NaN]`:
  a snapshot equal to the background has no fluctuation to relate the error to. `max_drift`
  (`src/metrics.jl:32`) returns `NaN` for `[0.0, 1.0]`, a series that starts at zero. In the
  smoke mode no data row of `linear_baselines.csv` has a `NaN` error: the script test compares
  each row's errors with their maximum, which a `NaN` fails. The full mode is not checked for it.
- **kind:** found late
- **found:** 2026-11-01

### K6 · No test shows that the online cost of a reduced step is measured after its warm-up.

- **location:** `src/reduction.jl:338`
- **evidence:** `reduced_step_time` runs `warmup` steps before the timed ones. A mutant that
  removes the call `timed_steps!(solstep, int, 1, warmup)` survives every test, because the timed
  steps give a time per step with or without it; only the size of that time differs.
- **kind:** not verified
- **found:** 2026-11-01

### K7 · `encode` and `decode` have the names of two functions that GeometricMachineLearning exports.

- **location:** `src/reduction.jl:65`
- **evidence:** GeometricMachineLearning (`main` at `d3892de5`) exports `encode` and `decode`
  (`src/GeometricMachineLearning.jl:282` there), with methods for its reduced systems
  (`src/reduced_system/reduced_system.jl:408-422` there). This package exports its own `encode`
  and `decode` of a `LinearReduction` and of an `AutoencoderReduction`, the same concept for a
  reduction. The package imports GeometricMachineLearning and calls its `encode` qualified
  (`src/autoencoders.jl:102`); a script that loads both packages and calls `encode`
  unqualified gets an `UndefVarError` for an ambiguous binding, so such a script qualifies the
  call.
- **kind:** found late
- **found:** 2026-11-01

### K8 · No test shows that the online cost of a DLRA step is measured after its warm-up.

- **location:** `src/dlra.jl:421`
- **evidence:** `dlra_step_time` runs `warmup` steps before the timed ones. A mutant that removes
  the line `_, X, S, V = timed_bug_steps(d, X, S, V, warmup)` survives `test/dlra.jl` and
  `test/integration/dlra_script.jl` (`mutate.jl`: SURVIVED, 469 of 469 pass), because the timed
  steps give a time per step with or without it; only the size of that time differs.
- **kind:** not verified
- **found:** 2026-11-01

### K9 · The row, the CSV writer and the cost loop of the DLRA solver repeat those of the linear baselines.

- **location:** `src/dlra.jl:374`
- **evidence:** `dlra_row` (`src/dlra.jl:374`) and `write_dlra` (`src/dlra.jl:496`) are
  `baseline_row` (`src/linear_baselines.jl:40`) and `write_baselines`
  (`src/linear_baselines.jl:163`) with another column tuple, and `dlra_costs` (`src/dlra.jl:462`)
  is `reduced_costs` (`src/linear_baselines.jl:123`) with another key, another function to time,
  and another file name, step and method label. A helper that takes the columns, and one that
  takes these settings, would remove about 40 lines; it changes `src/linear_baselines.jl`.
- **kind:** found late
- **found:** 2026-11-01

### K10 · The norm ratio of the canonical-flow record depends on the grid.

- **location:** `src/autoencoders.jl:460`
- **evidence:** `canonical_flow_comparison` divides the norm of the canonical flow of the
  discrete energy by the norm of the Arakawa field. The gradient of the energy carries the cell
  area `h_x h_v` and the field does not, so the ratio changes with the grid: for linear Landau at
  `t = 0`, `canonical_flow_comparison(case, initial_condition(case, grid(case, nx, nv)...))`
  gives 662.7 on 32 × 64 and 159.5 on 64 × 128. `autoencoders_hamiltonian.toml` writes `nx` and
  `nv` beside the ratio, so only ratios of one grid compare.
- **kind:** found late
- **found:** 2026-11-01

### K11 · The cosine of the canonical-flow record does not tell the canonical flow from the gradient.

- **location:** `src/autoencoders.jl:460`
- **evidence:** the canonical flow `J ∇H` and the gradient `∇H` have the same norm, and both are
  orthogonal to the Arakawa field to round-off: on the smoke grid at `t = 0` the cosines are
  `-1.96e-18` and `-1.39e-18`. A mutant that writes `∇H`, or `J ∇H` with the sign of one half
  flipped, in place of the flow changes neither written value beyond round-off. A cosine near
  zero is therefore no evidence that the parity split is not canonical; only the norm ratio
  carries information.
- **kind:** found late
- **found:** 2026-11-01

### K12 · The seeds of the weight-tied network give losses that differ only at round-off.

- **location:** `src/autoencoders.jl:193`
- **evidence:** the seed sets the layers between the frontend and the PSD lift of the core,
  whose decode–encode is the projection of that lift whatever those layers are. On the linear
  Landau run of `scripts/full_order.jl` (64 × 128, `n = 16`), the loss after the frontend's and
  the lift's `solve!` is 0.7912309074393263 for seeds 42 and 43. In the full mode of
  `scripts/autoencoders.jl`, the column `loss_stage_I` of the 36 weight-tied data rows is equal
  for seeds 42, 43 and 44 in each of the 12 groups of one case and one `n`, and `loss_final`
  differs between the seeds of a group by at most `1.6e-7` relative. The range over the three
  seeds is then near zero for the reconstruction. The seed changes the latent coordinates: on the
  smoke grid at `n = 4`, the encoded snapshot of seeds 42 and 43 differs by 0.114 at a norm of
  0.165.
- **kind:** found late
- **found:** 2026-11-01

### K14 · The decoder Jacobian of `decoder_jacobian` and of `ManifoldGalerkinField` infers `Any`, so each field call dispatches at run time.

- **location:** `src/autoencoders.jl:81`
- **evidence:** the same call is at `src/autoencoders.jl:342`. `ForwardDiff.jacobian(r.decoder,
  z)` has no config, so ForwardDiff picks its chunk size at run time, and
  `Base.return_types(decoder_jacobian, (typeof(r), Vector{Float64}))` gives `Any[Any]` for both
  networks at `n = 4` and `n = 32`. The field's own result is concrete. A `JacobianConfig` with a
  fixed chunk for `Float64` states changes `reduced_step_time` on 64 × 128 by a factor of
  0.991–1.007 (both networks, `n = 4` and `n = 32`, ratio of the medians of three runs, load
  21–25), within the noise of the machine. Such a config does not cover the dual states of the
  Newton Jacobian, which make 20 of 177 field calls in 10 steps at `n = 4`, 114 of 316
  (weight-tied) and 60 of 217 (untied) at `n = 32`. The online-cost columns of
  `autoencoders.csv` and `autoencoders_cost.toml` include this dispatch.
- **kind:** found late
- **found:** 2026-11-01

### K15 · `ManifoldGalerkinField` allocates two vectors of the grid's size on every call.

- **location:** `src/autoencoders.jl:344`
- **evidence:** the call allocates `f = r.lift * r.decoder(z)` and `ḟ = similar(f)`
  (`src/autoencoders.jl:348`), each `nx · nv` long, 64 KiB each on 64 × 128 for a `Float64`
  state. The `GalerkinField` of the linear baselines (`src/reduction.jl:205`) keeps buffers for
  a `Float64` state and allocates only for a state of another element type. The effect on
  the online cost is not measured.
- **kind:** found late
- **found:** 2026-11-01

### K16 · The 99.99 % POD mode count of the particle snapshots reaches the number of snapshots for labelled and for sorted snapshots, so that column does not separate the two.

- **location:** `src/particles.jl:37`
- **evidence:** the smoke run of `scripts/particles.jl` (512 particles, 21 snapshots over
  `[0, 10]`) writes to `particle_pod_modes.csv` `labelled,512,0.9999,21` and
  `sorted,512,0.9999,21`: both counts equal the number of snapshots, the largest count that the
  snapshot matrix allows. At 99 % the counts differ, 13 labelled against 11 sorted. The
  thresholds are the plan's (99 % and 99.99 %).
- **kind:** found late
- **found:** 2026-11-01

### K17 · No test reaches a training run of `train_frontend` that stops on a finite loss and a non-finite gradient.

- **location:** `src/particles.jl:308`
- **evidence:** the mutant `status = isfinite(loss(ps)) ? "ok" : "failed at step $taken"`, with
  `taken == steps &&` dropped, survives the `particles.jl` test file: such a run then reports
  `"ok"`, while the docstring of `train_frontend` gives `"failed at step s"`. On a non-finite
  loss, the edge that the tests reach, the mutant is equivalent, because `loss(ps)` is
  deterministic.
- **kind:** missing test
- **found:** 2026-11-01

### K18 · No test reaches the full mode of `run_particles` or of `--labelled-pod`, so no test shows that either takes its settings from `particle_mode(smoke)`.

- **location:** `src/particles.jl:499`
- **evidence:** the core test checks `particle_mode(false) == PARTICLE_FULL_MODE` and
  `particle_mode(true) == PARTICLE_SMOKE_MODE`, but no core test calls `run_particles`. The
  mutant `mode = particle_mode(smoke)` → `mode = PARTICLE_FULL_MODE` survives
  `test/particles.jl`; under it, the smoke test of the script runs the full mode for hours. In
  `scripts/particles.jl:46`, the mutant that passes `smoke = true` to `add_labelled_pod` in place
  of `smoke` survives `test/particles.jl` too, because its test of the script options calls
  `main` only with `--smoke`. The full mode is the script run, outside the suite; the full-mode
  runs of `scripts/particles.jl`, with and without `--labelled-pod`, went through both lines
  and wrote rows of 256 particles. A test of the mode choice that does not run the full mode
  needs a seam in `run_particles`, and the user asks for little effort on it.
- **kind:** missing test
- **found:** 2026-11-01

### K19 · `add_labelled_pod` truncates `particle_frontends.csv` when the POD of the stored run throws.

- **location:** `src/particles.jl:400`
- **evidence:** `add_labelled_pod` opens the file with `"w"` and computes `labelled_pod_rows`
  inside the `do` block, so an error of `pod_error` leaves the file with the kept rows only. A
  stored smoke run whose velocities hold one `NaN` makes `svdvals` throw
  `ArgumentError: invalid argument #4 to LAPACK call`; a file of 5 lines whose last row is
  `pod_labelled` then has 4 lines, and the old `pod_labelled` row is lost. Computing the rows
  before `open(path, "w")` closes it. A `pod_error` with `n` above the rank does not throw: it
  returns `0.0`.
- **kind:** defect
- **found:** 2026-11-01
