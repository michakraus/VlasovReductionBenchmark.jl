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

### K8 · The norm ratio of the canonical-flow record depends on the grid.

- **location:** `src/autoencoders.jl:460`
- **evidence:** `canonical_flow_comparison` divides the norm of the canonical flow of the
  discrete energy by the norm of the Arakawa field. The gradient of the energy carries the cell
  area `h_x h_v` and the field does not, so the ratio changes with the grid: for linear Landau at
  `t = 0` it is 662.7 on 32 × 64 and 159.5 on 64 × 128 (the round-1 critic review of this
  branch). `autoencoders_hamiltonian.toml` writes `nx` and `nv` beside the ratio, so only
  ratios of one grid compare.
- **kind:** not verified
- **found:** 2026-11-01

### K9 · The cosine of the canonical-flow record does not tell the canonical flow from the gradient.

- **location:** `src/autoencoders.jl:460`
- **evidence:** the canonical flow `J ∇H` and the gradient `∇H` have the same norm, and both are
  orthogonal to the Arakawa field to round-off: on the smoke grid at `t = 0` the cosines are
  `-1.96e-18` and `-1.39e-18`. A mutant that writes `∇H`, or `J ∇H` with the sign of one half
  flipped, in place of the flow changes neither written value beyond round-off. A cosine near
  zero is therefore no evidence that the parity split is not canonical; only the norm ratio
  carries information.
- **kind:** not verified
- **found:** 2026-11-01

### K10 · The seeds of the weight-tied network give losses that differ only at round-off.

- **location:** `src/autoencoders.jl:193`
- **evidence:** the seed sets the layers between the frontend and the PSD lift of the core,
  whose decode–encode is the projection of that lift whatever those layers are. The loss after
  the frontend's and the lift's `solve!` is bit-identical for seeds 42 and 43 (0.7433233237891245
  at `n = 16` on 64 × 128), and after 10 steps the last losses differ by `1.5e-11` (the round-1
  critic review of this branch). The range over the three seeds of the full mode is then near
  zero for the reconstruction. The seed changes the latent coordinates: on the smoke grid at
  `n = 4`, the encoded snapshot of seeds 42 and 43 differs by 0.114 at a norm of 0.165.
- **kind:** not verified
- **found:** 2026-11-01

### K11 · The memory of the full mode of `scripts/autoencoders.jl` is not measured.

- **location:** `src/autoencoders.jl:593`
- **evidence:** `run_autoencoders` keeps every trained reduction in `specs` until
  `autoencoder_costs` serializes them all for the fresh timing process. Each untied reduction
  holds its lift and its `Q` factor, two `8192 × 128` `Float64` matrices of 8 MiB each on
  64 × 128, so the up to 36 untied data rows of the full mode hold about 600 MiB in memory and
  on disk.
- **kind:** not verified
- **found:** 2026-11-01
