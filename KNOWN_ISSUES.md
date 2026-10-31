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
