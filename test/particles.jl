using HDF5
using LinearAlgebra: norm, svdvals
using Random: Xoshiro, randperm
using Test
using VlasovReductionBenchmark
using VlasovReductionBenchmark: charbonnier_relative, check_pic_energy, failed_runs,
                                frontend_rows, inverse_cdf, particle_E1, particle_mode,
                                train_subsample, training_step!, wrap_position
using VlasovReductionBenchmark: Adam, CompositeMethod, FactoredStiefelAdam, NeuralNetwork,
                                SymmetricSymplecticAutoencoder, TrainingOptimizer,
                                Zygote, optimization_step!

include("helpers/thrown.jl")

const L = 4π

@testset "Sample of the initial density" begin
    for M in (512, 4096)
        x, v, w = sample_particles(M, 0.5, 0.5)
        @test length(x) == length(v) == length(w) == M
        @test all(xi -> 0 ≤ xi < L, x)
        @test all(==(L / M), w)
        # The first Fourier mode of the empirical density, (2/M) Σ e^{-ikx}, has expectation α
        # under 1 + α cos kx and a standard deviation of about √2/√M per component; the bound
        # is about 3.5σ.
        @test abs(2 / M * sum(exp.(-im * 0.5 .* x)) - 0.5) ≤ 5 / sqrt(M)
        # the velocities are standard normal: mean and variance within 5σ
        @test abs(sum(v) / M) ≤ 5 / sqrt(M)
        @test abs(sum(abs2, v) / M - 1) ≤ 5 * sqrt(2 / M)
        # one generator, seeded the same, gives the same sample
        @test sample_particles(M, 0.5, 0.5) == (x, v, w)
    end
    @test sample_particles(64, 0.5, 0.5; seed = 43)[1] != sample_particles(64, 0.5, 0.5)[1]
    # the decided draw: from one Xoshiro(42), first the M positions, then the M velocities
    rng = Xoshiro(42)
    u = [rand(rng) for _ in 1:512]
    z = randn(rng, 512)
    x, v, _ = sample_particles(512, 0.5, 0.5)
    @test x == inverse_cdf.(u, 0.5, 0.5)
    @test v == z

    # the inverse of the CDF (x + α sin(kx)/k)/L of the density
    F(x) = (x + 0.5 * sin(0.5x) / 0.5) / L
    for u in (0.0, 1e-12, 0.1, 0.25, 0.5, 0.7, 0.999, 1 - 1e-12)
        x = inverse_cdf(u, 0.5, 0.5)
        @test 0 ≤ x < L
        @test F(x) ≈ u atol=4eps()
    end
    # the largest draw of `rand`, 1 - 2⁻⁵³, lies above F(prevfloat(L)), so the root is in the
    # last interval [prevfloat(L), L) of the bisection, and the position stays below L
    @test F(prevfloat(L)) < prevfloat(1.0)
    @test inverse_cdf(prevfloat(1.0), 0.5, 0.5) == prevfloat(L)
end

@testset "|E₁| of the particles" begin
    k = 0.5
    # one particle carrying the whole weight L: (2/(kL)) L = 2/k
    @test particle_E1([1.3], [L], k) ≈ 2 / k rtol=1e-14
    # two equal weights half a wavelength apart cancel
    @test particle_E1([0.0, L / 2], [1.0, 1.0], k) ≤ 1e-15
    # invariant under a wrap by L
    @test particle_E1([0.3, 2.0], [0.5, 1.5], k) ≈
          particle_E1([0.3 + L, 2.0 - L], [0.5, 1.5], k) rtol=1e-12
    # at t = 0 it is α/k in expectation, within 5/√M/k
    for M in (512, 4096)
        x, _, w = sample_particles(M, 0.5, k)
        @test abs(particle_E1(x, w, k) - 0.5 / k) ≤ 5 / sqrt(M) / k
    end
end

@testset "Wrap of a position" begin
    @test wrap_position(0.0, L) == 0.0
    @test wrap_position(L, L) == 0.0
    @test wrap_position(1.0 + L, L) ≈ 1.0 rtol=1e-14
    @test wrap_position(-1.0, L) ≈ L - 1 rtol=1e-14
    # mod(-1e-17, L) rounds to L itself, which is outside [0, L)
    @test mod(-1e-17, L) == L
    @test wrap_position(-1e-17, L) == 0.0
    @test wrap_position(-1e-17, L) < L
    @test wrap_position(-3L - 0.5, L) ≈ L - 0.5 rtol=1e-12
end

@testset "Sorted snapshots" begin
    rng = Xoshiro(1)
    X, V = rand(rng, 50, 7) .* L, randn(rng, 50, 7)
    Xs, Vs = sorted_snapshots(X, V)
    @test size(Xs) == size(Vs) == (50, 7)
    for j in 1:7
        @test issorted(Xs[:, j])
        # each sorted snapshot holds the same (x, v) pairs as the labelled one
        @test collect(zip(Xs[:, j], Vs[:, j])) == sort(collect(zip(X[:, j], V[:, j])))
    end
    # the sort is stable: equal positions keep their label order
    Xs, Vs = sorted_snapshots([1.0 2.0; 0.0 2.0; 1.0 2.0], [10.0 1.0; 20.0 2.0; 30.0 3.0])
    @test Xs == [0.0 2.0; 1.0 2.0; 1.0 2.0]
    @test Vs == [20.0 1.0; 10.0 2.0; 30.0 3.0]
    # and on 100 particles at two positions
    X = reshape(repeat([2.0, 1.0], 50), :, 1)
    Xs, Vs = sorted_snapshots(X, reshape(collect(1.0:100.0), :, 1))
    @test vec(Vs) == vcat(2.0:2.0:100.0, 1.0:2.0:99.0)
end

@testset "POD mode counts" begin
    # σ² = (9, 1): one mode holds 90 % of the energy, exactly
    S = [3.0 0.0; 0.0 1.0]
    @test pod_modes(S, 0.9) == 1
    @test pod_modes(S, 0.91) == 2
    @test pod_modes(S, 1.0) == 2
    @test pod_error(S, 1) ≈ sqrt(0.1) rtol=1e-14
    @test pod_error(S, 2) ≤ 1e-15
    @test pod_error(S, 0) ≈ 1 rtol=1e-14

    # A synthetic particle set whose sorted snapshots have rank 2 by construction: the sorted
    # positions s are the same in every snapshot, and the velocities are c_t g with Σ_t c_t = 0.
    # The two singular values are σ₁ = ‖s‖ √Nt and σ₂ = ‖g‖ ‖c‖ = σ₁ / 20, so one mode holds
    # 1/(1 + 1/400) = 99.75 % of the energy: one mode reaches 99 %, two reach 99.99 %. A count
    # on the singular values themselves, 1/(1 + 1/20) = 95.2 % for one mode, needs two for 99 %.
    M, Nt = 64, 21
    s = ((1:M) .- 0.5) .* (L / M)
    c = cos.(2π .* (0:(Nt - 1)) ./ Nt)
    g = randn(Xoshiro(2), M)
    g .*= norm(s) * sqrt(Nt) / 20 / (norm(g) * norm(c))
    X, V = repeat(s, 1, Nt), g * c'
    sorted = vcat(X, V)
    σ = svdvals(sorted)
    @test σ[1] ≈ norm(s) * sqrt(Nt) rtol=1e-12
    @test σ[2] ≈ σ[1] / 20 rtol=1e-12
    @test σ[3] ≤ 1e-12 * σ[1]
    @test pod_modes(sorted, 0.99) == 1
    @test pod_modes(sorted, 0.9999) == 2

    # the labels of each snapshot shuffled, with the velocities carried along
    rng = Xoshiro(3)
    perms = [randperm(rng, M) for _ in 1:Nt]
    Xl = stack(X[perms[t], t] for t in 1:Nt)
    Vl = stack(V[perms[t], t] for t in 1:Nt)
    @test pod_modes(vcat(Xl, Vl), 0.9999) > 2
    Xs, Vs = sorted_snapshots(Xl, Vl)
    @test pod_modes(vcat(Xs, Vs), 0.99) == 1
    @test pod_modes(vcat(Xs, Vs), 0.9999) == 2
end

@testset "Energy check of a run" begin
    @test check_pic_energy([0.0, 1e-4, -1e-3, 1e-3]) === nothing
    e = thrown(() -> check_pic_energy([0.0, 1e-4, -2e-3, 5e-3]))
    @test e isa ErrorException
    @test occursin(
        r"^nonlinear Landau damping with particles: the energy drifted by 0\.002 relative at step 2, above 0\.001$",
        e.msg)
    e = thrown(() -> check_pic_energy([0.0, NaN]))
    @test e isa ErrorException
    @test occursin(
        r"^nonlinear Landau damping with particles: the energy drifted by NaN relative at step 1, above 0\.001$",
        e.msg)

    # a run of four particles drifts by 1.2e-3 relative, first at step 30 (measured); it
    # throws, and writes no file
    mktempdir() do d
        e = thrown(() -> run_pic(4, 5.0; runs_dir = d))
        @test e isa ErrorException
        @test occursin(
            r"^nonlinear Landau damping with particles: the energy drifted by \S+ relative at step \d+, above 0\.001$",
            e.msg)
        @test isempty(readdir(d))
    end
end

@testset "Short PIC run" begin
    mktempdir() do d
        run = run_pic(512, 1.0; runs_dir = d)
        x0, v0, w = sample_particles(512, 0.5, 0.5)
        @test run.times ≈ 0:0.05:1 atol=1e-12
        @test run.snapshot_times ≈ [0.0, 0.5, 1.0] atol=1e-12
        @test size(run.positions) == size(run.velocities) == (512, 3)
        @test run.positions[:, 1] == x0
        @test run.velocities[:, 1] == v0
        @test run.weights == w
        # within 1 time unit, particles near x = 0 with v < 0 leave [0, L) unless wrapped
        @test all(xi -> 0 ≤ xi < L, run.positions)
        @test length(run.E1) == length(run.energy_drift) == length(run.momentum_drift) == 21
        # |E₁| every step; at a snapshot step it is the mode of the stored positions
        for (j, s) in enumerate((1, 11, 21))
            @test run.E1[s] ≈ particle_E1(run.positions[:, j], w, 0.5) rtol=1e-12
        end
        @test run.energy_drift[1] == 0
        @test maximum(abs, run.energy_drift) ≤ 1e-3
        # the run moves: the energy and the positions change
        @test maximum(abs, run.energy_drift) > 0
        @test run.positions[:, 3] != run.positions[:, 1]
        @test run.momentum_drift[1] == 0
        @test run.momentum_drift[end] ≈
              sum(w .* run.velocities[:, 3]) - sum(w .* run.velocities[:, 1]) atol=1e-14

        path = joinpath(d, "particles_nonlinear_landau.h5")
        @test readdir(d) == ["particles_nonlinear_landau.h5"]
        h5open(path) do f
            @test read(f, "positions") == run.positions
            @test read(f, "velocities") == run.velocities
            @test eltype(read(f, "positions")) == Float64
            @test read(f, "weights") == w
            @test read(f, "E1") == run.E1
            @test read(f, "times") == run.times
            @test read(f, "snapshot_times") == run.snapshot_times
            @test read(f, "energy_drift") == run.energy_drift
            @test read(f, "momentum_drift") == run.momentum_drift
            a = attrs(f)
            @test (a["case"], a["alpha"], a["k"], a["particles"], a["dt"], a["tend"]) ==
                  ("nonlinear_landau", 0.5, 0.5, 512, 0.05, 1.0)
            @test (a["ncells"], a["order"], a["seed"]) == (32, 4, 42)
            @test a["xdomain"] == [0.0, L]
        end
    end
end

@testset "Charbonnier relative error" begin
    Z = [3.0 0.0; 4.0 0.0]
    @test charbonnier_relative(Z, Z) == 0
    Ẑ = Z .+ [0.0 2.0; 0.0 0.0]
    @test charbonnier_relative(Ẑ, Z) ≈ (sqrt(4 + 1e-6) - 1e-3) / sqrt(25 + 1e-6) rtol=1e-14
end

@testset "Training of a frontend" begin
    rng = Xoshiro(5)
    M, B = 8, 6
    Z = vcat(sort(rand(rng, M, B) .* L; dims = 1), randn(rng, M, B))
    for fe in (:sort, :slot, :pairwise)
        untrained = train_frontend(fe, Z, 4; seed = 42, steps = 0)
        trained = train_frontend(fe, Z, 4; seed = 42, steps = 20)
        @test untrained.status == trained.status == "ok"
        @test trained.particles == M
        @test untrained.seconds_per_step ≥ 0
        @test isfinite(untrained.seconds_per_step)
        @test 0 < trained.relative_error < untrained.relative_error
        @test trained.seconds_per_step > 0
        # the seed sets the network, and only the seed
        @test train_frontend(fe, Z, 4; seed = 42, steps = 20).relative_error ==
              trained.relative_error
        @test train_frontend(fe, Z, 4; seed = 43, steps = 20).relative_error !=
              trained.relative_error
    end
    # a non-finite loss ends the run with its step
    Zbad = copy(Z)
    Zbad[1, 1] = NaN
    failed = train_frontend(:sort, Zbad, 4; seed = 42, steps = 5)
    @test failed.status == "failed at step 0"
    @test isnan(failed.relative_error)
    # with no step to take, the loss after the last step is checked
    @test train_frontend(:sort, Zbad, 4; seed = 42, steps = 0).status == "failed at step 0"
    # the time per step divides by the steps taken: both runs stop after one failed gradient,
    # so a division by the 50 planned steps would give a time 50 times smaller (the fastest
    # of five runs each, so that the noise of the machine stays far below that factor)
    fastest(steps) = minimum(
        train_frontend(:sort, Zbad, 4; seed = 42, steps).seconds_per_step for _ in 1:5)
    @test fastest(50) > fastest(1) / 5

    # the network: the frontend and the latent dimension n, seeded by `seed`
    relative_error(nn) = norm(nn.model(Z, nn.params) .- Z) / norm(Z)
    for fe in (:sort, :slot, :pairwise), n in (4, 8)

        nn = NeuralNetwork(SymmetricSymplecticAutoencoder(M, 1, n; frontend = fe), Float64;
            rng = Xoshiro(42))
        @test train_frontend(fe, Z, n; seed = 42, steps = 0).relative_error ≈
              relative_error(nn) rtol=1e-12
    end
    @test allunique([train_frontend(fe, Z, 4; seed = 42, steps = 0).relative_error
                     for fe in (:sort, :slot, :pairwise)])

    # one step: FactoredStiefelAdam and Adam at the step size 1e-3, on the gradient of the
    # Charbonnier relative error
    nn = NeuralNetwork(SymmetricSymplecticAutoencoder(M, 1, 4; frontend = :slot), Float64;
        rng = Xoshiro(42))
    optimizer = TrainingOptimizer(nn.params;
        algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
        linesearch = 1e-3)
    before = relative_error(nn)
    gradient = Zygote.gradient(p -> charbonnier_relative(nn.model(Z, p), Z), nn.params)[1]
    optimization_step!(nn.params, optimizer, gradient)
    @test relative_error(nn) != before
    @test train_frontend(:slot, Z, 4; seed = 42, steps = 1).relative_error ≈
          relative_error(nn) rtol=1e-12
end

@testset "One training step" begin
    rng = Xoshiro(7)
    M, B = 8, 6
    Z = vcat(sort(rand(rng, M, B) .* L; dims = 1), randn(rng, M, B))
    nn = NeuralNetwork(SymmetricSymplecticAutoencoder(M, 1, 4; frontend = :sort), Float64;
        rng = Xoshiro(42))
    optimizer = TrainingOptimizer(nn.params;
        algorithm = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam()),
        linesearch = 1e-3)
    loss(p) = charbonnier_relative(nn.model(Z, p), Z)
    l₀ = loss(nn.params)

    # a finite loss whose gradient is not finite: d/dw √|w - c| at w = c
    c = nn.params.L1.weight[1, 1]
    singular(p) = loss(p) + sqrt(abs(p.L1.weight[1, 1] - c))
    @test isfinite(singular(nn.params))
    @test training_step!(nn.params, optimizer, singular) == false
    @test loss(nn.params) == l₀

    # a non-finite loss
    @test training_step!(nn.params, optimizer, p -> loss(p) + NaN) == false
    @test loss(nn.params) == l₀

    # a finite loss and gradient: the step moves the parameters
    @test training_step!(nn.params, optimizer, loss) == true
    @test loss(nn.params) != l₀
end

@testset "Count of the failed training runs" begin
    rows = [
        (method = "sort", status = "ok"), (method = "slot", status = "failed at step 3"),
        (method = "pairwise", status = "failed at step 0"), (method = "pod", status = "ok")]
    @test failed_runs(rows) == (2, 3)
    @test failed_runs(rows[[1, 4]]) == (0, 1)
end

@testset "Rows of the frontends" begin
    rng = Xoshiro(6)
    M, Nt, Mt = 40, 5, 8
    positions, velocities = rand(rng, M, Nt) .* L, randn(rng, M, Nt)
    # the subsample is the Mt particles of smallest initial position, labels fixed in time
    Z = train_subsample(positions, velocities, Mt)
    first_eight = sortperm(positions[:, 1])[1:Mt]
    @test Z == vcat(positions[first_eight, :], velocities[first_eight, :])

    rows = frontend_rows(positions, velocities; train_particles = Mt, latent_dims = (4,),
        seeds = (42, 43), steps = 2)
    @test [(r.method, r.n, r.seed) for r in rows] ==
          [("sort", 4, "42"), ("sort", 4, "43"), ("slot", 4, "42"), ("slot", 4, "43"),
        ("pairwise", 4, "42"), ("pairwise", 4, "43"), ("pod", 4, "none")]
    # every row, :pairwise's included, is on the same Mt particles
    @test all(r -> r.particles == Mt, rows)
    @test rows[1].relative_error ==
          train_frontend(:sort, Z, 4; seed = 42, steps = 2).relative_error
    # the baseline: POD with n modes on the same Mt particles, sorted in each snapshot
    Xs, Vs = sorted_snapshots(Z[1:Mt, :], Z[(Mt + 1):end, :])
    σ² = svdvals(vcat(Xs, Vs)) .^ 2
    @test rows[end].relative_error ≈ sqrt(sum(σ²[5:end]) / sum(σ²)) rtol=1e-12
    @test rows[end].status == "ok"
end

include("../scripts/particles.jl")

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/particles\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    for args in (["--runs-dir", "a"], ["--runs-dir", "a", "--results-dir", "b", "--x"],
        ["--runs-dir", "a", "--results-dir"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
end

@testset "Settings of the two modes" begin
    @test VlasovReductionBenchmark.PARTICLE_FULL_MODE ==
          (particles = 4096, train_particles = 256, tend = 40.0,
        latent_dims = (4, 8, 16, 32), steps = 2000, seeds = (42, 43, 44))
    @test VlasovReductionBenchmark.PARTICLE_SMOKE_MODE ==
          (particles = 512, train_particles = 64, tend = 10.0, latent_dims = (4, 8),
        steps = 100, seeds = (42,))
    @test VlasovReductionBenchmark.PIC_ENERGY_TOLERANCE == 1e-3
    # the script's full mode is the full settings, and `--smoke` the smoke settings
    @test particle_mode(false) == VlasovReductionBenchmark.PARTICLE_FULL_MODE
    @test particle_mode(true) == VlasovReductionBenchmark.PARTICLE_SMOKE_MODE
end
