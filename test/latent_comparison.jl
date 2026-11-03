using HDF5: attrs, h5open
using LinearAlgebra: cond, norm
using Printf: @sprintf
using VlasovReductionBenchmark
using VlasovReductionBenchmark: LATENT_COLUMNS, LinearReduction, ParityCoordinates,
                                dual_rows, galerkin_row, invariants, k0_fraction,
                                latent_rows, sympnet_row, weight_tied_rows
using Test

include("helpers/thrown.jl")

const NX, NV = 32, 64
const K = 0.5

# A two-stream run on the smoke grid to t = 6, 13 snapshots.
const CASE = VlasovCase("two_stream_long", two_stream_profile, 0.05, K, 6.0)
const TS = run_case(CASE, NX, NV).snapshots

# The columns of a row that hold an error or a drift.
const METRICS = (
    "error_in_sample", "error_extrapolation", "max_energy_drift", "max_l1_drift",
    "max_l2_drift", "max_mass_drift")

# The columns of a singular SympNet row that hold NaN.
const NAN_COLUMNS = (METRICS..., "offline_seconds", "loss_first", "loss_last")

# The three statements of the paragraph of the page.
const STATEMENTS = (
    "Dual PSD is canonical only for the Vlasov bracket linearised at the initial background",
    "POD is not a row of the comparison.",
    "Dual PSD has the reconstruction error of PSD without the x-average")

# A mode of 7 training snapshots, two seeds and short trainings.
const MODE = (m_train = 7, seeds = (42, 43), epochs = 3, batch_size = 32, n_layers = 4,
    autoencoder_steps = 5)

# The 7 snapshots of the fixture of test/dual_psd.jl whose modes all hold cos(kx) and none
# sin(kx): the odd–even block of the latent pairing at n = 4 is round-off, so
# dual_psd_reduction throws.
function cosine_fixture()
    x, v = grid(CASE, NX, NV)
    p = maxwellian.(v)
    c = cos.(K .* x)
    modes = (c .* p', c .* (v .^ 2 .* p)', c .* (v .^ 4 .* p)', c .* (v .^ 6 .* p)')
    coefficients(s) = (0.01s, 0.003s^2, 0.002s^3, 0.001s^4)
    return stack([p' .+ sum(coefficients(s) .* modes) for s in 0:6])
end

# The numbers of a cell of the page, from the `rows` of the CSV file: medians of the rows of
# status ok, as the sorted middle, and the counts.
function expected_cell(rows)
    ok = filter(r -> r["status"] == "ok", rows)
    singular = count(r -> r["status"] == "singular", rows)
    function middle(c)
        x = sort([parse(Float64, r[c]) for r in ok])
        m = length(x)
        return isodd(m) ? x[(m + 1) ÷ 2] : (x[m ÷ 2] + x[m ÷ 2 + 1]) / 2
    end
    medians = isempty(ok) ? fill(singular == length(rows) ? "singular" : "—", 4) :
              [@sprintf("%.4g", middle(c))
               for c in ("error_extrapolation", "error_in_sample", "max_energy_drift",
        "max_mass_drift")]
    return join(medians, " / ") * "; $(length(rows) - length(ok) - singular) diverged, " *
           "$singular singular"
end

@testset "The two modes" begin
    full, smoke = latent_comparison_mode(false), latent_comparison_mode(true)
    @test full == (grid = (64, 128), long_tend = 100.0, ns = (4, 8, 16, 32),
        seeds = (42, 43, 44), epochs = 2000, batch_size = 32, n_layers = 4,
        autoencoder_steps = 2000, m_train = 101)
    @test smoke ==
          (grid = (32, 64), long_tend = 40.0, ns = (4,), seeds = (42,), epochs = 20,
        batch_size = 32, n_layers = 4, autoencoder_steps = 100, m_train = 41)
    # m_train counts the snapshots with t ≤ t_end / 2, every 0.5
    @test full.m_train == count(≤(50.0), 0.0:0.5:100.0)
    @test smoke.m_train == count(≤(20.0), 0.0:0.5:40.0)
end

@testset "k = 0 fraction" begin
    # the fluctuation A(v) + B(v) cos(kx): Σᵢ cos²(kxᵢ) = nx / 2 and Σᵢ cos(kxᵢ) = 0 on the grid
    x, v = grid(CASE, NX, NV)
    A, B = exp.(-v .^ 2), v .* exp.(-v .^ 2 / 2)
    f0 = maxwellian.(v)
    S = stack([f0' .+ s .* (A' .+ cos.(K .* x) .* B') for s in 1:3])
    @test k0_fraction(S, f0) ≈
          sqrt(NX * sum(abs2, A) / (NX * sum(abs2, A) + NX / 2 * sum(abs2, B))) rtol=1e-12
    # no x-average part, and nothing but one
    @test k0_fraction(stack([f0' .+ cos.(K .* x) .* B' for s in 1:2]), f0) ≤ 1e-14
    @test k0_fraction(stack([f0' .+ ones(NX) .* A' for s in 1:2]), f0) ≈ 1 rtol=1e-14
end

@testset "A singular dual PSD" begin
    S = cosine_fixture()
    f0 = background(S[:, :, 1])
    e = thrown(() -> dual_psd_reduction(S, f0, 4, K))
    @test e isa ArgumentError && occursin("round-off", sprint(showerror, e))
    B = latent_pairing(psd_nonzero_reduction(S, f0, 4), K)[1:2, 3:4]
    common = Dict("dataset" => "cosine", "n" => "4", "k0_fraction" => "0.5")
    rows = dual_rows(common, CASE, S, f0, 4, MODE)
    @test length(rows) == 2
    @test [r["seed"] for r in rows] == ["42", "43"]
    for r in rows
        @test sort(collect(keys(r))) == sort(collect(LATENT_COLUMNS))
        @test (r["dataset"], r["n"], r["reduction"], r["dynamics"], r["status"]) ==
              ("cosine", "4", "dual_psd", "sympnet", "singular")
        @test parse(Float64, r["pairing_condition"]) == cond(B)
        @test r["k0_fraction"] == "0.5"
        @test all(c -> r[c] == "NaN", NAN_COLUMNS)
    end
    # a zero background gives a zero pairing: the block is exactly singular, and its condition
    # number infinite
    e = thrown(() -> dual_psd_reduction(TS[:, :, 1:7], zeros(NV), 4, K))
    @test e isa ArgumentError && occursin("round-off", sprint(showerror, e))
    rows = dual_rows(common, CASE, TS, zeros(NV), 4, MODE)
    @test all(r -> (r["status"], r["pairing_condition"]) == ("singular", "Inf"), rows)
    # on the two-stream run the block at n = 2 is 1 × 1 and round-off
    g0 = background(TS[:, :, 1])
    B = latent_pairing(psd_nonzero_reduction(TS[:, :, 1:7], g0, 2), K)[1:1, 2:2]
    rows = dual_rows(common, CASE, TS, g0, 2, MODE)
    @test all(r -> r["status"] == "singular", rows)
    @test all(r -> parse(Float64, r["pairing_condition"]) == cond(B), rows)

    # another exception of the build propagates
    stub(S, f0, n, k) = throw(DimensionMismatch("stub"))
    @test thrown(() -> dual_rows(common, CASE, S, f0, 4, MODE; build = stub)) isa
          DimensionMismatch
    # an ArgumentError of psd_nonzero_reduction is no singular pairing
    e = thrown(() -> dual_rows(common, CASE, S, f0, 5, MODE))
    @test e isa ArgumentError && occursin("even number of modes", sprint(showerror, e))
end

@testset "The rows of one run" begin
    rows = latent_rows(CASE, TS, 4, MODE)
    key(r) = (r["reduction"], r["dynamics"], r["seed"])
    # a SympNet row per reduction and seed, a Galerkin row of each linear reduction without a
    # seed and of the network per seed, and no Galerkin row of dual PSD
    reductions = ("psd", "psd_nonzero", "dual_psd", "weight_tied")
    sympnets = [(red, "sympnet", s) for red in reductions for s in ("42", "43")]
    galerkins = [("psd", "galerkin", ""), ("psd_nonzero", "galerkin", ""),
        ("weight_tied", "galerkin", "42"), ("weight_tied", "galerkin", "43")]
    @test sort(key.(rows)) == sort([sympnets; galerkins])
    @test all(r -> r["dataset"] == "two_stream_long" && r["n"] == "4", rows)
    @test all(r -> r["status"] == "ok", rows)
    @test all(r -> sort(collect(keys(r))) == sort(collect(LATENT_COLUMNS)), rows)
    f0 = background(TS[:, :, 1])
    train = TS[:, :, 1:7]
    @test all(r -> parse(Float64, r["k0_fraction"]) ≈ k0_fraction(train, f0), rows)
    κ = pairing_condition(dual_psd_reduction(train, f0, 4, K))
    for r in rows
        if r["reduction"] == "dual_psd"
            @test parse(Float64, r["pairing_condition"]) ≈ κ
        else
            @test isempty(r["pairing_condition"])
        end
    end
    for r in rows
        sympnet = r["dynamics"] == "sympnet"
        @test isempty(r["loss_first"]) == !sympnet
        @test isempty(r["loss_last"]) == !sympnet
        @test parse(Float64, r["offline_seconds"]) > 0
    end
    pick(red, dyn, seed) = only(filter(r -> key(r) == (red, dyn, seed), rows))

    # the SympNet row of PSD from the training snapshots only
    r = psd_reduction(train, f0, 4)
    trained = train_latent_sympnet(latent_snapshots(r, train), 7; epochs = 3, seed = 43)
    run = latent_run(r, CASE, TS, trained.network, 7)
    row = pick("psd", "sympnet", "43")
    @test parse(Float64, row["error_in_sample"]) ≈ run.l2_error_in_sample rtol=1e-12
    @test parse(Float64, row["error_extrapolation"]) ≈ run.l2_error_extrapolation rtol=1e-12
    @test parse(Float64, row["max_energy_drift"]) ≈ run.energy_drift rtol=1e-12
    @test parse(Float64, row["loss_first"]) ≈ first(trained.history) rtol=1e-12
    @test parse(Float64, row["loss_last"]) ≈ last(trained.history) rtol=1e-12
    @test run.l2_error_extrapolation > 1e-6

    # the SympNet rows of dual PSD and of the network, from the training snapshots only
    for (red, reduction) in (("dual_psd", dual_psd_reduction(train, f0, 4, K)),
        ("weight_tied", train_weight_tied(train, f0, 4; seed = 42, steps = 5).reduction))
        nn = train_latent_sympnet(latent_snapshots(reduction, train), 7; epochs = 3,
            seed = 42).network
        expected = latent_run(reduction, CASE, TS, nn, 7)
        row = pick(red, "sympnet", "42")
        for (c, value) in (("error_in_sample", expected.l2_error_in_sample),
            ("error_extrapolation", expected.l2_error_extrapolation))
            @test isapprox(parse(Float64, row[c]), value; rtol = 1e-12)
        end
    end

    # the Galerkin row of PSD: the errors relative to the fluctuation and the drifts of the
    # invariants of the decoded snapshots, computed here
    galerkin = run_reduced(r, CASE, NX, NV, TS[:, :, 1])
    D = galerkin.snapshots
    e = [norm(D[:, :, s] - TS[:, :, s]) / norm(TS[:, :, s] .- f0') for s in 1:13]
    model = vlasov_model(CASE, NX, NV)
    xs, vs = model.distribution.x, model.distribution.v
    series = [invariants(D[:, :, s], model.work, K, xs, vs, step(xs), step(vs))
              for s in 1:13]
    function drift(name)
        values = [getfield(d, name) for d in series]
        return maximum(abs.(values ./ values[1] .- 1))
    end
    row = pick("psd", "galerkin", "")
    @test row["status"] == galerkin.status
    @test parse(Float64, row["error_in_sample"]) ≈ sqrt(sum(abs2, e[1:7]) / 7) rtol=1e-12
    @test parse(Float64, row["error_extrapolation"]) ≈ sqrt(sum(abs2, e[8:13]) / 6) rtol=1e-12
    for (c, name) in (("max_energy_drift", :energy), ("max_l1_drift", :l1),
        ("max_l2_drift", :l2), ("max_mass_drift", :mass))
        @test isapprox(parse(Float64, row[c]), drift(name); rtol = 1e-8, atol = 1e-15)
    end
    @test drift(:energy) > 1e-8
end

@testset "Failed models and the offline time" begin
    common = Dict("dataset" => "two_stream_long", "n" => "4", "k0_fraction" => "0.5")
    f0 = background(TS[:, :, 1])
    # a training of no step has no loss that fell: both rows of each seed carry its status
    rows = weight_tied_rows(common, CASE, TS, f0, 4, merge(MODE, (autoencoder_steps = 0,)))
    @test sort([(r["dynamics"], r["seed"]) for r in rows]) ==
          [("galerkin", "42"), ("galerkin", "43"), ("sympnet", "42"), ("sympnet", "43")]
    for r in rows
        @test startswith(r["status"], "failed: the loss did not fall")
        @test all(c -> r[c] == "NaN", (METRICS..., "offline_seconds"))
        @test r["loss_first"] == r["loss_last"] == (r["dynamics"] == "sympnet" ? "NaN" : "")
        @test isempty(r["pairing_condition"])
    end

    # a basis with a NaN fails a Galerkin step: the status of run_reduced and NaN
    r = psd_reduction(TS[:, :, 1:7], f0, 4)
    bad = LinearReduction(ParityCoordinates(), fill(NaN, size(r.basis)), f0, NX, NV)
    status = run_reduced(bad, CASE, NX, NV, TS[:, :, 1]).status
    @test startswith(status, "failed at step")
    row = galerkin_row(common, "psd", bad, CASE, TS, nothing, 7; seconds = 0.5)
    @test row["status"] == status
    @test all(c -> row[c] == "NaN", METRICS)
    @test row["offline_seconds"] == "0.5"

    # the offline time of a SympNet row is the time of the reduction plus that of the training
    row = sympnet_row(common, "psd", r, CASE, TS, 42, MODE; seconds = 1000.0)
    @test 1000.0 < parse(Float64, row["offline_seconds"]) < 1060.0
end

@testset "A run file of another end time" begin
    mode = (grid = (NX, NV), long_tend = 30.0, ns = ())
    mktempdir() do dir
        file = joinpath(dir, "two_stream_long_alpha0.05.h5")
        function store(tend)
            h5open(file, "w") do f
                f["snapshots"] = fill(7.0, NX, NV, 3)
                f["snapshot_times"] = [0.0, 0.5, 1.0]
                a = attrs(f)
                a["nx"], a["nv"], a["tend"], a["alpha"] = NX, NV, tend, 0.05
            end
        end
        store(40.0)
        rows = redirect_stdout(() -> latent_comparison_rows(dir; mode), devnull)
        @test isempty(rows)
        @test h5open(f -> attrs(f)["tend"], file) == 30.0
        S = h5open(f -> read(f, "snapshots"), file)
        @test size(S) == (NX, NV, 61)
        @test !all(==(7.0), S)
        others = ("two_stream_beams_long_alpha0.05.h5", "nonlinear_landau_long_alpha0.5.h5")
        @test all(f -> isfile(joinpath(dir, f)), others)
        # a file of the same end time is read, not run again
        store(30.0)
        redirect_stdout(() -> latent_comparison_rows(dir; mode), devnull)
        @test all(==(7.0), h5open(f -> read(f, "snapshots"), file))
    end
end

@testset "The page" begin
    header = join(LATENT_COLUMNS, ',')
    # dataset, n, reduction, dynamics, seed, status, the six metrics from the extrapolation
    # error `e` (in sample e/10, extrapolation e, energy e/100, L¹ and L² e/2, mass 1e-14 e),
    # pairing_condition, k0_fraction, offline_seconds and the two losses
    line(d, n, red, dyn, seed, status, e, κ = "") = join(
        [d, n, red, dyn, seed, status, string.(e .* [0.1, 1, 0.01, 0.5, 0.5, 1e-14])...,
            κ, d == "d1" ? "0.125" : "0.25", "1.5", dyn == "sympnet" ? "0.3" : "",
            dyn == "sympnet" ? "0.1" : ""], ',')
    lines = [header,
        # the mean of 0.1, 0.2, 0.9 is 0.4, the median 0.2
        line("d1", "4", "psd", "sympnet", "42", "ok", 0.1),
        line("d1", "4", "psd", "sympnet", "43", "ok", 0.2),
        line("d1", "4", "psd", "sympnet", "44", "ok", 0.9),
        line("d1", "8", "psd", "sympnet", "42", "ok", 0.5),
        line("d1", "8", "psd", "sympnet", "43", "diverged", NaN),
        line("d1", "8", "psd", "sympnet", "44", "ok", 0.75),
        line("d1", "4", "psd", "galerkin", "", "ok", 0.3),
        line("d1", "8", "psd", "galerkin", "", "failed at step 3", NaN),
        line("d1", "4", "dual_psd", "sympnet", "42", "singular", NaN, "1e17"),
        line("d1", "4", "dual_psd", "sympnet", "43", "singular", NaN, "Inf"),
        line("d1", "8", "dual_psd", "sympnet", "42", "ok", 0.25, "12.5"),
        line("d1", "8", "dual_psd", "sympnet", "43", "singular", NaN, "3e16"),
        line("d1", "4", "weight_tied", "galerkin", "42", "ok", 0.4),
        line("d2", "4", "psd", "sympnet", "42", "ok", 0.6)]
    mktempdir() do dir
        write(joinpath(dir, "latent_dynamics.csv"), join(lines, '\n') * '\n')
        page = read(write_latent_page(dir), String)
        rows = VlasovReductionBenchmark.read_csv_rows(joinpath(dir, "latent_dynamics.csv"))
        cell(d, red, dyn, n) = expected_cell(filter(
            r -> (r["dataset"], r["reduction"], r["dynamics"], r["n"]) == (d, red, dyn, n),
            rows))
        # the table row of d1 at n = 4 and 8
        table(red, dyn) = "| $red | $dyn | $(cell("d1", red, dyn, "4")) | " *
                          "$(cell("d1", red, dyn, "8")) |"
        @test cell("d1", "psd", "sympnet", "4") ==
              "0.2 / 0.02 / 0.002 / 2e-15; 0 diverged, 0 singular"
        @test occursin("## d1\n", page) && occursin("## d2\n", page)
        @test occursin("| reduction | dynamics | n = 4 | n = 8 |\n|:--|:--|:--|:--|\n", page)
        for (red, dyn) in (("psd", "sympnet"), ("psd", "galerkin"), ("dual_psd", "sympnet"))
            @test occursin(table(red, dyn), page)
        end
        @test cell("d1", "psd", "sympnet", "8") ==
              "0.625 / 0.0625 / 0.00625 / 6.25e-15; 1 diverged, 0 singular"
        @test cell("d1", "psd", "galerkin", "8") == "— / — / — / —; 1 diverged, 0 singular"
        @test cell("d1", "dual_psd", "sympnet", "4") ==
              "singular / singular / singular / singular; 0 diverged, 2 singular"
        # the network has no row at n = 8: its cell is a dash
        w4 = cell("d1", "weight_tied", "galerkin", "4")
        @test occursin("| weight_tied | galerkin | $w4 | — |", page)
        @test !occursin("| weight_tied | sympnet |", page)
        @test !occursin("| psd_nonzero |", page)
        @test occursin("| psd | sympnet | $(cell("d2", "psd", "sympnet", "4")) |", page)
        @test occursin("`k0_fraction`: 0.125\n", page)
        @test occursin("`k0_fraction`: 0.25\n", page)
        @test occursin("- n = 4, seed 42: 1e+17 (singular)\n", page)
        @test occursin("- n = 4, seed 43: ∞ (singular)\n", page)
        @test occursin("- n = 8, seed 42: 12.5 (ok)\n", page)
        @test occursin("- n = 8, seed 43: 3e+16 (singular)\n", page)
        @test all(s -> occursin(s, page), STATEMENTS)
    end
end
