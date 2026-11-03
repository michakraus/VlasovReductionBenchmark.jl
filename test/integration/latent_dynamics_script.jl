using HDF5: h5open
using Printf: @sprintf
using VlasovReductionBenchmark
using VlasovReductionBenchmark: LATENT_COLUMNS, read_csv_rows
using Test

include("../helpers/thrown.jl")
include("../../scripts/latent_dynamics.jl")

const DATASETS = ("two_stream_long", "two_stream_beams_long", "nonlinear_landau_long")

# The cell of the page from the `rows` of one run, reduction, dynamics and n: the medians of the
# rows of status ok, as the sorted middle, and the counts of the other rows.
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

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/latent_dynamics\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    for args in (["--runs-dir", "a"], ["--runs-dir", "a", "--results-dir", "b", "--x"],
        ["--runs-dir", "a", "--results-dir"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
end

@testset "Smoke run of the script" begin
    mktempdir() do root
        runs, results = joinpath(root, "runs"), joinpath(root, "results")
        output = joinpath(root, "output.txt")
        open(output, "w") do io
            redirect_stdout(io) do
                main(["--runs-dir", runs, "--results-dir", results, "--smoke"])
            end
        end
        printed = read(output, String)
        csv = joinpath(results, "latent_dynamics.csv")
        @test first(eachline(csv)) == join(LATENT_COLUMNS, ',')
        rows = read_csv_rows(csv)

        # the row set: 3 runs × 1 size × (4 SympNet rows + 3 Galerkin rows)
        @test length(rows) == 21
        @test occursin("latent_dynamics.csv: 21 rows, ", printed)
        @test sort([(r["dataset"], r["n"], r["reduction"], r["dynamics"], r["seed"])
                    for r in rows]) ==
              sort([[(d, "4", red, "sympnet", "42")
                     for d in DATASETS
                     for red in ("psd", "psd_nonzero", "dual_psd", "weight_tied")];
                    [(d, "4", red, "galerkin", seed)
                     for d in DATASETS
                     for (red, seed) in (("psd", ""), ("psd_nonzero", ""),
                        ("weight_tied", "42"))]])
        admitted(s) = s in ("ok", "diverged", "singular") || startswith(s, "failed at step")
        @test all(r -> admitted(r["status"]), rows)
        @test all(r -> isempty(r["pairing_condition"]) == (r["reduction"] != "dual_psd"), rows)
        # one k0_fraction per run
        k0s(d) = unique(r["k0_fraction"] for r in rows if r["dataset"] == d)
        @test all(d -> length(k0s(d)) == 1, DATASETS)

        # the SympNet row of PSD on two-stream, from psd_reduction of the training snapshots
        # and the latent-dynamics functions with the same seed
        case = long_run_cases(40.0)[1]
        S = h5open(f -> read(f, "snapshots"), joinpath(runs, "two_stream_long_alpha0.05.h5"))
        @test size(S) == (32, 64, 81)
        f0 = background(S[:, :, 1])
        function psd_run(r)
            trained = train_latent_sympnet(latent_snapshots(r, S[:, :, 1:41]), 41;
                epochs = 20, seed = 42, n_layers = 4, batch_size = 32)
            return latent_run(r, case, S, trained.network, 41)
        end
        run = psd_run(psd_reduction(S[:, :, 1:41], f0, 4))
        row = only(filter(
            r -> (r["dataset"], r["reduction"], r["dynamics"]) ==
                 ("two_stream_long", "psd", "sympnet"), rows))
        @test row["status"] == run.status == "ok"
        @test parse(Float64, row["error_in_sample"]) ≈ run.l2_error_in_sample rtol=1e-12
        @test parse(Float64, row["error_extrapolation"]) ≈ run.l2_error_extrapolation rtol=1e-12
        # a basis from all the snapshots gives another extrapolation error
        everything = psd_run(psd_reduction(S, f0, 4))
        @test abs(everything.l2_error_extrapolation - run.l2_error_extrapolation) >
              1e-6 * run.l2_error_extrapolation

        # every cell of the page is recomputed from the CSV file
        page = read(joinpath(results, "latent_dynamics.md"), String)
        for d in DATASETS
            @test occursin("## $d\n", page)
            for red in ("psd", "psd_nonzero", "dual_psd", "weight_tied"),
                dyn in ("sympnet", "galerkin")

                these = filter(
                    r -> (r["dataset"], r["reduction"], r["dynamics"]) == (d, red, dyn), rows)
                isempty(these) && continue
                @test occursin("| $red | $dyn | $(expected_cell(these)) |", page)
            end
            dual = only(filter(r -> (r["dataset"], r["reduction"]) == (d, "dual_psd"), rows))
            κ = parse(Float64, dual["pairing_condition"])
            @test occursin(
                "- n = 4, seed 42: $(isinf(κ) ? "∞" : @sprintf("%.4g", κ)) ($(dual["status"]))",
                page)
            k0 = parse(Float64, only(k0s(d)))
            @test occursin("`k0_fraction`: $(@sprintf("%.4g", k0))\n", page)
        end
        @test !occursin("| dual_psd | galerkin |", page)
        statements = (
            "Dual PSD is canonical only for the Vlasov bracket linearised at the initial " *
            "background",
            "POD is not a row of the comparison.",
            "Dual PSD has the reconstruction error of PSD without the x-average")
        @test all(s -> occursin(s, page), statements)
    end
end
