using GeometricMachineLearning: GeometricMachineLearning, LowRankStiefelLayer
using HDF5: h5open
using LinearAlgebra: Diagonal, norm, svd
using Printf: @sprintf
using VlasovReductionBenchmark
using VlasovReductionBenchmark: PARITY_MODES, channel_network, joint_loss, read_csv_rows,
                                reconstruction_errors, train_network!, weight_tied_network,
                                weight_tied_reduction
using Test

include("../helpers/thrown.jl")
include("../../scripts/channels.jl")

const GML = GeometricMachineLearning

relative(a, b) = norm(a - b) / norm(b)

@testset "Script options" begin
    @test parse_options(["--runs-dir", "a", "--results-dir", "b"]) ==
          (runs_dir = "a", results_dir = "b", smoke = false)
    @test parse_options(["--smoke", "--results-dir", "b", "--runs-dir", "a"]) ==
          (runs_dir = "a", results_dir = "b", smoke = true)
    usage = r"^usage: julia --project scripts/channels\.jl --runs-dir <d> --results-dir <r> \[--smoke\]$"
    for args in (["--runs-dir", "a"], ["--runs-dir", "a", "--results-dir", "b", "--x"],
        ["--runs-dir", "a", "--results-dir"])
        e = thrown(() -> parse_options(args))
        @test e isa ArgumentError
        @test occursin(usage, e.msg)
    end
end

# The fluctuation snapshots of the long run `file` in `runs`, around the background of the first.
function long_run(runs, file)
    S = h5open(f -> read(f, "snapshots"), joinpath(runs, file))
    f0 = background(S[:, :, 1])
    return S, f0, S .- f0'
end

# The rank-`r` factors of each slice of `G`, and their dense states, computed here.
function factors(G, r)
    F = [svd(G[:, :, s]) for s in axes(G, 3)]
    x = (X = stack(f.U[:, 1:r] for f in F), S = stack(Diagonal(f.S[1:r]) for f in F),
        V = stack(f.V[:, 1:r] for f in F))
    return x, stack(f.U[:, 1:r] * Diagonal(f.S[1:r]) * f.V[:, 1:r]' for f in F)
end

# The per-snapshot relative errors of `R` against `G`.
snapshot_errors(R, G) = [relative(R[:, :, s], G[:, :, s]) for s in axes(G, 3)]

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
        datasets = ["advection_half", "advection_full", "two_stream_long",
            "two_stream_beams_long", "nonlinear_landau_long"]

        splits = read_csv_rows(joinpath(results, "channel_splits.csv"))
        dlra = read_csv_rows(joinpath(results, "channel_dlra.csv"))
        joint = read_csv_rows(joinpath(results, "channel_joint.csv"))
        @test first(eachline(joinpath(results, "channel_splits.csv"))) ==
              join(VlasovReductionBenchmark.CHANNEL_SPLIT_COLUMNS, ',')
        @test first(eachline(joinpath(results, "channel_dlra.csv"))) ==
              join(VlasovReductionBenchmark.CHANNEL_DLRA_COLUMNS, ',')
        @test first(eachline(joinpath(results, "channel_joint.csv"))) ==
              join(VlasovReductionBenchmark.CHANNEL_JOINT_COLUMNS, ',')
        # five datasets, the seven splits of n ∈ {4, 8}, one input rank
        @test length(splits) == 35
        @test sort([(r["dataset"], r["n"], r["mx"], r["mv"]) for r in splits]) ==
              sort([(d, string(n), string(mx), string(n ÷ mx)) for d in datasets
                    for n in (4, 8) for mx in 1:n if n % mx == 0])
        @test length(dlra) == 10
        @test sort([(r["dataset"], r["n"], r["r_in"]) for r in dlra]) ==
              sort([(d, n, "4") for d in datasets for n in ("4", "8")])
        @test length(joint) == 10
        @test sort([(r["dataset"], r["n"], r["r_in"]) for r in joint]) ==
              sort([(d, n, "4") for d in datasets for n in ("4", "8")])
        @test all(r -> r["status"] == "ok", vcat(splits, dlra, joint))
        @test all(r -> r["seed"] == "42", vcat(splits, dlra, joint))
        @test occursin("channel_splits.csv: 35 rows, 0 failed", printed)
        @test occursin("channel_dlra.csv: 10 rows, 0 failed", printed)
        @test occursin("channel_joint.csv: 10 rows, 0 failed", printed)

        # per dataset and n: I3's split, the best split by its Stage I error, and a trained
        # error on exactly those two
        for d in datasets, n in (4, 8)

            these = filter(r -> r["dataset"] == d && r["n"] == string(n), splits)
            @test [(r["mx"], r["mv"]) for r in these if r["parity_modes"] == "yes"] ==
                  [string.(PARITY_MODES[n])]
            stage1 = [parse(Float64, r["stage1_error"]) for r in these]
            @test [r["best"] == "yes" for r in these] ==
                  (1:length(these) .== argmin(stage1))
            for r in these
                trained = r["parity_modes"] == "yes" || r["best"] == "yes"
                @test isempty(r["trained_error"]) == !trained
                trained &&
                    @test parse(Float64, r["loss_last"]) < parse(Float64, r["loss_first"])
            end
        end

        # one row of each file by value, on the two-stream run at n = 8, I3's split (2, 4)
        S, f0, G = long_run(runs, "two_stream_long_alpha0.05.h5")
        @test size(S) == (32, 64, 81)
        pick(rows) = only(filter(
            r -> r["dataset"] == "two_stream_long" && r["n"] == "8" &&
                 (r["mx"], r["mv"]) == ("2", "4"),
            rows))

        # the Stage I and the trained error from I3's network and I3's training
        r = pick(splits)
        stage1 = reconstruction_errors("two_stream_long",
            weight_tied_reduction(weight_tied_network(G, 8; seed = 42), f0, 32, 64), S)
        @test argmax(stage1.snapshot_errors) > 1
        @test parse(Float64, r["stage1_error"]) ≈ stage1.error rtol=1e-10
        @test parse(Float64, r["stage1_max_snapshot_error"]) ≈ stage1.max_snapshot_error rtol=1e-10
        trained = train_weight_tied(S, f0, 8; seed = 42, steps = 100)
        after = reconstruction_errors("two_stream_long", trained.reduction, S)
        @test argmax(after.snapshot_errors) > 1
        @test parse(Float64, r["trained_error"]) ≈ after.error rtol=1e-10
        @test parse(Float64, r["trained_max_snapshot_error"]) ≈ after.max_snapshot_error rtol=1e-10
        @test abs(after.error - stage1.error) > 1e-6
        # the frontend error from GML's layer and its inverse
        nn = trained.network
        l, ps = nn.model.frontends.grid,
        weight_tied_network(G, 8; seed = 42).params.frontends.grid
        R = GML.inverse_layer(l)(l(G, ps), ps)
        @test argmax(snapshot_errors(R, G)) > 1
        @test parse(Float64, r["frontend_error"]) ≈ relative(R, G) rtol=1e-10
        @test parse(Float64, r["frontend_max_snapshot_error"]) ≈
              maximum(snapshot_errors(R, G)) rtol=1e-10

        # the DLRA channel: GML's layer with the trained bases, the trained core and the layer's
        # inverse, on each snapshot's factors
        r = pick(dlra)
        x, input = factors(G, 4)
        ll = LowRankStiefelLayer(32, 64, 2, 4)
        lps = (Φx = copy(parent(nn.params.frontends.grid.Φx)),
            Φv = copy(parent(nn.params.frontends.grid.Φv)))
        encoder, decoder = GML.encoder(nn), GML.decoder(nn)
        R = stack(axes(G, 3)) do s
            c = ll((X = x.X[:, :, s], S = x.S[:, :, s], V = x.V[:, :, s]), lps)
            ξ = decoder(encoder(vcat(c.q, c.p)))
            d = GML.inverse_layer(ll)((q = ξ[1:8], p = ξ[9:16]), lps)
            d.X * d.S * d.V'
        end
        @test argmax(snapshot_errors(R, G)) > 1
        @test r["r_eff"] == "2"
        @test parse(Float64, r["error"]) ≈ relative(R, G) rtol=1e-10
        @test parse(Float64, r["max_snapshot_error"]) ≈ maximum(snapshot_errors(R, G)) rtol=1e-10
        @test parse(Float64, r["input_error"]) ≈ relative(R, input) rtol=1e-10
        @test parse(Float64, r["input_max_snapshot_error"]) ≈
              maximum(snapshot_errors(R, input)) rtol=1e-10

        # the two-channel network: the same training, and GML's encode, decode and translate
        r = pick(joint)
        dense(d) = stack(d.X[:, :, s] * d.S[:, :, s] * d.V[:, :, s]' for s in axes(G, 3))
        two = channel_network(G, 2, 4; seed = 42, dlra = dense(x))
        train_network!(ps -> joint_loss(two.model, ps, G, x), two.params, 100)
        expected = Dict(
            "grid_" => GML.decode(two, :grid, GML.encode(two, :grid, G)),
            "dlra_" => dense(GML.decode(two, :dlra, GML.encode(two, :dlra, x))),
            "grid_to_dlra_" => dense(GML.translate(two, :grid, :dlra, G)),
            "dlra_to_grid_" => GML.translate(two, :dlra, :grid, x))
        for (prefix, R) in expected
            @test argmax(snapshot_errors(R, G)) > 1
            @test parse(Float64, r[prefix * "error"]) ≈ relative(R, G) rtol=1e-10
            @test parse(Float64, r[prefix * "max_snapshot_error"]) ≈
                  maximum(snapshot_errors(R, G)) rtol=1e-10
        end
        @test r["r_eff"] == "2"

        # every number of the page, outside the names in backticks, is a value of a CSV file at
        # the four significant digits of the page
        page = read(joinpath(results, "channels.md"), String)
        known = Set{String}()
        for rs in (splits, dlra, joint), row in rs, field in values(row)
            v = tryparse(Float64, field)
            v === nothing || push!(known, field, @sprintf("%.4g", v))
        end
        numbers = [m.match
                   for m in eachmatch(r"\d+(\.\d+)?(e[-+]?\d+)?",
            replace(page, r"`[^`]*`" => ""))]
        @test !isempty(numbers)
        @test all(in(known), numbers)
        for d in datasets
            @test occursin("## $d\n", page)
        end
        @test occursin("No row failed.", page)
    end
end
