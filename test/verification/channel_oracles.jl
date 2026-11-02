using GeometricMachineLearning: GeometricMachineLearning, BilinearStiefelLayer
using LinearAlgebra: norm
using VlasovReductionBenchmark
using VlasovReductionBenchmark: channel_network, channel_rows, dense_states, dlra_network,
                                dlra_row, fluctuation, joint_loss, joint_row,
                                reconstruction_errors, split_rows, train_network!,
                                truncated_factors, weight_tied_network,
                                weight_tied_reduction
using Test

const GML = GeometricMachineLearning

# Every array of a nested NamedTuple or Tuple of parameters, in order.
arrays(x::AbstractArray) = [Array(parent(x))]
arrays(x::Union{NamedTuple, Tuple}) = reduce(vcat, [arrays(v) for v in values(x)])

# The first `m` snapshots of the advection over half a recurrence time of the smoke grid.
function advection_fixture(m)
    times = advection_times(64, 0.5; fraction = 0.5)[1:m]
    S = advection_snapshots(32, 64, times)
    f0 = background(S[:, :, 1])
    return S, f0, fluctuation(S, f0)
end

relative(a, b) = norm(a - b) / norm(b)

@testset "Rows of the splits" begin
    S, f0, G = advection_fixture(12)
    rows, nn = split_rows("fixture", G, 4; seed = 42, steps = 5)
    @test [(r["mx"], r["mv"]) for r in rows] == [("1", "4"), ("2", "2"), ("4", "1")]
    @test [r["parity_modes"] for r in rows] == ["no", "yes", "no"]
    @test all(
        r -> (r["dataset"], r["n"], r["seed"], r["status"]) ==
             ("fixture", "4", "42", "ok"), rows)
    stage1 = [parse(Float64, r["stage1_error"]) for r in rows]
    @test [r["best"] for r in rows] == [i == argmin(stage1) ? "yes" : "no" for i in 1:3]

    # the Stage I error at the split of PARITY_MODES is that of weight_tied_network, through
    # the lift of weight_tied_reduction
    i3 = rows[2]
    reference = reconstruction_errors("fixture",
        weight_tied_reduction(weight_tied_network(G, 4; seed = 42), f0, 32, 64), S)
    @test parse(Float64, i3["stage1_error"]) ≈ reference.error rtol=1e-10
    @test parse(Float64, i3["stage1_max_snapshot_error"]) ≈ reference.max_snapshot_error rtol=1e-10
    # its trained error is that of train_weight_tied, and differs from the Stage I error
    trained = train_weight_tied(S, f0, 4; seed = 42, steps = 5)
    reference = reconstruction_errors("fixture", trained.reduction, S)
    @test parse(Float64, i3["trained_error"]) ≈ reference.error rtol=1e-10
    @test parse(Float64, i3["trained_max_snapshot_error"]) ≈ reference.max_snapshot_error rtol=1e-10
    @test parse(Float64, i3["trained_error"]) < parse(Float64, i3["stage1_error"])
    @test parse(Float64, i3["loss_first"]) == trained.training.first
    @test parse(Float64, i3["loss_last"]) == trained.training.last
    @test parse(Float64, i3["offline_seconds"]) > 0
    @test arrays(nn.params.frontends) == arrays(trained.network.params.frontends)
    @test arrays(nn.params.core) == arrays(trained.network.params.core)
    # a split that is neither that of PARITY_MODES nor the best is not trained
    for r in rows
        if r["parity_modes"] == r["best"] == "no"
            @test r["trained_error"] == r["loss_first"] == r["offline_seconds"] == ""
        else
            @test r["trained_error"] != ""
        end
    end

    # the frontend error of the first split, from GML's layer and its inverse
    l = BilinearStiefelLayer(32, 64, 1, 4)
    ps = (Φx = zeros(32, 1), Φv = zeros(32, 4))
    GML.solve!(l, ps, G)
    R = GML.inverse_layer(l)(l(G, ps), ps)
    @test parse(Float64, rows[1]["frontend_error"]) ≈ relative(R, G) rtol=1e-10

    # a training that fails keeps its status and leaves the trained error empty
    rows, nn = split_rows("fixture", G, 4; seed = 42, steps = 0)
    @test nn === nothing
    for r in rows
        r["parity_modes"] == r["best"] == "no" && continue
        @test occursin(r"^failed", r["status"])
        @test r["trained_error"] == r["trained_max_snapshot_error"] == ""
        @test r["stage1_error"] != ""
    end
end

@testset "Rows of the DLRA channel" begin
    _, _, G = advection_fixture(12)
    rows, nn = split_rows("fixture", G, 8; seed = 42, steps = 3)
    # here the best split, (4, 2), is not the (2, 4) of PARITY_MODES, and both are trained
    @test [(r["mx"], r["mv"]) for r in rows if r["best"] == "yes"] == [("4", "2")]
    @test [r["trained_error"] != "" for r in rows] == [false, true, true, false]
    row = dlra_row("fixture", G, 8, 4, nn, "ok"; seed = 42)
    @test (row["mx"], row["mv"], row["r_in"], row["seed"], row["status"]) ==
          ("2", "4", "4", "42", "ok")
    # GML's decode rank min(r_out, m_x, 2 m_v) at r_out = m_x m_v
    @test row["r_eff"] == "2"
    # the errors from GML's encode and decode of the DLRA channel of a copy of the network
    two = dlra_network(nn)
    x = truncated_factors(G, 4)
    D = zeros(32, 64, 12)
    for s in 1:12
        xs = (X = x.X[:, :, s], S = x.S[:, :, s], V = x.V[:, :, s])
        d = GML.decode(two, :dlra, GML.encode(two, :dlra, xs))
        D[:, :, s] = d.X * d.S * d.V'
    end
    @test parse(Float64, row["error"]) ≈ relative(D, G) rtol=1e-10
    input = stack(x.X[:, :, s] * x.S[:, :, s] * x.V[:, :, s]' for s in 1:12)
    @test parse(Float64, row["input_error"]) ≈ relative(D, input) rtol=1e-10
    @test parse(Float64, row["max_snapshot_error"]) ≈
          maximum(relative(D[:, :, s], G[:, :, s]) for s in 1:12) rtol=1e-10
    @test parse(Float64, row["input_max_snapshot_error"]) ≈
          maximum(relative(D[:, :, s], input[:, :, s]) for s in 1:12) rtol=1e-10
    # a failed training of the grid network leaves the errors empty
    row = dlra_row("fixture", G, 8, 4, nothing, "failed at step 0"; seed = 42)
    @test row["status"] == "failed at step 0"
    @test row["error"] == row["input_error"] == row["r_eff"] == ""
end

@testset "Rows of the two-channel network" begin
    _, _, G = advection_fixture(12)
    x = truncated_factors(G, 4)
    row = joint_row("fixture", G, 4, 4; seed = 42, steps = 3)
    @test (row["mx"], row["mv"], row["r_in"], row["r_eff"], row["status"]) ==
          ("2", "2", "4", "2", "ok")
    @test parse(Float64, row["loss_last"]) < parse(Float64, row["loss_first"])
    # each error against the fluctuation, from GML's encode, decode and translate of the same
    # training
    nn = channel_network(G, 2, 2; seed = 42, dlra = dense_states(x))
    train_network!(ps -> joint_loss(nn.model, ps, G, x), nn.params, 3)
    dense(d) = stack(d.X[:, :, s] * d.S[:, :, s] * d.V[:, :, s]' for s in 1:12)
    expected = Dict(
        "grid_error" => relative(GML.decode(nn, :grid, GML.encode(nn, :grid, G)), G),
        "dlra_error" => relative(dense(GML.decode(nn, :dlra, GML.encode(nn, :dlra, x))), G),
        "grid_to_dlra_error" => relative(dense(GML.translate(nn, :grid, :dlra, G)), G),
        "dlra_to_grid_error" => relative(GML.translate(nn, :dlra, :grid, x), G))
    for (column, value) in expected
        @test parse(Float64, row[column]) ≈ value rtol=1e-10
    end
    # a translation is not the round trip of its source channel
    @test abs(expected["grid_to_dlra_error"] - expected["grid_error"]) > 1e-6
    @test abs(expected["dlra_to_grid_error"] - expected["dlra_error"]) > 1e-6
    row = joint_row("fixture", G, 4, 4; seed = 42, steps = 0)
    @test occursin(r"^failed", row["status"])
    @test row["grid_error"] == row["dlra_to_grid_error"] == row["r_eff"] == ""
end

@testset "The rows of one dataset and n" begin
    _, _, G = advection_fixture(12)
    mode = (seed = 42, steps = 2, dlra_ranks = (2, 3), joint_rank = 3)
    rows = channel_rows("fixture", G, 4, mode)
    @test length(rows.splits) == 3
    @test [(r["r_in"], r["status"]) for r in rows.dlra] == [("2", "ok"), ("3", "ok")]
    @test all(r -> r["error"] != "", rows.dlra)
    @test (rows.joint["r_in"], rows.joint["status"]) == ("3", "ok")
    # the DLRA rows take the status of the training at the split of PARITY_MODES
    rows = channel_rows("fixture", G, 4, merge(mode, (steps = 0,)))
    i3 = only(r for r in rows.splits if r["parity_modes"] == "yes")
    @test occursin(r"^failed", i3["status"])
    @test all(r -> r["status"] == i3["status"] && r["error"] == "", rows.dlra)
    @test occursin(r"^failed", rows.joint["status"])
end
