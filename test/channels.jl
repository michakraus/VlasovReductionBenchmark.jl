using GeometricMachineLearning: GeometricMachineLearning, CharbonnierRelativeLoss
using HDF5: attrs, h5open
using LinearAlgebra: Diagonal, norm, qr
using Random: Xoshiro
using VlasovReductionBenchmark
using VlasovReductionBenchmark: PARITY_MODES, channel_datasets, channel_network,
                                channels_mode,
                                dense_states, dlra_network, fluctuation,
                                frontend_reconstruction, grid_splits, joint_loss,
                                rank_error,
                                relative_errors, truncated_factors, weight_tied_network,
                                write_channels_page
using Test

const GML = GeometricMachineLearning

# Every array of a nested NamedTuple or Tuple of parameters, in order.
arrays(x::AbstractArray) = [Array(parent(x))]
arrays(x::Union{NamedTuple, Tuple}) = reduce(vcat, [arrays(v) for v in values(x)])

# The fluctuations of the first `m` snapshots of the advection over half a recurrence time of
# the smoke grid, around the background of the first.
function advection_fixture(m)
    times = advection_times(64, 0.5; fraction = 0.5)[1:m]
    S = advection_snapshots(32, 64, times)
    return fluctuation(S, background(S[:, :, 1]))
end

relative(a, b) = norm(a - b) / norm(b)

@testset "Splits of the grid channel" begin
    @test grid_splits(4, 32, 64) == [(1, 4), (2, 2), (4, 1)]
    @test grid_splits(8, 32, 64) == [(1, 8), (2, 4), (4, 2), (8, 1)]
    @test sum(length(grid_splits(n, 64, 128)) for n in (4, 8, 16, 32)) == 18
    @test sum(length(grid_splits(n, 32, 64)) for n in (4, 8)) == 7
    # the bounds m_x ≤ n_x and m_v ≤ n_v / 2
    @test grid_splits(32, 16, 128) == [(1, 32), (2, 16), (4, 8), (8, 4), (16, 2)]
    @test grid_splits(32, 64, 8) == [(8, 4), (16, 2), (32, 1)]
end

@testset "Frontend error on a fixture of known modes" begin
    nx, nv = 64, 128
    x, v = grid(linear_landau(), nx, nv)
    U = Matrix(qr(hcat(cos.(0.5 .* x), sin.(0.5 .* x), cos.(x), sin.(x))).Q)
    c = [4.0, 3.0, 2.0, 1.0]
    # the even and the odd extension of one half-grid vector, so that one column of Φ_v holds
    # both parities
    φ = exp.(-v[(nv ÷ 2 + 1):end] .^ 2 ./ 2) .* (1 .+ v[(nv ÷ 2 + 1):end])
    w₊, w₋ = vcat(reverse(φ), φ), vcat(-reverse(φ), φ)
    @test w₊ == reverse(w₊)
    @test w₋ == -reverse(w₋)
    G = stack(c[s] * U[:, s] * (w₊ + w₋)' for s in 1:4)
    e41 = relative_errors(G, frontend_reconstruction(channel_network(G, 4, 1; seed = 42), G))
    @test e41.error < 1e-12
    @test e41.max_snapshot_error < 1e-12
    e14 = relative_errors(G, frontend_reconstruction(channel_network(G, 1, 4; seed = 42), G))
    @test abs(e14.error - sqrt(sum(abs2, c[2:end]) / sum(abs2, c))) ≤ 1e-12
    # the first snapshot holds the one x mode, the others lose theirs
    @test abs(e14.max_snapshot_error - 1) ≤ 1e-12
end

@testset "Relative errors" begin
    G = zeros(2, 2, 2)
    G[:, :, 1] = [3.0 0.0; 0.0 4.0]
    G[:, :, 2] = [1.0 0.0; 0.0 0.0]
    R = copy(G)
    R[1, 1, 1] = 0.0
    R[1, 1, 2] = 0.0
    e = relative_errors(G, R)
    @test e.error ≈ sqrt(9 + 1) / sqrt(26) rtol=1e-15
    # the maximum is that of the second snapshot
    @test e.max_snapshot_error ≈ 1 rtol=1e-15
end

@testset "The network at the split of PARITY_MODES is weight_tied_network" begin
    G = advection_fixture(12)
    for n in (4, 8)
        mine = channel_network(G, PARITY_MODES[n]...; seed = 42)
        theirs = weight_tied_network(G, n; seed = 42)
        @test arrays(mine.params.frontends) == arrays(theirs.params.frontends)
        @test arrays(mine.params.core) == arrays(theirs.params.core)
    end
end

@testset "Truncated factors" begin
    σ = [3.0, 2.0, 1.0, 0.5]
    U = Matrix(qr(randn(Xoshiro(1), 8, 4)).Q)
    V = Matrix(qr(randn(Xoshiro(2), 6, 4)).Q)
    G = stack([U * Diagonal(σ) * V', U[:, 1:2] * Diagonal([1.0, 0.25]) * V[:, 1:2]'])
    x = truncated_factors(G, 2)
    @test (size(x.X), size(x.S), size(x.V)) == ((8, 2, 2), (2, 2, 2), (6, 2, 2))
    D = dense_states(x)
    @test relative(D[:, :, 1], G[:, :, 1]) ≈ sqrt((1 + 0.25) / sum(abs2, σ)) rtol=1e-13
    @test relative(D[:, :, 1], G[:, :, 1]) ≈ rank_error(G[:, :, 1], 2) rtol=1e-13
    @test relative(D[:, :, 2], G[:, :, 2]) ≤ 1e-14
    # a rank above that of the second snapshot keeps the trailing factors, and the snapshot
    D = dense_states(truncated_factors(G, 4))
    @test relative(D, G) ≤ 1e-14
end

@testset "The DLRA channel encodes as the grid channel" begin
    G = advection_fixture(12)
    rng = Xoshiro(7)
    for (mx, mv) in ((2, 4), (4, 2), (1, 8))
        grid_nn = channel_network(G, mx, mv; seed = 42)
        nn = dlra_network(grid_nn)
        r = 5
        X, Sm, V = randn(rng, 32, r), randn(rng, r, r), randn(rng, 64, r)
        zg = GML.encode(grid_nn, :grid, X * Sm * V')
        zd = GML.encode(nn, :dlra, (X = X, S = Sm, V = V))
        @test relative(zd, zg) ≤ 1e-12
        Ox = Matrix(qr(randn(rng, r, r)).Q)
        Ov = Matrix(qr(randn(rng, r, r)).Q)
        zo = GML.encode(nn, :dlra, (X = X * Ox, S = Ox' * Sm * Ov, V = V * Ov))
        @test relative(zo, zd) ≤ 1e-12
        # the grid channel of the copy is that of the grid network
        @test GML.encode(nn, :grid, G) == GML.encode(grid_nn, :grid, G)
    end
end

@testset "Two channels, one core" begin
    G = advection_fixture(12)
    x = truncated_factors(G, 4)
    nn = channel_network(G, 2, 2; seed = 42, dlra = dense_states(x))
    loss = CharbonnierRelativeLoss()
    single = (loss(nn.model, nn.params, :grid, G), loss(nn.model, nn.params, :dlra, x))
    @test all(>(0), single)
    @test abs(joint_loss(nn.model, nn.params, G, x) - sum(single)) ≤ 1e-12 * sum(single)
    # the DLRA frontend's bases are the Tucker bases of the dense rank-4 states
    reference = channel_network(dense_states(x), 2, 2; seed = 42)
    @test arrays(nn.params.frontends.dlra) == arrays(reference.params.frontends.grid)
    @test arrays(nn.params.frontends.grid) ==
          arrays(channel_network(G, 2, 2; seed = 42).params.frontends.grid)
end

@testset "Channels mode" begin
    @test channels_mode(false) ==
          (grid = (64, 128), ns = (4, 8, 16, 32), steps = 2000, seed = 42,
        dlra_ranks = (4, 8, 16), joint_rank = 8, long_tend = 100.0,
        advection_fractions = (advection_half = 0.5, advection_full = 1.0))
    @test channels_mode(true) ==
          (grid = (32, 64), ns = (4, 8), steps = 100, seed = 42, dlra_ranks = (4,),
        joint_rank = 4, long_tend = 40.0,
        advection_fractions = (advection_half = 0.5, advection_full = 1.0))
end

@testset "Long runs read or written" begin
    mode = (grid = (32, 64), advection_fractions = (advection_half = 0.5,), long_tend = 1.0)
    mktempdir() do dir
        names = ["two_stream_long_alpha0.05.h5", "two_stream_beams_long_alpha0.05.h5",
            "nonlinear_landau_long_alpha0.5.h5"]
        read_back() = redirect_stdout(() -> channel_datasets(mode, dir), devnull)
        datasets = read_back()
        @test [d[1] for d in datasets] == ["advection_half", "two_stream_long",
            "two_stream_beams_long", "nonlinear_landau_long"]
        @test all(isfile(joinpath(dir, f)) for f in names)
        @test size(datasets[2][2]) == (32, 64, 3)
        # a stored run of the same grid, amplitude and end time is read, not run; one of
        # another grid, amplitude or end time is run again and written over
        function store(file; nx = 32, nv = 64, tend = 1.0,
                alpha = endswith(file, "0.5.h5") ? 0.5 : 0.05)
            h5open(joinpath(dir, file), "w") do f
                f["snapshots"] = fill(7.0, 32, 64, 3)
                f["snapshot_times"] = [0.0, 0.5, 1.0]
                a = attrs(f)
                a["nx"], a["nv"], a["tend"], a["alpha"] = nx, nv, tend, alpha
            end
        end
        stored(d) = all(==(7.0), d[2])
        store(names[1])
        store(names[2]; nx = 16)
        store(names[3]; tend = 2.0)
        datasets = read_back()
        @test stored(datasets[2])
        @test datasets[2][3] == [0.0, 0.5, 1.0]
        @test !any(stored, datasets[3:4])
        @test h5open(f -> attrs(f)["nx"], joinpath(dir, names[2])) == 32
        @test h5open(f -> attrs(f)["tend"], joinpath(dir, names[3])) == 1.0
        store(names[1]; nv = 32)
        store(names[2]; alpha = 0.5)
        store(names[3])
        datasets = read_back()
        @test !any(stored, datasets[2:3])
        @test stored(datasets[4])
    end
end

@testset "Page" begin
    mktempdir() do dir
        write(joinpath(dir, "channel_splits.csv"),
            "dataset,n,mx,mv,seed,parity_modes,best,status,frontend_error," *
            "frontend_max_snapshot_error,stage1_error,stage1_max_snapshot_error," *
            "trained_error,trained_max_snapshot_error,offline_seconds,loss_first,loss_last\n" *
            "d1,4,1,4,42,no,yes,ok,0.5,0.75,0.625,0.875,0.25,0.5,1.5,0.1,0.05\n" *
            "d1,4,2,2,42,yes,no,failed at step 3,0.375,0.5,0.6875,0.75,,,1.5,0.1,NaN\n" *
            "d1,4,4,1,42,no,no,ok,0.95,0.95,0.96,0.96,,,,,\n")
        write(joinpath(dir, "channel_dlra.csv"),
            "dataset,n,mx,mv,r_in,r_eff,seed,status,error,max_snapshot_error," *
            "input_error,input_max_snapshot_error\n" *
            "d1,4,2,2,4,,42,failed at step 3,,,,\n")
        write(joinpath(dir, "channel_joint.csv"),
            "dataset,n,mx,mv,r_in,r_eff,seed,status,grid_error,grid_max_snapshot_error," *
            "dlra_error,dlra_max_snapshot_error,grid_to_dlra_error," *
            "grid_to_dlra_max_snapshot_error,dlra_to_grid_error," *
            "dlra_to_grid_max_snapshot_error,offline_seconds,loss_first,loss_last\n" *
            "d1,4,2,2,4,2,42,ok,0.125,0.25,0.1875,0.25,0.3125,0.5,0.4375,0.5,2.5,0.2,0.1\n")
        page = read(write_channels_page(dir), String)
        @test occursin("## d1\n", page)
        @test occursin("| 4 | (1, 4) | 0.5 | 0.625 | 0.25 | best |", page)
        @test occursin("| 4 | (2, 2) | 0.375 | 0.6875 | — | `PARITY_MODES` |", page)
        @test occursin("| 4 | (4, 1) | 0.95 | 0.96 | — |  |", page)
        @test occursin("| 4 | (2, 2) | 4 | — | — | — |", page)
        @test occursin("| 4 | (2, 2) | 4 | 0.125 | 0.1875 | 0.3125 | 0.4375 |", page)
        @test occursin(
            "- `channel_splits.csv`: d1, n = 4, (m_x, m_v) = (2, 2): failed at step 3", page)
        @test occursin(
            "- `channel_dlra.csv`: d1, n = 4, (m_x, m_v) = (2, 2): failed at step 3", page)
        @test !occursin("No row failed.", page)
    end
end
