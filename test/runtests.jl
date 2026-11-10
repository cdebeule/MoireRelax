# Tests of MoireRelax: invariants of the method, and reference values that guard against unintended changes.
# Run with `julia --project=. -e 'using Pkg; Pkg.test()'` from the repository root.

using MoireRelax, Test, StaticArrays, LinearAlgebra, Random, Aqua
using MoireRelax: FFTW

Random.seed!(1)

# graphene (PRB 98, 224102): Lamé constants and stacking potential [c1, c2, c3, p1, p3] (meV per cell)
const μ, λ = 47352.0, 69518.0 - 47352.0
const Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0.0, 0.0]
const P = SVector{5}(Vg)
const K1 = [2 / (λ + 2μ)  2 / μ]                    # compliances of u = u1 - u2
const flags = FFTW.ESTIMATE                         # reproducible FFT plans

# the stacking energy of a bilayer, u = u1 - u2
bilayer_potential(moire, ws) = (dVdu, u) -> stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)

function relax_bilayer(ws, θ; solver = :lbfgs, ϵ0 = 0.0, u0 = ComplexF64[])
    moire = MoireGeometry(moire_matrix(θ, ϵ0), ws)
    C = fill(stiffness(Vg, ws), 1, 1)
    return relax_moire(bilayer_potential(moire, ws), K1, C, moire, ws; solver, u0, verbose = false), moire
end

@testset "MoireRelax" begin

@testset "lattice and geometry" begin
    lat = Hexagonal()
    (a1, a2), (b1, b2) = lattice_vectors(lat), reciprocal_vectors(lat)
    @test [dot(b, a) for b in (b1, b2), a in (a1, a2)] ≈ 2π * I
    @test cell_area(lat) ≈ abs(a1[1] * a2[2] - a1[2] * a2[1])
    @test (a1, a2) == (MoireRelax.a1, MoireRelax.a2) && Acell == cell_area(lat)
    # the rotation by 2π/3 maps the lattice to itself, and its reciprocal version is its inverse transpose
    R = MoireRelax.rotation(lat)
    A = [a1 a2]
    @test A * R ≈ [cosd(120) -sind(120); sind(120) cosd(120)] * A
    @test MoireRelax.reciprocal_rotation(lat) == transpose(inv(R))
    @test R^3 == I
    # the shells
    @test MoireRelax.shell(lat, (1, 0)) == 1 && MoireRelax.shell(lat, (2, 1)) == 2 && MoireRelax.shell(lat, (1, -1)) == 2
    bvecs = make_bvecs(4)
    @test size(bvecs, 2) == 3 * 4 * 5 / 2                            # 3s representatives of the 6s vectors of shell s
    @test issorted(MoireRelax.shell.(Ref(lat), eachcol(bvecs)))
    @test allunique(eachcol([bvecs -bvecs]))                          # one of each ±g
    # twist and moiré length
    for θ in (0.5, 1.3, 5.0)
        @test twist_angle(moire_matrix(θ)) ≈ θ
        ws = Workspace(3, 1; flags)
        @test MoireGeometry(moire_matrix(θ), ws).L ≈ 1 / (2sind(θ / 2))
    end
    @test_throws ArgumentError MoireGeometry([1.0 0.1; 0.0 1.0] - I, WorkspaceC3(3, 1; flags))
    # the geometry of saved modes
    ws = Workspace(5, 1; flags)
    @test MoireGeometry(moire_matrix(2.0), Float64.(ws.bvecs)).g ≈ MoireGeometry(moire_matrix(2.0), ws).g
end

@testset "stacking potential" begin
    # the D3 form against its Fourier series Σ_G 2 Re(V_G e^{iG⋅ϕ}) over the three stars, and its analytic Hessian
    b1, b2 = reciprocal_vectors(Hexagonal())
    Vg3 = [1.3, -0.4, 0.2, 0.7, -0.1]
    c1, c2, c3, p1, p3 = Vg3
    terms = [(b1, c1 + im * p1), (b2, c1 + im * p1), (-b1 - b2, c1 + im * p1),
             (b1 + 2b2, c2), (-2b1 - b2, c2), (b1 - b2, c2),
             (2b1, c3 + im * p3), (2b2, c3 + im * p3), (-2b1 - 2b2, c3 + im * p3)]
    series(ϕ) = sum(2real(c * cis(dot(G, ϕ))) for (G, c) in terms)
    dseries(ϕ) = sum(2real(im * c * cis(dot(G, ϕ))) * G for (G, c) in terms)
    d2series(ϕ) = sum(-2real(c * cis(dot(G, ϕ))) * G * G' for (G, c) in terms)
    for _ in 1:5
        ϕ = SA[randn(), randn()]
        V, dV = VdV_fast(ϕ, Vg3...)
        @test V ≈ series(ϕ)
        @test dV ≈ dseries(ϕ)
        @test hessian(Vg3, ϕ) ≈ d2series(ϕ)                             # analytic
        @test hessian(ϕ -> VdV_fast(ϕ, Vg3...), ϕ) ≈ d2series(ϕ) rtol = 1e-5   # a callable: central differences
    end
    # graphene: the minimum is AB, (a1 + a2)/3, with the curvature of the analytic Hessian
    a1, a2 = lattice_vectors(Hexagonal())
    ϕmin = minimum_stacking(Vg)
    @test ϕmin ≈ (a1 + a2) / 3 || ϕmin ≈ -(a1 + a2) / 3
    @test norm(last(VdV_fast(ϕmin, P...))) < 1e-8
    Hmin = hessian(Vg, ϕmin)
    @test Hmin ≈ Hmin'
    @test isposdef(Symmetric(Hmin))
    @test stiffness(Vg) ≈ tr(Hmin) / 2
    @test stiffness(Vg, Workspace(5, 1; flags)) ≈ stiffness(Vg)
    # any callable V(ϕ) -> (V, ∇V) serves as the potential
    Vfun(ϕ) = VdV_fast(ϕ, P...)
    @test minimum_stacking(Vfun) ≈ ϕmin
    @test stiffness(Vfun) ≈ stiffness(Vg) rtol = 1e-6                 # finite-difference Hessian for a callable
    @test curvature(Vfun, ϕmin) ≈ curvature(Vg, ϕmin) rtol = 1e-6
end

@testset "workspaces and grid" begin
    ws, wsC3 = Workspace(10, 1; flags), WorkspaceC3(10, 1; flags)
    @test ws.lattice == Hexagonal() == wsC3.lattice
    @test multiplicity(ws) == 1 && multiplicity(wsC3) == 3
    @test size(wsC3.bvecs, 2) == size(ws.bvecs, 2) ÷ 3
    @test wsC3.bvecs == ws.bvecs[:, 1:3:end]
    # the C3 orbits cover the grid once
    N = wsC3.N
    orb = grid_orbits(Hexagonal(), N)
    @test sort(unique(vec(orb))) == 1:N^2
    @test sum(length(unique(col)) for col in eachcol(orb)) == N^2
    @test orb == wsC3.orb
    # a random C3 state is C3-symmetric on the grid: u(Rr) = R u(r), i.e. D[R p] = ω D[p]
    moire = MoireGeometry(moire_matrix(1.0), wsC3)
    D = to_grid!(wsC3, randn(ComplexF64, size(wsC3.bvecs, 2), 2), moire)
    @test maximum(abs(D[orb[r, k]] - ω3[r] * D[orb[1, k]]) for r in 2:3, k in axes(orb, 2)) < 1e-12
    # to_grid! and from_grid! are adjoint: the gradient of ⟨V⟩ on the grid for V = Re(f̄ D) is from_grid!(f)
    moire = MoireGeometry(moire_matrix(1.0), ws)
    n = size(ws.bvecs, 2)
    u, du = randn(ComplexF64, n, 2), randn(ComplexF64, n, 2)
    F = randn(ComplexF64, ws.N, ws.N)                           # forces on the grid
    E(u) = real(sum(conj(F) .* to_grid!(ws, u, moire))) / ws.N^2
    ws.D .= F
    dVdu = from_grid!(zeros(ComplexF64, n, 2), ws, moire)
    @test real(sum(conj(dVdu) .* du)) ≈ E(u + du) - E(u) rtol = 1e-8
end

@testset "stacking energy and gradient" begin
    for ws in (Workspace(8, 1; flags), WorkspaceC3(8, 1; flags), WorkspaceC3(6, 2; flags))
        nf = ws.nf
        moire = MoireGeometry(moire_matrix(1.5), ws)
        f1(ϕ0, w) = VdV_fast(ϕ0 + w, P...)
        function f2(ϕ0, w1, w2)
            (v1, d1), (v2, d2) = VdV_fast(ϕ0 + w1, P...), VdV_fast(ϕ0 - w2, P...)
            return v1 + v2, d1, -d2
        end
        pot!(dVdu, u) = stacking_energy!(nf == 1 ? f1 : f2, dVdu, u, moire, ws)
        n = size(ws.bvecs, 2)
        u = 0.1randn(ComplexF64, n, 2nf)
        # the gradient per mode against finite differences (C3: the stored amplitude stands for 3 modes)
        dVdu = similar(u)
        E0 = pot!(dVdu, u)
        @test E0 ≈ pot!(nothing, u)
        for _ in 1:3
            du = 1e-5randn(ComplexF64, n, 2nf)
            @test 2multiplicity(ws) * real(sum(conj(dVdu) .* du)) ≈ pot!(nothing, u + du) - pot!(nothing, u - du) atol = 1e-12
        end
        # the LBFGS objective in the preconditioned variables, against check_gradient
        K = repeat(K1, 1, nf)
        C = Matrix(stiffness(Vg, ws) * I, nf, nf)
        S, = precond(K, C, moire.L, moire; mult = multiplicity(ws))
        fg! = MoireRelax.LBFGSObjective(pot!, similar(u), similar(u), similar(u), S, K, moire.L, multiplicity(ws))
        @test check_gradient(fg!, 0.01randn(2n * 2nf)) < 1e-5
    end
end

@testset "relaxation" begin
    θ = 1.0
    res, moire = relax_bilayer(Workspace(20, 1; flags), θ)
    @test keys(res) == (:u, :E_el, :E_V, :converged, :iterations)
    @test res.iterations > 0
    u, E_el, E_V, converged = res                                   # positional destructuring works too
    @test converged
    # verbose = false prints nothing
    ws = Workspace(8, 1; flags)
    moire8 = MoireGeometry(moire_matrix(θ), ws)
    mktemp() do path, io
        redirect_stdout(io) do
            relax_moire(bilayer_potential(moire8, ws), K1, fill(stiffness(Vg, ws), 1, 1), moire8, ws; verbose = false)
        end
        flush(io)
        @test filesize(path) == 0
    end
    @test E_el ≈ 1.2194649506 atol = 1e-7                           # reference values (graphene, 1°, 20 shells)
    @test E_V ≈ -2.7376206292 atol = 1e-7
    @test E_el ≈ elastic_energy(u, K1, moire.L)
    # DIIS and LBFGS agree
    (ud, E_eld, E_Vd, convd), = relax_bilayer(Workspace(20, 1; flags), θ; solver = :diis)
    @test convd
    @test E_eld + E_Vd ≈ E_el + E_V atol = 1e-8
    @test maximum(abs, ud - u) < 1e-5 * maximum(abs, u)
    # C3 symmetry gives the same state
    (uc, E_elc, E_Vc, convc), moirec = relax_bilayer(WorkspaceC3(20, 1; flags), θ)
    @test convc
    @test E_elc ≈ E_el atol = 1e-9
    @test E_Vc ≈ E_V atol = 1e-9
    wsc, ws = WorkspaceC3(20, 1; flags), Workspace(20, 1; flags)
    @test maximum(abs, to_grid!(wsc, uc, moirec) - to_grid!(ws, u, moire)) < 1e-5 * maximum(abs, ws.D)
    @test maximum(abs, uc - u[1:3:end, :]) < 1e-5 * maximum(abs, u)  # the stored C3 amplitudes are the stars' first modes
    # a start from fewer shells and an extrapolation
    (u8,), = relax_bilayer(Workspace(8, 1; flags), θ)
    (u20, E_el20, E_V20), = relax_bilayer(Workspace(20, 1; flags), θ; u0 = u8)
    @test E_el20 + E_V20 ≈ E_el + E_V atol = 1e-8
    @test extrapolate(θ, [1.2, 1.1], [u, u]) ≈ u
    @test isempty(extrapolate(θ, Float64[], Matrix{ComplexF64}[]))
    # the trilayer of layer fields
    ws3 = WorkspaceC3(20, 3; flags)
    moire3 = MoireGeometry(moire_matrix(θ), ws3)
    k = stiffness(Vg, ws3)
    function V3(ϕ0, u1, u2, u3)
        v12, d12 = VdV_fast(ϕ0 + u1 - u2, P...)
        v23, d23 = VdV_fast(-ϕ0 + u2 - u3, P...)
        return v12 + v23, d12, d23 - d12, -d23
    end
    u3, E_el3, E_V3, conv3 = relax_moire((d, u) -> stacking_energy!(V3, d, u, moire3, ws3), repeat(K1 / 2, 1, 3),
                                         chain_laplacian([k, k]), moire3, ws3)
    @test conv3
    @test E_el3 ≈ 2.8292067119 atol = 1e-7
    @test E_V3 ≈ -6.9405792468 atol = 1e-7
    D3 = to_grid!(ws3, u3, moire3)
    @test maximum(abs, D3[:, :, 1] - D3[:, :, 3]) < 1e-6             # mirror symmetry: the outer layers move alike
    @test maximum(abs, D3[:, :, 1] + D3[:, :, 2] + D3[:, :, 3]) < 1e-8 # and the center of mass stays put
end

@testset "supercells" begin
    θ, ϵ0 = 2.0, 0.01
    ws1 = Workspace(8, 1; flags)
    (u1, E_el1, E_V1), moire1 = relax_bilayer(ws1, θ; ϵ0)
    for S in ([2 0; 0 2], [1 -1; 1 2], [1 -1; 0 2])
        wsS = Workspace(8, 1; S, flags)
        d = det(S)
        @test count(moire_periodic(wsS)) == size(ws1.bvecs, 2)      # the moiré-periodic modes are those of the cell
        @test size(wsS.bvecs, 2) > d * size(ws1.bvecs, 2) ÷ 2         # plus the fractional ones between them
        # the transfer of the moiré-cell solution and back
        uS0 = transfer_modes(u1, ws1, wsS)
        @test supercell_fraction(uS0, wsS) == 0
        @test transfer_modes(uS0, wsS, ws1) ≈ u1
        # from a periodic start, the relaxation stays periodic and ends at the energy of the moiré cell
        (uS, E_elS, E_VS, convS), = relax_bilayer(wsS, θ; ϵ0, u0 = uS0)
        @test convS
        @test E_elS ≈ E_el1 atol = 1e-9
        @test E_VS ≈ E_V1 atol = 1e-9
        @test supercell_fraction(uS, wsS) < 1e-12
        @test length(wsS.ϕ0) == prod(wsS.N)
    end
    # a saved run: the geometry from the saved modes
    wsS = Workspace(8, 1; S = [1 -1; 1 2], flags)
    @test MoireGeometry(moire_matrix(θ, ϵ0), wsS.bvecs).g ≈ MoireGeometry(moire_matrix(θ, ϵ0), wsS).g
    @test_throws ArgumentError Workspace(8, 1; S = [1 1; 1 1], flags)   # det S must be positive
end

@testset "real space" begin
    θ = 2.0
    ws = WorkspaceC3(10, 1; flags)
    (u,), moire = relax_bilayer(ws, θ)
    M, (a1, a2) = moire_matrix(θ), lattice_vectors(Hexagonal())
    # G holds the moiré reciprocal lattice vectors: g of the mode (n1, n2) is n1 G1 + n2 G2
    @test all(moire.g[m] ≈ complex((moire.G * ws.bvecs[:, m])...) for m in eachindex(moire.g))
    # the coordinates in the cell of the point r = s1 L1 + s2 L2 (L_i = M⁻¹ a_i / L) are (s1, s2)
    r = (0.3 * (M \ a1) + 0.45 * (M \ a2)) / moire.L
    @test cell_coordinates(moire, r) ≈ [0.3, 0.45]
    @test rigid_disregistry(moire, Hexagonal(), r) ≈ 0.3a1 + 0.45a2
    @test cell_coordinates(moire, SA[0.0, 0.0]) == [0, 0]
    # fields interpolated from a fine grid against direct sums over the modes
    wsf = WorkspaceC3(10, 1, 128; flags)
    D = to_grid!(wsf, u, moire)[:, :, 1]
    xs, ys = range(-0.8, 0.8, 17), range(-0.6, 0.9, 13)
    ws1 = Workspace(10, 1; flags)                        # all modes of the stars, three per stored amplitude
    moire1 = MoireGeometry(M, ws1)
    star(m) = (m - 1) ÷ 3 + 1
    w = [-im * moire1.g[m] * moire1.invdotgg[m] * (u[star(m), 1] + im * u[star(m), 2]) for m in eachindex(moire1.g)]
    wm = [im * moire1.g[m] * moire1.invdotgg[m] * (conj(u[star(m), 1]) + im * conj(u[star(m), 2])) for m in eachindex(moire1.g)]
    Ex, Ey = cis.(real.(moire1.g) .* xs'), cis.(imag.(moire1.g) .* ys')
    uref = transpose(Ex) * (w .* Ey) + transpose(conj.(Ex)) * (wm .* conj.(Ey))
    uxy = grid_to_points(D, wsf, moire, xs, ys)
    @test maximum(abs, uxy - uref) < 2e-3 * maximum(abs, uref)
    rot = scalar_to_grid!(wsf, u[:, 2] ./ (2moire.L), moire)
    rotref = field_synthesizer(moire1, xs, ys)(repeat(u[:, 2] ./ (2moire.L), inner = 3))
    @test maximum(abs, grid_to_points(rot, wsf, moire, xs, ys) - rotref) < 2e-3 * maximum(abs, rotref)
    # the stacking energy on the grid, interpolated, against V at the interpolated stacking
    V = [first(VdV_fast(ϕ0 + SA[real(w), imag(w)], P...)) for (ϕ0, w) in zip(wsf.ϕ0, D)]
    Vref = [first(VdV_fast(rigid_disregistry(moire, Hexagonal(), SA[x, y]) + SA[real(uref[i, j]), imag(uref[i, j])], P...))
            for (i, x) in enumerate(xs), (j, y) in enumerate(ys)]
    @test maximum(abs, grid_to_points(V, wsf, moire, xs, ys) - Vref) < 2e-3 * (maximum(Vref) - minimum(Vref))
    # a supercell grid: the same periodic field interpolated from the grid of a √3 × √3 supercell (the C3 amplitudes
    # expanded to the three modes of each star, then transferred to the supercell modes)
    wsS = Workspace(10, 1, 224; S = [1 -1; 1 2], flags)
    moireS = MoireGeometry(M, wsS)
    DS = to_grid!(wsS, transfer_modes(repeat(u, inner = (3, 1)), ws1, wsS), moireS)[:, :, 1]
    @test maximum(abs, grid_to_points(DS, wsS, moireS, xs, ys) - uref) < 2e-3 * maximum(abs, uref)
    @test_throws DimensionMismatch grid_to_points(D[1:10, :], wsf, moire, xs, ys)
    @test_throws DimensionMismatch scalar_to_grid!(wsf, u[1:5, 2], moire)
end

@testset "harmonic model" begin
    @test chain_laplacian([1.0, 2.0]) == [1 -1 0; -1 3 -2; 0 -2 2]
    @test chain_laplacian([1.0, 1.0], [0.5]; nnn_start = 1) == [1.5 -1 -0.5; -1 2 -1; -0.5 -1 1.5]
    @test_throws ArgumentError chain_laplacian([1.0, 1.0], [0.5])
    @test coupling_laplacian(2, [(1, 2, 3.0)]) == [3 -3; -3 3]
    # precond: S Sᵀ = (2A)⁻¹ for the harmonic Hessian A, and Sinv is its inverse
    ws = Workspace(4, 2; flags)
    moire = MoireGeometry(moire_matrix(3.0), ws)
    K, C = [1e-5 2e-5 3e-5 4e-5], chain_laplacian([100.0])
    S, Sinv = precond(K, C, moire.L, moire)
    for m in eachindex(moire.g), t in 1:2
        A = Diagonal(1 ./ (K[t:2:end] * moire.L^2)) + C * moire.invdotgg[m]
        @test S[:, :, m, t] * S[:, :, m, t]' ≈ inv(2A)
        @test Sinv[:, :, m, t] * S[:, :, m, t] ≈ I
    end
    # damping: P = (I + L²/|g|² diag(K) C)⁻¹
    Pd = damping!(zeros(2, 2, length(moire.g), 2), K, C, moire.L, moire)
    for m in eachindex(moire.g), t in 1:2
        @test Pd[:, :, m, t] ≈ inv(I + moire.L^2 * moire.invdotgg[m] * Diagonal(K[t:2:end]) * C)
    end
end

@testset "arguments" begin
    @test_throws ArgumentError Workspace(0, 1; flags)
    @test_throws ArgumentError WorkspaceC3(3, 0; flags)
    @test_throws ArgumentError WorkspaceC3(3, 1; S = [2 0; 0 2], flags)
    @test_throws ArgumentError Workspace(3, 1, 5; flags)             # grid too small for 3 shells
    ws = Workspace(4, 1; flags)
    moire = MoireGeometry(moire_matrix(2.0), ws)
    pot! = bilayer_potential(moire, ws)
    C = fill(stiffness(Vg, ws), 1, 1)
    @test_throws DimensionMismatch relax_moire(pot!, [1e-5 2e-5 3e-5], C, moire, ws; verbose = false)
    @test_throws DimensionMismatch relax_moire(pot!, K1, [1.0 0.0], moire, ws; verbose = false)
    @test_throws DimensionMismatch relax_moire(pot!, K1, C, moire, ws; u0 = zeros(ComplexF64, 3, 4), verbose = false)
    @test_throws ArgumentError relax_moire(pot!, K1, C, moire, ws; solver = :newton, verbose = false)
    @test_throws ArgumentError relax_moire((d, u) -> "oops", K1, C, moire, ws; verbose = false)
    @test_throws ArgumentError relax_moire((d, u) -> nothing, K1, C, moire, ws; solver = :diis, verbose = false)
end

@testset "utilities" begin
    moire = MoireGeometry(moire_matrix(2.0), Workspace(3, 1; flags))
    xs, ys = range(-0.5, 0.5, 7), range(-0.5, 0.5, 5)
    c = randn(ComplexF64, length(moire.g))
    f = field_synthesizer(moire, xs, ys)(c)
    @test size(f) == (7, 5)
    @test f[4, 3] ≈ 2real(sum(c))                                    # the origin
    @test f[2, 4] ≈ 2real(sum(c .* cis.(real.(moire.g) * xs[2] + imag.(moire.g) * ys[4])))
    @test length(coef_rows(Vg, " (m)")) == 5
    redirect_stdout(devnull) do
        print_params("parameters", Any[("θ", 1.0, "deg"), ("N", 3), coef_rows(Vg, "")...])
    end
end

@testset "code quality (Aqua)" begin
    Aqua.test_all(MoireRelax)
end

end
