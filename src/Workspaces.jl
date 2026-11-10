# Workspaces without symmetry (Workspace, also for supercells) and with C3 symmetry (WorkspaceC3): the modes, the
# real-space grid of the cell, the FFTs between them and the stacking energy on the grid (C3, supercells: reference)

struct Workspace{LAT<:Lattice,T,B,NI,PB,PF} <: AbstractWorkspace
    lattice::LAT                    # the monolayer lattice
    nf::Int                         # number of displacement fields on the grid
    S::Matrix{Int}                  # supercell vectors (columns) in units of the moiré lattice vectors; I: moiré cell
    bvecs::Matrix{B}                # moiré reciprocal coordinates (n1, n2) of the representative wavevectors
    N::NI                           # grid points per direction (a supercell: (N1, N2))
    kp::Vector{CartesianIndex{2}}   # array position of +g_m
    km::Vector{CartesianIndex{2}}   # array position of -g_m
    ϕ0::Matrix{SVector{2,T}}        # rigid disregistry s1 a1 + s2 a2 on the grid
    W::Array{Complex{T},3}          # u_x + i u_y of each field at ±g; zero elsewhere
    D::Array{Complex{T},3}          # u_x + i u_y of each field on the grid, then the forces, then their FFT
    pB::PB                          # out-of-place backward FFT: W → D
    pF::PF                          # in-place forward FFT of D
end

# with C3 symmetry: one mode per star {g, Rg, R²g} and one grid point per C3 orbit (reference)
struct WorkspaceC3{LAT<:Lattice,T,PB,PF} <: AbstractWorkspace
    lattice::LAT                    # the monolayer lattice
    nf::Int                         # number of displacement fields on the grid
    bvecs::Matrix{Int}              # one representative (n1, n2) per star {g, Rg, R²g}
    N::Int                          # grid points per direction
    kp::Matrix{CartesianIndex{2}}   # array position of +R^(r-1) g_m: kp[r, m]
    km::Matrix{CartesianIndex{2}}   # array position of -R^(r-1) g_m: km[r, m]
    ϕ0::Matrix{SVector{2,T}}        # rigid disregistry s1 a1 + s2 a2 on the grid
    W::Array{Complex{T},3}          # u_x + i u_y of each field at ±R^(r-1) g; zero elsewhere
    D::Array{Complex{T},3}          # u_x + i u_y of each field on the grid, then the forces, then their FFT
    pB::PB                          # out-of-place backward FFT: W → D
    pF::PF                          # in-place forward FFT of D
    orb::Matrix{Int}                # C3 orbits of the grid (grid_orbits)
end

"The phases ``(1, \\omega, \\omega^2)``, ``\\omega = e^{2\\pi i/3}``, of ``u_x + i u_y`` at ``R^{r-1} g``, r = 1, 2, 3, of a C3-symmetric field."
const ω3 = (complex(1.0), cis(2π / 3), cis(4π / 3))

"""
    Workspace(N_shell, nf, N = 0; lattice = Hexagonal(), S = nothing, pad = 1, flags = FFTW.MEASURE, fft_threads = 1)

The modes of `N_shell` shells of the lattice, one of each pair ±g, and an N × N real-space grid of the moiré cell for
`nf` displacement fields, with the FFT plans between them (`to_grid!`, `from_grid!`); no symmetry is assumed. Without
`N`, the grid is the smallest FFT-friendly size ≥ pad (2N_shell + 1); pad = 1.5 to 2 reduces aliasing at small angles.
With a supercell `S` (columns: the supercell vectors in units of the moiré lattice vectors), the modes within `N_shell`
shells of the moiré reciprocal lattice, by shell, and an N1 × N2 grid of the supercell (reference, Supercells).
`flags` and `fft_threads` go to the FFT plans. The fields `lattice`, `nf`, `S`, `bvecs` (the reciprocal coordinates
of the modes, one per column), `N` and `ϕ0` (the rigid disregistry on the grid) are part of the interface; the FFT
buffers `W` and `D` are owned by the workspace, so use one workspace per concurrent relaxation.
"""
function Workspace(N_shell::Integer, nf::Integer, N::Integer = 0; lattice::Lattice = Hexagonal(), S = nothing, kw...)
    Smat, bvecs, Ns, kp, km, ϕ0, W, D, pB, pF = workspace_fields(1, lattice, N_shell, nf, N, S; kw...)
    return Workspace(lattice, nf, Smat, bvecs, S === nothing ? Ns[1] : Ns, vec(kp), vec(km), ϕ0, W, D, pB, pF)
end

"""
    WorkspaceC3(N_shell, nf, N = 0; lattice = Hexagonal(), pad = 1, flags = FFTW.MEASURE, fft_threads = 1)

As `Workspace`, for fields with C3 symmetry, u(Rr) = R u(r): one mode per star {g, Rg, R²g} (the first of each star of
`Workspace`, `bvecs[:, 1:3:end]`), and the stacking energy evaluated on one point of each C3 orbit of the grid (`orb`,
from `grid_orbits`). About three times fewer modes and grid points; for the moiré cell only (reference, C3 symmetry).
"""
function WorkspaceC3(N_shell::Integer, nf::Integer, N::Integer = 0; lattice::Lattice = Hexagonal(), S = nothing, kw...)
    _, bvecs, Ns, kp, km, ϕ0, W, D, pB, pF = workspace_fields(3, lattice, N_shell, nf, N, S; kw...)
    return WorkspaceC3(lattice, nf, bvecs, Ns[1], kp, km, ϕ0, W, D, pB, pF, grid_orbits(lattice, Ns[1]))
end

# fields of both workspaces: the modes and grid of the cell (Supercells.jl), one mode in nrot (1 or 3) of
# cell_wavevectors, with the array positions kp[r, m], km[r, m] of ±R^(r-1) g_m for r = 1, …, nrot
function workspace_fields(nrot, lat, N_shell, nf, N, S; pad::Real = 1, flags = FFTW.MEASURE, fft_threads::Integer = 1)
    N_shell >= 1 || throw(ArgumentError("N_shell must be at least 1, got $N_shell"))
    nf >= 1 || throw(ArgumentError("nf, the number of fields, must be at least 1, got $nf"))
    S === nothing || nrot == 1 ||
        throw(ArgumentError("WorkspaceC3 is for the moiré cell only: use Workspace(N_shell, nf; S) for a supercell"))
    Smat = S === nothing ? [1 0; 0 1] : Matrix{Int}(S)
    m, bvecs, Ns = cell_modes(lat, N_shell, S, N, pad)
    m, bvecs = m[:, 1:nrot:end], bvecs[:, 1:nrot:end]
    R = reciprocal_rotation(lat)    # the rotation by 2π/3 of the wavevectors (n1, n2)
    kp = Matrix{CartesianIndex{2}}(undef, nrot, size(m, 2))
    km = similar(kp)
    for k in axes(m, 2)
        n = SA[m[1, k], m[2, k]]
        for r in 1:nrot
            kp[r, k], km[r, k] = grid_index(n[1], n[2], Ns), grid_index(-n[1], -n[2], Ns)
            n = R * n
        end
    end
    allunique([vec(kp); vec(km)]) ||
        throw(ArgumentError("grid $Ns too small for the modes: N must exceed twice the largest grid index of a mode"))

    W, D = zeros(ComplexF64, Ns..., nf), zeros(ComplexF64, Ns..., nf)
    pB = plan_bfft(W, (1, 2); flags, num_threads = fft_threads)
    pF = plan_fft!(D, (1, 2); flags, num_threads = fft_threads)
    fill!(W, 0) # planning may overwrite W
    return Smat, bvecs, Ns, kp, km, cell_disregistry(lat, Smat, Ns), W, D, pB, pF
end

"""
    make_bvecs(N_shell)

The representative reciprocal vectors (n1, n2) of `N_shell` hexagonal shells, one of each pair ±g, as the columns of a
2 × 3Σs matrix: for each (n1, n2) with 0 ≤ n2 < n1 ≤ N_shell, its rotations by 0, 2π/3 and 4π/3 in turn, so that the
shells (max(|n1|, |n2|, |n1 - n2|) = s) come in increasing order and every third column starts a star.
"""
function make_bvecs(N_shell::Integer)
    bvecs = Int[]
    for n1 in 1:N_shell, n2 in 0:n1-1
        append!(bvecs, n1, n2, -n2, n1 - n2, n2 - n1, -n1)
    end
    return reshape(bvecs, 2, :)
end

"""
    multiplicity(ws)

The number of modes that share each stored amplitude: 1 for `Workspace`, 3 for `WorkspaceC3`.
"""
multiplicity(::Workspace) = 1
multiplicity(::WorkspaceC3) = 3

# array positions of +R^(r-1) g_m and -R^(r-1) g_m (Workspace: r = 1 only)
kpos(ws::Workspace, r, m) = ws.kp[m], ws.km[m]
kpos(ws::WorkspaceC3, r, m) = ws.kp[r, m], ws.km[r, m]

"""
    grid_orbits(lattice, N)

The C3 orbits of the N × N grid of the cell of `lattice`: `orb[r, k]` is the linear index of ``R^{r-1} p_k``, with
``p_k`` the first point of orbit k and R the rotation by 2π/3 (`rotation(lattice)`, in the basis a1, a2). A point that R
leaves fixed is an orbit of one point, whose index appears three times in its column (reference, C3 symmetry).
"""
function grid_orbits(lat::Lattice, N::Integer)
    R = rotation(lat)   # the rotation by 2π/3 of the grid points (i1, i2)
    seen = falses(N, N)
    orb = Int[]
    for j2 in 0:N-1, j1 in 0:N-1
        seen[j1+1, j2+1] && continue
        i = SA[j1, j2]
        for _ in 1:3
            seen[i[1]+1, i[2]+1] = true
            push!(orb, i[1] + 1 + N * i[2])
            i = mod.(R * i, N)
        end
    end
    return reshape(orb, 3, :)
end

"""
    to_grid!(ws, u, moire) -> ws.D

The displacement fields on the grid of `ws`, ``u_x + i u_y`` (units a) of each field in `ws.D` (N × N × nf), from the
mode amplitudes `u` (columns u_para, u_perp of each field; reference, Conventions), by one FFT. Overwrites `ws.W` and
`ws.D`.
"""
function to_grid!(ws::AbstractWorkspace, u::AbstractMatrix, moire::MoireGeometry)
    (; W, D, pB, nf) = ws
    (; g, invdotgg) = moire
    for f in 1:nf, m in eachindex(g)
        cu = -im * g[m] * invdotgg[m] # at +g: u_x + i u_y = cu (u_para + i u_perp)
        cp, cm = cu * (u[m, 2f - 1] + im * u[m, 2f]), -cu * (conj(u[m, 2f - 1]) + im * conj(u[m, 2f]))
        for r in 1:multiplicity(ws)
            kp, km = kpos(ws, r, m)
            W[kp, f], W[km, f] = ω3[r] * cp, ω3[r] * cm
        end
    end
    return mul!(D, pB, W)
end

"""
    from_grid!(dVdu, ws, moire) -> dVdu

The gradient ``\\partial\\langle V\\rangle/\\partial(\\mathrm{Re}\\,u) + i\\,\\partial\\langle V\\rangle/\\partial(\\mathrm{Im}\\,u)``
per mode (C3: per star, the gradient of one of its modes) from the forces ``\\partial V/\\partial u_x + i\\,\\partial
V/\\partial u_y`` on each field in `ws.D`, by one FFT in place. The adjoint of `to_grid!`.
"""
function from_grid!(dVdu::AbstractMatrix, ws::AbstractWorkspace, moire::MoireGeometry)
    (; D, pF, nf) = ws
    (; g, invdotgg) = moire
    N2 = length(ws.ϕ0) # number of grid points
    pF * D
    for f in 1:nf, m in eachindex(g)
        kp, km = kpos(ws, 1, m)
        Fp, Fm = conj(g[m]) * D[kp, f], g[m] * conj(D[km, f])
        dVdu[m, 2f - 1] = im * (Fp + Fm) * invdotgg[m] / N2
        dVdu[m, 2f] = (Fp - Fm) * invdotgg[m] / N2
    end
    return dVdu
end

### Stacking energy on the grid

"""
    stacking_energy!(f, dVdu, u, moire, ws) -> ⟨V⟩

Stacking energy ⟨V⟩ on the grid and, unless `dVdu === nothing`, its gradient `dVdu` = ∂⟨V⟩/∂(Re u) + i ∂⟨V⟩/∂(Im u)
per mode, from the local energy `f(ϕ0, u_1, …, u_nf) -> (V, ∂V/∂u_1, …, ∂V/∂u_nf)` at each grid point (2-vectors:
the rigid disregistry ϕ0 and the fields u_f). The grid points are shared out between the threads, so `f` must not write
to shared state. WorkspaceC3 evaluates f on one point per C3 orbit (reference).
"""
function stacking_energy!(f::F, dVdu, u, moire, ws::AbstractWorkspace) where {F}
    D = to_grid!(ws, u, moire)
    E = _stacking_energy!(f, D, ws, Val(ws.nf))   # Val: the number of fields is known when compiling the loop
    dVdu === nothing || from_grid!(dVdu, ws, moire)
    return E / length(ws.ϕ0)
end

# every grid point p (linear index), with the field i at D[p + (i - 1) N²]
function _stacking_energy!(f::F, D, ws::Workspace, nf::Val) where {F}
    N2, ϕ0 = length(ws.ϕ0), ws.ϕ0
    E = 0.0
    @batch reduction = ((+, E),) for p in 1:N2
        E += point_energy!(f, D, ϕ0, p, N2, nf)
    end
    return E
end

# one point of each C3 orbit, and the forces on the other points as rotated copies
function _stacking_energy!(f::F, D, ws::WorkspaceC3, nf::Val) where {F}
    N2, ϕ0, orb = length(ws.ϕ0), ws.ϕ0, ws.orb
    E = 0.0
    @batch reduction = ((+, E),) for k in axes(orb, 2)
        E += orbit_energy!(f, D, ϕ0, orb, k, N2, nf)
    end
    return E
end

# one field: the two loops above with their bodies written out, 5 to 15% faster with threads than with point_energy!
# and orbit_energy!, which are faster for two fields (tests with two threads)
function _stacking_energy!(f::F, D, ws::Workspace, nf::Val{1}) where {F}
    N2, ϕ0 = length(ws.ϕ0), ws.ϕ0
    E = 0.0
    @batch reduction = ((+, E),) for p in 1:N2
        v, z = local_energy(f, ϕ0[p], D, p, N2, nf)
        D[p] = z[1]
        E += v
    end
    return E
end

function _stacking_energy!(f::F, D, ws::WorkspaceC3, nf::Val{1}) where {F}
    N2, ϕ0, orb = length(ws.ϕ0), ws.ϕ0, ws.orb
    E = 0.0
    @batch reduction = ((+, E),) for k in axes(orb, 2)
        p = orb[1, k]
        v, z = local_energy(f, ϕ0[p], D, p, N2, nf)
        for r in 3:-1:1                                 # r = 1 last, for a C3-invariant point
            D[orb[r, k]] = ω3[r] * z[1]
        end
        E += (orb[2, k] == p ? 1 : 3) * v               # a C3-invariant point is an orbit of its own
    end
    return E
end

# V at the grid point p, with its forces written back at p
@inline function point_energy!(f::F, D, ϕ0, p, N2, nf) where {F}
    v, z = local_energy(f, ϕ0[p], D, p, N2, nf)
    for i in eachindex(z)
        D[p + (i - 1) * N2] = z[i]
    end
    return v
end

# V of the C3 orbit k, from its first point p, with the rotated forces written on all its points
@inline function orbit_energy!(f::F, D, ϕ0, orb, k, N2, nf) where {F}
    p = orb[1, k]
    v, z = local_energy(f, ϕ0[p], D, p, N2, nf)
    for r in 3:-1:1, i in eachindex(z)                  # r = 1 last, for a C3-invariant point
        D[orb[r, k] + (i - 1) * N2] = ω3[r] * z[i]
    end
    return (orb[2, k] == p ? 1 : 3) * v                 # a C3-invariant point is an orbit of its own
end

# f at the grid point p, with the fields u_f as 2-vectors -> (V, the forces ∂V/∂u_f as complex numbers). @inbounds
# (p ≤ N², i ≤ nf): with the bounds check, the threaded loops are slower
@inline function local_energy(f::F, ϕ0, D, p, N2, nf::Val) where {F}
    u = ntuple(i -> (w = @inbounds D[p + (i - 1) * N2]; SA[real(w), imag(w)]), nf)
    VdV = @inline f(ϕ0, u...)   # @inline: as fast as a hand-written loop (f is too large to be inlined by default)
    return first(VdV), map(d -> complex(d[1], d[2]), Base.tail(VdV))
end
