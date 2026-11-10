# MoireRelax.jl: lattice relaxation in moiré materials with continuum elasticity theory
# Contents, conventions, usage and the derivations behind the code: docs/reference.md

module MoireRelax

using LinearAlgebra, StaticArrays, Printf, Polyester, FFTW, Optim
using NLSolversBase: OnceDifferentiable, only_fg!
using PrecompileTools: @compile_workload

export Optim # scripts can call Optim.converged(res), Optim.minimizer(res), …
export Lattice, Hexagonal, lattice_vectors, reciprocal_vectors, cell_area
export a1, a2, b1, b2, Acell, MoireGeometry, moire_matrix, twist_angle
export VdV_fast, hessian, curvature, minimum_stacking, stiffness
export extrapolate, print_params, coef_rows
export AbstractWorkspace, Workspace, WorkspaceC3, make_bvecs, multiplicity, grid_orbits, ω3
export to_grid!, from_grid!, stacking_energy!
export coupling_laplacian, chain_laplacian, coupling_modes, damping!, damped_step!, precond, transform!
export MixerDIIS, mix!, get_error
export relax_moire, elastic_energy, minimize_lbfgs, check_gradient
export supercell_wavevectors, transfer_modes, moire_periodic, supercell_fraction, field_synthesizer
export scalar_to_grid!, grid_to_points, cell_coordinates, rigid_disregistry

### Lattice

"""
    Lattice

The monolayer lattice: its lattice and reciprocal vectors (`lattice_vectors`, `reciprocal_vectors`), its cell area
(`cell_area`), the shells of its reciprocal vectors and its rotation symmetry. A workspace carries the lattice
(`ws.lattice`), and everything that depends on it asks the lattice; the rest of the package works per mode and per grid
point (reference, Conventions). `Hexagonal` is the lattice provided.
"""
abstract type Lattice end

"""
    Hexagonal()

The hexagonal (triangular) lattice of graphene, hBN and the transition-metal dichalcogenides, with zigzag along x:
``a_1 = (-1/2, \\sqrt3/2)``, ``a_2 = (-1, 0)`` in units of the lattice constant. The default lattice of the workspaces.
"""
struct Hexagonal <: Lattice end

"""
    lattice_vectors(lattice) -> (a1, a2)

The lattice vectors of `lattice` (units of the lattice constant), as `SVector{2}`.
"""
lattice_vectors(::Hexagonal) = (SA[-1/2, sqrt(3)/2], SA[-1, 0])

"""
    reciprocal_vectors(lattice) -> (b1, b2)

The reciprocal vectors of `lattice`, with ``b_i \\cdot a_j = 2\\pi \\delta_{ij}``.
"""
reciprocal_vectors(::Hexagonal) = (SA[0, 4π/sqrt(3)], SA[-2π, -2π/sqrt(3)])

"""
    cell_area(lattice)

The area of the cell of `lattice` (units of the lattice constant squared).
"""
cell_area(::Hexagonal) = sqrt(3) / 2

# the shell of the reciprocal vector n1 b1 + n2 b2: shell s of the hexagonal lattice has max(|n1|, |n2|, |n1 - n2|) = s
shell(::Hexagonal, n) = hexnorm(n)
hexnorm(n) = max(abs(n[1]), abs(n[2]), abs(n[1] - n[2]))

# the representative wavevectors (n1, n2) of N_shell shells, one of each ±g, ordered by shell (make_bvecs)
cell_wavevectors(::Hexagonal, N_shell) = make_bvecs(N_shell)

# the rotation by 2π/3 in the basis a1, a2 (R a1 = -a1 + a2, R a2 = -a1): the symmetry of WorkspaceC3
rotation(::Hexagonal) = SA[-1 -1; 1 0]

# the same rotation in the basis b1, b2: the inverse transpose, which is an integer matrix for a lattice symmetry
reciprocal_rotation(lat::Lattice) = round.(Int, inv(transpose(rotation(lat))))

"The lattice vectors of `Hexagonal()`: `a1` = (-1/2, √3/2), `a2` = (-1, 0), in units of the lattice constant."
const a1 = lattice_vectors(Hexagonal())[1]
@doc (@doc a1) const a2 = lattice_vectors(Hexagonal())[2]
"The reciprocal vectors of `Hexagonal()`: `b1` = (0, 4π/√3), `b2` = (-2π, -2π/√3), with ``b_i \\cdot a_j = 2\\pi \\delta_{ij}``."
const b1 = reciprocal_vectors(Hexagonal())[1]
@doc (@doc b1) const b2 = reciprocal_vectors(Hexagonal())[2]
"The area √3/2 of the cell of `Hexagonal()`, in units of the lattice constant squared."
const Acell = cell_area(Hexagonal())

"""
    AbstractWorkspace

The workspaces: `Workspace` (no symmetry, also for supercells) and `WorkspaceC3` (C3 symmetry), in Workspaces.jl.
Other packages can add their own.
"""
abstract type AbstractWorkspace end

### Moiré geometry

"""
    MoireGeometry(M, ws)
    MoireGeometry(M, bvecs; lattice = Hexagonal())

The moiré of the relative deformation `M` of two layers (`moire_matrix`), on the modes of the workspace `ws` or of the
reciprocal coordinates `bvecs` (e.g. those saved with a run): the moiré length `L`, the mean length of the moiré lattice
vectors ``M^{-1} a_1``, ``M^{-1} a_2`` (units a); the moiré reciprocal vectors `g` ``= L M^\\mathsf{T} b`` of the modes
as ``g_x + i g_y`` (units 1/L), with `invdotgg` ``= 1/|g|^2``; and `G`, the matrix whose columns are the moiré
reciprocal lattice vectors ``g_i = L M^\\mathsf{T} b_i`` (units 1/L), so that ``g_i \\cdot r = b_i \\cdot \\phi_0`` for a
point `r` (units L) with the rigid disregistry ``\\phi_0 = M r`` (`cell_coordinates`). With `WorkspaceC3`, `M` must be
a scaling times a rotation.
"""
struct MoireGeometry{T}
    L::T                    # mean length of the moiré lattice vectors M⁻¹a1, M⁻¹a2 (units a)
    g::Vector{Complex{T}}   # g_x + i g_y of the moiré reciprocal vectors (units 1/L)
    invdotgg::Vector{T}     # 1/|g|²
    G::SMatrix{2,2,T,4}     # columns: the moiré reciprocal lattice vectors g_i = L Mᵀ bᵢ (units 1/L)
end

function MoireGeometry(M::AbstractMatrix, ws::AbstractWorkspace)
    ws isa WorkspaceC3 && !(M[1, 1] ≈ M[2, 2] && M[1, 2] ≈ -M[2, 1]) &&
        throw(ArgumentError("C3 symmetry requires M = scaling × rotation (twist and isotropic mismatch)"))
    (a1, a2), (b1, b2) = lattice_vectors(ws.lattice), reciprocal_vectors(ws.lattice)
    L = (norm(M \ a1) + norm(M \ a2)) / 2
    G = L * SMatrix{2,2}(transpose(M)) * [b1 b2]
    bx = ws.bvecs[1, :] * b1[1] + ws.bvecs[2, :] * b2[1]
    by = ws.bvecs[1, :] * b1[2] + ws.bvecs[2, :] * b2[2]
    g = @. L * complex(M[1, 1] * bx + M[2, 1] * by, M[1, 2] * bx + M[2, 2] * by) # L Mᵀb
    return MoireGeometry(L, g, (@. 1 / abs2(g)), G)
end

"""
    cell_coordinates(moire, r) -> s

The coordinates ``s = (s_1, s_2)`` in the moiré cell of the point `r` (units L): the rigid disregistry there is
``\\phi_0 = s_1 a_1 + s_2 a_2``, and the grid of a workspace holds the points ``s_i = (k - 1)/N_i``, ``k = 1, \\dots,
N_i`` (for a supercell, the grid coordinates are ``t = S^{-1} s``). Integer shifts of `s` are the same point of the
cell.
"""
cell_coordinates(moire::MoireGeometry, r) = transpose(moire.G) * SVector{2}(r) / 2π

"""
    rigid_disregistry(moire, lattice, r)

The rigid disregistry ``\\phi_0 = M r = s_1 a_1 + s_2 a_2`` (units a) at the point `r` (units L) of the moiré, for the
lattice vectors of `lattice` (`cell_coordinates`).
"""
function rigid_disregistry(moire::MoireGeometry, lat::Lattice, r)
    s, (a1, a2) = cell_coordinates(moire, r), lattice_vectors(lat)
    return s[1] * a1 + s[2] * a2
end

MoireGeometry(M::AbstractMatrix, bvecs::AbstractMatrix{<:Real}; lattice::Lattice = Hexagonal()) =
    MoireGeometry(M, ModeSet(Float64.(bvecs), lattice))
struct ModeSet{LAT<:Lattice} <: AbstractWorkspace  # the modes of a saved run, for MoireGeometry
    bvecs::Matrix{Float64}
    lattice::LAT
end

# rotation by θ (degrees)
rot(θ) = [cosd(θ) -sind(θ); sind(θ) cosd(θ)]

"""
    moire_matrix(θ, ϵ0 = 0.0)

The relative deformation ``M = (1 + \\epsilon_0/2) R(\\theta/2) - (1 - \\epsilon_0/2) R(-\\theta/2)`` of two layers
twisted by `θ` (degrees) with the isotropic lattice mismatch `ϵ0`; the rigid disregistry is ``\\phi_0 = M r``.
"""
moire_matrix(θ, ϵ0 = 0.0) = (1 + ϵ0 / 2) * rot(θ / 2) - (1 - ϵ0 / 2) * rot(-θ / 2)

"""
    twist_angle(M)

The twist angle (degrees) of the relative deformation `M`, from its antisymmetric part.
"""
twist_angle(M) = 2asind((M[2, 1] - M[1, 2]) / 4)

### Stacking potential

# the sines and cosines of the first three stars at ϕ, with x1 = b1⋅ϕ, x2 = b2⋅ϕ (shared by all series at the same ϕ)
@inline function harmonics(ϕ)
    sa, ca = sincos(dot(b1, ϕ))                             # sincos(x1)
    sb, cb = sincos(dot(b2, ϕ))                             # sincos(x2)
    sab, cab = sa * cb + ca * sb, ca * cb - sa * sb         # sincos(x1 + x2)
    s2a, c2a = 2sa * ca, ca^2 - sa^2                        # sincos(2x1)
    s2b, c2b = 2sb * cb, cb^2 - sb^2                        # sincos(2x2)
    s2ab, c2ab = 2sab * cab, cab^2 - sab^2                  # sincos(2x1 + 2x2)
    sa2b, ca2b = sa * c2b + ca * s2b, ca * c2b - sa * s2b   # sincos(x1 + 2x2)
    sb2a, cb2a = sb * c2a + cb * s2a, cb * c2a - sb * s2a   # sincos(x2 + 2x1)
    samb, camb = sa * cb - ca * sb, ca * cb + sa * sb       # sincos(x1 - x2)
    return sa, ca, sb, cb, sab, cab, s2a, c2a, s2b, c2b, s2ab, c2ab, sa2b, ca2b, sb2a, cb2a, samb, camb
end

# the D3 series V from the harmonics H, with c1 = Re(V1), c2 = V2, c3 = Re(V3), p1 = Im(V1), p3 = Im(V3)
@inline function series_value(H, c1, c2, c3, p1, p3)
    sa, ca, sb, cb, sab, cab, s2a, c2a, s2b, c2b, s2ab, c2ab, sa2b, ca2b, sb2a, cb2a, samb, camb = H
    return 2c1 * (ca + cb + cab) - 2p1 * (sa + sb - sab) + 2c2 * (ca2b + cb2a + camb) +
           2c3 * (c2a + c2b + c2ab) - 2p3 * (s2a + s2b - s2ab)
end

# V and ∂V/∂ϕ = A b1 + B b2 (the terms along b1 + b2, b1 + 2b2, b2 + 2b1 and b1 - b2 shared out between b1 and b2)
@inline function series(H, c1, c2, c3, p1, p3)
    sa, ca, sb, cb, sab, cab, s2a, c2a, s2b, c2b, s2ab, c2ab, sa2b, ca2b, sb2a, cb2a, samb, camb = H
    A = -2c1 * (sa + sab) - 2p1 * (ca - cab) - 2c2 * (sa2b + 2sb2a + samb) - 4c3 * (s2a + s2ab) - 4p3 * (c2a - c2ab)
    B = -2c1 * (sb + sab) - 2p1 * (cb - cab) - 2c2 * (2sa2b + sb2a - samb) - 4c3 * (s2b + s2ab) - 4p3 * (c2b - c2ab)
    return series_value(H, c1, c2, c3, p1, p3), A * b1 + B * b2
end

# the Hessian ∇∇V of the D3 series: -Σ_G 2 Re(V_G e^{iG⋅ϕ}) G Gᵀ over the stars {b1, b2, -b1-b2} (V1 = c1 + i p1),
# {b1+2b2, -2b1-b2, b1-b2} (V2 = c2) and {2b1, 2b2, -2b1-2b2} (V3 = c3 + i p3)
@inline function series_hessian(H, c1, c2, c3, p1, p3)
    sa, ca, sb, cb, sab, cab, s2a, c2a, s2b, c2b, s2ab, c2ab, sa2b, ca2b, sb2a, cb2a, samb, camb = H
    B1, B2, B12 = b1 * b1', b2 * b2', (b1 + b2) * (b1 + b2)'
    B1p2, B2p1, B1m2 = (b1 + 2b2) * (b1 + 2b2)', (2b1 + b2) * (2b1 + b2)', (b1 - b2) * (b1 - b2)'
    return -2 * ((c1 * ca - p1 * sa) * B1 + (c1 * cb - p1 * sb) * B2 + (c1 * cab + p1 * sab) * B12 +
                 c2 * (ca2b * B1p2 + cb2a * B2p1 + camb * B1m2) +
                 4 * ((c3 * c2a - p3 * s2a) * B1 + (c3 * c2b - p3 * s2b) * B2 + (c3 * c2ab + p3 * s2ab) * B12))
end

"""
    VdV_fast(ϕ, c1, c2, c3, p1, p3) -> (V, ∇V)

The stacking potential in the D3 form at the disregistry `ϕ` (units a), and its gradient (an `SVector{2}`): the first
three stars of the hexagonal reciprocal lattice with the coefficients `Vg = [c1, c2, c3, p1, p3]`, ``c_1 = \\mathrm{Re}
V_1``, ``c_2 = V_2``, ``c_3 = \\mathrm{Re} V_3``, ``p_1 = \\mathrm{Im} V_1``, ``p_3 = \\mathrm{Im} V_3`` (reference,
Geometry and stacking potential). Pass the coefficients as a static vector `P` (`VdV_fast(ϕ, P...)`), so that the call
is type stable.
"""
@inline VdV_fast(ϕ::AbstractVector{<:Real}, c1::Real, c2::Real, c3::Real, p1::Real, p3::Real) =
    series(harmonics(ϕ), c1, c2, c3, p1, p3)

# a stacking potential as a function ϕ -> (V, ∇V): the coefficients Vg of the D3 form, or any such callable
as_potential(Vg::AbstractVector) = (P = SVector{5,Float64}(Vg); ϕ -> VdV_fast(ϕ, P...))
as_potential(V) = V

"""
    hessian(Vg, ϕ)
    hessian(V, ϕ; h = 1e-4)

The Hessian ``\\nabla\\nabla V`` of the stacking potential at `ϕ`: analytic for the coefficients `Vg` of the D3 form,
and from central differences of the gradient (step `h`) for any callable `V(ϕ) -> (V, ∇V)`.
"""
hessian(Vg::AbstractVector, ϕ) = series_hessian(harmonics(ϕ), SVector{5,Float64}(Vg)...)
function hessian(V, ϕ; h = 1e-4)
    dV(ϕ) = last(V(ϕ))
    return hcat(dV(ϕ + SA[h, 0]) - dV(ϕ - SA[h, 0]), dV(ϕ + SA[0, h]) - dV(ϕ - SA[0, h])) / 2h
end

"""
    curvature(V, ϕ)

The interface stiffness ``k = \\mathrm{tr}(\\nabla\\nabla V)/2`` (meV/cell/a²) of the stacking potential `V` (the
coefficients `Vg` of the D3 form, or a callable `V(ϕ) -> (V, ∇V)`) at `ϕ`, the mean curvature of `V` there; warns if `V`
is not convex at `ϕ`.
"""
function curvature(V, ϕ)
    H = hessian(V, ϕ)
    minimum(eigvals(Symmetric((H + H') / 2))) > 0 || @warn "stacking potential is not convex at ϕ = $ϕ"
    return tr(H) / 2
end

"""
    minimum_stacking(V, ws; tol = 1e-12)
    minimum_stacking(V, ϕ0 = cell_grid(60); tol = 1e-12)

The lowest stacking of the potential `V` (the coefficients `Vg` of the D3 form, or a callable `V(ϕ) -> (V, ∇V)`): the
lowest of the points `ϕ0` (the grid of the workspace `ws`, or by default a 60 × 60 grid of the hexagonal cell), refined
by Newton steps until they are shorter than `tol`; warns if they do not converge.
"""
minimum_stacking(V, ws::AbstractWorkspace; kw...) = minimum_stacking(V, ws.ϕ0; kw...)
function minimum_stacking(V, ϕ0::AbstractArray = cell_grid(60); tol = 1e-12)
    f = as_potential(V)
    values = [first(f(ϕ)) for ϕ in ϕ0]
    ϕ = ϕ0[argmin(values)]
    for _ in 1:20
        step = hessian(V, ϕ) \ last(f(ϕ))
        ϕ -= step
        norm(step) < tol && break
    end
    norm(last(f(ϕ))) <= 1e-8 * (1 + maximum(abs, values)) || @warn "minimum_stacking: Newton did not converge (ϕ = $ϕ)"
    return SVector{2,Float64}(ϕ)
end

"""
    stiffness(V, ws = cell_grid(60))

The interface stiffness `curvature(V, ϕ)` at the lowest stacking `minimum_stacking(V, ws)` of the potential `V`.
"""
stiffness(V, ws = cell_grid(60)) = curvature(V, minimum_stacking(V, ws))

# the stackings s1 a1 + s2 a2 of an N × N grid of the hexagonal cell
cell_grid(N::Integer) = cell_disregistry(Hexagonal(), [1 0; 0 1], (N, N))

include(joinpath(@__DIR__, "Supercells.jl"))    # supercells: modes, grid, disregistry; transfer_modes
include(joinpath(@__DIR__, "Workspaces.jl"))    # Workspace, WorkspaceC3, to_grid!, from_grid!, stacking_energy!
include(joinpath(@__DIR__, "Harmonic.jl"))      # harmonic model: coupling matrices, DIIS damping, LBFGS precond
include(joinpath(@__DIR__, "DIIS.jl"))          # MixerDIIS, mix!, get_error
include(joinpath(@__DIR__, "Relax.jl"))         # relax_moire, minimize_lbfgs, check_gradient

### Utilities

"""
    extrapolate(x, xn, un)

The Lagrange extrapolation to `x` of the solutions `un` at the points `xn` (e.g. the last solutions of a sweep over the
angle, as the start of the next); `ComplexF64[]` (start from zero) without points.
"""
extrapolate(x, xn, un) = isempty(un) ? ComplexF64[] :
    sum(prod((x - xn[j]) / (xn[i] - xn[j]) for j in eachindex(xn) if j != i; init = 1.0) * un[i] for i in eachindex(xn))

"""
    print_params(title, rows)

Prints a box of parameters: each row is `(name, value)` or `(name, value, unit)`.
"""
function print_params(title, rows)
    println("|=================================================|")
    @printf("|%s|\n", rpad(lpad(title, (49 + length(title)) ÷ 2, '='), 49, '='))
    println("|=================================================|")
    for row in rows
        @printf("| %-9s:%24s %-12s |\n", row[1], row[2], length(row) > 2 ? row[3] : "")
    end
    println("|=================================================|")
end

"""
    coef_rows(Vg, tag, unit = "meV/cell")

The rows of `print_params` for the coefficients `[c1, c2, c3, p1, p3]` of a stacking potential, named `c1tag`, ….
"""
coef_rows(Vg, tag, unit = "meV/cell") = [(name * tag, v, unit) for (name, v) in zip(("c1", "c2", "c3", "p1", "p3"), Vg)]

"""
    field_synthesizer(moire, xs, ys)

The function ``c \\mapsto 2\\,\\mathrm{Re} \\sum_g c_g e^{i g \\cdot r}`` of a real field from its coefficients `c` on
the modes of `moire` (one per pair ±g), on the grid `xs` × `ys` (units L), for plots.
"""
function field_synthesizer(moire::MoireGeometry, xs, ys)
    Ex, Ey = cis.(real.(moire.g) .* xs'), cis.(imag.(moire.g) .* ys')         # e^{i gx x}, e^{i gy y}: modes × grid
    return c -> 2real.(transpose(Ex) * (c .* Ey))
end

include(joinpath(@__DIR__, "Precompile.jl"))    # workload run while the package is precompiled

end # module
