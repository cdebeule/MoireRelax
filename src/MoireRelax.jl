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

### Lattice

# The monolayer lattice: its lattice and reciprocal vectors, its cell area, the shells of its reciprocal vectors and
# its rotation symmetry. A workspace carries the lattice, and everything that depends on it asks the lattice (README,
# Conventions); the rest of the package works per mode and per grid point.
abstract type Lattice end

# the hexagonal (triangular) lattice of graphene, hBN, …, with zigzag along x
struct Hexagonal <: Lattice end

lattice_vectors(::Hexagonal) = (SA[-1/2, sqrt(3)/2], SA[-1, 0])             # a1, a2 (units a)
reciprocal_vectors(::Hexagonal) = (SA[0, 4π/sqrt(3)], SA[-2π, -2π/sqrt(3)])   # b1, b2, with bᵢ⋅aⱼ = 2π δᵢⱼ
cell_area(::Hexagonal) = sqrt(3) / 2                                        # units a²

# the shell of the reciprocal vector n1 b1 + n2 b2: shell s of the hexagonal lattice has max(|n1|, |n2|, |n1 - n2|) = s
shell(::Hexagonal, n) = hexnorm(n)
hexnorm(n) = max(abs(n[1]), abs(n[2]), abs(n[1] - n[2]))

# the representative wavevectors (n1, n2) of N_shell shells, one of each ±g, ordered by shell (make_bvecs)
cell_wavevectors(::Hexagonal, N_shell) = make_bvecs(N_shell)

# the rotation by 2π/3 in the basis a1, a2 (R a1 = -a1 + a2, R a2 = -a1): the symmetry of WorkspaceC3
rotation(::Hexagonal) = SA[-1 -1; 1 0]

# the same rotation in the basis b1, b2: the inverse transpose, which is an integer matrix for a lattice symmetry
reciprocal_rotation(lat::Lattice) = round.(Int, inv(transpose(rotation(lat))))

const a1, a2 = lattice_vectors(Hexagonal())     # the hexagonal lattice, for the D3 stacking potential below
const b1, b2 = reciprocal_vectors(Hexagonal())
const Acell = cell_area(Hexagonal())

# workspaces without symmetry (Workspace, also for supercells) and with C3 symmetry (WorkspaceC3), in Workspaces.jl;
# other packages can add their own
abstract type AbstractWorkspace end

### Moiré geometry

struct MoireGeometry{T}
    L::T                    # mean length of the moiré lattice vectors M⁻¹a1, M⁻¹a2 (units a)
    g::Vector{Complex{T}}   # g_x + i g_y of the moiré reciprocal vectors (units 1/L)
    invdotgg::Vector{T}     # 1/|g|²
end

function MoireGeometry(M::AbstractMatrix, ws::AbstractWorkspace)
    ws isa WorkspaceC3 && !(M[1, 1] ≈ M[2, 2] && M[1, 2] ≈ -M[2, 1]) &&
        error("C3 symmetry requires M = scaling × rotation (twist and isotropic mismatch)")
    (a1, a2), (b1, b2) = lattice_vectors(ws.lattice), reciprocal_vectors(ws.lattice)
    L = (norm(M \ a1) + norm(M \ a2)) / 2
    bx = ws.bvecs[1, :] * b1[1] + ws.bvecs[2, :] * b2[1]
    by = ws.bvecs[1, :] * b1[2] + ws.bvecs[2, :] * b2[2]
    g = @. L * complex(M[1, 1] * bx + M[2, 1] * by, M[1, 2] * bx + M[2, 2] * by) # L Mᵀb
    return MoireGeometry(L, g, @. 1 / abs2(g))
end

# the geometry of modes given by their moiré reciprocal coordinates, e.g. the bvecs saved with a run
MoireGeometry(M::AbstractMatrix, bvecs::AbstractMatrix{<:Real}; lattice::Lattice = Hexagonal()) =
    MoireGeometry(M, ModeSet(Float64.(bvecs), lattice))
struct ModeSet{LAT<:Lattice} <: AbstractWorkspace
    bvecs::Matrix{Float64}
    lattice::LAT
end

# rotation by θ (degrees)
rot(θ) = [cosd(θ) -sind(θ); sind(θ) cosd(θ)]

# relative deformation M of two layers twisted by θ (degrees) with the isotropic lattice mismatch ϵ0
moire_matrix(θ, ϵ0 = 0.0) = (1 + ϵ0 / 2) * rot(θ / 2) - (1 - ϵ0 / 2) * rot(-θ / 2)

# twist angle θ (degrees) from the antisymmetric part of M
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

# stacking potential (D3) V and its gradient dV at ϕ
@inline VdV_fast(ϕ::AbstractVector{<:Real}, c1::Real, c2::Real, c3::Real, p1::Real, p3::Real) =
    series(harmonics(ϕ), c1, c2, c3, p1, p3)

# Hessian of V at ϕ, from central differences of the gradient
function hessian(Vg, ϕ; h = 1e-4)
    dV(ϕ) = last(VdV_fast(ϕ, Vg...))
    return hcat(dV(ϕ + SA[h, 0]) - dV(ϕ - SA[h, 0]), dV(ϕ + SA[0, h]) - dV(ϕ - SA[0, h])) / 2h
end

# interface stiffness k = tr(∇∇V)/2 (meV/cell/a²), the mean curvature of V at ϕ
function curvature(Vg, ϕ)
    H = hessian(Vg, ϕ)
    minimum(eigvals(Symmetric((H + H') / 2))) > 0 || @warn "stacking potential is not convex at ϕ = $ϕ"
    return tr(H) / 2
end

# lowest stacking of V: the lowest of the points ϕ0 (the grid of a workspace, or by default a 60 × 60 grid of the
# cell), refined by Newton steps (warns if they do not converge)
minimum_stacking(Vg, ws; kw...) = minimum_stacking(Vg, ws.ϕ0; kw...)
function minimum_stacking(Vg, ϕ0::AbstractArray = cell_grid(60); tol = 1e-12)
    V(ϕ) = first(VdV_fast(ϕ, Vg...))
    dV(ϕ) = last(VdV_fast(ϕ, Vg...))
    ϕ = ϕ0[argmin(V.(ϕ0))]
    for _ in 1:20
        step = hessian(Vg, ϕ) \ dV(ϕ)
        ϕ -= step
        norm(step) < tol && break
    end
    norm(dV(ϕ)) <= 1e-8 * (1 + maximum(abs, Vg)) || @warn "minimum_stacking: Newton did not converge (ϕ = $ϕ)"
    return SVector{2,Float64}(ϕ)
end

# interface stiffness k at the minimum stacking of V (found on the grid of ws, or on a 60 × 60 grid of the cell)
stiffness(Vg, ws = cell_grid(60)) = curvature(Vg, minimum_stacking(Vg, ws))

# the stackings s1 a1 + s2 a2 of an N × N grid of the hexagonal cell
cell_grid(N::Integer) = cell_disregistry(Hexagonal(), [1 0; 0 1], (N, N))

include(joinpath(@__DIR__, "Supercells.jl"))    # supercells: modes, grid, disregistry; transfer_modes
include(joinpath(@__DIR__, "Workspaces.jl"))    # Workspace, WorkspaceC3, to_grid!, from_grid!, stacking_energy!
include(joinpath(@__DIR__, "Harmonic.jl"))      # harmonic model: coupling matrices, DIIS damping, LBFGS precond
include(joinpath(@__DIR__, "DIIS.jl"))          # MixerDIIS, mix!, get_error
include(joinpath(@__DIR__, "Relax.jl"))         # relax_moire, minimize_lbfgs, check_gradient

### Utilities

# Lagrange extrapolation of un from the points xn to x; empty (start from zero) without points
extrapolate(x, xn, un) = isempty(un) ? ComplexF64[] :
    sum(prod((x - xn[j]) / (xn[i] - xn[j]) for j in eachindex(xn) if j != i; init = 1.0) * un[i] for i in eachindex(xn))

# parameter box; each row is (name, value) or (name, value, unit)
function print_params(title, rows)
    println("|=================================================|")
    @printf("|%s|\n", rpad(lpad(title, (49 + length(title)) ÷ 2, '='), 49, '='))
    println("|=================================================|")
    for row in rows
        @printf("| %-9s:%24s %-12s |\n", row[1], row[2], length(row) > 2 ? row[3] : "")
    end
    println("|=================================================|")
end

# rows of the parameter box for the coefficients [c1, c2, c3, p1, p3] of a stacking potential
coef_rows(Vg, tag, unit = "meV/cell") = [(name * tag, v, unit) for (name, v) in zip(("c1", "c2", "c3", "p1", "p3"), Vg)]

# the synthesis c -> 2 Re Σ_g c_g e^{i g⋅r} of real fields from their coefficients on the modes of moire, on the grid
# xs × ys (units L), for plots
function field_synthesizer(moire::MoireGeometry, xs, ys)
    Ex, Ey = cis.(real.(moire.g) .* xs'), cis.(imag.(moire.g) .* ys')         # e^{i gx x}, e^{i gy y}: modes × grid
    return c -> 2real.(transpose(Ex) * (c .* Ey))
end

include(joinpath(@__DIR__, "Precompile.jl"))    # workload run while the package is precompiled

end # module
