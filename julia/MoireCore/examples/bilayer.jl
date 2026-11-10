# Twisted bilayer graphene: relaxation of the relative displacement u = u1 - u2 at a few twist angles, with C3 symmetry.
# Run from the package folder with `julia -t auto --project=. examples/bilayer.jl`.

using MoireCore, StaticArrays, Printf

# graphene (PRB 98, 224102): Lamé constants and stacking potential [c1, c2, c3, p1, p3], in meV per cell
const μ, λ = 47352.0, 69518.0 - 47352.0
const Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0.0, 0.0]

# one relaxation at the twist angle θ (degrees), from the start u0 (zero if empty)
function relax_bilayer(θ, Vg; N_shell = 30, solver = :lbfgs, u0 = ComplexF64[])
    ws = WorkspaceC3(N_shell, 1)                        # one field: u = u1 - u2
    moire = MoireGeometry(moire_matrix(θ), ws)
    K = [2 / (λ + 2μ)  2 / μ]                           # compliances of u (the two layers in series)
    C = fill(stiffness(Vg, ws), 1, 1)                   # curvature of V at its minimum (AB) preconditions the solver
    P = SVector{5}(Vg)                                  # static, so that VdV_fast(ϕ, P...) is type stable
    potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)
    u, E_el, E_V, converged = relax_moire(potential!, K, C, moire, ws; solver, u0, label = @sprintf("θ = %.2f°", θ))
    umax = maximum(abs, to_grid!(ws, u, moire))         # largest |u1 - u2| on the grid (units of a)
    return u, E_el, E_V, umax
end

# sweep the angles downward, starting each from the extrapolation of the previous solutions
θs = [1.2, 1.0, 0.8, 0.6]
sols = Matrix{ComplexF64}[]
for θ in θs
    n = length(sols)
    u0 = extrapolate(θ, θs[max(1, n - 2):n], sols[max(1, n - 2):n])    # the last three solutions
    u, E_el, E_V, umax = relax_bilayer(θ, Vg; u0)
    push!(sols, u)
    @printf("    max |u1 - u2| = %.3f a   (%.1f%% of the AB-BA distance a/√3)\n", umax, 100umax * sqrt(3))
end
