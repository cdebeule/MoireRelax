# Alternating-twist trilayer graphene (layers at +θ/2, -θ/2, +θ/2): relaxation of the three layer displacements.
# Run from the repository root with `julia -t auto --project=examples examples/trilayer.jl` (or `--project=.`).

using MoireRelax, StaticArrays, Printf, LinearAlgebra

# graphene (PRB 98, 224102): Lamé constants and stacking potential [c1, c2, c3, p1, p3], in meV per cell
const μ, λ = 47352.0, 69518.0 - 47352.0
const Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0.0, 0.0]

function relax_trilayer(θ, Vg; N_shell = 30, solver = :lbfgs)
    nf = 3                                              # the fields are the layer displacements u1, u2, u3
    ws = WorkspaceC3(N_shell, nf)
    moire = MoireGeometry(moire_matrix(θ), ws)
    K = repeat([1 / (λ + 2μ)  1 / μ], 1, nf)            # compliances (dilation, rotation) of each layer
    k = stiffness(Vg, ws)                               # interface stiffness at the minimum of V
    C = chain_laplacian([k, k])                         # harmonic coupling of the two interfaces (1, 2) and (2, 3)
    P = SVector{5}(Vg)
    # local stacking energy: the disregistry is ϕ0 + u1 - u2 at the lower interface and -ϕ0 + u2 - u3 at the upper one
    function V(ϕ0, u1, u2, u3)
        v12, d12 = VdV_fast(ϕ0 + u1 - u2, P...)
        v23, d23 = VdV_fast(-ϕ0 + u2 - u3, P...)
        return v12 + v23, d12, d23 - d12, -d23          # V, ∂V/∂u1, ∂V/∂u2, ∂V/∂u3
    end
    potential!(dVdu, u) = stacking_energy!(V, dVdu, u, moire, ws)
    u, E_el, E_V, converged = relax_moire(potential!, K, C, moire, ws; solver, label = @sprintf("θ = %.2f°", θ))
    D = to_grid!(ws, u, moire)                          # u_x + i u_y of each layer on the grid (units of a)
    return u, E_el, E_V, [maximum(abs, view(D, :, :, f)) for f in 1:nf]
end

u, E_el, E_V, umax = relax_trilayer(1.0, Vg)
@printf("    largest displacement of each layer: %.4f, %.4f, %.4f a\n", umax...)
println("    (the outer layers move equally, the middle layer about twice as much and the other way)")
