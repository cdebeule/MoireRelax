# Twisted bilayer graphene: relaxation of the relative displacement u = u1 - u2 at a few twist angles, with C3 symmetry,
# and a figure of the relaxed moiré at the smallest angle: the stacking energy, the displacement and the local twist.
# Run from the repository root with `julia -t auto --project=examples examples/bilayer.jl` (examples/Project.toml has
# CairoMakie; see the README). The figure is written to bilayer_<θ>.png in the current folder.

using MoireRelax, StaticArrays, Printf, CairoMakie

# graphene (PRB 98, 224102): Lamé constants and stacking potential [c1, c2, c3, p1, p3], in meV per cell
const μ, λ = 47352.0, 69518.0 - 47352.0
const Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0.0, 0.0]
const P = SVector{5}(Vg)                                # static, so that VdV_fast(ϕ, P...) is type stable
const N_shell = 30

# one relaxation at the twist angle θ (degrees), from the start u0 (zero if empty)
function relax_bilayer(θ; solver = :lbfgs, u0 = ComplexF64[])
    ws = WorkspaceC3(N_shell, 1)                        # one field: u = u1 - u2
    moire = MoireGeometry(moire_matrix(θ), ws)
    K = [2 / (λ + 2μ)  2 / μ]                           # compliances of u (the two layers in series)
    C = fill(stiffness(Vg, ws), 1, 1)                   # curvature of V at its minimum (AB) preconditions the solver
    potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)
    res = relax_moire(potential!, K, C, moire, ws; solver, u0, label = @sprintf("θ = %.2f°", θ))
    umax = maximum(abs, to_grid!(ws, res.u, moire))     # largest |u1 - u2| on the grid (units of a)
    return res.u, res.E_el, res.E_V, umax
end

### Sweep: the angles downward, each started from the extrapolation of the previous solutions

θs = [1.2, 1.0, 0.8, 0.6]
sols, Es = Matrix{ComplexF64}[], Float64[]
for θ in θs
    n = length(sols)
    u0 = extrapolate(θ, θs[max(1, n - 2):n], sols[max(1, n - 2):n])    # the last three solutions
    u, E_el, E_V, umax = relax_bilayer(θ; u0)
    push!(sols, u)
    push!(Es, E_el + E_V)
    @printf("    max |u1 - u2| = %.3f a   (%.1f%% of the AB-BA distance a/√3)\n", umax, 100umax * sqrt(3))
end

### Real-space fields of the relaxed moiré at the smallest angle

θ, u = θs[end], sols[end]
wsf = WorkspaceC3(N_shell, 1, 192)                      # the same modes on a finer grid, for smooth maps
moire = MoireGeometry(moire_matrix(θ), wsf)
D = to_grid!(wsf, u, moire)[:, :, 1]                    # u_x + i u_y (units a) on the grid of the moiré cell
V = [first(VdV_fast(ϕ0 + SA[real(w), imag(w)], P...)) for (ϕ0, w) in zip(wsf.ϕ0, D)]   # V at the relaxed stacking
twist = θ .+ rad2deg.(scalar_to_grid!(wsf, u[:, 2] ./ (2moire.L), moire))   # local twist angle θ + ½ ẑ⋅∇×u

xs = ys = range(-1.1, 1.1, 221)                         # Cartesian points (units of the moiré period L)
Vxy = grid_to_points(V, wsf, moire, xs, ys)             # the grid of the cell interpolated to the points
uxy = grid_to_points(D, wsf, moire, xs, ys)
twistxy = grid_to_points(twist, wsf, moire, xs, ys)

### Figure

# labels as LaTeX strings: the parts between $ are math, the rest text (Makie renders both with MathTeXEngine)
fig = Figure(size = (1500, 520), fontsize = 16)
ax1 = Axis(fig[1, 1]; title = L"stacking energy $V$", xlabel = L"x / L", ylabel = L"y / L")
Colorbar(fig[1, 2], heatmap!(ax1, xs, ys, Vxy; colormap = :viridis); label = L"$V$ (meV/cell)")
ax2 = Axis(fig[1, 3]; title = L"displacement $u = u_1 - u_2$", xlabel = L"x / L")
Colorbar(fig[1, 4], heatmap!(ax2, xs, ys, abs.(uxy); colormap = :magma); label = L"$|u|$ ($a$)")
sub = 5:8:length(xs)                                    # arrows on a coarse subgrid, centered on their points,
umax = maximum(abs, uxy)                                # the longest 0.07 L (a bit less than their spacing)
arrows2d!(ax2, xs[sub], ys[sub], real.(uxy[sub, sub]), imag.(uxy[sub, sub]); align = :center,
          lengthscale = 0.07 / umax, minshaftlength = 0, shaftwidth = 1.5, tipwidth = 5, tiplength = 6, color = :black)
ax3 = Axis(fig[1, 5]; title = L"local twist angle $\theta + \frac{1}{2} \hat{z} \cdot \nabla \times u$", xlabel = L"x / L")
Colorbar(fig[1, 6], heatmap!(ax3, xs, ys, twistxy; colormap = :viridis); label = L"$\theta_\mathrm{loc}$ ($^\circ$)")
for c in (1, 3, 5)
    colsize!(fig.layout, c, Aspect(1, 1.0))             # square axes that fill their cells: the colorbars match them
end
θstr, Lstr, Estr = @sprintf("%.2f", θ), @sprintf("%.1f", moire.L), @sprintf("%.3f", Es[end])
Label(fig[0, 1:6], L"twisted bilayer graphene, $\theta = %$θstr^\circ$, $L = %$Lstr$ $a$, $E = %$Estr$ meV/cell";
      fontsize = 18, tellwidth = false)                 # over all columns, centered
resize_to_layout!(fig)                                  # the figure takes the size of its content
file = @sprintf("bilayer_%.2f.png", θ)
save(file, fig; px_per_unit = 2)
println("figure saved as ", file)
