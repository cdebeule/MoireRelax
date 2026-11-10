# Twisted bilayer graphene swept over the twist angle, from 2° to 0.01°: the Fourier components of the relaxation,
# the rotation and the dilation of u = u1 - u2 with one amplitude per C3 star, and the energies against the angle.
# The number of shells follows the angle (N_shell θ ≳ 4 for the domain walls), so the smallest angles use 400 shells
# on a 1215² grid, about 2 s each; the whole sweep takes about half a minute.
# Run from the repository root with `julia -t auto --project=examples examples/bilayer_sweep.jl`. The figures are
# written to bilayer_sweep_modes.png and bilayer_sweep_energies.png, the data to bilayer_sweep.txt.

using MoireRelax, StaticArrays, Printf, CairoMakie
using MoireRelax: FFTW

# graphene (PRB 98, 224102): Lamé constants and stacking potential [c1, c2, c3, p1, p3], in meV per cell
const μ, λ = 47352.0, 69518.0 - 47352.0
const Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0.0, 0.0]
const P = SVector{5}(Vg)
const K = [2 / (λ + 2μ)  2 / μ]                         # compliances of u = u1 - u2 (the two layers in series)
const V_AB = first(VdV_fast(minimum_stacking(Vg), P...))  # the stacking energy of the AB domains

nshell(θ) = max(20, ceil(Int, 4 / θ))                   # shells that resolve the walls at the angle θ (degrees)
padrows(u, n) = vcat(u, zeros(ComplexF64, n - size(u, 1), size(u, 2)))   # a solution of fewer shells on n modes

### Sweep: the angles downward, each started from the extrapolation (in log θ) of the last three solutions

θs = exp10.(range(log10(2.0), log10(0.01), 24))
nstar = 10                                              # the stars of the first 4 shells are plotted
sols, E_el, E_V = Matrix{ComplexF64}[], Float64[], Float64[]
bvecs = Matrix{Int}(undef, 2, 0)
for (i, θ) in enumerate(θs)
    ws = WorkspaceC3(nshell(θ), 1; pad = 1.5, flags = FFTW.ESTIMATE)   # a new grid at each angle, planned quickly
    moire = MoireGeometry(moire_matrix(θ), ws)
    C = fill(stiffness(Vg, ws), 1, 1)
    potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)
    n, prev = size(ws.bvecs, 2), max(1, i - 3):i-1
    u0 = extrapolate(log(θ), log.(θs[prev]), [padrows(sols[j], n) for j in prev])
    res = relax_moire(potential!, K, C, moire, ws; u0, label = @sprintf("θ = %.3f°  %3d shells", θ, nshell(θ)))
    push!(sols, res.u)
    push!(E_el, res.E_el)
    push!(E_V, res.E_V)
    global bvecs = ws.bvecs[:, 1:nstar]                 # the same first stars at every angle (ordered by shell)
end
# the amplitudes are real here (a pure twist of a symmetric bilayer); columns: the angles
u_perp = [real(sols[i][k, 2]) for k in 1:nstar, i in eachindex(θs)]   # rotation, u⊥ = L (ẑ⋅∇×u)_g
u_para = [real(sols[i][k, 1]) for k in 1:nstar, i in eachindex(θs)]   # dilation, u∥ = L (∇⋅u)_g

open("bilayer_sweep.txt", "w") do io
    stars = join(("($(b[1]),$(b[2]))" for b in eachcol(bvecs)), " ")
    println(io, "# theta(deg) E_el E_V (meV/cell), then u_perp and u_para of the stars ", stars)
    for i in eachindex(θs)
        println(io, join((θs[i], E_el[i], E_V[i], u_perp[:, i]..., u_para[:, i]...), " "))
    end
end

### Figure 1: the Fourier components against the angle (the mirror pairs, e.g. (3,1) and (3,2), have the same
### rotation and opposite dilations)

labels = (xlabelsize = 21, ylabelsize = 21, xticklabelsize = 17, yticklabelsize = 17, titlesize = 20)
xlabel = L"$\theta$ ($^\circ$)"
colors = resample_cmap(:viridis, nstar + 1)[1:nstar]
fig = Figure(size = (1150, 480), fontsize = 16)
ax1 = Axis(fig[1, 1]; xscale = log10, title = L"rotation, one amplitude per star $\mathbf{g}$", xlabel,
           ylabel = L"$u_\perp = L (\hat{z} \cdot \nabla \times \mathbf{u})_\mathbf{g}$", labels...)
ax2 = Axis(fig[1, 2]; xscale = log10, title = L"dilation", xlabel,
           ylabel = L"$u_\parallel = L (\nabla \cdot \mathbf{u})_\mathbf{g}$", labels...)
for k in 1:nstar
    n1, n2 = bvecs[1, k], bvecs[2, k]
    lines!(ax1, θs, u_perp[k, :]; color = colors[k], linewidth = 2, label = L"$(%$n1, %$n2)$")
    lines!(ax2, θs, u_para[k, :]; color = colors[k], linewidth = 2)
end
axislegend(ax1, L"$\mathbf{g} = n_1 \mathbf{G}_1 + n_2 \mathbf{G}_2$"; position = :rt, nbanks = 2, labelsize = 15)
resize_to_layout!(fig)
save("bilayer_sweep_modes.png", fig; px_per_unit = 2)

### Figure 2: the energies against the angle, and the energies of the walls on log scales (both go as θ, the wall
### length per area, once the domains have formed)

fig2 = Figure(size = (1150, 480), fontsize = 16)
ax3 = Axis(fig2[1, 1]; xscale = log10, title = L"energies per monolayer cell", xlabel, ylabel = L"meV/cell", labels...)
lines!(ax3, θs, E_el .+ E_V; color = :black, linewidth = 2, label = L"$E = E_\mathrm{el} + \langle V \rangle$")
lines!(ax3, θs, E_el; linewidth = 2, label = L"$E_\mathrm{el}$")
lines!(ax3, θs, E_V; linewidth = 2, label = L"$\langle V \rangle$")
hlines!(ax3, [V_AB]; color = :gray, linestyle = :dash, label = L"$V_\mathrm{AB}$")
axislegend(ax3; position = :rb, labelsize = 15)
ax4 = Axis(fig2[1, 2]; xscale = log10, yscale = log10, title = L"energies of the walls", xlabel, ylabel = L"meV/cell",
           labels...)
lines!(ax4, θs, E_el; linewidth = 2, label = L"$E_\mathrm{el}$")
lines!(ax4, θs, E_V .- V_AB; linewidth = 2, label = L"$\langle V \rangle - V_\mathrm{AB}$")
lines!(ax4, θs, E_el[end] .* θs ./ θs[end]; color = :gray, linestyle = :dash, label = L"$\propto \theta$")
axislegend(ax4; position = :lt, labelsize = 15)
resize_to_layout!(fig2)
save("bilayer_sweep_energies.png", fig2; px_per_unit = 2)
println("figures saved as bilayer_sweep_modes.png and bilayer_sweep_energies.png, the data as bilayer_sweep.txt")
