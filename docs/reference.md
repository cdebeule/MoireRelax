# MoireRelax: reference

MoireRelax relaxes the in-plane displacements of the layers of a moiré of monolayers with hexagonal symmetry
(graphene, hBN, transition-metal dichalcogenides, …): it minimizes their elastic energy plus the stacking energy of
the interfaces, by LBFGS or DIIS. The relaxation is computed on one moiré cell with periodic boundary conditions, or
on a supercell of several moiré cells ([Supercells](#supercells)), in which the relaxation can break the moiré
translations: the displacements are expanded in the reciprocal vectors of the cell, and the stacking energy is
evaluated on a real-space grid of the cell (FFTs). It handles any number of layers and interfaces (bilayers,
trilayers, stacks), with or without C3 symmetry. The method is described in
[Refs. 1 and 2](../README.md#references).

This document is the reference of the package: its conventions, the method behind each part and every function. The
comments in the code are kept to one line; the longer explanations are collected here. The equations are LaTeX, which
GitHub typesets. For the installation and a first example, see the [README](../README.md).

**Contents:** [Usage](#usage) · [Files](#files) · [Conventions](#conventions) ·
[Geometry and stacking potential](#geometry-and-stacking-potential) · [Workspaces](#workspaces) ·
[Supercells](#supercells) · [Harmonic model](#harmonic-model-of-the-interlayer-coupling) · [Solvers](#solvers) ·
[Precompilation](#precompilation)

## Usage

Scripts load the package with `using MoireRelax`. Start Julia with threads (`julia -t auto`, or `-t 8`): the threads
share out the grid points of the stacking energy (`stacking_energy!`).

At the first `using MoireRelax` (and again after a change of its files), Julia compiles the package, about 35 s, with
a workload that runs the solvers ([Precompilation](#precompilation)). Later runs load the compiled code:
`using MoireRelax` takes about 1 s and the first `relax_moire` about 1 s, most of it for the script's own
`potential!`, which is still compiled in each run. In an interactive session, `using Revise` before `using MoireRelax`
applies changes of the files without restarting Julia.

### Example

A twisted bilayer, relaxed with C3 symmetry; `examples/bilayer.jl` sweeps the angle with this function, and
`examples/trilayer.jl` relaxes the three layers of an alternating-twist trilayer:

```julia
using MoireRelax, StaticArrays

# relaxation of a twisted bilayer of two equal layers at the angle θ (degrees), with C3 symmetry
function relax_bilayer(θ, μ, λ, Vg; N_shell = 30, solver = :lbfgs)
    ws = WorkspaceC3(N_shell, 1)                    # one field: the relative displacement u = u1 - u2
    moire = MoireGeometry(moire_matrix(θ), ws)
    K = [2 / (λ + 2μ)  2 / μ]                       # compliances of u1 - u2 (the two layers in series)
    C = fill(stiffness(Vg, ws), 1, 1)               # Hessian of the harmonic interlayer energy
    P = SVector{5}(Vg)                              # static, so that VdV_fast(ϕ, P...) is type stable
    potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)
    return relax_moire(potential!, K, C, moire, ws; solver)
end

# graphene (PRB 98, 224102): Lamé constants μ, λ and stacking potential [c1, c2, c3, p1, p3] (meV/cell)
u, E_el, E_V, converged = relax_bilayer(1.0, 47352, 69518 - 47352, [4.064 / 2, -0.374 / 2, -0.095 / 2, 0, 0])
```

`u` holds the mode amplitudes ([Conventions](#conventions)); `to_grid!(ws, u, moire)` gives the displacement field on
the grid of the moiré cell, and `field_synthesizer` evaluates it on any grid for plots.

## Files

| File | Contents |
|---|---|
| `src/MoireRelax.jl` | the module and its exports; the lattice (`Lattice`, `Hexagonal`) and the moiré geometry (`MoireGeometry`, `moire_matrix`, `twist_angle`); the D3 stacking potential (`VdV_fast`, `hessian`, `curvature`, `minimum_stacking`, `stiffness`); utilities (`extrapolate`, `print_params`, `coef_rows`, `field_synthesizer`); includes the files below |
| `src/Supercells.jl` | the modes, grid and rigid disregistry of a cell of the lattice (`supercell_wavevectors`, `cell_modes`, `grid_index`, `cell_disregistry`), on which both workspaces are built; `transfer_modes`, `moire_periodic`, `supercell_fraction` |
| `src/Workspaces.jl` | `Workspace` (also for supercells) and `WorkspaceC3`: the modes, the real-space grid of the cell and the FFTs between them (`to_grid!`, `from_grid!`); C3 orbits (`grid_orbits`); the stacking energy on the grid (`stacking_energy!`) |
| `src/Harmonic.jl` | the harmonic model of the interlayer coupling: coupling matrices (`coupling_laplacian`, `chain_laplacian`), DIIS damping (`damping!`, `damped_step!`), LBFGS change of variables (`precond`, `transform!`) |
| `src/DIIS.jl` | DIIS (Pulay) mixing: `MixerDIIS`, `mix!`, `get_error` |
| `src/Relax.jl` | `relax_moire`: minimization of the elastic plus stacking energy by LBFGS or DIIS; `elastic_energy`, `minimize_lbfgs`, `check_gradient` |
| `src/Precompile.jl` | the workload run while the package is precompiled |
| `Project.toml` | name, identifier (UUID) and dependencies of the package |
| `examples/bilayer.jl`, `examples/trilayer.jl` | a twisted bilayer swept over the angle, and an alternating-twist trilayer relaxed layer by layer |
| `docs/reference.md` | this document |
| `test/runtests.jl` | the tests: the invariants of the method (gradients against finite differences, C3 against no symmetry, LBFGS against DIIS, a supercell against the moiré cell, the D3 potential against its Fourier series and analytic Hessian), the argument checks, reference energies, and the code-quality checks of Aqua.jl; run by `Pkg.test()` and, on GitHub, by the CI workflow on Linux, macOS and Windows |

## Conventions

- lengths in $a$ (the lattice constant, or the mean one of two materials), energies in meV per monolayer cell
- the cell: the relaxation is periodic with the moiré cell (or with a supercell of it), so the displacement fields are
  Fourier series in the reciprocal vectors of the cell, cut off at `N_shell` shells, and all averages $\langle \cdot \rangle$
  are over the cell; energies are given per monolayer cell, so they compare between cells
- the lattice: the monolayer lattice is a type (`Lattice`) that the workspace carries (`ws.lattice`), and everything
  that depends on it, the lattice and reciprocal vectors (`lattice_vectors`, `reciprocal_vectors`), the cell area
  (`cell_area`), the shells of the reciprocal vectors and the rotation of `WorkspaceC3`, asks the lattice; the rest of
  the package works per mode and per grid point. `Hexagonal()`, the default, is the lattice of graphene, hBN and the
  transition-metal dichalcogenides, and the only one provided; the D3 stacking potential below is written in its
  reciprocal vectors
- displacements and moiré wavevectors in complex notation $v = v_x + i v_y$, with $g$ in units $1/L$; the lattice
  vectors and the disregistry $\phi$ are real 2-vectors (`SVector{2}`)
- fields: the $n_f$ displacement fields $u_f$ ($f = 1, \dots, n_f$) that the solver relaxes: the layer displacements of
  a stack, or the combinations of them that the stacking energy depends on, e.g. $u = u_1 - u_2$ (bilayer) or
  $u_1 - u_3$ and $u_1 - 2u_2 + u_3$ (trilayer of identical layers). The elastic energy must be diagonal in the fields:
  $E_\mathrm{el} = \sum |u|^2/(K L^2)$ with one compliance $K$ per column (a layer: $1/(\lambda + 2\mu)$ and $1/\mu$;
  $u_1 - u_2$: the sum of the two layers' compliances)
- compliances `K`: one row for all modes ($1 \times 2n_f$), or one row per mode ($n \times 2n_f$, e.g. for an
  effective bilayer)
- amplitudes: $u_\parallel$ and $u_\perp$ (`u_para`, `u_perp` in the code) are the dilation and the rotation,

  $$u_f(r) = \sum_g u_{f,g} e^{i g \cdot r}, \qquad u_{f,g} = -i \frac{g}{|g|^2} (u_\parallel + i u_\perp),$$

  such that $u_\parallel = L (\nabla \cdot u_f)_g$ and $u_\perp = L (\hat z \cdot \nabla \times u_f)_g$
- columns $2f - 1$ and $2f$ of `u` hold $u_\parallel$ and $u_\perp$ of field $f$, one row per mode
- one representative $g_m$ per pair $\pm g$; the amplitudes at $-g_m$ are the complex conjugates (real fields)

## Geometry and stacking potential

The lattice vectors of `Hexagonal()` are $a_1 = (-1/2, \sqrt{3}/2)$ and $a_2 = (-1, 0)$ (zigzag along $x$), and its
reciprocal vectors $b_1 = (0, 4\pi/\sqrt{3})$ and $b_2 = (-2\pi, -2\pi/\sqrt{3})$, with $b_i \cdot a_j = 2\pi \delta_{ij}$
(`lattice_vectors`, `reciprocal_vectors`; also the constants `a1`, `a2`, `b1`, `b2`); the area of the cell is
$A_\mathrm{cell} = \sqrt{3}/2$ (`cell_area`, `Acell`). The lattice also defines the shells of its reciprocal vectors,
shell $s$ having $\max(|n_1|, |n_2|, |n_1 - n_2|) = s$ (`shell`), their list by shell (`cell_wavevectors`), and the
rotation by $2\pi/3$ in the basis $a_1, a_2$ (`rotation`; `reciprocal_rotation` in the basis $b_1, b_2$).

The moiré follows from the relative deformation $M$ of two layers: the rigid disregistry is $\phi_0 = M r$. For a twist
$\theta$ and an isotropic lattice mismatch $\epsilon_0$ (`moire_matrix(θ, ϵ0)`, angles in degrees),

$$M = \Big(1 + \frac{\epsilon_0}{2}\Big) R(\theta/2) - \Big(1 - \frac{\epsilon_0}{2}\Big) R(-\theta/2),$$

and `twist_angle(M)` reads $\theta$ from the antisymmetric part of $M$. `MoireGeometry(M, ws)` holds the moiré length
$L$, the mean length of the moiré lattice vectors $M^{-1} a_1$ and $M^{-1} a_2$, and the moiré reciprocal vectors
$g = L M^\mathsf{T} b$ of the modes of `ws` (units $1/L$), such that $g \cdot r = b \cdot \phi_0$ for $r$ in units $L$.

The stacking potential of an interface has the D3 form, with $x_i = b_i \cdot \phi$:

$$\begin{aligned} V(\phi) = {} & 2 c_1 \big[\cos x_1 + \cos x_2 + \cos(x_1 + x_2)\big] - 2 p_1 \big[\sin x_1 + \sin x_2 - \sin(x_1 + x_2)\big] \\ & + 2 c_2 \big[\cos(x_1 + 2 x_2) + \cos(2 x_1 + x_2) + \cos(x_1 - x_2)\big] \\ & + 2 c_3 \big[\cos 2 x_1 + \cos 2 x_2 + \cos(2 x_1 + 2 x_2)\big] - 2 p_3 \big[\sin 2 x_1 + \sin 2 x_2 - \sin(2 x_1 + 2 x_2)\big], \end{aligned}$$

with the coefficients `Vg = [c1, c2, c3, p1, p3]`: $c_1 = \operatorname{Re} V_1$, $c_2 = V_2$,
$c_3 = \operatorname{Re} V_3$, $p_1 = \operatorname{Im} V_1$, $p_3 = \operatorname{Im} V_3$ (meV/cell).

- `VdV_fast(ϕ, c1, c2, c3, p1, p3)`: $V$ and its gradient at $\phi$; pass the coefficients as a static vector `P`
  (`VdV_fast(ϕ, P...)`). It is `series(harmonics(ϕ), c...)`: `harmonics` gives the sines and cosines of the three
  stars once, which several series at the same $\phi$ can share (internal functions)
- `hessian(Vg, ϕ)`: the Hessian of $V$ at $\phi$, analytic, $-\sum_G 2\operatorname{Re}(V_G e^{iG\cdot\phi})\,GG^\mathsf{T}$
  over the three stars
- `curvature(Vg, ϕ)`: the interface stiffness $k = \operatorname{tr}(\nabla \nabla V)/2$ (meV/cell/$a^2$), the mean
  curvature of $V$ at $\phi$ (warns if $V$ is not convex there)
- `minimum_stacking(Vg, ws)`: the lowest stacking, from the lowest grid point refined by Newton steps;
  `stiffness(Vg, ws)` is the curvature there. Without a workspace (`stiffness(Vg)`), a 60 × 60 grid of the cell
  (`cell_grid(60)`) serves as the starting points
- the same four functions take, in place of the coefficients `Vg`, any callable `V(ϕ) -> (V, ∇V)` as the potential,
  e.g. `ϕ -> VdV_fast(ϕ, P...)` or a function of another form; its Hessian is then taken by central differences of the
  gradient (`hessian(V, ϕ; h = 1e-4)`, relative accuracy about $10^{-7}$)
- utilities: `extrapolate(x, xn, un)` extrapolates (Lagrange) the solutions `un` at the points `xn` to the next point
  `x` of a sweep, as its starting point; `print_params(title, rows)` prints a parameter box, and `coef_rows(Vg, tag)`
  gives its rows for the coefficients of a stacking potential

## Workspaces

`Workspace(N_shell, nf, N = 0; lattice = Hexagonal(), S = nothing, pad = 1, flags = FFTW.MEASURE, fft_threads = 1)`
holds the modes of `N_shell` shells of the lattice and an $N \times N$ grid for `nf` fields (with a supercell `S`:
[Supercells](#supercells)). Without `N`, the grid is the smallest FFT-friendly size
$\ge \mathrm{pad} (2 N_\mathrm{shell} + 1)$: `pad = 1` is the minimal grid; `pad = 1.5` to `2` reduces the aliasing of
the nonlinear forces onto the modes, which matters for sharp domain walls (small angles). `WorkspaceC3` takes the same
arguments. `flags = FFTW.MEASURE` (the default) lets FFTW choose its algorithm by timing, so two runs can differ at
the level of $10^{-16}$; `flags = FFTW.ESTIMATE` makes them reproducible, at the same speed within noise. Scripts use
the fields `ws.lattice`, `ws.N`, `ws.nf`, `ws.bvecs`, `ws.ϕ0` (the rigid disregistry $s_1 a_1 + s_2 a_2$ on the grid)
and, with C3, `ws.orb`. A workspace owns its FFT buffers (`ws.W`, `ws.D`), which every `to_grid!`, `from_grid!` and
`stacking_energy!` overwrites: use one workspace per concurrent relaxation, and read `ws.D` right after `to_grid!`.
The arguments are checked (`N_shell` and `nf` at least 1, a grid large enough for the modes, a supercell only without
symmetry), with an `ArgumentError` otherwise.

`to_grid!(ws, u, moire)` returns $u_x + i u_y$ of each field on the grid (the array `ws.D`, $N \times N \times n_f$),
and `from_grid!(dVdu, ws, moire)` turns the forces in `ws.D` into the gradient per mode.

`make_bvecs(N_shell)` lists the representative b vectors: $(n_1, n_2)$ rotated by $0$, $2\pi/3$, $4\pi/3$; the $6s$
vectors $\pm g$ of shell $s$ have $\max(|n_1|, |n_2|, |n_1 - n_2|) = s$. The shells come in increasing order, so the
modes of fewer shells are the first rows of `u`, which is how `relax_moire` starts from a `u0` of fewer shells.

### C3 symmetry

With C3 symmetry, $u(Rr) = R u(r)$ for the rotation $R$ by $2\pi/3$, so the amplitudes $u_\parallel$, $u_\perp$ at
$g$, $Rg$ and $R^2 g$ are equal: one mode per star $\lbrace g, Rg, R^2 g \rbrace$, at which $u_x + i u_y$ picks up the
phases $\omega_3 = (1, \omega, \omega^2)$, $\omega = e^{2\pi i/3}$. On the grid, $\phi(Rr) = R \phi(r)$, so the
stacking energy and its force need only be evaluated on one point of each C3 orbit; the forces on the other points are
rotated copies.

C3 symmetry requires $M$ = scaling × rotation (twist and isotropic mismatch), which `MoireGeometry` checks. Each stored
amplitude is shared by the three modes of its star (`multiplicity(ws) = 3`): the elastic energy counts it three times,
and the derivative of $\langle V \rangle$ with respect to it is 3 times the per-mode gradient from `from_grid!`.

`grid_orbits(lattice, N)` returns the C3 orbits of the $N \times N$ grid: `orb[r, k]` is the linear index of $R^{r-1} p_k$, with
$p_k$ the first point of orbit $k$ and $R: (i_1, i_2) \mapsto (-i_1 - i_2, i_1) \bmod N$ the rotation by $2\pi/3$ of
$s_1 a_1 + s_2 a_2$ (`rotation(lattice)`). The points that $R$ leaves fixed (the origin, and $(N/3, N/3)$ and $(2N/3, 2N/3)$ when 3 divides
$N$) are orbits of one point, whose index appears three times in their column of `orb`. `stacking_energy!` counts them
once and keeps their unrotated force.

### Stacking energy on the grid

```julia
stacking_energy!(f, dVdu, u, moire, ws) -> ⟨V⟩
```

computes the stacking energy $\langle V \rangle = \frac{1}{N^2} \sum_r V(r)$ on the grid and, unless
`dVdu === nothing`, its gradient per mode,

$$\texttt{dVdu} = \frac{\partial \langle V \rangle}{\partial (\mathrm{Re}\ u)} + i \frac{\partial \langle V \rangle}{\partial (\mathrm{Im}\ u)}.$$

The model gives only the local energy at a grid point,

$$f(\phi_0, u_1, \dots, u_{n_f}) \to \Big(V,\ \frac{\partial V}{\partial u_1},\ \dots,\ \frac{\partial V}{\partial u_{n_f}}\Big),$$

with the rigid disregistry $\phi_0$, the fields $u_f$ and the forces $\partial V / \partial u_f$ as 2-vectors
(`SVector{2}`). `stacking_energy!` does the rest: the FFTs (`to_grid!`, `from_grid!`), the threads (Polyester's
`@batch`) and, with `WorkspaceC3`, the evaluation on one point per orbit, the rotated forces and the invariant points.
So one `potential!` serves both workspaces. Two examples:

```julia
# bilayer: one field, u = u1 - u2
potential!(dVdu, u, moire, ws, P) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)

# trilayer: two fields, u_a and u_s
function potential!(dVdu, u, moire, ws, P1, P2, δ, n::Integer)
    return stacking_energy!(dVdu, u, moire, ws) do ϕ0, ua, us
        ua += δ
        v12, d12 = VdV_fast(ϕ0 + (ua + us) / 2, P1...)
        v23, d23 = VdV_fast(-n * ϕ0 + (ua - us) / 2, P1...)
        v13, d13 = VdV_fast((1 - n) * ϕ0 + ua, P2...)
        return v12 + v23 + v13, (d12 + d23) / 2 + d13, (d12 - d23) / 2   # V, ∂V/∂u_a, ∂V/∂u_s
    end
end
```

A layer-by-layer loop over a variable number of fields takes them as `ϕ0, ul...`. The local
energy is inlined into the grid loop, so it runs as fast as a hand-written loop. The parameters it captures are best
static vectors, numbers or tuples (a tuple of the stacking vectors was up to 15% faster than a `Vector`). It must not
write to shared state, since the grid points are shared out between the threads.

## Supercells

A supercell of the moiré cell lets the relaxation break the moiré translations, e.g. stripes of doubled period.
`Workspace(N_shell, nf, N = 0; S)` takes it as the integer matrix $S$ whose columns are the supercell vectors in units of
the moiré lattice vectors $L_1 = M^{-1} a_1$, $L_2 = M^{-1} a_2$ (60° apart): $A_j = S_{1j} L_1 + S_{2j} L_2$, and
$\det S$ is the number of moiré cells. `S = nothing` (the default) is the moiré cell as before.

- Modes: the reciprocal vectors of the supercell are $x_1 G_1 + x_2 G_2$, with $G_i = M^\mathsf{T} b_i$ the moiré
  reciprocal vectors and $x = S^{-\mathsf{T}} m$ for integers $m$: the moiré reciprocal vectors ($x$ integer) and the
  fractional ones between them. `ws.bvecs` holds $x$ (floats), one of each $\pm g$, all those within `N_shell` shells of the moiré reciprocal
  lattice (the resolution of the moiré cell), ordered by shell (`supercell_wavevectors(lattice, S, N_shell)` returns $m$
  and $x$).
- Grid: $N_1 \times N_2$ points of the supercell, $N_i$ the smallest FFT-friendly size $\ge \mathrm{pad}(2 \max |m_i| + 1)$
  (`ws.N = (N1, N2)`; one `N` gives $N \times N$). At the grid point $t$ (supercell coordinates), the moiré coordinates are
  $s = S t$ and the rigid disregistry $\phi_0 = s_1 a_1 + s_2 a_2$; with $g \cdot r = 2\pi\, m \cdot t$, the FFT index of
  a mode is $m \bmod N$.
- Everything else (`MoireGeometry`, `stacking_energy!`, `relax_moire` with both solvers, the harmonic model) works per
  mode and per grid point, so it takes a supercell workspace as it is. Energies stay per monolayer cell, so they compare
  directly between cells. `WorkspaceC3` is for the moiré cell only.
- Symmetry breaking: the gradient of a moiré-periodic state is moiré-periodic, so a relaxation from a periodic start
  (e.g. zero) stays periodic and ends at the energy of the moiré cell (within $10^{-15}$ meV/cell for
  $2 \times 2$, $1 \times \sqrt 3$ and $\sqrt 3 \times \sqrt 3$). To find a state of lower symmetry, start from a
  perturbed one.

Common supercells: the hexagonal $A_1 = m L_1 + k L_2$, $A_2$ = $A_1$ rotated by 60° ($S = [m\ {-k};\ k\ \ m{+}k]$,
$\det S = m^2 + mk + k^2$), e.g. $\sqrt 3 \times \sqrt 3$ `[1 -1; 1 2]` (the K points of the moiré Brillouin zone),
$2 \times 2$ `[2 0; 0 2]` (the M points), $3 \times 3$ `[3 0; 0 3]`; and the rectangular $p \times q\sqrt 3$ for stripes
along $L_1$ ($S = [p\ {-q};\ 0\ \ 2q]$, $\det S = 2pq$), e.g. $1 \times \sqrt 3$ `[1 -1; 0 2]` (one M point).

- `transfer_modes(X, wsa, wsb)`: amplitudes (one row per mode, any columns) of the workspace `wsa` on the modes of
  `wsb`, matched by their reciprocal coordinates (the amplitudes at $-g$ are the complex conjugates), zero on the modes
  that `wsa` lacks; e.g. a moiré-cell solution as the start of a supercell run. `wsa` may be anything with the field
  `bvecs`, e.g. `(; bvecs = params["bvecs"])` of a saved run.
- `moire_periodic(ws)`: which modes are moiré-periodic (integer $x$); `supercell_fraction(c, ws)`: the fraction of
  $\sum |c|^2$ in the other modes, i.e. how much a state breaks the moiré translations
- `MoireGeometry(M, bvecs)`: the geometry of modes given by their reciprocal coordinates, e.g. those saved with a run
- `field_synthesizer(moire, xs, ys)`: the function $c \mapsto 2 \operatorname{Re} \sum_g c_g e^{i g \cdot r}$ of real
  fields from their coefficients, on the grid `xs` × `ys` (units $L$), for plots
- internal, shared by the workspaces (and by packages that build on MoireRelax): `cell_modes` (the modes and grid size
  of a cell, the moiré cell included), `grid_index` (the FFT position of a mode), `cell_disregistry` (the rigid
  disregistry on the grid of a cell), `shell` (the shell of a reciprocal vector of the lattice)

## Harmonic model of the interlayer coupling

The harmonic model preconditions both solvers.

**1. Interface stiffness.** Near its minimum, the stacking energy of the interface between the layers $i$ and $j$ is

$$V \approx V_\mathrm{min} + \frac{k}{2} |u_i - u_j|^2,$$

with $k$ (meV/cell/$a^2$) the curvature of $V$ at its minimum (`curvature`, `stiffness`). At a C3-symmetric stacking
the Hessian of $V$ is $k$ times the identity, so $k$ does not depend on the direction of $u_i - u_j$.

**2. Layer fields.** When the fields are the layer displacements, the layers are the nodes of a graph, and each pair of
coupled layers $(i, j)$ is an edge with weight $k_{ij}$: the interfaces $(l, l + 1)$ and, if present, the next-nearest
pairs $(l, l + 2)$. Summed over the edges, the harmonic interlayer energy is

$$E_\mathrm{int} = \sum_\mathrm{edges} \frac{k_{ij}}{2} |u_i - u_j|^2 = \frac{1}{2} u^\dagger C u, \qquad C_{ij} = \frac{\partial^2 E_\mathrm{int}}{\partial u_i \partial u_j},$$

i.e. $C$ is the Hessian of $E_\mathrm{int}$ with respect to the layer displacements. Since the interfaces are isotropic,
$C$ acts on the $x$ and $y$ components separately and in the same way:
$\partial^2 E_\mathrm{int} / \partial u_{i,\alpha} \partial u_{j,\beta} = C_{ij} \delta_{\alpha\beta}$
($\alpha, \beta = x, y$).

In terms of the graph, $C_{ii}$ is the sum of the weights of the edges at layer $i$ and $C_{ij} = -k_{ij}$ for an edge
$(i, j)$: $C$ is the weighted Laplacian of the graph, which for a stack is a discrete Laplacian along the layer index.
Examples: 2 layers with the interface $k$, and 3 layers with the interfaces $k_1$ $(1, 2)$ and $k_2$ $(2, 3)$ and the
next-nearest pair $k_3$ $(1, 3)$:

$$C = \begin{pmatrix} k & -k \\ -k & k \end{pmatrix}, \qquad C = \begin{pmatrix} k_1 + k_3 & -k_1 & -k_3 \\ -k_1 & k_1 + k_2 & -k_2 \\ -k_3 & -k_2 & k_2 + k_3 \end{pmatrix}.$$

$C$ is symmetric and positive semidefinite for $k \ge 0$, and its rows sum to zero: a common translation of all layers
costs no interlayer energy.

**3. Other fields.** In general, $C$ is the Hessian of $E_\mathrm{int}$ with respect to the fields. When the layer
displacements are linear in the fields, $u_\mathrm{layers} = T u_\mathrm{fields}$, it is
$C = T^\mathsf{T} C_\mathrm{layers} T$, with $C_\mathrm{layers}$ the Laplacian above, and the compliances of the
fields follow from $T^\mathsf{T} \operatorname{diag}(1/K_\mathrm{layers}) T = \operatorname{diag}(1/K_\mathrm{fields})$,
which must be diagonal. Such a $C$ need not be a Laplacian: its rows need not sum to zero. Examples: the 2 layers above
with the field $u_1 - u_2$, and the 3 layers above with $k_1 = k_2 = k$ and the fields $u_1 - u_3$, $u_1 - 2u_2 + u_3$:

$$C = (k), \qquad C = \operatorname{diag}(k/2 + k_3,\ k/2).$$

**4. Modes.** For each mode $g_m$ and column type $t$ ($t = 1$: dilation, $t = 2$: rotation), collect the amplitudes
and the compliances of the $n_f$ fields in vectors (a layer: $K_t = 1/(\lambda + 2\mu)$, $1/\mu$):

$$u_{m,t} = \big(u[m, t],\ u[m, t + 2],\ \dots,\ u[m, 2n_f - 2 + t]\big), \qquad K_t = \big(K[t],\ K[t + 2],\ \dots,\ K[2n_f - 2 + t]\big).$$

In these variables, the elastic energy of field $f$ and the harmonic interlayer energy are

$$\sum_{m,t} \frac{|u_{m,t}^{(f)}|^2}{K_t^{(f)} L^2} \qquad \text{and} \qquad \sum_{m,t} \frac{u_{m,t}^\dagger C u_{m,t}}{|g_m|^2},$$

where $1/|g_m|^2$ appears because the amplitudes are dilations and rotations (the displacement is
amplitude$/|g_m|$). Modes and column types do not mix, so the harmonic energy splits into small $n_f \times n_f$
problems:

$$E = \sum_{m,t} u_{m,t}^\dagger A_{m,t} u_{m,t}, \qquad A_{m,t} = \operatorname{diag}\Big(\frac{1}{K_t L^2}\Big) + \frac{C}{|g_m|^2},$$

with the gradient $\partial E / \partial(\mathrm{Re}\ u) + i \partial E / \partial(\mathrm{Im}\ u) = 2Au$. Short
wavelengths (large $|g_m|$): the elastic term dominates and the fields relax independently. Long wavelengths:
$C/|g_m|^2$ dominates and holds the layers together.

**5. Eigendecomposition.** With $D_t = \operatorname{diag}(K_t)$ and
$D_t^{1/2} C D_t^{1/2} = Q \Lambda Q^\mathsf{T}$ (`coupling_modes`),

$$A_{m,t} = D_t^{-1/2} Q \Big(\frac{1}{L^2} + \frac{\Lambda}{|g_m|^2}\Big) Q^\mathsf{T} D_t^{-1/2}.$$

$Q$ does not depend on $g_m$, so one eigendecomposition per column type serves all modes.

### Functions

- `coupling_laplacian(nlayers, couplings)`: the weighted Laplacian of the graph of layers. Each coupling `(i, j, k)`
  adds $k |u_i - u_j|^2 / 2$ to the harmonic energy, i.e. $k$ to $C_{ii}$ and $C_{jj}$, and $-k$ to $C_{ij}$ and
  $C_{ji}$.
- `chain_laplacian(k_nn, k_nnn = Float64[]; nnn_start)`: the Laplacian of a stack of layers
  $1, \dots, n_\mathrm{layers}$, with $n_\mathrm{layers}$ = `length(k_nn) + 1`:
  - `k_nn[l]` is the stiffness of the interface between the neighboring layers $l$ and $l + 1$
  - `k_nnn[j]` is the stiffness of the $j$-th next-nearest pair: the pairs are $(s, s + 2)$, $(s + 1, s + 3)$, … with
    $s$ = `nnn_start`, the lower layer of the first pair. `nnn_start` must be given when `k_nnn` is not empty. Example:
    5 layers, `nnn_start = 2` and `k_nnn = [ka, kb]` couple the layers 2 and 4 (`ka`) and 3 and 5 (`kb`), e.g. when
    layer 1 is a different material and the next-nearest coupling exists only among the layers 2, …, 5
- `coupling_modes(K, C, t)`: `sK` $= \sqrt{K_t}$ (square roots of the compliances of the fields) and
  `sK .* C .* sK'` $= Q \Lambda Q^\mathsf{T}$, i.e. $D_t^{1/2} C D_t^{1/2} = Q \Lambda Q^\mathsf{T}$. With $K$ one row
  per mode, `damping!` and `precond` decompose once per mode.
- `damping!(P, K, C, L, moire)`: the damping matrices `P[:, :, m, t]`,

  $$P_{m,t} = \Big(I + \frac{L^2}{|g_m|^2} \operatorname{diag}(K_t) C\Big)^{-1} = \big(L^2 \operatorname{diag}(K_t) A_{m,t}\big)^{-1},$$

  for the DIIS fixed point $u = -\frac{K L^2}{2}$ `dVdu`: the damped step $u + P (u_\mathrm{next} - u)$ is the Newton
  step for harmonic interfaces. With $V = D_t^{1/2} Q$, $\operatorname{diag}(K_t) C = V \Lambda V^{-1}$ and
  $P = V \big(I + L^2 \Lambda / |g_m|^2\big)^{-1} V^{-1}$.
- `damped_step!(out, u, G, P, β)`: $u + \beta P (G - u)$ for each mode and column type; `out` may be `u` or `G`
- `precond(K, C, L, moire; mult = 1)`: the change of variables $u = S x$ for LBFGS, such that the Hessian in $x$ is the
  identity for harmonic interfaces. With respect to the real and imaginary parts, the Hessian in $u$ is $2A$, and

  $$S = D_t^{1/2} Q \Big(\frac{2}{L^2} + \frac{2\Lambda}{|g_m|^2}\Big)^{-1/2}$$

  gives $S^\mathsf{T} 2A S = I$, i.e. $S S^\mathsf{T} = (2A)^{-1}$; `Sinv[i, f, m, t]` is its inverse. For a single
  field ($n_f = 1$) with $C = (k)$, e.g. the relative displacement of a bilayer, $S$ is the scalar
  $\sqrt{(K L^2/2) / (1 + L^2 K k / |g_m|^2)}$. When `mult` modes share each amplitude (C3: `mult = 3`), the energy and
  its Hessian are `mult` times larger and $S$ is divided by $\sqrt{\mathrm{mult}}$.
- `transform!(out, S, x; transpose = false)`: `out[m, 2f - 2 + t] = Σ_i S[f, i, m, t] x[m, 2i - 2 + t]`, or with
  $S^\mathsf{T}$ for `transpose = true` (gradients: $\partial E / \partial x = S^\mathsf{T} \partial E / \partial u$)

## Solvers

```julia
relax_moire(potential!, K, C, moire, ws; solver = :lbfgs, u0 = ComplexF64[], label = "", verbose = true, kwargs...)
    -> (; u, E_el, E_V, converged, iterations)
```

minimizes $E = E_\mathrm{el} + E_V$ over the mode amplitudes `u` ($n \times 2n_f$, columns $u_\parallel$, $u_\perp$ of
each field), with

- $E_\mathrm{el} = \mathrm{mult} \sum |u|^2 / (K L^2)$ (`elastic_energy`), with `K` the compliances of the columns and
  mult = `multiplicity(ws)` the number of modes that share each amplitude (3 for `WorkspaceC3`)
- $E_V$ = `potential!(dVdu, u)`, the model's stacking energy $\langle V \rangle$, usually from `stacking_energy!`,
  which also fills its gradient `dVdu` per mode (Stacking energy on the grid) unless `dVdu === nothing`

$C$ is the Hessian of the harmonic interlayer energy with respect to the fields (e.g. $(k)$ for the relative
displacement of a bilayer, `chain_laplacian(…)` for the layers of a stack), used to precondition both solvers. The
starting point is `u0` (zero if empty; it may come from fewer shells, whose modes are the first rows; rows beyond the
modes of the workspace are dropped). The result is a named tuple, so `res.u`, `res.iterations`, … as well as
`u, E_el, E_V, converged = relax_moire(…)` work. With `verbose` (the default), each call prints one line with `label`,
the solver, the number of iterations and the energies; it warns if the solver did not converge. The sizes of `K`, `C`
and `u0` are checked (`DimensionMismatch`), and `potential!` must return the energy as a real number
(`ArgumentError` otherwise).

### LBFGS (solver = :lbfgs)

LBFGS in $u = S x$ (`precond`). Keywords: `rtol = 1e-6` (gradient tolerance, relative to the size of the solution),
`max_iter = 300`, `m = 10` (history) and `atol = nothing` (absolute gradient tolerance in the variables $x$; if given,
it replaces `rtol`). `converged` is the gradient criterion (`Optim.g_converged`): Optim also stops when the energy
stops changing, which then counts as not converged and gives the warning. Further keywords go to `Optim.Options`
(e.g. `show_trace = true` prints the iterations).

`minimize_lbfgs(fg!, x0; rtol, atol, max_iter, m, kwargs...)` is LBFGS in variables $x$ scaled such that the Hessian
is $\approx 1$. The gradient tolerance is relative to the size of the solution, estimated by
$x_0 - \partial E / \partial x (x_0)$ (a Newton step for a unit Hessian); optimize reuses this first evaluation (its
first call, at $x_0$, is answered from `F0`, `G0`: Optim 1.x evaluates the starting point again otherwise). With
`atol`, the gradient tolerance is `atol` instead (absolute, in the variables $x$). Further keywords go to
`Optim.Options` (e.g. `show_trace = true`).

### DIIS (solver = :diis)

The damped fixed-point update $u \leftarrow u + P (u_\mathrm{next} - u)$ (`damping!`), with
$u_\mathrm{next} = -\frac{K L^2}{2}$ `dVdu`, by simple mixing first and DIIS after. Keywords: `rtol = 1e-6` and
`atol = 1e-14` (largest change of $u$ per iteration, relative and absolute), `max_iter_relax = 10`, `rtol_relax = 1e-2`,
`atol_relax = 1e-14` (switch to DIIS after `max_iter_relax` iterations of simple mixing, or earlier once the change is
below these), `max_iter = 300` (DIIS iterations after that),
`β_simplemix = 1`, `restart = 20`, `maxdim = 20`, `show_trace = false` (print the changes per iteration).

The matrix products of the DIIS mixer run on BLAS threads, so two runs can differ at the level of $10^{-14}$; with
`OPENBLAS_NUM_THREADS=1` they are reproducible.

### DIIS mixing

Pulay mixing (direct inversion in the iterative subspace,
[reference](https://www.chem.fsu.edu/~deprince/programming_projects/diis/)) for a fixed-point problem $x = F(x)$, in
`src/DIIS.jl`, which also works on its own. `MixerDIIS(n; maxdim = 20, S = ComplexF64)` stores up to `maxdim` outputs
$F(x_i)$ and residuals $r_i = F(x_i) - x_i$ of vectors of length `n` with elements of type `S`, and their overlaps. For
complex vectors, the overlaps are $\mathrm{Re} \langle r_i, r_j \rangle$: the real and imaginary parts are independent
real unknowns.

`mix!(k, x_in, x_out, mixer; β = 1, β_diis = 1, ridge = 1e-12)`, given `x_out` $= F(x_\mathrm{in})$, overwrites `x_in`
with the next input:

- History: `k` = 1, 2, … counts the calls since the last restart, and the history holds `min(k, maxdim)` pairs; once it
  is full, a new pair replaces the one with the largest residual. A restart (`k = 1`) keeps only the new pair.
- DIIS step: the coefficients $c_i$ minimize $|\sum_i c_i r_i|^2$ with $\sum_i c_i = 1$, i.e.

  $$\begin{pmatrix} G & 1 \\ 1^\mathsf{T} & 0 \end{pmatrix} \begin{pmatrix} c \\ \lambda \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \qquad G_{ij} = \mathrm{Re} \langle r_i, r_j \rangle,$$

  and the next input is $\sum_i c_i (x_i + \beta_\mathrm{diis} r_i)$; for $\beta_\mathrm{diis} = 1$ this is
  $\sum_i c_i F(x_i)$.
- Conditioning: the overlaps are scaled by the largest $|r_i|^2$, and `ridge` is added to their diagonal, which keeps
  the system solvable when the residuals are nearly linearly dependent.
- Fallback: with a single pair or a failed DIIS system, this is simple mixing
  $x_\mathrm{in} \leftarrow \beta x_\mathrm{out} + (1 - \beta) x_\mathrm{in}$.

`get_error(A, B)` returns the largest change $|a_i - b_i|$, absolute and relative to the largest $|b_i|$ ($B = 0$ gives
a large relative change, not NaN); the DIIS solver of `relax_moire` stops on it.

### Gradient check

`check_gradient(fg!, x; h = 1e-6, n = 5)` compares the gradient of `fg!(F, G, x)` (Optim's `only_fg!` convention:
returns the energy, fills `G` unless `G === nothing`) with central differences along `n` random unit directions, and
returns the largest relative error. Errors $\lesssim 10^{-6}$ indicate a correct gradient. Use a point $x$ away from
the minimum: there the gradient is $\approx 0$ and the relative error is dominated by noise.

## Precompilation

While the package is precompiled (Installation and usage), the workload in `src/Precompile.jl` runs both solvers on
both workspaces, with the usual keyword sets, and on a supercell, and the utilities, so that their compiled
code is cached with the package. A model's `potential!` is still compiled at its first call in each script, and so are
supercells with more than one field.

- The workload runs in PrecompileTools' `@compile_workload`: only while the package is precompiled (loading it without
  precompilation, e.g. `julia --compiled-modules=no`, does not run it), and it caches all the code compiled meanwhile,
  also that of other packages reached by runtime dispatch (Optim's internals), which a plain call of the workload does
  not cache. That takes the first `relax_moire` of a script from 2.5 to 0.9 s.
- `relax_moire` calls the solvers through `invokelatest`, a call that inference does not follow, so that only the
  solver in use is compiled.
- The LBFGS objective `LBFGSObjective` holds the model's `potential!` as a field of type `Any`, so that the solver is
  compiled once for all models, which lets the package precompile it.
- `Base.@nospecializeinfer` with `@nospecialize(potential!)` on `relax_moire` and its solvers: neither compilation nor
  inference specialize on the model, so that the precompiled code serves any `potential!` without being compiled again.
- Julia compiles the package again after a change of any of its files, since `include` records each included file as a
  dependency; no `include_dependency` is needed. A change of MoireRelax also recompiles the packages that depend on
  it, at their next `using`.
- During heavy editing, the workload can be switched off with the preference `precompile_workload = false` of
  PrecompileTools (e.g. in `LocalPreferences.toml`, under `[MoireRelax]`): compiling the package then takes a few
  seconds, and the solvers compile at their first calls instead.
