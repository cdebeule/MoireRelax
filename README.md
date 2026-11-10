# MoireRelax

Lattice relaxation in moiré materials with continuum elasticity theory.

- **`julia/MoireCore`** — a Julia package that relaxes the in-plane displacements of the layers of a moiré: it
  minimizes the elastic energy of the layers plus the stacking energy of their interfaces, for any number of layers
  (twisted bilayers, trilayers, stacks), with or without C3 symmetry, on the moiré cell or on a supercell of it. The
  displacements are expanded in the moiré reciprocal vectors, the stacking energy is evaluated on a real-space grid
  (FFTs), and the energy is minimized by preconditioned LBFGS or DIIS. Its [README](julia/MoireCore/README.md)
  documents the conventions, the method and every function; `examples/` has a twisted bilayer and a trilayer.
- **`LAMMPS/`** — LAMMPS [3] input script and interatomic potentials used in Ref. [1].

## Installation

MoireCore needs Julia 1.10 or later (tested on Julia 1.13 with Optim 2.3). In Julia:

```julia
using Pkg
Pkg.add(url = "https://github.com/cdebeule/MoireRelax", subdir = "julia/MoireCore")
```

Or clone the repository and run the examples in the package's own environment:

```sh
git clone https://github.com/cdebeule/MoireRelax
cd MoireRelax/julia/MoireCore
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=. examples/bilayer.jl
```

## Example

Twisted bilayer graphene at 1°, relaxed with C3 symmetry:

```julia
using MoireCore, StaticArrays

θ = 1.0                                             # twist angle (degrees)
μ, λ = 47352.0, 69518.0 - 47352.0                   # Lamé constants of graphene (meV per cell)
Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0, 0]      # stacking potential [c1, c2, c3, p1, p3] (meV per cell)

ws = WorkspaceC3(30, 1)                             # 30 shells of moiré harmonics, one field u = u1 - u2
moire = MoireGeometry(moire_matrix(θ), ws)
K = [2 / (λ + 2μ)  2 / μ]                           # compliances of u (dilation, rotation)
C = fill(stiffness(Vg, ws), 1, 1)                   # interface stiffness, to precondition the solver
P = SVector{5}(Vg)
potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)

u, E_el, E_V, converged = relax_moire(potential!, K, C, moire, ws)
D = to_grid!(ws, u, moire)                          # u_x + i u_y on the grid of the moiré cell (units of a)
```

This prints the elastic and stacking energies per cell (`E_el = 1.219`, `E_V = -2.738` meV) and returns the mode
amplitudes `u`; `examples/bilayer.jl` sweeps the angle and `examples/trilayer.jl` does the same for three layers.

## Citation

If you use this code, please cite

[1] C. De Beule, G. N. Pallewela, M. M. Al Ezzi, L. Peng, E. J. Mele, and S. Adam,
*Theory for lattice relaxation in marginally twisted bilayers*, [arXiv:2503.19162](https://arxiv.org/abs/2503.19162) (2025).

[2] C. De Beule, Y. Lai, L. Peng, D. Bennett, and S. Adam,
*Lattice relaxation in moiré heterobilayers*, [arXiv:2605.17805](https://arxiv.org/abs/2605.17805) (2026).

[3] [LAMMPS Molecular Dynamics Simulator](https://www.lammps.org), for the files in `LAMMPS/`.

## License

MIT, see [LICENSE](LICENSE).
