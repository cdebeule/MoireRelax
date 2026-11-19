# MoireRelax

[![CI](https://github.com/cdebeule/MoireRelax/actions/workflows/CI.yml/badge.svg)](https://github.com/cdebeule/MoireRelax/actions/workflows/CI.yml)

<div align="justify">

A Julia package for lattice relaxation in moiré materials with continuum elasticity theory. Version 0.1: the
interface may still change before version 1.0.

MoireRelax computes how the layers of a moiré material relax in plane. The layers are monolayers with a hexagonal
lattice (graphene, hBN, transition-metal dichalcogenides, …), and the package minimizes the elastic energy of the
layers plus the stacking energy of their interfaces, the interlayer energy that depends on the local stacking. It
handles any number of layers (twisted bilayers, trilayers, stacks), with or without imposing the threefold (C3)
rotation symmetry of the moiré. The relaxation is computed on one moiré cell with periodic boundary conditions, or on
a supercell of several moiré cells for relaxations that break the moiré periodicity. The displacements are expanded in
a Fourier series on the reciprocal lattice of the cell, the stacking energy is evaluated on a real-space grid, with
fast Fourier transforms (FFT) between the two, and the total energy is minimized by preconditioned LBFGS or DIIS. The
method is described in Refs. [1] and [2].

- [`docs/reference.md`](docs/reference.md) documents the conventions, the method and every function.
- [`examples/`](examples) has complete scripts for twisted bilayers and an alternating-twist trilayer
  ([Example](#example)). The package computes the displacement fields and evaluates them at any points; plotting is
  left to the user, and the bilayer examples show how to do it with CairoMakie.

## Installation

MoireRelax needs Julia 1.10 or later (tested on Julia 1.13 with Optim 2.3). In Julia:

```julia
using Pkg
Pkg.add(url = "https://github.com/cdebeule/MoireRelax")
```

To run the examples, clone the repository and install their environment, which also adds CairoMakie for the figures
(Julia 1.11 or later; on Julia 1.10, run `Pkg.develop(path = "."); Pkg.add("CairoMakie")` in the `examples`
environment instead of `instantiate`):

```sh
git clone https://github.com/cdebeule/MoireRelax
cd MoireRelax
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=examples examples/bilayer.jl
```

Start Julia with several threads (`-t auto`) to evaluate the stacking energy in parallel. The tests run with
`julia --project=. -e 'using Pkg; Pkg.test()'`.

## Example

Twisted bilayer graphene at 1°, relaxed with C3 symmetry on the moiré cell, with the graphene parameters of Carr
*et al.* [4]. Lengths are in units of the lattice constant *a* of the monolayer, and energies in meV per monolayer
unit cell (not per moiré cell).

```julia
using MoireRelax, StaticArrays

θ = 1.0                                             # twist angle (degrees)
μ, λ = 47352.0, 69518.0 - 47352.0                   # Lamé constants of graphene (meV per monolayer cell)
Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0, 0]      # stacking potential: Fourier components (meV per monolayer cell)

ws = WorkspaceC3(30, 1)                             # 30 shells, one field u = u1 - u2
moire = MoireGeometry(moire_matrix(θ), ws)          # the moiré of the twist θ
K = [2 / (λ + 2μ)  2 / μ]                           # compliances of u, for its dilation and rotation parts
C = fill(stiffness(Vg, ws), 1, 1)                   # interface stiffness, to precondition the solver
P = SVector{5}(Vg)                                  # the same as a static vector, for speed
# the stacking energy, averaged over the grid, at the local stacking ϕ0 + u (ϕ0: rigid layers), and its gradient
potential!(dVdu, u) = stacking_energy!((ϕ0, w) -> VdV_fast(ϕ0 + w, P...), dVdu, u, moire, ws)

u, E_el, E_V, converged = relax_moire(potential!, K, C, moire, ws)
D = to_grid!(ws, u, moire)                          # u_x + i u_y on a real-space grid of the moiré cell (units of a)
```

This prints the elastic and stacking energies (`E_el = 1.219` and `E_V = -2.738` meV per monolayer cell) and returns
`u`, the Fourier amplitudes of the relative displacement of the layers. The scripts in [`examples/`](examples) go
further: `bilayer.jl` relaxes a few angles and draws the relaxed moiré, `bilayer_sweep.jl` follows the Fourier
components and the energies from 2° to 0.01°, and `trilayer.jl` relaxes the three layers of an alternating-twist
trilayer. They write their figures and data to `output/`.

## References

The continuum theory implemented here is developed in

[1] C. De Beule, G. N. Pallewela, M. M. Al Ezzi, L. Peng, E. J. Mele, and S. Adam,
*Theory for lattice relaxation in marginally twisted bilayers*, [arXiv:2503.19162](https://arxiv.org/abs/2503.19162) (2025).

[2] C. De Beule, Y. Lai, L. Peng, D. Bennett, and S. Adam,
*Lattice relaxation in moiré heterobilayers*, [arXiv:2605.17805](https://arxiv.org/abs/2605.17805) (2026).

It builds on earlier continuum theories of lattice relaxation in moiré materials, in particular

[3] N. N. T. Nam and M. Koshino,
*Lattice relaxation and energy band modulation in twisted bilayer graphene*,
[Phys. Rev. B **96**, 075311](https://doi.org/10.1103/PhysRevB.96.075311) (2017).

[4] S. Carr, D. Massatt, S. B. Torrisi, P. Cazeaux, M. Luskin, and E. Kaxiras,
*Relaxation and domain formation in incommensurate two-dimensional heterostructures*,
[Phys. Rev. B **98**, 224102](https://doi.org/10.1103/PhysRevB.98.224102) (2018),
also the source of the parameters of twisted bilayer graphene in the example and in `examples/`.

[5] S. Dai, Y. Xiang, and D. J. Srolovitz,
*Twisted bilayer graphene: Moiré with a twist*,
[Nano Lett. **16**, 5923](https://doi.org/10.1021/acs.nanolett.6b02870) (2016).

[6] J. Jung, A. M. DaSilva, A. H. MacDonald, and S. Adam,
*Origin of band gaps in graphene on hexagonal boron nitride*,
[Nat. Commun. **6**, 6308](https://doi.org/10.1038/ncomms7308) (2015).

[7] V. V. Enaldiev, V. Zólyomi, C. Yelgel, S. J. Magorrian, and V. I. Fal'ko,
*Stacking domains and dislocation networks in marginally twisted bilayers of transition metal dichalcogenides*,
[Phys. Rev. Lett. **124**, 206101](https://doi.org/10.1103/PhysRevLett.124.206101) (2020).

First-principles stacking energies of graphene, hBN and graphene/hBN bilayers are given, e.g., in

[8] S. Zhou, J. Han, S. Dai, J. Sun, and D. J. Srolovitz,
*van der Waals bilayer energetics: Generalized stacking-fault energy of graphene, boron nitride, and graphene/boron
nitride bilayers*, [Phys. Rev. B **92**, 155438](https://doi.org/10.1103/PhysRevB.92.155438) (2015).

The reconstruction these theories describe has been imaged, e.g., in

[9] J. S. Alden, A. W. Tsen, P. Y. Huang, R. Hovden, L. Brown, J. Park, D. A. Muller, and P. L. McEuen,
*Strain solitons and topological defects in bilayer graphene*,
[Proc. Natl. Acad. Sci. USA **110**, 11256](https://doi.org/10.1073/pnas.1309394110) (2013).

[10] H. Yoo *et al.*,
*Atomic and electronic reconstruction at the van der Waals interface in twisted bilayer graphene*,
[Nat. Mater. **18**, 448](https://doi.org/10.1038/s41563-019-0346-z) (2019).

[11] A. Weston *et al.*,
*Atomic reconstruction in twisted bilayers of transition metal dichalcogenides*,
[Nat. Nanotechnol. **15**, 592](https://doi.org/10.1038/s41565-020-0682-9) (2020).

A paper on the package itself is in preparation; how to cite it will be added here.

## License

MIT, see [LICENSE](LICENSE).

</div>
