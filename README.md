# MoireRelax

[![CI](https://github.com/cdebeule/MoireRelax/actions/workflows/CI.yml/badge.svg)](https://github.com/cdebeule/MoireRelax/actions/workflows/CI.yml)

<div align="justify">

A Julia package for lattice relaxation in moiré materials with continuum elasticity theory. Version 0.1: the
interface may still change before version 1.0.

MoireRelax relaxes the in-plane displacements of the layers of a moiré of monolayers with hexagonal symmetry
(graphene, hBN, transition-metal dichalcogenides, …): it minimizes the elastic energy of the layers plus the stacking
energy of their interfaces, for any number of layers (twisted bilayers, trilayers, stacks), with or without C3
symmetry. The relaxation is computed on one moiré cell with periodic boundary conditions, or on a supercell of several
moiré cells for relaxations that break the moiré periodicity: the displacements are expanded in the reciprocal vectors
of the cell, the stacking energy is evaluated on a real-space grid of the cell using the fast Fourier transform (FFT),
and the energy is minimized by preconditioned LBFGS or DIIS. The method is described in Refs. [1] and [2].

- [`docs/reference.md`](docs/reference.md) documents the conventions, the method and every function.
- [`examples/`](examples) has a twisted bilayer at a few angles with a figure of the relaxed moiré, the same swept
  from 2° to 0.01° with the Fourier components of the relaxation and the energies against the angle, and an
  alternating-twist trilayer. The package computes the fields and brings them to any points; the figures are made by
  the user, and the bilayer examples show how, with CairoMakie.

## Installation

MoireRelax needs Julia 1.10 or later (tested on Julia 1.13 with Optim 2.3). In Julia:

```julia
using Pkg
Pkg.add(url = "https://github.com/cdebeule/MoireRelax")
```

Or clone the repository and run the examples in their environment, which adds CairoMakie for the figures (Julia 1.11
or later for `instantiate`; on 1.10, `Pkg.develop(path = "."); Pkg.add("CairoMakie")` in that environment instead):

```sh
git clone https://github.com/cdebeule/MoireRelax
cd MoireRelax
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
julia -t auto --project=examples examples/bilayer.jl
```

Start Julia with threads (`-t auto`): the grid points of the stacking energy are shared out between them.
The tests run with `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Example

Twisted bilayer graphene at 1°, relaxed with C3 symmetry on the moiré cell, with the graphene parameters of Carr
*et al.* [4]:

```julia
using MoireRelax, StaticArrays

θ = 1.0                                             # twist angle (degrees)
μ, λ = 47352.0, 69518.0 - 47352.0                   # Lamé constants of graphene (meV per cell) [4]
Vg = [4.064 / 2, -0.374 / 2, -0.095 / 2, 0, 0]      # stacking potential [c1, c2, c3, p1, p3] (meV per cell) [4]

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
amplitudes `u`. The examples go further: `bilayer.jl` relaxes a few angles and draws the relaxed moiré,
`bilayer_sweep.jl` follows the Fourier components and the energies from 2° to 0.01°, and `trilayer.jl` relaxes the
three layers of an alternating-twist trilayer; they write their figures and data to `output/`.

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
