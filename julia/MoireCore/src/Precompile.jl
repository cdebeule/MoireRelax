# Precompilation workload: runs only while the package is precompiled (README.md, Precompilation)

# runs both solvers on both workspaces, with the keyword sets of the scripts, and on a supercell; and the utilities
function precompile_workload()
    Vg = [1.0, 0.0, 0.0, 0.5, 0.0]
    P = SVector{5,Float64}(Vg)
    # local stacking energy V(ϕ0 + u_f) of each field, for one and two fields
    V1(ϕ0, w) = VdV_fast(ϕ0 + w, P...)
    function V2(ϕ0, w1, w2)
        (v1, d1), (v2, d2) = V1(ϕ0, w1), V1(ϕ0, w2)
        return v1 + v2, d1, d2
    end
    for ws in (Workspace(3, 1), WorkspaceC3(3, 1), Workspace(3, 2), WorkspaceC3(3, 2))
        nf, k = ws.nf, stiffness(Vg, ws)
        K, C = fill(1e-4, 1, 2nf), Matrix(k * I, nf, nf)
        moire = MoireGeometry(moire_matrix(5.0, 0.01), ws)
        pot!(dVdu, u) = stacking_energy!(nf == 1 ? V1 : V2, dVdu, u, moire, ws)
        redirect_stdout(devnull) do
            for solver in (:lbfgs, :diis)
                u, = relax_moire(pot!, K, C, moire, ws; solver, label = "θ")
                relax_moire(pot!, K, C, moire, ws; solver, u0 = extrapolate(1.0, [2.0, 3.0], [u, u]), label = "θ")
            end
            for opts in ((; rtol = 1e-8, max_iter = 10, m = 20), (; atol = 1e-9, max_iter = 10),
                         (; max_iter = 10, m = 20))
                relax_moire(pot!, K, C, moire, ws; solver = :lbfgs, u0 = ComplexF64[], label = "θ", opts...)
            end
            print_params("MoireCore", [("a", 1.0, "b"); ("c", "d"); ("e", 1); coef_rows(Vg, " (x)")])
            print_params("MoireCore", Any[("a", 1.0, "b"), ("c", "d"), ("e", 1)])
        end
    end
    # a supercell (one field, default keywords; other cases compile at their first use): a moiré-cell solution as the
    # start, the indicators, saved modes, fields
    ws1, wsS = Workspace(2, 1), Workspace(2, 1; S = [2 0; 0 1])
    K, C = fill(1e-4, 1, 2), fill(stiffness(Vg, wsS), 1, 1)
    moireS = MoireGeometry(moire_matrix(5.0, 0.01), wsS)
    potS!(dVdu, u) = stacking_energy!(V1, dVdu, u, moireS, wsS)
    u = transfer_modes(randn(ComplexF64, length(ws1.kp), 2), ws1, wsS)
    redirect_stdout(devnull) do
        for solver in (:lbfgs, :diis)
            relax_moire(potS!, K, C, moireS, wsS; solver, u0 = extrapolate(1.0, [2.0], [u]), label = "θ")
        end
    end
    supercell_fraction(u, wsS), supercell_fraction(u[:, 1], wsS), transfer_modes(u, (; bvecs = wsS.bvecs), ws1)
    moire = MoireGeometry(moire_matrix(5.0, 0.01), Float64.(ws1.bvecs))
    field_synthesizer(moire, range(-0.5, 0.5, 5), range(-0.5, 0.5, 5))(randn(ComplexF64, length(moire.g)))
    chain_laplacian([1.0, 2.0, 3.0], [0.5]; nnn_start = 2)
    twist_angle(moire_matrix(1.0))
    return nothing
end

# runs only while the package is precompiled, and caches all the code compiled meanwhile, also that of other packages
# reached by runtime dispatch (Optim's internals)
@compile_workload precompile_workload()
