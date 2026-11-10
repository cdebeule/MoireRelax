# Supercells of the moiré cell: their modes, grid and rigid disregistry (for Workspace and the workspaces of other
# packages), and the transfer of solutions between cells (reference, Supercells)

"""
    supercell_wavevectors(lattice, S, N_shell) -> (m, x)

The representative wavevectors of the supercell `S` (one of each pair ±g): the integers `m` (grid indices) with
reciprocal coordinates ``x = S^{-\\mathsf{T}} m`` within `N_shell` shells of the moiré reciprocal lattice, sorted by
shell; one column each (reference, Supercells).
"""
function supercell_wavevectors(lat::Lattice, S, N_shell)
    d = S[1, 1] * S[2, 2] - S[1, 2] * S[2, 1]
    d > 0 || throw(ArgumentError("det S must be positive (the supercell vectors in a right-handed order), got $d"))
    m_max = ((abs(S[1, 1]) + abs(S[2, 1])) * N_shell, (abs(S[1, 2]) + abs(S[2, 2])) * N_shell)   # |m_i| = |(Sᵀx)_i|
    ms, ps = Tuple{Int,Int}[], Tuple{Int,Int}[]
    for m2 in -m_max[2]:m_max[2], m1 in -m_max[1]:m_max[1]
        (m1 > 0 || (m1 == 0 && m2 > 0)) || continue                       # one of each ±g
        p = (S[2, 2] * m1 - S[2, 1] * m2, S[1, 1] * m2 - S[1, 2] * m1)      # det(S) S⁻ᵀ m
        shell(lat, p) ≤ d * N_shell && (push!(ms, (m1, m2)); push!(ps, p))
    end
    order = sortperm(ps; by = p -> (shell(lat, p), p))
    return [ms[k][i] for i in 1:2, k in order], [ps[k][i] / d for i in 1:2, k in order]
end

# the modes and grid of the moiré cell (S = nothing: the modes of cell_wavevectors, in their order) or of the supercell
# S (by shell): the grid indices m of the modes, their reciprocal coordinates x (= m for the moiré cell) and the grid
# size, per direction the smallest FFT-friendly size ≥ pad (2 max|m| + 1), or N × N for N > 0
function cell_modes(lat::Lattice, N_shell, S, N, pad)
    fftsize(n) = N > 0 ? N : nextprod((2, 3, 5, 7), ceil(Int, pad * (2n + 1)))
    m, x = S === nothing ? (cell_wavevectors(lat, N_shell), cell_wavevectors(lat, N_shell)) :
           supercell_wavevectors(lat, S, N_shell)
    return m, x, (fftsize(maximum(abs, m[1, :])), fftsize(maximum(abs, m[2, :])))
end

# the array position of the wavevector with grid indices (m1, m2) on a grid of size Ns (FFT order)
grid_index(m1, m2, Ns) = CartesianIndex(mod(m1, Ns[1]) + 1, mod(m2, Ns[2]) + 1)

# the rigid disregistry ϕ0 = s1 a1 + s2 a2 on the grid of the cell S: s = S t at the grid point t (cell coordinates)
function cell_disregistry(lat::Lattice, S, Ns)
    a1, a2 = lattice_vectors(lat)
    t1, t2 = (0:Ns[1]-1) ./ Ns[1], (0:Ns[2]-1) ./ Ns[2]
    return [(S[1, 1] * s + S[1, 2] * t) * a1 + (S[2, 1] * s + S[2, 2] * t) * a2 for s in t1, t in t2]
end

"""
    transfer_modes(X, wsa, wsb) -> Y

The amplitudes X (one row per mode, any number of columns) of the workspace wsa on the modes of wsb, matched by their
reciprocal coordinates (the amplitudes at -g are the complex conjugates), and zero on the modes that wsa lacks; e.g. a
moiré-cell solution as the start of a supercell run. wsa may be anything with the field bvecs, e.g.
(; bvecs = params["bvecs"]) of a saved run.
"""
function transfer_modes(X, wsa, wsb)
    # keys: 27720x is an integer for det S ≤ 12 (27720 = lcm(1, …, 12)); beyond, the same rational gives the same double
    # (correctly rounded division), and two different modes share a key only if the lcm of the determinants exceeds 27720
    key(x) = (round(Int, 27720x[1]), round(Int, 27720x[2]))
    pos = Dict(key(x) => m for (m, x) in enumerate(eachcol(wsa.bvecs)))
    Y = zeros(ComplexF64, size(wsb.bvecs, 2), size(X, 2))
    for (m, x) in enumerate(eachcol(wsb.bvecs))
        if haskey(pos, key(x))
            Y[m, :] = X[pos[key(x)], :]
        elseif haskey(pos, key(-x))
            Y[m, :] = conj.(X[pos[key(-x)], :])
        end
    end
    return Y
end

"""
    moire_periodic(ws)

For each mode of `ws`, whether it is moiré-periodic (integer reciprocal coordinates).
"""
moire_periodic(ws) = [all(abs.(x .- round.(x)) .< 1e-9) for x in eachcol(ws.bvecs)]

"""
    supercell_fraction(c, ws)

The fraction of Σ|c|² in the modes of `ws` that are not moiré-periodic (rows of `c`: the modes), 0 for c = 0: how much
a supercell state breaks the moiré translations.
"""
function supercell_fraction(c, ws)
    total = sum(abs2, c)
    return total > 0 ? sum(abs2, c[.!moire_periodic(ws), :]) / total : 0.0
end
