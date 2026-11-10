# Harmonic model of the interlayer coupling, used to precondition the solvers (derivation: reference)

"""
    coupling_laplacian(nlayers, couplings)

The Hessian `C` of the harmonic interlayer energy with respect to the layer displacements, the weighted Laplacian of
the graph of layers: each coupling `(i, j, k)` adds k |u_i - u_j|²/2 to the energy, i.e. k to `C[i, i]` and `C[j, j]`
and -k to `C[i, j]` and `C[j, i]` (reference, Harmonic model).
"""
function coupling_laplacian(nlayers::Integer, couplings)
    C = zeros(nlayers, nlayers)
    for (i, j, k) in couplings
        C[i, i] += k
        C[j, j] += k
        C[i, j] -= k
        C[j, i] -= k
    end
    return C
end

"""
    chain_laplacian(k_nn, k_nnn = Float64[]; nnn_start)

`coupling_laplacian` of a stack of `length(k_nn) + 1` layers: `k_nn[l]` couples the layers l and l + 1, and `k_nnn[j]`
the next-nearest pair (s + j - 1, s + j + 1) with s = `nnn_start`, the lower layer of the first such pair (required
when `k_nnn` is not empty).
"""
function chain_laplacian(k_nn, k_nnn = Float64[]; nnn_start = nothing)
    nlayers = length(k_nn) + 1
    isempty(k_nnn) || nnn_start !== nothing ||
        throw(ArgumentError("chain_laplacian: give nnn_start, the lower layer of the first next-nearest pair"))
    s = something(nnn_start, 1)
    isempty(k_nnn) || s + length(k_nnn) + 1 <= nlayers ||
        throw(ArgumentError("chain_laplacian: next-nearest pairs beyond layer $nlayers"))
    pairs = [[(l, l + 1, k) for (l, k) in enumerate(k_nn)]; [(s + j - 1, s + j + 1, k) for (j, k) in enumerate(k_nnn)]]
    return coupling_laplacian(nlayers, pairs)
end

"""
    coupling_modes(K, C, t) -> (sK, Λ, Q)

The coupling modes of the column type `t` (1: dilation, 2: rotation): `sK` = √K_t, the square roots of the compliances
of the fields in the row `K`, and the eigendecomposition `sK .* C .* sK'` = Q Λ Qᵀ of the harmonic Hessian `C` in the
scaled variables (reference, Harmonic model).
"""
function coupling_modes(K, C, t)
    sK = sqrt.(K[t:2:end])
    Λ, Q = eigen(Symmetric(sK .* C .* sK'))
    return sK, Λ, Q
end

# coupling modes of column type t for every mode: one decomposition if K is one row for all modes, else one per mode
coupling_modes_per_g(K, C, t, n) =
    size(K, 1) == 1 ? [coupling_modes(vec(K), C, t)] : [coupling_modes(K[m, :], C, t) for m in 1:n]

"""
    damping!(P, K, C, L, moire) -> P

Fills the damping matrices `P[:, :, m, t]` = (I + L²/|g_m|² diag(K_t) C)⁻¹ (nf × nf × n × 2) of the DIIS solver, for
which the damped step u + P (u_next - u) of the fixed point u_next = -(K L²/2) dVdu is the Newton step for harmonic
interfaces (reference, Harmonic model).
"""
function damping!(P, K, C, L, moire)
    nf = size(C, 1)
    for t in 1:2
        modes = coupling_modes_per_g(K, C, t, length(moire.invdotgg))
        for m in eachindex(moire.invdotgg)
            sK, Λ, Q = modes[min(m, end)]
            V, Vinv = sK .* Q, Q' ./ sK'
            for b in 1:nf, a in 1:nf
                P[a, b, m, t] = sum(V[a, i] * Vinv[i, b] / (1 + L^2 * moire.invdotgg[m] * Λ[i]) for i in 1:nf)
            end
        end
    end
    return P
end

"""
    damped_step!(out, u, G, P, β) -> out

The damped step `out` = u + β P (G - u) for each mode and column type, with the damping matrices `P` of `damping!`;
`out` may be `u` or `G`.
"""
function damped_step!(out, u, G, P, β)
    nf = size(P, 1)
    r = zeros(eltype(u), nf)
    for t in 1:2, m in axes(u, 1)
        for b in 1:nf
            r[b] = G[m, 2b - 2 + t] - u[m, 2b - 2 + t]
        end
        for a in 1:nf
            out[m, 2a - 2 + t] = u[m, 2a - 2 + t] + β * sum(P[a, b, m, t] * r[b] for b in 1:nf)
        end
    end
    return out
end

"""
    precond(K, C, L, moire; mult = 1) -> (S, Sinv)

The change of variables u = S x of the LBFGS solver, such that the Hessian in x is the identity for harmonic
interfaces: `S[f, i, m, t]` for each mode m and column type t, and its inverse `Sinv[i, f, m, t]`; `mult` is the number
of modes that share each amplitude (reference, Harmonic model).
"""
function precond(K, C, L, moire; mult = 1)
    nf, n = size(C, 1), length(moire.invdotgg)
    S, Sinv = zeros(nf, nf, n, 2), zeros(nf, nf, n, 2)
    for t in 1:2
        modes = coupling_modes_per_g(K, C, t, n)
        for m in 1:n, i in 1:nf
            sK, Λ, Q = modes[min(m, end)]
            d = sqrt(mult * (2 / L^2 + 2Λ[i] * moire.invdotgg[m]))
            for f in 1:nf
                S[f, i, m, t] = sK[f] * Q[f, i] / d
                Sinv[i, f, m, t] = d * Q[f, i] / sK[f]
            end
        end
    end
    return S, Sinv
end

"""
    transform!(out, S, x; transpose = false) -> out

`out[m, 2f - 2 + t]` = Σ_i `S[f, i, m, t] x[m, 2i - 2 + t]` for the matrices `S` of `precond`, or with Sᵀ for
`transpose = true` (gradients: ∂E/∂x = Sᵀ ∂E/∂u); `out` must not alias `x`.
"""
transform!(out, S, x; transpose::Bool = false) = _transform!(out, S, x, Val(transpose))

function _transform!(out, S, x, ::Val{tr}) where {tr} # Val(transpose) makes the choice at compile time
    nf = size(S, 1)
    @inbounds for t in 1:2, m in axes(x, 1), f in 1:nf
        acc = zero(eltype(out))
        for i in 1:nf
            acc += (tr ? S[i, f, m, t] : S[f, i, m, t]) * x[m, 2i - 2 + t]
        end
        out[m, 2f - 2 + t] = acc
    end
    return out
end
