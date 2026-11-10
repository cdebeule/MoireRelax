# DIIS.jl: Pulay mixing for fixed-point problems x = F(x) (README). Included by MoireCore.jl; also works on its own.

using LinearAlgebra

"""
    MixerDIIS(n; maxdim = 20, S = ComplexF64)

Storage for `mix!` (Pulay mixing): up to `maxdim` outputs and residuals of vectors of length `n` with elements of type
`S`, and their overlaps Re⟨rᵢ, rⱼ⟩ (README).
"""
struct MixerDIIS{T<:Real,S<:Number}
    X::Matrix{S}                            # stored outputs F(xᵢ)
    R::Matrix{S}                            # their residuals rᵢ = F(xᵢ) - xᵢ
    G::Matrix{T}                            # overlaps G[i, j] = Re⟨rᵢ, rⱼ⟩
    err::Vector{T}                          # |rᵢ|
end

MixerDIIS(n::Integer; maxdim::Integer = 20, S::Type{<:Number} = ComplexF64) =
    MixerDIIS(zeros(S, n, maxdim), zeros(S, n, maxdim), zeros(real(S), maxdim, maxdim), zeros(real(S), maxdim))

"""
    mix!(k, x_in, x_out, mixer; β = 1, β_diis = 1, ridge = 1e-12)

DIIS step for x = F(x): given `x_out = F(x_in)`, overwrites `x_in` with the next input, built from the stored outputs
F(xᵢ) and residuals rᵢ = F(xᵢ) - xᵢ; `k` = 1, 2, … counts the calls since the last restart (README).
"""
@views function mix!(k::Integer, x_in::AbstractVector, x_out::AbstractVector, mixer::MixerDIIS{T};
                     β::Real = 1, β_diis::Real = 1, ridge::Real = 1e-12) where {T}
    (; X, R, G, err) = mixer
    maxdim = size(X, 2)
    i = k <= maxdim ? k : argmax(err)           # slot of the new pair
    @. X[:, i] = x_out
    @. R[:, i] = x_out - x_in
    m = min(k, maxdim)                          # size of the history
    G[1:m, i] .= real.(R[:, 1:m]' * R[:, i])
    G[i, 1:m] .= G[1:m, i]
    err[i] = sqrt(G[i, i])

    # minimize cᵀ G c with Σ c = 1:  [G 1; 1ᵀ 0] [c; λ] = [0; 1]
    c = nothing
    if m > 1
        A = [G[1:m, 1:m] / maximum(G[j, j] for j in 1:m) + ridge * I  ones(T, m); ones(T, 1, m)  zero(T)]
        c = try A \ [zeros(T, m); one(T)] catch; nothing end
    end
    if isnothing(c) || !all(isfinite, c)
        @. x_in = β * x_out + (1 - β) * x_in                            # simple mixing
    else
        mul!(x_in, X[:, 1:m], c[1:m])                                   # Σ cᵢ F(xᵢ)
        β_diis == 1 || mul!(x_in, R[:, 1:m], c[1:m], β_diis - 1, true)  # - (1 - β_diis) Σ cᵢ rᵢ
    end
    return x_in
end

# largest change |aᵢ - bᵢ|, absolute and relative to the largest |bᵢ| (B = 0 gives a large relative change, not NaN)
function get_error(A::AbstractArray, B::AbstractArray)
    d2 = maximum(i -> abs2(A[i] - B[i]), eachindex(A, B))
    return sqrt(d2), sqrt(d2 / max(maximum(abs2, B), floatmin(Float64)))
end