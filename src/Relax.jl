# relax_moire: minimization of the elastic plus stacking energy by LBFGS or DIIS (solvers: README.md)

# elastic energy mult Σ |u|²/(K L²), with K one row of compliances for all modes (1 × 2nf) or one row per mode
function elastic_energy(u, K, L, mult = 1)
    E = 0.0
    for j in axes(u, 2), i in axes(u, 1)
        E += abs2(u[i, j]) / K[size(K, 1) == 1 ? 1 : i, j]
    end
    return mult * E / L^2
end

"""
    relax_moire(potential!, K, C, moire, ws; solver = :lbfgs, u0 = ComplexF64[], label = "", kwargs...)
        -> (u, E_el, E_V, converged)

Minimizes E = E_el + E_V over the mode amplitudes u (n × 2nf, columns u_para, u_perp of each field), with
E_el = mult Σ |u|²/(K L²) and E_V = potential!(dVdu, u), the model's stacking energy ⟨V⟩, which also fills
dVdu = ∂⟨V⟩/∂(Re u) + i ∂⟨V⟩/∂(Im u) per mode unless dVdu === nothing. C is the Hessian of the harmonic interlayer
energy with respect to the fields, used to precondition both solvers. The starting point is u0 (zero if empty; it may
come from fewer shells).

  - solver = :lbfgs: rtol = 1e-6, atol = nothing, max_iter = 300, m = 10
  - solver = :diis: rtol = 1e-6, atol = 1e-14, max_iter = 300, max_iter_relax = 10, rtol_relax = 1e-2,
    atol_relax = 1e-14, β_simplemix = 1, restart = 20, maxdim = 20, verbose = false

The keywords are explained in README.md.
"""
Base.@nospecializeinfer function relax_moire(@nospecialize(potential!), K, C, moire, ws; solver::Symbol = :lbfgs,
                                             u0 = ComplexF64[], label = "", kwargs...)
    n, nf = length(moire.g), size(C, 1)
    u = zeros(ComplexF64, n, 2nf)
    n0 = min(n, size(u0, 1))
    n0 > 0 && (u[1:n0, :] .= u0[1:n0, :])
    mult = multiplicity(ws)
    # invokelatest: a call that inference does not follow, so that only the solver in use is compiled
    if solver === :lbfgs
        E_el, E_V, converged, iters = invokelatest(_relax_lbfgs!, u, potential!, K, C, moire, mult; kwargs...)::
                                      Tuple{Float64,Float64,Bool,Int}
    elseif solver === :diis
        E_el, E_V, converged, iters = invokelatest(_relax_diis!, u, potential!, K, C, moire, mult; label, kwargs...)::
                                      Tuple{Float64,Float64,Bool,Int}
    else
        error("unknown solver $solver: use :lbfgs or :diis")
    end
    converged || @warn "relax_moire did not converge" solver label
    @printf("%s   %-5s %4d iterations   E_el = %.6f   E_V = %.6f   E = %.6f\n",
            label, solver, iters, E_el, E_V, E_el + E_V)
    flush(stdout)
    return u, E_el, E_V, converged
end

### LBFGS

# LBFGS in the variables x of precond, u = S x
Base.@nospecializeinfer function _relax_lbfgs!(u, @nospecialize(potential!), K, C, moire, mult; rtol = 1e-6,
                                               atol = nothing, max_iter = 300, m = 10)
    n, ncol, L = size(u, 1), size(u, 2), moire.L
    S, Sinv = precond(K, C, L, moire; mult)
    x0 = zeros(2n * ncol)
    transform!(cplx(x0, n, ncol), Sinv, u)
    fg! = LBFGSObjective(potential!, u, similar(u), similar(u), S, K, L, mult)
    res = minimize_lbfgs(fg!, x0; rtol, atol, max_iter, m)
    transform!(u, S, cplx(Optim.minimizer(res), n, ncol))
    E_el = elastic_energy(u, K, L, mult)
    return E_el, Optim.minimum(res) - E_el, Optim.converged(res), Optim.iterations(res)
end

cplx(x, n, ncol) = reshape(reinterpret(ComplexF64, x), n, ncol)  # x for Optim: real and imaginary parts

# the energy E(x) of _relax_lbfgs! and its gradient G = ∂E/∂x, as fg!(F, G, x) for Optim (only_fg!)
struct LBFGSObjective{TU,TS,TK}
    potential!::Any # Any, so that the solver is compiled once for all models (README, Precompilation)
    u::TU
    gu::TU
    dVdu::TU
    S::TS
    K::TK
    L::Float64
    mult::Int
end

function (o::LBFGSObjective)(F, G, x)
    (; u, gu, dVdu, S, K, L, mult) = o
    n, ncol = size(u)
    transform!(u, S, cplx(x, n, ncol))
    E_V = o.potential!(G === nothing ? nothing : dVdu, u)::Float64
    if G !== nothing
        @. gu = mult * (2u / (K * L^2) + dVdu)                  # ∂E/∂(Re u) + i ∂E/∂(Im u)
        transform!(cplx(G, n, ncol), S, gu; transpose = true)   # ∂E/∂x = Sᵀ ∂E/∂u
    end
    return elastic_energy(u, K, L, mult) + E_V
end

# LBFGS in variables x scaled such that the Hessian is ≈ 1, with a gradient tolerance relative to the solution (README)
function minimize_lbfgs(fg!, x0; rtol = 1e-6, atol = nothing, max_iter = 300, m = 10, kwargs...)
    G0 = similar(x0)
    F0 = fg!(0.0, G0, x0)
    g_tol = something(atol, rtol * max(maximum(abs, x0 .- G0), eps()))
    fresh = Ref(true) # the first call of optimize, at x0, is answered from F0, G0
    fg0!(F, G, x) = fresh[] && (fresh[] = false; x == x0) ? (G === nothing || copyto!(G, G0); F0) : fg!(F, G, x)
    d = OnceDifferentiable(only_fg!(fg0!), x0, 0.0)
    return optimize(d, x0, LBFGS(; m), Optim.Options(; g_tol, iterations = max_iter, kwargs...))
end

### DIIS

# damped fixed-point iteration u ← u + P (u_next - u), u_next = -(K L²/2) dVdu: simple mixing first, DIIS after
Base.@nospecializeinfer function _relax_diis!(u, @nospecialize(potential!), K, C, moire, mult; rtol = 1e-6,
                                              atol = 1e-14, max_iter = 300, max_iter_relax = 10, rtol_relax = 1e-2,
                                              atol_relax = 1e-14, β_simplemix = 1.0, restart = 20, maxdim = 20,
                                              verbose = false, label = "")
    n, nf, L = size(u, 1), size(C, 1), moire.L
    P = damping!(zeros(nf, nf, n, 2), K, C, L, moire)
    mixer = MixerDIIS(length(u); maxdim)
    u_next, dVdu = similar(u), similar(u)
    E_V, converged, diis, k, iters = 0.0, false, false, 0, 0
    for iter in 1:(max_iter_relax + max_iter)
        iters = iter
        E_V = potential!(dVdu, u)::Float64
        @. u_next = -K * L^2 / 2 * dVdu                     # fixed point u = u_next ⟺ ∂E/∂u = 0
        max_atol, max_rtol = get_error(u, u_next)
        isfinite(max_atol) || error("DIIS diverged at iteration $iter ($label)")
        verbose && @printf("    iteration %4d   atol = %.4e   rtol = %.4e\n", iter, max_atol, max_rtol)
        converged = max_atol <= atol || max_rtol <= rtol
        converged && break
        diis = diis || iter > max_iter_relax || max_atol <= atol_relax || max_rtol <= rtol_relax
        if diis
            # DIIS (Pulay) mixing of the damped update
            k += 1
            damped_step!(u_next, u, u_next, P, 1)
            mix!(mod1(k, restart), vec(u), vec(u_next), mixer; β = β_simplemix)
        else
            # simple mixing of the damped update (brings a poor starting point close enough for DIIS)
            damped_step!(u, u, u_next, P, β_simplemix)
        end
    end
    converged || (E_V = potential!(nothing, u)::Float64)    # the last evaluation was before the last update
    return elastic_energy(u, K, L, mult), E_V, converged, iters
end

### Checks

"""
    check_gradient(fg!, x; h = 1e-6, n = 5) -> largest relative error

Compares the gradient of `fg!(F, G, x)` (Optim's only_fg! convention) with central differences along `n` random unit
directions. Errors ≲ 1e-6 indicate a correct gradient; use a point `x` away from the minimum (README).
"""
function check_gradient(fg!, x; h = 1e-6, n = 5)
    G = similar(x)
    fg!(1.0, G, x)
    err = 0.0
    for _ in 1:n
        d = normalize(randn(length(x)))
        fd = (fg!(1.0, nothing, x .+ h .* d) - fg!(1.0, nothing, x .- h .* d)) / 2h
        ad = dot(G, d)
        err = max(err, abs(fd - ad) / max(abs(fd), abs(ad), eps()))
    end
    return err
end
