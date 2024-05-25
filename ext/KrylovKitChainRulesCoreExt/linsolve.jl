function ChainRulesCore.rrule(config::RuleConfig,
                              ::typeof(linsolve),
                              f,
                              b,
                              x₀,
                              alg_primal,
                              a₀,
                              a₁; alg_rrule=alg_primal)
    (x, info) = linsolve(f, b, x₀, alg_primal, a₀, a₁)
    T, fᴴ, construct∂f = _prepare_inputs(config, f, (x,), alg_primal)

    function linsolve_pullback(X̄)
        x̄ = unthunk(X̄[1])
        ∂self = NoTangent()
        ∂x₀ = ZeroTangent()
        ∂algorithm = NoTangent()
        ∂b, reverse_info = linsolve(fᴴ, x̄, (zero(a₀) * zero(a₁)) * x̄, alg_rrule, conj(a₀),
                                    conj(a₁))
        if info.converged > 0 && reverse_info.converged == 0 && alg_rrule.verbosity >= 0
            @warn "`linsolve` cotangent problem did not converge, whereas the primal linear problem di: normres = $(reverse_info.normres)"
        end

        ∂f = construct∂f((scale(∂b, -conj(a₁)),))
        ∂a₀ = @thunk(-inner(x, ∂b))
        if a₀ == zero(a₀) && a₁ == one(a₁)
            ∂a₁ = @thunk(-inner(b, ∂b))
        else
            ∂a₁ = @thunk(-inner(add(b, x, -a₀ / a₁, +1 / a₁), ∂b))
        end
        return ∂self, ∂f, ∂b, ∂x₀, ∂algorithm, ∂a₀, ∂a₁
    end
    return (x, info), linsolve_pullback
end

# frule - currently untested

function ChainRulesCore.frule((_, ΔA, Δb, Δx₀, _, Δa₀, Δa₁)::Tuple, ::typeof(linsolve),
                              A::AbstractMatrix, b::AbstractVector, x₀, algorithm, a₀, a₁)
    (x, info) = linsolve(A, b, x₀, algorithm, a₀, a₁)

    if Δb isa ChainRulesCore.AbstractZero
        rhs = zerovector(b)
    else
        rhs = scale(Δb, (1 - Δa₁))
    end
    if !iszero(Δa₀)
        rhs = add!!(rhs, x, -Δa₀)
    end
    if !iszero(ΔA)
        rhs = mul!(rhs, ΔA, x, -a₁, true)
    end
    (Δx, forward_info) = linsolve(A, rhs, zerovector(rhs), algorithm, a₀, a₁)
    if info.converged > 0 && forward_info.converged == 0 && alg_rrule.verbosity >= 0
        @warn "The tangent linear problem did not converge, whereas the primal linear problem did."
    end
    return (x, info), (Δx, NoTangent())
end

function ChainRulesCore.frule(config::RuleConfig{>:HasForwardsMode}, tangents,
                              ::typeof(linsolve),
                              A::AbstractMatrix, b::AbstractVector, x₀, algorithm, a₀, a₁)
    return frule(tangents, linsolve, A, b, x₀, algorithm, a₀, a₁)
end

function ChainRulesCore.frule(config::RuleConfig{>:HasForwardsMode},
                              (_, Δf, Δb, Δx₀, _, Δa₀, Δa₁),
                              ::typeof(linsolve),
                              f, b, x₀, algorithm, a₀, a₁)
    (x, info) = linsolve(f, b, x₀, algorithm, a₀, a₁)

    if Δb isa AbstractZero
        rhs = zerovector(b)
    else
        rhs = scale(Δb, (1 - Δa₁))
    end
    if !iszero(Δa₀)
        rhs = add!!(rhs, x, -Δa₀)
    end
    if !(Δf isa AbstractZero)
        rhs = add!!(rhs, frule_via_ad(config, (Δf, ZeroTangent()), f, x), -a₀)
    end
    (Δx, forward_info) = linsolve(f, rhs, zerovector(rhs), algorithm, a₀, a₁)
    if info.converged > 0 && forward_info.converged == 0 && alg_rrule.verbosity >= 0
        @warn "The tangent linear problem did not converge, whereas the primal linear problem did."
    end
    return (x, info), (Δx, NoTangent())
end
