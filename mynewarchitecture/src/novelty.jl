
# novelty.jl — Controlled candidate-idea generation operators (spec §27).
#
# Novelty is a GENERATION property; correctness is decided later by evaluation
# (spec §27, §112). Nothing here mutates real memory (counterfactuals included,
# spec §30). Each operator emits its own typed trace event.

const _INVERSE_PREDICATE = Dict(:causes=>:caused_by, :requires=>:required_by,
                                :supports=>:supported_by, :part_of=>:has_part,
                                :enables=>:enabled_by, :prevents=>:prevented_by,
                                :affects=>:affected_by)

_shared_domain(agent, a::Int32, b::Int32) = !isempty(intersect(concept_domains(agent,a), concept_domains(agent,b)))
_different_domain(agent, a::Int32, b::Int32) = !_shared_domain(agent, a, b)

# 1. Direct retrieval / deduction ──────────────────────────────────────────
function gen_direct(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for r in agent.semantic.relations
        (r.subject in relevant || r.object in relevant) || continue
        push!(out, new_hypothesis(agent, r; supporting=[r.subject, r.object],
              assumptions=["direct stored relation"], strategy=:deductive, origin=:direct))
    end
    return out
end

# 2. Transitive causal deduction ────────────────────────────────────────────
function gen_transitive(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for r1 in agent.semantic.relations, r2 in agent.semantic.relations
        (r1.predicate == :causes && r2.predicate == :causes && r1.object == r2.subject) || continue
        (r1.subject in relevant || r2.object in relevant) || continue
        claim = CognitiveRelation(r1.subject, :causes, r2.object, :deduced)
        push!(out, new_hypothesis(agent, claim; supporting=[r1.subject, r1.object, r2.object],
              assumptions=["causal transitivity"],
              consequences=["$(concept_name(agent,r1.subject)) may indirectly cause $(concept_name(agent,r2.object))"],
              strategy=:deductive, origin=:deduction))
    end
    return out
end

# 3. Concept recombination (cross-domain when domains differ) ────────────────
function gen_recombination(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    causes = [r for r in agent.semantic.relations if r.predicate == :causes && r.subject in relevant]
    supports = [r for r in agent.semantic.relations if r.predicate == :supports && r.subject in relevant]
    for c in causes, s in supports
        c.object == s.object && continue
        claim = CognitiveRelation(c.subject, :supports, s.object, :recombined)
        cross = _different_domain(agent, c.subject, s.object)
        emit!(state.trace, cross ? CROSS_DOMAIN_RECOMBINATION : HYPOTHESIS_CREATED;
              source_a=c.subject, source_b=s.object,
              shared="directed influence", candidate="$(concept_name(agent,c.subject)) supports $(concept_name(agent,s.object))")
        push!(out, new_hypothesis(agent, claim; supporting=[c.subject, c.object, s.object],
              assumptions=[cross ? "cross-domain recombination" : "concept recombination"],
              consequences=["new relational candidate"],
              strategy=:recombination, origin=cross ? :cross_domain : :recombination))
    end
    return out
end

# 4. Analogical transfer A:B :: C:? (spec §29) ──────────────────────────────
function gen_analogy(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for base in agent.semantic.relations
        base.subject in relevant && continue          # base should be the known pair
        for c in relevant
            c == base.subject && continue
            # If C shares a domain or a relation-type with base.subject, transfer base.predicate.
            (_shared_domain(agent, c, base.subject) || relation_exists(agent, c, base.predicate, base.object)) || continue
            relation_exists(agent, c, base.predicate, base.object) && continue
            claim = CognitiveRelation(c, base.predicate, base.object, :analogy)
            emit!(state.trace, ANALOGY_FORMED; a=base.subject, b=base.object, c=c,
                  relation=base.predicate)
            push!(out, new_hypothesis(agent, claim; supporting=[c, base.subject, base.object],
                  assumptions=["structural analogy $(concept_name(agent,base.subject)):$(concept_name(agent,base.object)) :: $(concept_name(agent,c)):?"],
                  strategy=:analogy, origin=:analogy))
        end
    end
    return out
end

# 5. Abstraction from shared structure (spec §64) ───────────────────────────
function gen_abstraction(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    by_object = Dict{Tuple{Int32,Symbol},Vector{Int32}}()
    for r in agent.semantic.relations
        push!(get!(by_object, (r.object, r.predicate), Int32[]), r.subject)
    end
    for ((obj, pred), subjects) in by_object
        length(subjects) < 2 && continue
        any(s -> s in relevant, subjects) || obj in relevant || continue
        a, b = subjects[1], subjects[2]
        claim = CognitiveRelation(a, :similar_to, b, :deduced)
        emit!(state.trace, ABSTRACTION_FORMED; shared_predicate=pred, shared_object=obj, members=length(subjects))
        push!(out, new_hypothesis(agent, claim; supporting=[a, b, obj],
              assumptions=["abstraction: both $pred $(concept_name(agent,obj))"],
              strategy=:abstraction, origin=:abstraction))
    end
    return out
end

# 6. Counterfactual (does not mutate memory, spec §30) ──────────────────────
function gen_counterfactual(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for r in agent.semantic.relations
        (r.predicate == :causes && (r.subject in relevant || r.object in relevant)) || continue
        claim = CognitiveRelation(r.subject, :prevents, r.object, :counterfactual)
        emit!(state.trace, COUNTERFACTUAL_EXPLORED; subject=r.subject, object=r.object,
              question="what if $(concept_name(agent,r.subject)) were absent")
        push!(out, new_hypothesis(agent, claim; supporting=[r.subject, r.object], opposing=[r.subject, r.object],
              assumptions=["counterfactual: original cause removed"],
              consequences=["$(concept_name(agent,r.object)) may not occur"],
              strategy=:counterfactual, origin=:counterfactual))
    end
    return out
end

# 7. Inverse relation ───────────────────────────────────────────────────────
function gen_inverse(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for r in agent.semantic.relations
        (r.subject in relevant || r.object in relevant) || continue
        inv = get(_INVERSE_PREDICATE, r.predicate, nothing)
        inv === nothing && continue
        claim = CognitiveRelation(r.object, inv, r.subject, :recombined)
        push!(out, new_hypothesis(agent, claim; supporting=[r.subject, r.object],
              assumptions=["inverse of $(r.predicate)"], strategy=:recombination, origin=:inverse))
    end
    return out
end

# 8. Causal reversal (deliberately testable; often contradicted) ────────────
function gen_causal_reversal(agent::CognitiveAgent, state::CognitiveState, relevant::Set{Int32})
    out = CognitiveHypothesis[]
    for r in agent.semantic.relations
        (r.predicate == :causes && (r.subject in relevant || r.object in relevant)) || continue
        claim = CognitiveRelation(r.object, :causes, r.subject, :recombined)
        push!(out, new_hypothesis(agent, claim; supporting=[r.subject, r.object],
              assumptions=["reversed causal direction"], strategy=:counterfactual, origin=:causal_reversal))
    end
    return out
end

const GENERATION_OPERATORS = [gen_direct, gen_transitive, gen_recombination, gen_analogy,
                              gen_abstraction, gen_counterfactual, gen_inverse, gen_causal_reversal]

"""
Run all generation operators under the reasoning budget (spec §35). Novel
operators are gated by `max_recombinations`; total hypotheses by `max_hypotheses`
and `hypothesis_limit`. Deduplicates by hypothesis signature.
"""
function generate_hypotheses!(agent::CognitiveAgent, state::CognitiveState)
    relevant = Set(vcat(state.concepts, state.retrieved_concepts))
    budget = agent.config.budget
    limit = min(agent.config.hypothesis_limit, budget.max_hypotheses)
    seen = Set{UInt64}()
    for op in GENERATION_OPERATORS
        length(state.hypotheses) >= limit && break
        novel = op !== gen_direct && op !== gen_transitive
        if novel && state.recombinations_used >= budget.max_recombinations
            continue
        end
        produced = op(agent, state, relevant)
        for h in produced
            length(state.hypotheses) >= limit && break
            sig = hypothesis_signature(h)
            sig in seen && continue
            push!(seen, sig)
            novel && (state.recombinations_used += 1)
            push!(state.hypotheses, h)
            state.hypotheses_used += 1
            emit!(state.trace, HYPOTHESIS_CREATED; hypothesis=h.id, origin=h.origin, strategy=h.strategy,
                  claim="$(concept_name(agent,h.claim.subject)) $(h.claim.predicate) $(concept_name(agent,h.claim.object))")
            push!(state.strategy_sequence, h.strategy)
        end
    end
    return state.hypotheses
end

