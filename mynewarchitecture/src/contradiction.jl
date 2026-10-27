
# contradiction.jl — Explicit contradiction detection and a recurring registry (spec §31).

"""Two relations conflict when they assert opposing predicates over the same pair."""
function relations_conflict(a::CognitiveRelation, b::CognitiveRelation)
    a.subject == b.subject && a.object == b.object || return false
    (a.predicate == :causes  && b.predicate in (:prevents, :contradicts, :inhibits, :blocks)) && return true
    (b.predicate == :causes  && a.predicate in (:prevents, :contradicts, :inhibits, :blocks)) && return true
    (a.predicate == :supports && b.predicate in (:contradicts, :opposite_of))                 && return true
    (b.predicate == :supports && a.predicate in (:contradicts, :opposite_of))                 && return true
    (a.predicate == :opposite_of && b.predicate == :similar_to)                               && return true
    (b.predicate == :opposite_of && a.predicate == :similar_to)                               && return true
    return false
end

find_conflicts(agent::CognitiveAgent, claim::CognitiveRelation) =
    [r for r in agent.semantic.relations if relations_conflict(claim, r)]

function register_contradiction!(agent::CognitiveAgent, claim::CognitiveRelation, other::CognitiveRelation, desc::String)
    reg = agent.contradictions
    for rec in reg.records
        if rec.subject == claim.subject && rec.object == claim.object &&
           rec.predicate_a == claim.predicate && rec.predicate_b == other.predicate
            rec.occurrences += 1
            return rec
        end
    end
    rec = ContradictionRecord(claim.subject, claim.object, claim.predicate, other.predicate, 1, desc)
    push!(reg.records, rec)
    return rec
end

"""Test hypotheses against semantic conflicts, learned ternary state, and experience."""
function test_hypotheses!(agent::CognitiveAgent, state::CognitiveState)
    for h in state.hypotheses
        conflicts = find_conflicts(agent, h.claim)
        hard_conflicts = [c for c in conflicts if synaptic_support(agent, c) > 0f0]
        contested_conflicts = [c for c in conflicts if synaptic_support(agent, c) == 0f0]
        learned_opposition = synaptic_support(agent, h.claim) < 0f0
        hard_count = length(hard_conflicts) + (learned_opposition ? 1 : 0)
        score_hypothesis!(agent, h, state; contradiction_count=hard_count,
                          contested_count=length(contested_conflicts))

        if hard_count > 0
            h.status = :contradicted
            h.confidence = 0f0
            if learned_opposition
                desc = "learned ternary pathway opposes $(concept_name(agent,h.claim.subject)) $(h.claim.predicate) $(concept_name(agent,h.claim.object))"
                push!(state.contradictions, desc)
                emit!(state.trace, CONTRADICTION_FOUND; hypothesis=h.id, detail=desc,
                      synaptic_support=h.scores.synaptic_support)
            end
            for conflict in hard_conflicts
                desc = "$(concept_name(agent,h.claim.subject)) $(h.claim.predicate) $(concept_name(agent,h.claim.object)) conflicts with learned $(conflict.predicate)"
                register_contradiction!(agent, h.claim, conflict, desc)
                push!(state.contradictions, desc)
                emit!(state.trace, CONTRADICTION_FOUND; hypothesis=h.id, detail=desc,
                      conflicting_predicate=conflict.predicate)
            end
        elseif direct_support(agent, h.claim) && h.scores.synaptic_support > 0f0
            h.status = :supported
            h.confidence = hypothesis_confidence(h.scores, 0)
            emit!(state.trace, HYPOTHESIS_SUPPORTED; hypothesis=h.id,
                  synaptic_support=h.scores.synaptic_support)
        else
            (transitive, _) = transitive_causal_support(agent, h.claim)
            if transitive && h.scores.synaptic_support > 0f0
                h.status = :supported
                h.confidence = hypothesis_confidence(h.scores, 0)
                emit!(state.trace, HYPOTHESIS_SUPPORTED; hypothesis=h.id,
                      synaptic_support=h.scores.synaptic_support)
            elseif direct_support(agent, h.claim) || transitive || !isempty(contested_conflicts) ||
                   h.origin in (:counterfactual, :causal_reversal)
                h.status = :weak
                h.confidence = hypothesis_confidence(h.scores, 0) * 0.75f0
                emit!(state.trace, HYPOTHESIS_WEAKENED; hypothesis=h.id,
                      contested=length(contested_conflicts),
                      synaptic_support=h.scores.synaptic_support,
                      experience=h.scores.experience)
            else
                h.status = :active
                h.confidence = hypothesis_confidence(h.scores, 0)
            end
        end
    end
    return state
end

