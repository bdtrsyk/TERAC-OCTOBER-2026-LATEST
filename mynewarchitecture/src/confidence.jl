
# confidence.jl — Dynamic confidence (spec §36).
#
# Confidence is a computed function of evidence, not a constant or random(). It is
# a transient Float32 auxiliary quantity, never a learned weight.

"""
Per-hypothesis confidence from its documented score breakdown and the number of
contradictions against it. Influenced by memory evidence, logical compatibility,
concept reliability, and constraint satisfaction; reduced by contradictions.
"""
function hypothesis_confidence(sc::HypothesisScore, contradiction_count::Int)
    base = 0.45f0*sc.memory_evidence + 0.20f0*sc.logical_compatibility +
           0.15f0*sc.concept_compatibility + 0.20f0*sc.constraint_satisfaction +
           0.10f0*sc.synaptic_support + 0.10f0*sc.experience
    penalty = 0.5f0 * min(1f0, Float32(contradiction_count))
    return clamp(base - penalty, 0f0, 1f0)
end

"""
Episode-level confidence: the selected hypothesis's confidence, boosted by
agreement (how many other viable hypotheses assert the same claim) and reduced
when unresolved contradictions remain (spec §36).
"""
function state_confidence(state::CognitiveState)
    sel = state.current_hypothesis
    sel === nothing && return 0f0
    idx = findfirst(h -> h.id == sel, state.hypotheses)
    idx === nothing && return 0f0
    selected = state.hypotheses[idx]
    agree = count(h -> h.id != selected.id && h.status in (:supported, :active) &&
                       h.claim.subject == selected.claim.subject &&
                       h.claim.predicate == selected.claim.predicate &&
                       h.claim.object == selected.claim.object, state.hypotheses)
    agreement_boost = min(0.2f0, 0.1f0 * agree)
    contradiction_drag = isempty(state.contradictions) ? 0f0 : 0.1f0
    return clamp(selected.confidence + agreement_boost - contradiction_drag, 0f0, 1f0)
end

