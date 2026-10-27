
# self_correction.jl — Detect reasoning failures and repair them (spec §32, §129).
#
# Self-correction is real, not decorative: it inspects the current hypotheses and
# contradiction set, rejects untenable candidates (never hides them), records the
# failure so the same mistake is avoided later (spec §62), backtracks so a
# different candidate is chosen, and requests information when nothing viable
# remains rather than fabricating an answer. Every repair emits a typed event.

"""
Reject one hypothesis with a stated reason. Marks it `:rejected`, zeroes its
confidence, penalises the strategy that produced it, records the failure, updates
metrics, and emits `HYPOTHESIS_REJECTED`. Idempotent. Returns the hypothesis.
"""
function reject_hypothesis!(agent::CognitiveAgent, state::CognitiveState,
                            h::CognitiveHypothesis, reason::String)
    h.status == :rejected && return h
    h.status = :rejected
    h.confidence = 0f0
    agent.metrics.hypotheses_rejected += 1
    note_strategy_outcome!(agent, h.strategy, false; episode=Int32(agent.episodic.next_id))
    record_failure!(agent, agent.episodic.next_id, state.input, reason, h.strategy; claim=h.claim)
    emit!(state.trace, HYPOTHESIS_REJECTED; hypothesis=h.id, reason=reason,
          claim="$(concept_name(agent,h.claim.subject)) $(h.claim.predicate) $(concept_name(agent,h.claim.object))")
    return h
end

"""
Inspect the reasoning state and repair it (spec §32). Rejects every contradicted
hypothesis, backtracks (recording a `:backtrack` action and a `BACKTRACKED`
event), and — if no viable hypothesis survives — emits `INFORMATION_REQUESTED`
and records the dead end in failure memory. Returns the updated state. Increments
`metrics.self_corrections` whenever a repair actually happens.
"""
function self_correct!(agent::CognitiveAgent, state::CognitiveState)
    repaired = false

    # 1. Reject contradicted hypotheses. They are surfaced, never silently dropped.
    for h in state.hypotheses
        if h.status == :contradicted
            reject_hypothesis!(agent, state, h, "contradicted by stored knowledge")
            repaired = true
        end
    end

    # 2. Backtrack: register the action so a different candidate is chosen next,
    #    and count it against the backtrack budget (spec §35).
    if repaired
        state.backtracks_used += 1
        push!(state.previous_actions, :backtrack)
        agent.metrics.self_corrections += 1
        emit!(state.trace, BACKTRACKED; backtracks=state.backtracks_used,
              remaining=count(h -> h.status in (:new, :active, :supported, :weak), state.hypotheses))
    end

    # 3. If nothing viable remains, request information instead of fabricating.
    if !any_viable(state)
        strategy = isempty(state.strategy_sequence) ? :none : last(state.strategy_sequence)
        record_failure!(agent, agent.episodic.next_id, state.input,
                        "no viable hypothesis after self-correction", strategy)
        push!(state.previous_actions, :request_information)
        emit!(state.trace, INFORMATION_REQUESTED; about=state.goal,
              reason="no viable hypothesis after self-correction")
    end

    return state
end

