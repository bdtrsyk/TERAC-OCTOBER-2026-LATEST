
# meta_reasoning.jl — Controller, anti-loop, and reasoning budget (spec §33,§34,§35).
#
# The controller has inspectable state (the CognitiveState budget counters and
# signature sets) and its strategy decisions generate trace events.

"""Signature of the current cognitive state for loop detection (spec §33)."""
function state_signature(state::CognitiveState)
    active = sort(collect(state.active_units))
    viable = sort([Int(h.id) for h in state.hypotheses if h.status in (:active, :supported, :weak, :new)])
    return hash((state.phase, active, viable, state.current_hypothesis))
end

"""Record a phase transition (spec §24) and enforce the step budget (spec §35)."""
function transition!(state::CognitiveState, phase::ReasoningPhase)
    state.phase = phase
    push!(state.phase_history, phase)
    state.steps_used += 1
    emit!(state.trace, STATE_TRANSITION; phase=phase, step=state.steps_used)
    return state
end

"""Return true and flag the state when any reasoning budget is exhausted (spec §35)."""
function budget_exhausted!(agent::CognitiveAgent, state::CognitiveState)
    b = agent.config.budget
    exhausted = state.steps_used >= b.max_steps ||
                state.hypotheses_used >= b.max_hypotheses ||
                state.retrievals_used >= b.max_retrievals ||
                state.backtracks_used >= b.max_backtracks
    if exhausted && !state.budget_exhausted
        state.budget_exhausted = true
        agent.metrics.budget_exhaustions += 1
        emit!(state.trace, BUDGET_EXHAUSTED; steps=state.steps_used, hypotheses=state.hypotheses_used,
              retrievals=state.retrievals_used, backtracks=state.backtracks_used)
    end
    return exhausted
end

"""
Detect an unproductive loop: a state signature already visited this episode. An
explicitly requested repetition is NOT treated as a loop (spec §33 exception).
"""
function detect_loop!(state::CognitiveState)
    state.intent.requested_repetition && return false
    sig = state_signature(state)
    if sig in state.visited_states
        emit!(state.trace, LOOP_DETECTED; signature=sig)
        return true
    end
    push!(state.visited_states, sig)
    return false
end

any_viable(state::CognitiveState) = any(h -> h.status in (:active, :supported, :weak, :new), state.hypotheses)

"""
Meta-controller decision (spec §34). Returns one of
`:select, :backtrack, :retrieve_more, :switch_strategy, :stop`, and traces the
choice when it changes strategy.
"""
function meta_decide(agent::CognitiveAgent, state::CognitiveState)
    if budget_exhausted!(agent, state)
        return :stop
    end
    if !isempty(state.contradictions) && any(h -> h.status == :contradicted, state.hypotheses)
        return :backtrack
    end
    if !any_viable(state)
        if state.retrievals_used < agent.config.budget.max_retrievals &&
           state.recombinations_used < agent.config.budget.max_recombinations
            emit!(state.trace, STRATEGY_CHANGED; reason="no viable hypothesis", action=:retrieve_more)
            return :retrieve_more
        else
            return :stop
        end
    end
    if detect_loop!(state)
        emit!(state.trace, STRATEGY_CHANGED; reason="loop detected", action=:switch_strategy)
        return :switch_strategy
    end
    return :select
end

