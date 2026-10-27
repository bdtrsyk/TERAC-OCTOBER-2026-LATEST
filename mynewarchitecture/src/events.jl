
# events.jl — Typed reasoning-event system and ordered trace buffer.
#
# Spec §45/§46/§101/§47: reasoning transparency must be built from *typed*
# machine events that correspond to things that actually happened, preserving
# real temporal order. No fabricated chain-of-thought text.

"""Canonical event kinds. Every kind must be emitted only by real state changes."""
@enum EventKind begin
    EPISODE_STARTED
    GOAL_CREATED
    PARSED_INPUT
    INTENT_DETECTED
    UNIT_ACTIVATED
    CONCEPT_RETRIEVED
    MEMORY_RETRIEVED
    STATE_TRANSITION
    HYPOTHESIS_CREATED
    HYPOTHESIS_SUPPORTED
    HYPOTHESIS_WEAKENED
    HYPOTHESIS_REJECTED
    CONTRADICTION_FOUND
    ANALOGY_FORMED
    CROSS_DOMAIN_RECOMBINATION
    COUNTERFACTUAL_EXPLORED
    ABSTRACTION_FORMED
    BACKTRACKED
    STRATEGY_CHANGED
    LOOP_DETECTED
    BUDGET_EXHAUSTED
    HYPOTHESIS_SELECTED
    OUTPUT_GENERATED
    EVALUATION_COMPLETED
    REWARD_ISSUED
    PUNISHMENT_ISSUED
    CREDIT_ASSIGNED
    SYNAPSE_CHANGED
    MEMORY_UPDATED
    EPISODE_STORED
    SELF_EVALUATED
    INFORMATION_REQUESTED
end

"""
A single, ordered, machine-readable reasoning event. `step` is the monotonic
position within the episode; `data` carries typed key→value details.
"""
struct ReasoningEvent
    step::Int
    kind::EventKind
    data::Dict{Symbol,Any}
end

ReasoningEvent(step::Int, kind::EventKind) = ReasoningEvent(step, kind, Dict{Symbol,Any}())

"""Ordered buffer of reasoning events for one episode (or a run)."""
mutable struct TraceBuffer
    enabled::Bool
    events::Vector{ReasoningEvent}
    counter::Int
end
TraceBuffer(enabled::Bool=true) = TraceBuffer(enabled, ReasoningEvent[], 0)

"""Emit a typed event. Returns the created event (or `nothing` when disabled)."""
function emit!(tb::TraceBuffer, kind::EventKind; kwargs...)
    tb.enabled || return nothing
    tb.counter += 1
    data = Dict{Symbol,Any}(kwargs)
    ev = ReasoningEvent(tb.counter, kind, data)
    push!(tb.events, ev)
    return ev
end

count_kind(tb::TraceBuffer, kind::EventKind) = count(e -> e.kind == kind, tb.events)
has_event(tb::TraceBuffer, kind::EventKind) = any(e -> e.kind == kind, tb.events)
events_of(tb::TraceBuffer, kind::EventKind) = filter(e -> e.kind == kind, tb.events)

"""Human-readable rendering of the authentic event stream (spec §46)."""
function render_trace(tb::TraceBuffer; io::IO=stdout)
    for ev in tb.events
        parts = ["STEP $(lpad(ev.step,2,'0'))", "event = $(ev.kind)"]
        for k in sort(collect(keys(ev.data)))
            push!(parts, "  $k = $(ev.data[k])")
        end
        println(io, join(parts, "\n"))
        println(io)
    end
end

"""Compact one-line-per-event summary used by higher debug levels."""
function trace_lines(tb::TraceBuffer)
    [string(lpad(e.step,3,'0'), "  ", e.kind, isempty(e.data) ? "" : string("  ", NamedTuple(e.data)))
     for e in tb.events]
end

