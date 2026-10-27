# engine.jl -- HISTORICAL parent bundle (not loaded by the package).
# Merged (in order): core.jl, memory.jl, reasoning.jl, app.jl
# Superseded by src/ (per-module files). Kept only as the source that
# _extract.jl splits. Internal comments describe the older flat-file eras.


# ===== BEGIN core.jl =====

# core.jl -- consolidated bundle (auto-generated)
# Sources merged: config.jl, events.jl, types.jl, ternary.jl, network.jl, concepts.jl
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt


# ===== BEGIN config.jl =====

# config.jl — Configuration, debug levels, reasoning budgets, scaling presets,
# and the configurable quality→signal mapping.
#
# Everything here is auxiliary/control data. None of it is a learned parameter.

"""Debug verbosity levels (spec §83)."""
@enum DebugLevel QUIET=0 NORMAL=1 VERBOSE=2 TRACE=3 AUDIT=4

"""
Reasoning budgets (spec §35). The engine stops safely when any budget is
exhausted and reports `reasoning_budget_exhausted = true` rather than pretending
to have finished normally.
"""
Base.@kwdef struct ReasoningBudget
    max_steps::Int              = 24
    max_hypotheses::Int         = 12
    max_retrievals::Int         = 16
    max_backtracks::Int         = 6
    max_recombinations::Int     = 12
end

"""
Eligibility-trace parameters (spec §11). Eligibility is a *transient* float and
never a learned weight.
"""
Base.@kwdef struct EligibilityConfig
    decay::Float32   = 0.85f0     # multiplicative decay per reasoning step
    floor::Float32   = 0.02f0     # below this an eligible trace is cleared
    cap::Float32     = 1.0f0      # maximum eligibility
    gain::Float32    = 0.5f0      # per-activation increment
end

"""
Discrete plasticity thresholds (spec §10, §42). Accumulated learning pressure
must exceed a threshold before a synapse takes one ternary step. Stability makes
repeatedly-successful synapses harder to perturb.
"""
Base.@kwdef struct PlasticityConfig
    promote_threshold::Float32  = 0.9f0
    demote_threshold::Float32   = 0.9f0
    stability_gain::Float32     = 0.15f0   # how fast a rewarded synapse stabilises
    stability_decay::Float32    = 0.10f0   # how fast a punished synapse destabilises
    max_stability::Float32      = 3.0f0
    exploration_prob::Float32   = 0.03f0   # bounded stochastic exploration
end

const QUALITY_CLASSES = (
    :EXCELLENT, :GOOD, :PARTIAL, :NEUTRAL, :WRONG, :REPETITIVE,
    :REQUESTED_REPETITION, :IRRELEVANT, :CONTRADICTORY, :INCOMPLETE,
    :CONSTRAINT_FAILURE, :INSTRUCTION_FAILURE,
)

"""Default configurable quality→signal map (spec §14, §15). Positive values are
reward magnitudes; negative values are punishment magnitudes."""
function default_quality_signals()
    Dict{Symbol,Float32}(
        :EXCELLENT            =>  1.00f0,
        :GOOD                 =>  0.65f0,
        :PARTIAL              =>  0.20f0,
        :NEUTRAL              =>  0.00f0,
        :REQUESTED_REPETITION =>  0.35f0,
        :WRONG                => -0.75f0,
        :REPETITIVE           => -0.50f0,
        :IRRELEVANT           => -0.40f0,
        :CONTRADICTORY        => -0.80f0,
        :INCOMPLETE           => -0.30f0,
        :CONSTRAINT_FAILURE   => -0.60f0,
        :INSTRUCTION_FAILURE  => -0.70f0,
    )
end

"""
Top-level cognitive configuration. Scaling presets (spec §122, §123) are chosen
via `CognitiveConfig(scale=:tiny|:small|:medium|:large)`; dimensions are never
hard-coded into the algorithms.
"""
Base.@kwdef mutable struct CognitiveConfig
    scale::Symbol                       = :small
    working_memory_capacity::Int        = 16
    retrieval_limit::Int                = 10
    hypothesis_limit::Int               = 12
    unit_capacity_hint::Int             = 4_000     # advisory; graph grows lazily
    budget::ReasoningBudget             = ReasoningBudget()
    eligibility::EligibilityConfig      = EligibilityConfig()
    plasticity::PlasticityConfig        = PlasticityConfig()
    quality_signals::Dict{Symbol,Float32} = default_quality_signals()
    reward_enabled::Bool                = true
    punishment_enabled::Bool            = true
    memory_enabled::Bool                = true
    ternary_enabled::Bool               = true      # ablation toggle (spec §73)
    packed_ternary::Bool                = false     # use packed store (spec §50)
    debug_invariants::Bool              = true
    debug_level::DebugLevel             = NORMAL
    seed::Int                           = 42
    confidence_select_threshold::Float32 = 0.35f0   # min confidence to commit
end

const SCALE_PRESETS = Dict(
    :tiny   => (working_memory_capacity=8,  retrieval_limit=6,  hypothesis_limit=6,  unit_capacity_hint=512),
    :small  => (working_memory_capacity=16, retrieval_limit=10, hypothesis_limit=12, unit_capacity_hint=4_000),
    :medium => (working_memory_capacity=48, retrieval_limit=24, hypothesis_limit=24, unit_capacity_hint=50_000),
    :large  => (working_memory_capacity=96, retrieval_limit=48, hypothesis_limit=48, unit_capacity_hint=500_000),
)

"""Return a config for a named scale preset without redesigning the system."""
function config_for_scale(scale::Symbol; kwargs...)
    haskey(SCALE_PRESETS, scale) || error("unknown scale preset: $scale (choose :tiny,:small,:medium,:large)")
    p = SCALE_PRESETS[scale]
    cfg = CognitiveConfig(; scale=scale, working_memory_capacity=p.working_memory_capacity,
                          retrieval_limit=p.retrieval_limit, hypothesis_limit=p.hypothesis_limit,
                          unit_capacity_hint=p.unit_capacity_hint, kwargs...)
    return cfg
end

should_log(cfg::CognitiveConfig, level::DebugLevel) = Int(cfg.debug_level) >= Int(level)

# ===== END config.jl =====


# ===== BEGIN events.jl =====

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

# ===== END events.jl =====


# ===== BEGIN types.jl =====

# types.jl — Core cognitive data structures.
#
# This file intentionally contains NO neural-network layers and NO floating-point
# *learned* weights. The only persistent learned quantity is `SparseSynapse.value`
# which is strictly ternary (Int8 ∈ {-1,0,+1}). All Float32 fields here are
# transient/auxiliary (eligibility, confidence, scores, stability, statistics).

# ── Reasoning state machine phases (spec §24) ──────────────────────────────
@enum ReasoningPhase begin
    INITIALIZE
    PARSE
    REPRESENT
    RETRIEVE
    FORM_GOAL
    GENERATE_HYPOTHESES
    COMPARE
    TEST
    DETECT_CONTRADICTION
    BACKTRACK
    REFINE
    RECOMBINE
    VERIFY
    SELECT
    CONCLUDE
    SELF_EVALUATE
    REVISE
    FINISH
end

# ── Structured credit levels (spec §12) ────────────────────────────────────
@enum CreditLevel STRONG_CREDIT MODERATE_CREDIT WEAK_CREDIT NEUTRAL_CREDIT NEGATIVE_CREDIT

# ── Computational units and ternary synapses (spec §4, §5, §41, §42) ────────
mutable struct CognitiveUnit
    id::Int32
    state::Int8                    # discrete activation ∈ {-1,0,+1}
    history::Vector{Int8}
    concept_ids::Vector{Int32}
    activation_count::Int32
end
CognitiveUnit(id::Int32, concept_ids::Vector{Int32}) =
    CognitiveUnit(id, Int8(0), Int8[], concept_ids, Int32(0))

"""
A persistent learned synapse. `value` is strictly ternary. Every other field is
transient bookkeeping (eligibility) or non-learned metadata (history/stability).
"""
mutable struct SparseSynapse
    source::Int32
    target::Int32
    value::Int8                    # LEARNED STATE — always -1, 0, or +1
    eligibility::Float32           # transient attribution trace (not learned)
    pressure::Float32              # accumulated learning pressure toward a step
    stability::Float32             # resistance to perturbation (spec §42)
    creation_episode::Int32
    last_changed_episode::Int32
    times_activated::Int32
    times_rewarded::Int32
    times_punished::Int32
end
SparseSynapse(source::Int32, target::Int32, value::Int8, episode::Int32) =
    SparseSynapse(source, target, value, 0f0, 0f0, 0f0, episode, episode, 0, 0, 0)

mutable struct SparseTernaryNetwork
    units::Dict{Int32,CognitiveUnit}
    outgoing::Dict{Int32,Vector{SparseSynapse}}
    incoming::Dict{Int32,Vector{SparseSynapse}}
    synapse_count::Int
end
SparseTernaryNetwork() = SparseTernaryNetwork(Dict{Int32,CognitiveUnit}(),
                                              Dict{Int32,Vector{SparseSynapse}}(),
                                              Dict{Int32,Vector{SparseSynapse}}(), 0)

# ── Concept graph (spec §20, §21, §28) ──────────────────────────────────────
mutable struct CognitiveConcept
    id::Int32
    name::String
    properties::Set{String}
    domain_tags::Set{String}
    frequency::Int32
    successes::Int32
    failures::Int32
    activation_count::Int32
end
CognitiveConcept(id::Int32, name::String; domain::String="general") =
    CognitiveConcept(id, name, Set{String}(), Set([domain]), 1, 0, 0, 0)

struct CognitiveRelation
    subject::Int32
    predicate::Symbol
    object::Int32
    provenance::Symbol             # :input, :deduced, :recombined, :analogy, :memory, :counterfactual
end

# ── Memory stores (spec §22) ────────────────────────────────────────────────
mutable struct MemoryItem
    text::String
    concept_ids::Vector{Int32}
    relevance::Float32
    step::Int32
    kind::Symbol                   # :observation, :goal, :retrieval, :conclusion
end
MemoryItem(text, ids, relevance, step) = MemoryItem(text, ids, relevance, step, :observation)

mutable struct WorkingMemory
    capacity::Int
    items::Vector{MemoryItem}
end
WorkingMemory(capacity::Int) = WorkingMemory(capacity, MemoryItem[])

mutable struct SemanticMemory
    concepts::Dict{Int32,CognitiveConcept}
    name_to_id::Dict{String,Int32}
    relations::Vector{CognitiveRelation}
    relation_set::Set{CognitiveRelation}
    next_concept_id::Int32
end
SemanticMemory() = SemanticMemory(Dict{Int32,CognitiveConcept}(), Dict{String,Int32}(),
                                  CognitiveRelation[], Set{CognitiveRelation}(), Int32(1))

# Procedural strategy memory (spec §40)
mutable struct StrategyRecord
    name::Symbol
    successes::Int32
    failures::Int32
    last_used_episode::Int32
end
StrategyRecord(name::Symbol) = StrategyRecord(name, 0, 0, 0)

mutable struct StrategyMemory
    records::Dict{Symbol,StrategyRecord}
end
StrategyMemory() = StrategyMemory(Dict{Symbol,StrategyRecord}())

# Failure memory (spec §62)
struct FailureRecord
    episode::Int
    context::String
    reason::String
    strategy::Symbol
    hypothesis_claim::Union{Nothing,CognitiveRelation}
    successful_alternative::Union{Nothing,String}
end

mutable struct FailureMemory
    records::Vector{FailureRecord}
end
FailureMemory() = FailureMemory(FailureRecord[])

# Contradiction registry (spec §31)
mutable struct ContradictionRecord
    subject::Int32
    object::Int32
    predicate_a::Symbol
    predicate_b::Symbol
    occurrences::Int32
    description::String
end

mutable struct ContradictionRegistry
    records::Vector{ContradictionRecord}
end
ContradictionRegistry() = ContradictionRegistry(ContradictionRecord[])

# ── User intent (spec §17) ──────────────────────────────────────────────────
mutable struct UserIntent
    goal::Symbol                       # :assert, :verify, :explain, :transform, :repeat, :unknown
    requested_repetition::Bool
    requested_repetition_count::Int
    requested_transformation::String
    requested_structure::String
    constraints::Vector{String}
    prohibited_behavior::Vector{String}
    requested_quantity::Int
    exclusions::Vector{String}
end
UserIntent(goal::Symbol) = UserIntent(goal, false, 0, "", "", String[], String[], 0, String[])

# ── Hypotheses (spec §25, §26) ──────────────────────────────────────────────
mutable struct HypothesisScore
    logical_compatibility::Float32
    constraint_satisfaction::Float32
    memory_evidence::Float32
    concept_compatibility::Float32
    synaptic_support::Float32
    experience::Float32
    contradiction_penalty::Float32
    simplicity::Float32
    novelty::Float32
    goal_relevance::Float32
end
HypothesisScore() = HypothesisScore(0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0,0f0)

mutable struct CognitiveHypothesis
    id::Int32
    claim::CognitiveRelation
    supporting_concepts::Vector{Int32}
    opposing_concepts::Vector{Int32}
    assumptions::Vector{String}
    predicted_consequences::Vector{String}
    scores::HypothesisScore
    score::Float32
    confidence::Float32
    status::Symbol                 # :new,:active,:supported,:weak,:contradicted,:rejected,:selected
    strategy::Symbol
    origin::Symbol                 # generation operator that produced it
end

# ── Learning bookkeeping (spec §115) ────────────────────────────────────────
struct CreditAssignment
    source::Int32
    target::Int32
    eligibility::Float32
    temporal_relevance::Float32
    causal_relevance::Float32
    credit::CreditLevel
    signal::Float32
    reason::String
end

struct SynapticChange
    source::Int32
    target::Int32
    old_state::Int8
    new_state::Int8
    eligibility::Float32
    credit::CreditLevel
    reason::String
end

# ── Evaluation result (spec §13, §19, §103) ─────────────────────────────────
struct EvaluationResult
    quality::Symbol
    intent::Symbol
    reward::Float32
    punishment::Float32
    repetition_detected::Bool
    repetition_requested::Bool
    correctness::Float32
    relevance::Float32
    consistency::Float32
    constraint_satisfaction::Float32
    evaluator_mode::Symbol
    expected_quality::Union{Nothing,Symbol}
    reasons::Vector{String}
end

# ── Reasoning state (working, per-episode) ──────────────────────────────────
mutable struct CognitiveState
    input::String
    goal::String
    intent::UserIntent
    concepts::Vector{Int32}
    active_units::Vector{Int32}
    active_pathways::Vector{Tuple{Int32,Int32}}
    retrieved_concepts::Vector{Int32}
    retrieved_memories::Vector{Int}
    hypotheses::Vector{CognitiveHypothesis}
    current_hypothesis::Union{Nothing,Int32}
    constraints::Vector{String}
    contradictions::Vector{String}
    phase::ReasoningPhase
    phase_history::Vector{ReasoningPhase}
    previous_actions::Vector{Symbol}
    strategy_sequence::Vector{Symbol}
    confidence::Float32
    step::Int
    trace::TraceBuffer
    # budget accounting
    steps_used::Int
    hypotheses_used::Int
    retrievals_used::Int
    backtracks_used::Int
    recombinations_used::Int
    budget_exhausted::Bool
    # anti-loop signatures
    visited_states::Set{UInt64}
    seen_hypotheses::Set{UInt64}
end

"""Look up the currently selected hypothesis, excluding rejected ones."""
function selected_hypothesis(state::CognitiveState)
    state.current_hypothesis === nothing && return nothing
    idx = findfirst(h -> h.id == state.current_hypothesis, state.hypotheses)
    idx === nothing && return nothing
    h = state.hypotheses[idx]
    return h.status == :rejected ? nothing : h
end

# ── Persistent episode object (spec §117) ───────────────────────────────────
mutable struct CognitiveEpisode
    id::Int
    input::String
    goal::String
    intent::UserIntent
    representation::Vector{Int32}
    retrieved_concepts::Vector{Int32}
    transitions::Vector{ReasoningPhase}
    hypotheses_generated::Vector{Int32}
    hypotheses_rejected::Vector{Int32}
    hypothesis_selected::Union{Nothing,Int32}
    events::Vector{ReasoningEvent}
    output::String
    evaluation::Union{Nothing,EvaluationResult}
    synaptic_changes::Vector{SynapticChange}
    memory_added_concepts::Vector{Int32}
    reward::Float32
    punishment::Float32
    strategy::Symbol
    budget_exhausted::Bool
    success::Bool
end

mutable struct EpisodicMemory
    episodes::Vector{CognitiveEpisode}
    next_id::Int
end
EpisodicMemory() = EpisodicMemory(CognitiveEpisode[], 1)

# ── Training data schema (spec §18) ─────────────────────────────────────────
struct TrainingExample
    id::Int
    input::String
    expected_output::Union{Nothing,String}
    quality_target::Union{Nothing,Symbol}
    user_intent::Union{Nothing,UserIntent}
    constraints::Vector{String}
    tags::Vector{Symbol}
    metadata::Dict{String,Any}
end

# ── Run-level metrics (spec §87) ────────────────────────────────────────────
mutable struct RunMetrics
    episodes::Int
    quality_counts::Dict{Symbol,Int}
    total_reward::Float32
    total_punishment::Float32
    ternary_transitions::Int
    synapses_changed_per_episode::Vector{Int}
    reasoning_steps::Vector{Int}
    reward_curve::Vector{Float32}
    punishment_curve::Vector{Float32}
    success_curve::Vector{Bool}
    self_corrections::Int
    novel_recombinations::Int
    contradictions_detected::Int
    hypotheses_rejected::Int
    budget_exhaustions::Int
    peak_bytes::Int
    runtime_seconds::Float64
end
RunMetrics() = RunMetrics(0, Dict{Symbol,Int}(), 0f0, 0f0, 0, Int[], Int[], Float32[], Float32[],
                          Bool[], 0, 0, 0, 0, 0, 0, 0.0)

# ── The agent (spec §85) ────────────────────────────────────────────────────
mutable struct CognitiveAgent
    config::CognitiveConfig
    network::SparseTernaryNetwork
    working::WorkingMemory
    semantic::SemanticMemory
    episodic::EpisodicMemory
    strategy::StrategyMemory
    failures::FailureMemory
    contradictions::ContradictionRegistry
    metrics::RunMetrics
    rng::Random.MersenneTwister
    last_output::String
    last_state::Union{Nothing,CognitiveState}
    last_evaluation::Union{Nothing,EvaluationResult}
    next_hypothesis_id::Int32
    architecture_version::VersionNumber
    serialization_version::Int
    created::String
end

# ===== END types.jl =====


# ===== BEGIN ternary.jl =====

# ternary.jl — Ternary value domain, invariant checks, packed storage, stats.
#
# The learned state domain is strictly {-1, 0, +1}. This file provides the
# canonical Int8 checks plus an OPTIONAL packed representation that must remain
# behaviorally equivalent to the Int8 reference (spec §50).

const TERNARY_VALUES = (Int8(-1), Int8(0), Int8(1))

@inline is_ternary(x::Integer) = x == -1 || x == 0 || x == 1

"""Throw an architecture-integrity error unless `x` is a valid ternary state."""
@inline function assert_ternary(x::Integer, where::AbstractString="synapse")
    is_ternary(x) || error("ArchitectureIntegrityError: non-ternary $where value = $x")
    return Int8(x)
end

"""Continuously verify the ternary invariant across the whole network (spec §4)."""
function assert_network_invariants(net::SparseTernaryNetwork)
    for edges in values(net.outgoing), e in edges
        is_ternary(e.value) || error("ArchitectureIntegrityError: synapse $(e.source)->$(e.target) has non-ternary value $(e.value)")
    end
    for u in values(net.units)
        is_ternary(u.state) || error("ArchitectureIntegrityError: unit $(u.id) has non-discrete state $(u.state)")
    end
    return true
end

"""Ternary distribution over all learned synapses (spec §89)."""
function ternary_distribution(net::SparseTernaryNetwork)
    neg = zero = pos = 0
    for edges in values(net.outgoing), e in edges
        e.value < 0 ? (neg += 1) : e.value > 0 ? (pos += 1) : (zero += 1)
    end
    total = neg + zero + pos
    pct(x) = total == 0 ? 0.0 : round(100 * x / total; digits=2)
    return (negative=neg, zero=zero, positive=pos, total=total,
            pct_negative=pct(neg), pct_zero=pct(zero), pct_positive=pct(pos))
end

# ── Optional packed ternary vector (2 bits per trit, 32 trits per UInt64) ────
# Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1. This is the compact native-style
# representation referenced in spec §50/§51; it is an optimization, never a
# replacement for the architecture.
const _TRITS_PER_WORD = 32

mutable struct PackedTernaryVector
    words::Vector{UInt64}
    len::Int
end

function PackedTernaryVector(n::Int)
    nwords = cld(n, _TRITS_PER_WORD)
    PackedTernaryVector(zeros(UInt64, nwords), n)
end

Base.length(p::PackedTernaryVector) = p.len

@inline function _encode(v::Integer)
    v == 0 && return UInt64(0)
    v == 1 && return UInt64(1)
    v == -1 && return UInt64(2)
    error("ArchitectureIntegrityError: cannot pack non-ternary value $v")
end

@inline _decode(bits::UInt64) = bits == 0 ? Int8(0) : bits == 1 ? Int8(1) : Int8(-1)

function Base.getindex(p::PackedTernaryVector, i::Int)
    (1 <= i <= p.len) || throw(BoundsError(p, i))
    w = (i - 1) ÷ _TRITS_PER_WORD + 1
    shift = ((i - 1) % _TRITS_PER_WORD) * 2
    bits = (p.words[w] >> shift) & UInt64(0b11)
    return _decode(bits)
end

function Base.setindex!(p::PackedTernaryVector, v::Integer, i::Int)
    (1 <= i <= p.len) || throw(BoundsError(p, i))
    w = (i - 1) ÷ _TRITS_PER_WORD + 1
    shift = ((i - 1) % _TRITS_PER_WORD) * 2
    mask = UInt64(0b11) << shift
    p.words[w] = (p.words[w] & ~mask) | (_encode(v) << shift)
    return v
end

"""Round-trip an Int8 vector through the packed store and confirm equivalence."""
function packed_roundtrip(values::AbstractVector{<:Integer})
    p = PackedTernaryVector(length(values))
    @inbounds for i in eachindex(values)
        p[i] = values[i]
    end
    out = Vector{Int8}(undef, length(values))
    @inbounds for i in eachindex(values)
        out[i] = p[i]
    end
    return out
end

"""Bytes used by the packed representation of `n` synapses vs the Int8 reference."""
function packed_storage_bytes(n::Int)
    packed = cld(n, _TRITS_PER_WORD) * 8
    reference = n                     # 1 byte per Int8
    naive_f64 = n * 8                 # if someone stored trits as Float64
    return (packed_bytes=packed, int8_bytes=reference, float64_bytes=naive_f64,
            bits_per_synapse=n == 0 ? 0.0 : round(packed * 8 / n; digits=3))
end

# ===== END ternary.jl =====


# ===== BEGIN network.jl =====

# network.jl — Sparse ternary network: units, synapses, discrete activation.
#
# Activation is genuinely computational: integer support/opposition over ternary
# synapses determines discrete unit states. There is no floating-point matmul and
# no learned float anywhere in this file.

function ensure_unit!(net::SparseTernaryNetwork, id::Int32, concept_ids::Vector{Int32}=Int32[id])
    if !haskey(net.units, id)
        net.units[id] = CognitiveUnit(id, concept_ids)
    end
    return net.units[id]
end

find_synapse(net::SparseTernaryNetwork, s::Int32, t::Int32) =
    (edges = get(net.outgoing, s, nothing); edges === nothing ? nothing :
     (i = findfirst(e -> e.target == t, edges); i === nothing ? nothing : edges[i]))

"""Create a ternary synapse without overwriting previously learned state."""
function connect!(net::SparseTernaryNetwork, s::Int32, t::Int32, value::Integer; episode::Int32=Int32(0))
    assert_ternary(value, "synapse")
    ensure_unit!(net, s); ensure_unit!(net, t)
    existing = find_synapse(net, s, t)
    existing !== nothing && return existing
    edge = SparseSynapse(s, t, Int8(value), episode)
    push!(get!(net.outgoing, s, SparseSynapse[]), edge)
    push!(get!(net.incoming, t, SparseSynapse[]), edge)
    net.synapse_count += 1
    return edge
end

out_edges(net::SparseTernaryNetwork, id::Int32) = get(net.outgoing, id, SparseSynapse[])
in_edges(net::SparseTernaryNetwork, id::Int32)  = get(net.incoming, id, SparseSynapse[])

function reset_activation!(net::SparseTernaryNetwork)
    for u in values(net.units)
        u.state = Int8(0)
    end
    return net
end

"""
Sparse discrete activation (spec §113). Seeds are set to +1; support is
propagated over `hops` passes. Each traversed synapse gains eligibility and its
`times_activated` counter increments. Returns `(active_ids, pathways)`.
"""
function activate!(net::SparseTernaryNetwork, seed_ids::Vector{Int32}, elig::EligibilityConfig;
                   hops::Int=2, trace::Union{Nothing,TraceBuffer}=nothing)
    reset_activation!(net)
    for id in seed_ids
        haskey(net.units, id) && (net.units[id].state = Int8(1))
    end
    active = Set{Int32}(filter(id -> haskey(net.units, id), seed_ids))
    pathways = Tuple{Int32,Int32}[]
    frontier = collect(active)
    for _ in 1:max(hops, 1)
        support = Dict{Int32,Int}()
        for src in frontier
            src_state = net.units[src].state
            src_state == 0 && continue
            for e in out_edges(net, src)
                # integer contribution: source polarity × synapse polarity
                support[e.target] = get(support, e.target, 0) + Int(src_state) * Int(e.value)
                e.eligibility = min(elig.cap, e.eligibility + elig.gain)
                e.times_activated += 1
                push!(pathways, (e.source, e.target))
            end
        end
        newly = Int32[]
        for (tgt, total) in support
            state = total > 0 ? Int8(1) : total < 0 ? Int8(-1) : Int8(0)
            u = net.units[tgt]
            if u.state != state
                u.state = state
            end
            push!(u.history, state)
            if state != 0 && !(tgt in active)
                push!(active, tgt); push!(newly, tgt)
                trace !== nothing && emit!(trace, UNIT_ACTIVATED; unit=tgt, state=state)
            end
        end
        isempty(newly) && break
        frontier = newly
    end
    return collect(active), pathways
end

active_unit_ids(net::SparseTernaryNetwork) = Int32[id for (id,u) in net.units if u.state != 0]
active_synapse_count(net::SparseTernaryNetwork) =
    sum(e -> e.eligibility > 0f0 ? 1 : 0, (e for edges in values(net.outgoing) for e in edges); init=0)

"""Decay all eligibility traces one step; clear those below the floor (spec §11)."""
function decay_eligibility!(net::SparseTernaryNetwork, elig::EligibilityConfig)
    for edges in values(net.outgoing), e in edges
        e.eligibility *= elig.decay
        e.eligibility < elig.floor && (e.eligibility = 0f0)
    end
    return net
end

# ===== END network.jl =====


# ===== BEGIN concepts.jl =====

# concepts.jl — Dynamic concept graph, relations, and domains (spec §20, §21, §28).
#
# Concepts acquire relationships through learning; the graph is never a static
# hard-coded dictionary. Each relation creates a real ternary synapse.

_concept_key(name::AbstractString) = lowercase(strip(replace(String(name), r"[^A-Za-z0-9_ \-]" => "")))

const OPPOSING_PREDICATES = Set{Symbol}([:prevents, :contradicts, :opposite_of, :inhibits, :blocks, :reduces])

predicate_polarity(p::Symbol) = p in OPPOSING_PREDICATES ? Int8(-1) : Int8(1)

"""Get-or-create a concept and its computational unit. Returns the concept id."""
function concept!(agent::CognitiveAgent, name::AbstractString; domain::String="general")
    key = _concept_key(name)
    isempty(key) && (key = "unknown")
    sm = agent.semantic
    if haskey(sm.name_to_id, key)
        id = sm.name_to_id[key]
        sm.concepts[id].frequency += 1
        push!(sm.concepts[id].domain_tags, domain)
        return id
    end
    id = sm.next_concept_id
    sm.next_concept_id += 1
    sm.name_to_id[key] = id
    sm.concepts[id] = CognitiveConcept(id, key; domain=domain)
    ensure_unit!(agent.network, id, Int32[id])
    return id
end

has_concept(agent::CognitiveAgent, name::AbstractString) = haskey(agent.semantic.name_to_id, _concept_key(name))
get_concept_id(agent::CognitiveAgent, name::AbstractString) = get(agent.semantic.name_to_id, _concept_key(name), Int32(0))

concept_name(agent::CognitiveAgent, id::Int32) =
    haskey(agent.semantic.concepts, id) ? agent.semantic.concepts[id].name : "unknown#$id"

concept_domains(agent::CognitiveAgent, id::Int32) =
    haskey(agent.semantic.concepts, id) ? agent.semantic.concepts[id].domain_tags : Set{String}()

"""Add a relation by concept ids. Creates the ternary synapse; returns the relation."""
function add_relation_ids!(agent::CognitiveAgent, s::Int32, predicate::Symbol, o::Int32;
                           provenance::Symbol=:input, episode::Int32=Int32(0))
    rel = CognitiveRelation(s, predicate, o, provenance)
    sm = agent.semantic
    if !(rel in sm.relation_set)
        push!(sm.relations, rel)
        push!(sm.relation_set, rel)
    end
    connect!(agent.network, s, o, Int8(0); episode=episode)
    return rel
end

"""Add a relation by names (concepts auto-created). Returns the relation."""
function add_relation!(agent::CognitiveAgent, subject::AbstractString, predicate::Symbol, object::AbstractString;
                       provenance::Symbol=:input, domain::String="general", episode::Int32=Int32(0))
    s = concept!(agent, subject; domain=domain)
    o = concept!(agent, object; domain=domain)
    return add_relation_ids!(agent, s, predicate, o; provenance=provenance, episode=episode)
end

relation_exists(agent::CognitiveAgent, rel::CognitiveRelation) = rel in agent.semantic.relation_set

function relation_exists(agent::CognitiveAgent, s::Int32, p::Symbol, o::Int32)
    for r in agent.semantic.relations
        r.subject == s && r.predicate == p && r.object == o && return true
    end
    return false
end

relations_about(agent::CognitiveAgent, id::Int32) =
    [r for r in agent.semantic.relations if r.subject == id || r.object == id]

relations_with_predicate(agent::CognitiveAgent, p::Symbol) =
    [r for r in agent.semantic.relations if r.predicate == p]

"""Record success/failure statistics on the concepts involved in a hypothesis."""
function credit_concepts!(agent::CognitiveAgent, ids::Vector{Int32}, success::Bool)
    for id in ids
        c = get(agent.semantic.concepts, id, nothing)
        c === nothing && continue
        success ? (c.successes += 1) : (c.failures += 1)
    end
end

concept_reliability(c::CognitiveConcept) = (c.successes + 1f0) / (c.successes + c.failures + 2f0)

# ===== END concepts.jl =====

# ===== END core.jl =====


# ===== BEGIN memory.jl =====

# memory.jl -- consolidated bundle (auto-generated)
# Sources merged: memory.jl, language.jl, hypotheses.jl, contradiction.jl, self_correction.jl, meta_reasoning.jl, confidence.jl, novelty.jl
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt


# ===== BEGIN memory.jl =====

# memory.jl — Working, semantic, episodic, strategy, and failure memory.
#
# Memory is structured, not a text log (spec §23). Retrieval is selective and the
# stores interact: semantic retrieval is biased by concept reliability, episodic
# recall surfaces prior successes, and failure memory lets the engine avoid
# repeating known-bad reasoning (spec §62, §63).

# ── Working memory (bounded, spec §22) ──────────────────────────────────────
function remember!(agent::CognitiveAgent, text::String, ids::Vector{Int32};
                   relevance::Float32=1f0, step::Int=0, kind::Symbol=:observation)
    wm = agent.working
    push!(wm.items, MemoryItem(text, copy(ids), relevance, Int32(step), kind))
    if length(wm.items) > wm.capacity
        # Evict the least-relevant item (structured eviction, not FIFO of text).
        idx = argmin([it.relevance for it in wm.items])
        deleteat!(wm.items, idx)
    end
    return wm.items[end]
end

working_memory_texts(agent::CognitiveAgent) = [it.text for it in agent.working.items]
clear_working_memory!(agent::CognitiveAgent) = (empty!(agent.working.items); agent)

# ── Semantic retrieval (selective, spec §23, §28) ───────────────────────────
"""
Score and return the most relevant concept ids for a query. Scoring combines
lexical identity, graph connectivity (relations linking to the query), learned
reliability, and cross-domain bridging so an exact keyword match is not required.
"""
function retrieve_concepts(agent::CognitiveAgent, query_ids::Vector{Int32};
                           limit::Int=agent.config.retrieval_limit,
                           trace::Union{Nothing,TraceBuffer}=nothing)
    agent.config.memory_enabled || return Int32[]
    query = Set(query_ids)
    query_domains = Set{String}()
    for q in query_ids
        union!(query_domains, concept_domains(agent, q))
    end
    scored = Tuple{Int32,Float32}[]
    for (id, concept) in agent.semantic.concepts
        lexical = id in query ? 2f0 : 0f0
        connected = 0
        for r in agent.semantic.relations
            ((r.subject == id && r.object in query) || (r.object == id && r.subject in query)) && (connected += 1)
        end
        # cross-domain bridge: a shared domain tag counts even without a direct edge
        bridge = isempty(intersect(concept.domain_tags, query_domains)) ? 0f0 : 0.5f0
        reliability = concept_reliability(concept)
        score = lexical + Float32(connected) + bridge + reliability
        score > 0f0 && push!(scored, (id, score))
    end
    sort!(scored; by=last, rev=true)
    ids = Int32[first(x) for x in scored[1:min(limit, length(scored))]]
    if trace !== nothing
        for id in ids
            emit!(trace, CONCEPT_RETRIEVED; concept=id, name=concept_name(agent, id))
        end
    end
    return ids
end

# ── Episodic recall (spec §63) ──────────────────────────────────────────────
"""Recall past episodes whose representation overlaps the query concepts."""
function retrieve_episodes(agent::CognitiveAgent, query_ids::Vector{Int32}; limit::Int=5, successful_only::Bool=false)
    query = Set(query_ids)
    scored = Tuple{Int,Int}[]
    for (i, ep) in enumerate(agent.episodic.episodes)
        successful_only && !ep.success && continue
        overlap = count(id -> id in query, ep.representation)
        overlap > 0 && push!(scored, (i, overlap))
    end
    sort!(scored; by=last, rev=true)
    return [agent.episodic.episodes[first(x)] for x in scored[1:min(limit, length(scored))]]
end

function store_episode!(agent::CognitiveAgent, ep::CognitiveEpisode)
    push!(agent.episodic.episodes, ep)
    agent.episodic.next_id = ep.id + 1
    return ep
end

# ── Procedural strategy memory (spec §40) ───────────────────────────────────
function strategy_record!(agent::CognitiveAgent, name::Symbol)
    get!(agent.strategy.records, name, StrategyRecord(name))
end

function note_strategy_outcome!(agent::CognitiveAgent, name::Symbol, success::Bool; episode::Int32=Int32(0))
    rec = strategy_record!(agent, name)
    success ? (rec.successes += 1) : (rec.failures += 1)
    rec.last_used_episode = episode
    return rec
end

strategy_success_rate(rec::StrategyRecord) = (rec.successes + 1f0) / (rec.successes + rec.failures + 2f0)
strategy_success_rate(agent::CognitiveAgent, name::Symbol) = strategy_success_rate(strategy_record!(agent, name))

"""Strategies ordered by historical success (used to bias selection, spec §40)."""
function preferred_strategies(agent::CognitiveAgent)
    recs = collect(values(agent.strategy.records))
    sort!(recs; by=strategy_success_rate, rev=true)
    return [r.name for r in recs]
end

# ── Failure memory (spec §62) ───────────────────────────────────────────────
function record_failure!(agent::CognitiveAgent, episode::Int, context::String, reason::String,
                         strategy::Symbol; claim::Union{Nothing,CognitiveRelation}=nothing,
                         alternative::Union{Nothing,String}=nothing)
    push!(agent.failures.records, FailureRecord(episode, context, reason, strategy, claim, alternative))
end

"""Recall prior failures relevant to the current context (avoid repeating them)."""
function recall_failures(agent::CognitiveAgent, context::AbstractString)
    key = lowercase(strip(String(context)))
    return [f for f in agent.failures.records if occursin(lowercase(f.context), key) || occursin(key, lowercase(f.context))]
end

# ===== END memory.jl =====


# ===== BEGIN language.jl =====

# language.jl — Replaceable local language interface (spec §38).
#
# Deliberately a transparent symbolic grammar (no LLM, no learned tokenizer).
# Produces structured representations; the reasoning engine never operates on raw
# text alone. Intent (spec §16, §17) is first-class.

const _NUMBER_WORDS = Dict("zero"=>0,"one"=>1,"two"=>2,"three"=>3,"four"=>4,"five"=>5,
                           "six"=>6,"seven"=>7,"eight"=>8,"nine"=>9,"ten"=>10)

# Ordered longest-first so multi-word predicates match before single words.
const _PREDICATE_PHRASES = [
    ("is opposite of", :opposite_of), ("is similar to", :similar_to),
    ("results in", :causes), ("leads to", :causes), ("related to", :related_to),
    ("part of", :part_of), ("is a", :is_a),
    ("causes", :causes), ("cause", :causes),
    ("prevents", :prevents), ("prevent", :prevents),
    ("requires", :requires), ("require", :requires),
    ("supports", :supports), ("support", :supports),
    ("contradicts", :contradicts), ("contradict", :contradicts),
    ("inhibits", :inhibits), ("inhibit", :inhibits),
    ("blocks", :blocks), ("block", :blocks),
    ("reduces", :reduces), ("reduce", :reduces),
    ("enables", :enables), ("enable", :enables),
    ("affects", :affects), ("affect", :affects),
]

const _STOPWORDS = Set(["the","a","an","of","to","in","on","at","and","or","that","this",
                        "then","with","for","does","do","is","are","can","why","what","how",
                        "will","would","it","its","as","by","be"])

tokenize(text::AbstractString) = split(lowercase(strip(String(text))))
normalize(text::AbstractString) = strip(replace(lowercase(String(text)), r"[?.!,]" => ""))
segment_sentences(text::AbstractString) = filter(!isempty, strip.(split(String(text), r"[.!?]+")))

function detect_repetition_request(text::AbstractString)
    lower = lowercase(String(text))
    requested = occursin(r"\b(repeat|say (it|that|this) again|say again|again exactly|reiterate)\b", lower)
    count = 1
    m = match(r"(\d+)\s*times?", lower)
    if m !== nothing
        count = parse(Int, m.captures[1])
    else
        for (w, n) in _NUMBER_WORDS
            if occursin(Regex("\\b$w\\s+times?\\b"), lower)
                count = n; break
            end
        end
    end
    return requested, count
end

detect_instruction(text::AbstractString) =
    occursin(r"\b(repeat|explain|list|convert|transform|rewrite|summariz|compare|solve|show)\b", lowercase(String(text)))

function extract_constraints(text::AbstractString)
    lower = lowercase(String(text))
    constraints = String[]
    structure = ""
    exclusions = String[]
    occursin(r"\bin json\b", lower)      && (structure = "json")
    occursin(r"\bas a list\b", lower)    && (structure = "list")
    occursin(r"\bone (word|sentence)\b", lower) && push!(constraints, "concise")
    occursin(r"\bonly\b", lower)         && push!(constraints, "only")
    m = match(r"\bwithout ([a-z ]+)", lower)
    m !== nothing && push!(exclusions, strip(m.captures[1]))
    me = match(r"\bexclude ([a-z ]+)", lower)
    me !== nothing && push!(exclusions, strip(me.captures[1]))
    return constraints, structure, exclusions
end

"""Parse explicit user intent (spec §17)."""
function parse_intent(text::AbstractString)
    lower = lowercase(strip(String(text)))
    requested, count = detect_repetition_request(text)
    goal = if requested
        :repeat
    elseif startswith(lower, "why")
        :explain
    elseif occursin(r"^(does|is|are|can|do|will|would)\b", lower)
        :verify
    elseif occursin(r"\b(transform|convert|rewrite|translate)\b", lower)
        :transform
    else
        :assert
    end
    constraints, structure, exclusions = extract_constraints(text)
    intent = UserIntent(goal)
    intent.requested_repetition = requested
    intent.requested_repetition_count = count
    intent.requested_structure = structure
    intent.constraints = constraints
    intent.exclusions = exclusions
    intent.requested_quantity = count
    return intent
end

"""Return `(predicate::Symbol, subject::String, object::String)` or `nothing`."""
function match_relation(normalized::AbstractString)
    for (phrase, pred) in _PREDICATE_PHRASES
        m = match(Regex("^(.+?)\\s+" * phrase * "\\s+(.+)\$"), normalized)
        m !== nothing && return (pred, strip(m.captures[1]), strip(m.captures[2]))
    end
    return nothing
end

content_words(normalized::AbstractString) =
    filter(w -> length(w) > 2 && !(w in _STOPWORDS), split(normalized))

"""
Full structured parse used by the reasoning engine. Creates concepts, asserts a
relation when the input is a declarative fact, and detects a query target when
the input is interrogative.
"""
function parse_input!(agent::CognitiveAgent, text::String; episode::Int32=Int32(0), domain::String="general")
    intent = parse_intent(text)
    normalized = normalize(text)
    is_query = intent.goal in (:verify, :explain)
    body = is_query ? strip(replace(normalized, r"^(why|does|is|are|can|do|will|would)\s+" => "")) : normalized
    body = strip(replace(body, r"\b(in json|as a list|in one (word|sentence)|one (word|sentence)|only)\b" => ""))
    body = strip(replace(body, r"\b(without|exclude)\s+[a-z0-9_\- ]+\b" => ""))

    matched = match_relation(body)
    if matched !== nothing
        pred, subj, obj = matched
        if is_query
            s = concept!(agent, subj; domain=domain); o = concept!(agent, obj; domain=domain)
            return (seed_ids=Int32[s,o], learned_relation=nothing,
                    query_target=(s, pred, o), intent=intent, is_query=true)
        else
            rel = add_relation!(agent, subj, pred, obj; provenance=:input, domain=domain, episode=episode)
            return (seed_ids=Int32[rel.subject, rel.object], learned_relation=rel,
                    query_target=nothing, intent=intent, is_query=false)
        end
    end

    words = content_words(body)
    ids = Int32[concept!(agent, w; domain=domain) for w in words]
    return (seed_ids=ids, learned_relation=nothing, query_target=nothing, intent=intent, is_query=is_query)
end

# ===== END language.jl =====


# ===== BEGIN hypotheses.jl =====

# hypotheses.jl — Hypothesis construction, support checks, multi-signal scoring.
#
# Each score component has a documented meaning (spec §26); nothing is a random
# number. Scores are transient Float32 evaluation quantities, not learned weights.

# Scoring weights (auxiliary, configurable-in-spirit; documented meanings).
const _W_LOGICAL      = 0.20f0   # claim fits known logical/predicate structure
const _W_CONSTRAINT   = 0.15f0   # satisfies stated constraints
const _W_MEMORY       = 0.25f0   # supported by stored/derivable evidence
const _W_CONCEPT      = 0.10f0   # involved concepts are historically reliable
const _W_SYNAPTIC     = 0.20f0   # support from persistent ternary states
const _W_EXPERIENCE   = 0.30f0   # prior success/failure with the same claim
const _W_SIMPLICITY   = 0.10f0   # fewer assumptions preferred
const _W_NOVELTY      = 0.05f0   # structural novelty (generation property only)
const _W_GOAL         = 0.15f0   # relevance to the explicit goal/query

const _NOVEL_ORIGINS = Set{Symbol}([:recombination, :analogy, :abstraction, :cross_domain,
                                    :counterfactual, :inverse, :causal_reversal,
                                    :constraint_relaxation, :constraint_tightening])
const _CANONICAL_INVERSES = Dict{Symbol,Symbol}(
    :caused_by=>:causes, :required_by=>:requires, :supported_by=>:supports,
    :has_part=>:part_of, :enabled_by=>:enables, :prevented_by=>:prevents,
    :affected_by=>:affects,
)

function canonical_claim(claim::CognitiveRelation)
    predicate = get(_CANONICAL_INVERSES, claim.predicate, nothing)
    predicate === nothing && return claim
    return CognitiveRelation(claim.object, predicate, claim.subject, claim.provenance)
end

function same_claim(a::CognitiveRelation, b::CognitiveRelation)
    left = canonical_claim(a); right = canonical_claim(b)
    return left.subject == right.subject && left.predicate == right.predicate && left.object == right.object
end

function _claim_from_text(agent::CognitiveAgent, text::String)
    parsed = match_relation(normalize(text))
    parsed === nothing && return nothing
    predicate, subject, object = parsed
    s = get_concept_id(agent, subject); o = get_concept_id(agent, object)
    (s == 0 || o == 0) && return nothing
    return canonical_claim(CognitiveRelation(s, predicate, o, :memory))
end

function failure_experience(agent::CognitiveAgent, claim::CognitiveRelation)
    target = canonical_claim(claim)
    for failure in agent.failures.records
        if failure.successful_alternative !== nothing
            alternative = _claim_from_text(agent, failure.successful_alternative)
            if alternative !== nothing && same_claim(alternative, target)
                return 1f0
            end
        end
    end
    for failure in agent.failures.records
        if failure.hypothesis_claim !== nothing && same_claim(failure.hypothesis_claim, target)
            return -1f0
        end
    end
    return 0f0
end

function _direct_synaptic_support(agent::CognitiveAgent, claim::CognitiveRelation)
    canonical = canonical_claim(claim)
    edge = find_synapse(agent.network, canonical.subject, canonical.object)
    edge === nothing && return 0f0
    return Float32(edge.value * predicate_polarity(canonical.predicate))
end

function synaptic_support(agent::CognitiveAgent, claim::CognitiveRelation)
    canonical = canonical_claim(claim)
    best = _direct_synaptic_support(agent, canonical)
    if canonical.predicate == :causes
        for r1 in agent.semantic.relations, r2 in agent.semantic.relations
            (r1.predicate == :causes && r2.predicate == :causes &&
             r1.subject == canonical.subject && r1.object == r2.subject &&
             r2.object == canonical.object) || continue
            path = 0.6f0 * min(_direct_synaptic_support(agent, r1),
                              _direct_synaptic_support(agent, r2))
            abs(path) > abs(best) && (best = path)
        end
    end
    return clamp(best, -1f0, 1f0)
end

"""Direct evidence: the exact canonical relation is already stored."""
function direct_support(agent::CognitiveAgent, claim::CognitiveRelation)
    canonical = canonical_claim(claim)
    return relation_exists(agent, canonical.subject, canonical.predicate, canonical.object)
end

"""Transitive causal support: A causes B and B causes C ⇒ A causes C is derivable."""
function transitive_causal_support(agent::CognitiveAgent, claim::CognitiveRelation)
    canonical = canonical_claim(claim)
    canonical.predicate == :causes || return (false, Int32(0))
    for r1 in agent.semantic.relations
        (r1.predicate == :causes && r1.subject == canonical.subject) || continue
        for r2 in agent.semantic.relations
            if r2.predicate == :causes && r2.subject == r1.object && r2.object == canonical.object
                return (true, r1.object)
            end
        end
    end
    return (false, Int32(0))
end

function new_hypothesis(agent::CognitiveAgent, claim::CognitiveRelation;
                        supporting::Vector{Int32}=Int32[], opposing::Vector{Int32}=Int32[],
                        assumptions::Vector{String}=String[], consequences::Vector{String}=String[],
                        strategy::Symbol=:deductive, origin::Symbol=:direct)
    CognitiveHypothesis(next_hypothesis_id!(agent), claim, supporting, opposing, assumptions,
                        consequences, HypothesisScore(), 0f0, 0f0, :new, strategy, origin)
end

hypothesis_signature(h::CognitiveHypothesis) =
    hash((h.claim.subject, h.claim.predicate, h.claim.object, h.strategy))

"""Compute the documented multi-signal score breakdown for a hypothesis (spec §26)."""
function score_hypothesis!(agent::CognitiveAgent, h::CognitiveHypothesis, state::CognitiveState;
                           contradiction_count::Int=0, contested_count::Int=0)
    claim = h.claim
    sc = h.scores
    supported_direct = direct_support(agent, claim)
    (t_ok, _) = transitive_causal_support(agent, claim)
    sc.synaptic_support = synaptic_support(agent, claim)
    sc.experience = failure_experience(agent, claim)

    sc.logical_compatibility = sc.synaptic_support > 0f0 ? 1f0 :
                               sc.synaptic_support < 0f0 ? 0.1f0 :
                               supported_direct ? 0.6f0 : t_ok ? 0.65f0 : 0.5f0

    if isempty(state.constraints)
        sc.constraint_satisfaction = 1f0
    else
        violated = count(c -> occursin("only", lowercase(c)) && length(h.assumptions) > 1, state.constraints)
        sc.constraint_satisfaction = 1f0 - Float32(violated) / length(state.constraints)
    end

    sc.memory_evidence = supported_direct ?
        (sc.synaptic_support > 0f0 ? 1f0 : sc.synaptic_support < 0f0 ? 0.2f0 : 0.6f0) :
        t_ok ? (sc.synaptic_support > 0f0 ? 0.7f0 : 0.5f0) : 0.2f0

    if isempty(h.supporting_concepts)
        sc.concept_compatibility = 0.5f0
    else
        rels = Float32[]
        for id in h.supporting_concepts
            c = get(agent.semantic.concepts, id, nothing)
            c !== nothing && push!(rels, concept_reliability(c))
        end
        sc.concept_compatibility = isempty(rels) ? 0.5f0 : sum(rels)/length(rels)
    end

    sc.contradiction_penalty = Float32(contradiction_count) * 0.8f0 +
                               Float32(contested_count) * 0.2f0
    sc.simplicity = 1f0 / (1f0 + Float32(length(h.assumptions)))
    sc.novelty = h.origin in _NOVEL_ORIGINS ? 0.7f0 : 0.1f0

    goal_ids = Set(state.concepts)
    involved = (claim.subject in goal_ids) + (claim.object in goal_ids)
    sc.goal_relevance = involved == 2 ? 1f0 : involved == 1 ? 0.3f0 : 0f0

    aggregate = _W_LOGICAL*sc.logical_compatibility + _W_CONSTRAINT*sc.constraint_satisfaction +
                _W_MEMORY*sc.memory_evidence + _W_CONCEPT*sc.concept_compatibility +
                _W_SYNAPTIC*sc.synaptic_support + _W_EXPERIENCE*sc.experience +
                _W_SIMPLICITY*sc.simplicity + _W_NOVELTY*sc.novelty + _W_GOAL*sc.goal_relevance
    h.score = aggregate - sc.contradiction_penalty
    return h.score
end

# ===== END hypotheses.jl =====


# ===== BEGIN contradiction.jl =====

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

# ===== END contradiction.jl =====


# ===== BEGIN self_correction.jl =====

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

# ===== END self_correction.jl =====


# ===== BEGIN meta_reasoning.jl =====

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

# ===== END meta_reasoning.jl =====


# ===== BEGIN confidence.jl =====

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

# ===== END confidence.jl =====


# ===== BEGIN novelty.jl =====

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

# ===== END novelty.jl =====

# ===== END memory.jl =====


# ===== BEGIN reasoning.jl =====

# reasoning.jl -- consolidated bundle (auto-generated)
# Sources merged: agent.jl, reasoning.jl, output_generation.jl, evaluator.jl, learning.jl, diagnostics.jl
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt


# ===== BEGIN agent.jl =====

# agent.jl — Agent construction and lifecycle helpers.

const ARCHITECTURE_VERSION = v"0.3.0"
const SERIALIZATION_VERSION = 3

"""Construct a fresh cognitive agent from a configuration (spec §85)."""
function cognitive_agent(config::CognitiveConfig=CognitiveConfig())
    CognitiveAgent(config,
                   SparseTernaryNetwork(),
                   WorkingMemory(config.working_memory_capacity),
                   SemanticMemory(),
                   EpisodicMemory(),
                   StrategyMemory(),
                   FailureMemory(),
                   ContradictionRegistry(),
                   RunMetrics(),
                   Random.MersenneTwister(config.seed),
                   "",
                   nothing,
                   nothing,
                   Int32(1),
                   ARCHITECTURE_VERSION,
                   SERIALIZATION_VERSION,
                   string(Dates.now()))
end

"""Construct an agent for a named scale preset (spec §123)."""
cognitive_agent(scale::Symbol; kwargs...) = cognitive_agent(config_for_scale(scale; kwargs...))

next_hypothesis_id!(agent::CognitiveAgent) = (id = agent.next_hypothesis_id; agent.next_hypothesis_id += Int32(1); id)

"""Reset transient per-episode network activation without touching learned state."""
function soft_reset!(agent::CognitiveAgent)
    reset_activation!(agent.network)
    return agent
end

"""Total learned synapse count."""
synapse_count(agent::CognitiveAgent) = agent.network.synapse_count
concept_count(agent::CognitiveAgent) = length(agent.semantic.concepts)
episode_count(agent::CognitiveAgent) = length(agent.episodic.episodes)

"""Verify architecture integrity on demand (spec §4, §48)."""
verify_integrity(agent::CognitiveAgent) = assert_network_invariants(agent.network)

# ===== END agent.jl =====


# ===== BEGIN reasoning.jl =====

# reasoning.jl — The explicit reasoning state machine (spec §24, §33, §34).
#
# This is the controller that ties the engine together: it walks the named
# ReasoningPhase states, activating the ternary network, retrieving memory,
# generating and testing hypotheses, self-correcting on contradiction, and
# selecting a conclusion under the reasoning budget. It performs NO learning and
# NO output rendering — those are separate concerns (evaluator/learning/output).
# The only "weights" touched are transient eligibility traces set by activation.

"""Build a fresh per-episode reasoning state (spec §23 working state)."""
function fresh_state(agent::CognitiveAgent, input::String, intent::UserIntent, goal::String)
    CognitiveState(input, goal, intent,
                   Int32[], Int32[], Tuple{Int32,Int32}[], Int32[], Int[],
                   CognitiveHypothesis[], nothing, copy(intent.constraints), String[],
                   INITIALIZE, ReasoningPhase[], Symbol[], Symbol[],
                   0f0, 0, TraceBuffer(true),
                   0, 0, 0, 0, 0, false,
                   Set{UInt64}(), Set{UInt64}())
end

"""State the explicit goal for this episode from the parsed input (spec §24 FORM_GOAL)."""
function form_goal(agent::CognitiveAgent, state::CognitiveState, parsed)
    if parsed.query_target !== nothing
        s, p, o = parsed.query_target
        return "establish whether $(concept_name(agent,s)) $p $(concept_name(agent,o))"
    elseif parsed.learned_relation !== nothing
        r = parsed.learned_relation
        return "assert $(concept_name(agent,r.subject)) $(r.predicate) $(concept_name(agent,r.object))"
    elseif state.intent.goal == :repeat
        return "repeat the previous conclusion"
    else
        return "interpret and respond to: $(state.input)"
    end
end

"""Broaden retrieval when no viable hypothesis was found (spec §34 retrieve-more)."""
function expand_retrieval!(agent::CognitiveAgent, state::CognitiveState)
    seeds = vcat(state.concepts, state.retrieved_concepts, collect(state.active_units))
    for id in retrieve_concepts(agent, seeds; limit=agent.config.retrieval_limit, trace=state.trace)
        id in state.retrieved_concepts && continue
        id in state.concepts && continue
        push!(state.retrieved_concepts, id)
    end
    return state
end

"""Commit to the best viable hypothesis using evidence, experience, and goal fit."""
function select_hypothesis!(agent::CognitiveAgent, state::CognitiveState;
                            target::Union{Nothing,Tuple{Int32,Symbol,Int32}}=nothing)
    viable = [h for h in state.hypotheses if h.status in (:supported, :active, :weak)]
    isempty(viable) && return nothing
    target_claim = target === nothing ? nothing : CognitiveRelation(target[1], target[2], target[3], :goal)
    matches(h) = target_claim !== nothing && same_claim(h.claim, target_claim)
    goal_ids = Set(state.concepts)
    function selection_score(h)
        involved = isempty(goal_ids) ? 2 : ((h.claim.subject in goal_ids) + (h.claim.object in goal_ids))
        off_topic_penalty = involved == 0 ? 1.0f0 : 0.0f0
        target_bonus = matches(h) && h.scores.experience >= 0f0 ? 0.85f0 : 0f0
        support_bonus = h.status == :supported ? 0.15f0 : 0f0
        direct_bonus = h.origin == :direct ? 0.10f0 : 0f0
        strategy_bias = 0.10f0 * (strategy_success_rate(agent, h.strategy) - 0.5f0)
        return h.score + target_bonus + support_bonus + direct_bonus + strategy_bias - off_topic_penalty
    end
    sort!(viable; by=selection_score, rev=true)
    best = viable[1]
    state.current_hypothesis = best.id
    best.status = :selected
    emit!(state.trace, HYPOTHESIS_SELECTED; hypothesis=best.id,
          score=round(best.score; digits=3), selection_score=round(selection_score(best); digits=3),
          confidence=round(best.confidence; digits=3), matches_goal=matches(best),
          synaptic_support=best.scores.synaptic_support, experience=best.scores.experience)
    return best
end

function mark_selected_pathways!(agent::CognitiveAgent, state::CognitiveState,
                                  selected::CognitiveHypothesis)
    claim = canonical_claim(selected.claim)
    output_pathway = (claim.subject, claim.object)
    supporting = Set(selected.supporting_concepts)
    pathways = Tuple{Int32,Int32}[]
    for relation in agent.semantic.relations
        canonical = canonical_claim(relation)
        pathway = (canonical.subject, canonical.object)
        pathway == output_pathway && continue
        canonical.subject in supporting && canonical.object in supporting || continue
        find_synapse(agent.network, pathway[1], pathway[2]) === nothing || push!(pathways, pathway)
    end
    output_edge = find_synapse(agent.network, output_pathway[1], output_pathway[2])
    output_edge === nothing && (output_edge = connect!(agent.network, output_pathway[1], output_pathway[2], Int8(0);
                                                   episode=Int32(agent.episodic.next_id)))
    push!(pathways, output_pathway)

    for pathway in unique(pathways)
        edge = find_synapse(agent.network, pathway[1], pathway[2])
        edge === nothing && continue
        edge.eligibility = min(agent.config.eligibility.cap,
                               edge.eligibility + agent.config.eligibility.gain)
        edge.times_activated += 1
        push!(state.active_pathways, pathway)
    end
    return state
end

"""
Internal self-evaluation (spec §37): the agent's own judgement of whether it
produced a confident, committed conclusion. This is distinct from the external
reward evaluator; it only reflects on the reasoning state and emits an event.
"""
function self_reflect!(agent::CognitiveAgent, state::CognitiveState)
    sel = state.current_hypothesis
    committed = sel !== nothing && state.confidence >= agent.config.confidence_select_threshold
    emit!(state.trace, SELF_EVALUATED; committed=committed,
          confidence=round(state.confidence; digits=3),
          viable=count(h -> h.status in (:supported, :active, :weak), state.hypotheses),
          rejected=count(h -> h.status == :rejected, state.hypotheses))
    return committed
end

"""
Run one reasoning episode over `input` and return the resulting `CognitiveState`
(spec §24). Performs the full phase walk under the reasoning budget; requested
repetition is short-circuited to the CONCLUDE phase because repeating is the
correct behaviour (intent-aware, spec §16/§17). No synapse learning happens here.
"""
function reason!(agent::CognitiveAgent, input::String; domain::String="general")
    soft_reset!(agent)
    episode = agent.episodic.next_id

    # ── INITIALIZE ──────────────────────────────────────────────────────────
    intent = parse_intent(input)
    state = fresh_state(agent, input, intent, "")
    emit!(state.trace, EPISODE_STARTED; episode=episode, input=input)
    emit!(state.trace, INTENT_DETECTED; goal=intent.goal,
          requested_repetition=intent.requested_repetition,
          repetition_count=intent.requested_repetition_count)
    transition!(state, INITIALIZE)

    # ── PARSE + REPRESENT ─────────────────────────────────────────────────────
    transition!(state, PARSE)
    parsed = parse_input!(agent, input; episode=Int32(episode), domain=domain)
    state.intent = parsed.intent
    state.constraints = copy(parsed.intent.constraints)
    state.concepts = collect(parsed.seed_ids)
    emit!(state.trace, PARSED_INPUT; concepts=state.concepts, is_query=parsed.is_query,
          learned=parsed.learned_relation !== nothing)
    remember!(agent, input, state.concepts; step=state.steps_used,
              kind = parsed.is_query ? :goal : :observation)

    transition!(state, REPRESENT)
    state.active_units, state.active_pathways =
        activate!(agent.network, state.concepts, agent.config.eligibility; hops=2, trace=state.trace)

    # ── RETRIEVE ──────────────────────────────────────────────────────────────
    transition!(state, RETRIEVE)
    state.retrieved_concepts = retrieve_concepts(agent, state.concepts; trace=state.trace)
    state.retrievals_used += 1
    recalled = retrieve_episodes(agent, state.concepts; limit=3)
    state.retrieved_memories = [ep.id for ep in recalled]
    for ep in recalled
        emit!(state.trace, MEMORY_RETRIEVED; episode=ep.id, goal=ep.goal)
    end

    # ── FORM_GOAL ─────────────────────────────────────────────────────────────
    transition!(state, FORM_GOAL)
    state.goal = form_goal(agent, state, parsed)
    emit!(state.trace, GOAL_CREATED; goal=state.goal)

    # Requested repetition: repeating is the intended, correct action (spec §16).
    if intent.requested_repetition
        transition!(state, CONCLUDE)
        state.confidence = 1f0
        transition!(state, FINISH)
        return state
    end

    # ── Reasoning cycle: GENERATE → TEST → (backtrack / retrieve-more) ──────────
    for _ in 1:4
        transition!(state, GENERATE_HYPOTHESES)
        generate_hypotheses!(agent, state)

        transition!(state, TEST)
        test_hypotheses!(agent, state)

        decision = meta_decide(agent, state)
        if decision === :backtrack
            transition!(state, DETECT_CONTRADICTION)
            transition!(state, BACKTRACK)
            self_correct!(agent, state)
            any_viable(state) && break
        elseif decision === :retrieve_more
            transition!(state, RETRIEVE)
            expand_retrieval!(agent, state)
            state.retrievals_used += 1
        elseif decision === :switch_strategy
            transition!(state, REFINE)
            expand_retrieval!(agent, state)
            state.retrievals_used += 1
        else                    # :select or :stop
            break
        end
    end

    # ── VERIFY (continuous ternary invariant, spec §4) ──────────────────────────
    transition!(state, VERIFY)
    agent.config.debug_invariants && verify_integrity(agent)

    # ── SELECT ──────────────────────────────────────────────────────────────────
    transition!(state, SELECT)
    selection_target = parsed.query_target
    if selection_target === nothing && parsed.learned_relation !== nothing
        relation = parsed.learned_relation
        selection_target = (relation.subject, relation.predicate, relation.object)
    end
    selected = select_hypothesis!(agent, state; target=selection_target)
    selected !== nothing && mark_selected_pathways!(agent, state, selected)

    # ── CONCLUDE ──────────────────────────────────────────────────────────────
    transition!(state, CONCLUDE)
    state.confidence = state_confidence(state)

    # ── SELF_EVALUATE ───────────────────────────────────────────────────────────
    transition!(state, SELF_EVALUATE)
    self_reflect!(agent, state)

    transition!(state, FINISH)
    agent.last_state = state
    return state
end

# ===== END reasoning.jl =====


# ===== BEGIN output_generation.jl =====

# output_generation.jl — Rule-driven language generated from structured state.


predicate_text(predicate::Symbol) = replace(string(predicate), "_" => " ")

function _json_escape(text::String)
    replace(text, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t")
end

function _format_conclusion(text::String, state::CognitiveState)
    if state.intent.requested_structure == "json"
        return "{\"conclusion\":\"$(_json_escape(text))\",\"confidence\":$(round(state.confidence; digits=3))}"
    elseif state.intent.requested_structure == "list"
        return "- $text"
    end
    return text
end

"""Render an answer solely from the selected structured hypothesis and intent."""
function render_output(agent::CognitiveAgent, state::CognitiveState)
    if state.intent.requested_repetition
        isempty(agent.last_output) && return "There is no previous conclusion to repeat."
        count = max(1, state.intent.requested_repetition_count)
        return join(fill(agent.last_output, count), "\n")
    end

    h = selected_hypothesis(state)
    h === nothing && return "Insufficient supported evidence; more information is required."

    subject = concept_name(agent, h.claim.subject)
    object = concept_name(agent, h.claim.object)
    relation = predicate_text(h.claim.predicate)
    conclusion = "$subject $relation $object"

    if state.intent.goal == :verify
        conclusion = "Yes. $conclusion."
    else
        conclusion = "Conclusion: $conclusion."
    end

    if h.claim.predicate == :causes && h.origin == :deduction
        supported, bridge = transitive_causal_support(agent, h.claim)
        if supported
            middle = concept_name(agent, bridge)
            conclusion *= " Evidence: $subject causes $middle; $middle causes $object."
        end
    elseif !isempty(h.assumptions) && state.intent.goal == :explain
        conclusion *= " Basis: $(join(h.assumptions, "; "))."
    end

    return _format_conclusion(conclusion, state)
end

# ===== END output_generation.jl =====


# ===== BEGIN evaluator.jl =====

# evaluator.jl — Behavioral, intent-aware, multi-mode outcome evaluation.

_canonical_text(text::AbstractString) = join(split(lowercase(replace(strip(String(text)), r"[^a-z0-9_\-]+" => " "))), " ")

function _expected_relation(text::String)
    candidate = replace(normalize(text), r"^(conclusion\s*:\s*|yes\s*\.\s*)" => "")
    first_sentence = first(split(candidate, '.'; limit=2))
    return match_relation(strip(first_sentence))
end

function repetition_detected(agent::CognitiveAgent, state::CognitiveState, output::String)
    isempty(agent.last_output) && return false
    if state.intent.requested_repetition
        count = max(1, state.intent.requested_repetition_count)
        expected = join(fill(agent.last_output, count), "\n")
        return _canonical_text(output) == _canonical_text(expected)
    end

    segments = filter(!isempty, _canonical_text.(segment_sentences(replace(output, '\n' => '.'))))
    internally_repeated = length(segments) > 1 && length(unique(segments)) < length(segments)
    same_input = !isempty(agent.episodic.episodes) &&
                 _canonical_text(agent.episodic.episodes[end].input) == _canonical_text(state.input)
    repeated_previous = same_input && _canonical_text(output) == _canonical_text(agent.last_output)
    return internally_repeated || repeated_previous
end

function _structural_correctness(agent::CognitiveAgent, state::CognitiveState, expected::String)
    h = selected_hypothesis(state)
    h === nothing && return 0f0
    relation = _expected_relation(expected)
    relation === nothing && return 0f0
    predicate, subject, object = relation
    return (_concept_key(subject) == concept_name(agent, h.claim.subject) &&
            predicate == h.claim.predicate &&
            _concept_key(object) == concept_name(agent, h.claim.object)) ? 1f0 : 0f0
end

function _textual_correctness(output::String, expected::String; exact::Bool=false)
    actual = _canonical_text(output)
    target = _canonical_text(expected)
    exact && return actual == target ? 1f0 : 0f0
    occursin(target, actual) && return 1f0
    expected_tokens = Set(split(target))
    isempty(expected_tokens) && return 0f0
    output_tokens = Set(split(actual))
    return Float32(length(intersect(expected_tokens, output_tokens)) / length(expected_tokens))
end

function _constraint_score(state::CognitiveState, output::String)
    checks = Bool[]
    reasons = String[]
    if state.intent.requested_structure == "json"
        valid = startswith(strip(output), "{") && endswith(strip(output), "}")
        push!(checks, valid); !valid && push!(reasons, "requested JSON structure was not produced")
    elseif state.intent.requested_structure == "list"
        lines = filter(!isempty, strip.(split(output, '\n')))
        valid = !isempty(lines) && all(startswith(line, "- ") for line in lines)
        push!(checks, valid); !valid && push!(reasons, "requested list structure was not produced")
    end
    for exclusion in state.intent.exclusions
        valid = !occursin(_canonical_text(exclusion), _canonical_text(output))
        push!(checks, valid); !valid && push!(reasons, "excluded content appeared: $exclusion")
    end
    if "concise" in state.intent.constraints || "concise" in state.constraints
        valid = length(segment_sentences(output)) <= 1
        push!(checks, valid); !valid && push!(reasons, "requested concise form was exceeded")
    end
    score = isempty(checks) ? 1f0 : Float32(count(identity, checks) / length(checks))
    return score, reasons
end

function _evaluation_mode(mode::Symbol, expected_output::Union{Nothing,String}, state::Union{Nothing,CognitiveState}=nothing)
    canonical = Symbol(replace(string(mode), "-" => "_"))
    canonical != :auto && return canonical
    if expected_output === nothing
        if state !== nothing && (!isempty(state.intent.constraints) || !isempty(state.intent.exclusions) || !isempty(state.intent.requested_structure))
            return :constraint_based
        end
        return :rule_based
    end
    return _expected_relation(expected_output) === nothing ? :rule_based : :structural
end

"""Evaluate behavior and retain both actual and expected quality (spec §13–§19, §59, §103)."""
function evaluate_output(agent::CognitiveAgent, state::CognitiveState, output::String;
                         expected_output::Union{Nothing,String}=nothing,
                         expected_quality::Union{Nothing,Symbol}=nothing,
                         evaluator_mode::Symbol=:auto)
    expected_quality !== nothing && !(expected_quality in QUALITY_CLASSES) &&
        error("unknown expected quality: $expected_quality")
    mode = _evaluation_mode(evaluator_mode, expected_output, state)
    mode in (:exact_match, :rule_based, :constraint_based, :structural,
             :semantic_structure, :human_feedback) || error("unknown evaluator mode: $evaluator_mode")

    requested = state.intent.requested_repetition
    repeated = repetition_detected(agent, state, output)
    h = selected_hypothesis(state)
    selected_consistent = requested || (h !== nothing && h.status ∉ (:rejected, :contradicted))
    consistency = selected_consistent && isempty(state.contradictions) ? 1f0 :
                  selected_consistent ? 0.7f0 : 0f0
    relevance = requested ? 1f0 : h === nothing ? 0f0 : h.scores.goal_relevance
    constraint, constraint_reasons = _constraint_score(state, output)

    correctness = if requested
        repeated ? 1f0 : 0f0
    elseif expected_output === nothing
        h === nothing ? 0f0 : clamp(state.confidence, 0f0, 1f0)
    elseif mode == :exact_match
        _textual_correctness(output, expected_output; exact=true)
    elseif mode in (:structural, :semantic_structure)
        structural = _structural_correctness(agent, state, expected_output)
        max(structural, _textual_correctness(output, expected_output))
    else
        _textual_correctness(output, expected_output)
    end

    instruction_ok = state.intent.goal != :transform &&
                     !(state.intent.requested_repetition && isempty(agent.last_output))
    quality = if mode == :human_feedback
        expected_quality === nothing && error("human-feedback evaluation requires expected_quality")
        expected_quality
    elseif requested && repeated
        :REQUESTED_REPETITION
    elseif repeated
        :REPETITIVE
    elseif !instruction_ok
        :INSTRUCTION_FAILURE
    elseif constraint < 1f0
        :CONSTRAINT_FAILURE
    elseif !isempty(state.contradictions) && !selected_consistent
        :CONTRADICTORY
    elseif expected_output !== nothing && correctness < 0.999f0
        :WRONG
    elseif h === nothing
        :INCOMPLETE
    elseif relevance < 0.25f0
        :IRRELEVANT
    elseif correctness >= 0.999f0 && state.confidence >= 0.80f0
        :EXCELLENT
    elseif correctness >= 0.65f0 && state.confidence >= 0.45f0
        :GOOD
    elseif correctness > 0f0 || state.confidence > 0f0
        :PARTIAL
    else
        :NEUTRAL
    end

    signal = get(agent.config.quality_signals, quality, 0f0)
    reward = signal > 0f0 && agent.config.reward_enabled ? signal : 0f0
    punishment = signal < 0f0 && agent.config.punishment_enabled ? -signal : 0f0
    reasons = String["actual quality=$quality", "intent=$(state.intent.goal)", "evaluator mode=$mode"]
    append!(reasons, constraint_reasons)
    repeated && push!(reasons, requested ? "repetition explicitly requested and fulfilled" :
                                       "semantically equivalent content repeated without a repetition instruction")
    expected_output !== nothing && push!(reasons, correctness >= 0.999f0 ?
        "behavioral target matched" : "behavioral target did not match")
    !selected_consistent && push!(reasons, "no consistent selected hypothesis")
    signal > 0f0 && !agent.config.reward_enabled && push!(reasons, "reward suppressed by config (reward_enabled=false)")
    signal < 0f0 && !agent.config.punishment_enabled && push!(reasons, "punishment suppressed by config (punishment_enabled=false)")
    if expected_quality !== nothing
        push!(reasons, expected_quality == quality ? "actual quality matched expected quality" :
              "quality discrepancy: expected $expected_quality, actual $quality")
    end

    return EvaluationResult(quality, state.intent.goal, reward, punishment, repeated, requested,
                            correctness, relevance, consistency, constraint, mode,
                            expected_quality, reasons)
end

# ===== END evaluator.jl =====


# ===== BEGIN learning.jl =====

# learning.jl — Eligibility credit assignment and discrete ternary plasticity.

function credit_level(relevance::Float32)
    relevance >= 0.80f0 ? STRONG_CREDIT : relevance >= 0.55f0 ? MODERATE_CREDIT :
    relevance >= 0.20f0 ? WEAK_CREDIT : NEUTRAL_CREDIT
end
credit_level(edge::SparseSynapse) = credit_level(edge.eligibility)

function relation_polarity(agent::CognitiveAgent, edge::SparseSynapse,
                           selected::Union{Nothing,CognitiveHypothesis}=nothing)
    if selected !== nothing
        claim = canonical_claim(selected.claim)
        claim.subject == edge.source && claim.object == edge.target &&
            return predicate_polarity(claim.predicate)
    end
    relations = CognitiveRelation[]
    for relation in agent.semantic.relations
        canonical = canonical_claim(relation)
        canonical.subject == edge.source && canonical.object == edge.target && push!(relations, canonical)
    end
    isempty(relations) && return edge.value
    matching = findfirst(r -> predicate_polarity(r.predicate) == edge.value, relations)
    return predicate_polarity(relations[matching === nothing ? 1 : matching].predicate)
end

"""Assign local temporal and causal credit only to pathways used in this episode."""
function assign_credit(agent::CognitiveAgent, state::CognitiveState,
                       evaluation::EvaluationResult)
    signal = evaluation.reward > 0f0 ? evaluation.reward : -evaluation.punishment
    isempty(state.active_pathways) && return CreditAssignment[]
    selected = selected_hypothesis(state)
    relevant_concepts = selected === nothing ? Set{Int32}() :
        Set(vcat(selected.supporting_concepts, Int32[selected.claim.subject, selected.claim.object]))
    first_position = Dict{Tuple{Int32,Int32},Int}()
    last_position = Dict{Tuple{Int32,Int32},Int}()
    for (index, pathway) in enumerate(state.active_pathways)
        !haskey(first_position, pathway) && (first_position[pathway] = index)
        last_position[pathway] = index
    end

    assignments = CreditAssignment[]
    path_count = length(state.active_pathways)
    for (pathway, last_pos) in sort!(collect(last_position); by=last)
        edge = find_synapse(agent.network, pathway[1], pathway[2])
        edge === nothing && continue
        first_pos = first_position[pathway]
        temporal = Float32(0.5f0 * first_pos / path_count + 0.5f0 * last_pos / path_count)
        endpoints = (edge.source in relevant_concepts) + (edge.target in relevant_concepts)
        causal = endpoints == 2 ? 1f0 : endpoints == 1 ? 0.5f0 : 0f0
        relevance = clamp(0.55f0 * edge.eligibility + 0.25f0 * temporal + 0.20f0 * causal, 0f0, 1f0)
        base = credit_level(relevance)
        level = signal < 0f0 && base != NEUTRAL_CREDIT ? NEGATIVE_CREDIT : base
        reason = "eligibility=$(round(edge.eligibility; digits=3)), temporal=$(round(temporal; digits=3)), causal=$(round(causal; digits=3))"
        assignment = CreditAssignment(edge.source, edge.target, edge.eligibility,
                                      temporal, causal, level, signal, reason)
        push!(assignments, assignment)
        emit!(state.trace, CREDIT_ASSIGNED; source=edge.source, target=edge.target,
              eligibility=edge.eligibility, temporal_relevance=temporal,
              causal_relevance=causal, credit=level, signal=signal, reason=reason)
    end
    return assignments
end

function _step_toward(value::Int8, target::Int8)
    @assert is_ternary(value) && is_ternary(target) "non-ternary plasticity input: $value → $target"
    value == target ? value : Int8(Int(value) + sign(Int(target) - Int(value)))
end

"""Apply one-step ternary transitions to eligible pathways; never mutate a float weight."""
function apply_plasticity!(agent::CognitiveAgent, state::CognitiveState,
                           evaluation::EvaluationResult)
    assignments = assign_credit(agent, state, evaluation)
    signal = evaluation.reward > 0f0 ? evaluation.reward : -evaluation.punishment
    changes = SynapticChange[]
    cfg = agent.config.plasticity
    selected = selected_hypothesis(state)

    if signal != 0f0 && agent.config.ternary_enabled
        for assignment in assignments
            assignment.credit == NEUTRAL_CREDIT && continue
            edge = find_synapse(agent.network, assignment.source, assignment.target)
            edge === nothing && continue
            old = edge.value
            desired = relation_polarity(agent, edge, selected)

            signal > 0f0 ? (edge.times_rewarded += 1) : (edge.times_punished += 1)
            target = signal > 0f0 ? desired : old == 0 ? Int8(-desired) : Int8(0)
            if old == target
                edge.pressure = 0f0
                signal > 0f0 && (edge.stability = min(cfg.max_stability,
                    edge.stability + cfg.stability_gain * abs(signal) * assignment.eligibility))
                continue
            end

            sign(edge.pressure) != sign(signal) && (edge.pressure = 0f0)
            multiplier = 0.75f0 + 0.75f0 * assignment.causal_relevance +
                         0.25f0 * assignment.temporal_relevance
            edge.pressure += signal * assignment.eligibility * multiplier
            resistance = old == 0 ? 0f0 : edge.stability
            threshold = (signal > 0f0 ? cfg.promote_threshold : cfg.demote_threshold) *
                        (1f0 + resistance)
            explored = evaluation.quality == :PARTIAL && old == 0 &&
                       abs(edge.pressure) >= threshold / 2f0 && rand(agent.rng) < cfg.exploration_prob
            if abs(edge.pressure) >= threshold || explored
                edge.value = assert_ternary(_step_toward(old, target), "plasticity transition")
                edge.pressure = 0f0
                edge.last_changed_episode = Int32(agent.episodic.next_id)
                if signal > 0f0
                    edge.stability = min(cfg.max_stability,
                        edge.stability + cfg.stability_gain * abs(signal) * assignment.eligibility)
                else
                    edge.stability = max(0f0,
                        edge.stability - cfg.stability_decay * abs(signal) * assignment.eligibility)
                end
                reason = explored ? "bounded exploration toward context-compatible state" :
                         signal > 0f0 ? "reward reinforced the successful ternary configuration" :
                                        "punishment moved the responsible pathway away from its failed state"
                change = SynapticChange(edge.source, edge.target, old, edge.value,
                                        assignment.eligibility, assignment.credit, reason)
                push!(changes, change)
                emit!(state.trace, SYNAPSE_CHANGED; source=edge.source, target=edge.target,
                      old=old, new=edge.value, eligibility=edge.eligibility,
                      credit=assignment.credit, reason=reason)
            end
        end
    end

    decay_eligibility!(agent.network, agent.config.eligibility)
    agent.config.debug_invariants && assert_network_invariants(agent.network)
    return changes
end

function complete_episode!(agent::CognitiveAgent, state::CognitiveState;
                           expected_output::Union{Nothing,String}=nothing,
                           expected_quality::Union{Nothing,Symbol}=nothing,
                           evaluator_mode::Symbol=:auto)
    output = render_output(agent, state)
    emit!(state.trace, OUTPUT_GENERATED; output=output)
    evaluation = evaluate_output(agent, state, output; expected_output=expected_output,
                                 expected_quality=expected_quality, evaluator_mode=evaluator_mode)
    emit!(state.trace, EVALUATION_COMPLETED; quality=evaluation.quality,
          expected_quality=evaluation.expected_quality, mode=evaluation.evaluator_mode,
          reasons=evaluation.reasons)
    evaluation.reward > 0f0 && emit!(state.trace, REWARD_ISSUED; amount=evaluation.reward)
    evaluation.punishment > 0f0 && emit!(state.trace, PUNISHMENT_ISSUED; amount=evaluation.punishment)
    changes = apply_plasticity!(agent, state, evaluation)

    h = selected_hypothesis(state)
    positive = evaluation.reward > 0f0 && evaluation.reward > evaluation.punishment
    nonneutral = evaluation.reward > 0f0 || evaluation.punishment > 0f0
    memory_added = Int32[]
    if h !== nothing
        nonneutral && credit_concepts!(agent, h.supporting_concepts, positive)
        nonneutral && note_strategy_outcome!(agent, h.strategy, positive;
                                             episode=Int32(agent.episodic.next_id))
        if positive && h.origin != :direct && !relation_exists(agent, h.claim)
            add_relation_ids!(agent, h.claim.subject, h.claim.predicate, h.claim.object;
                              provenance=:deduced, episode=Int32(agent.episodic.next_id))
            append!(memory_added, Int32[h.claim.subject, h.claim.object])
        end
    end
    if evaluation.punishment > 0f0
        strategy = h === nothing ? :none : h.strategy
        alternative = evaluation.quality == :WRONG ? expected_output : nothing
        record_failure!(agent, agent.episodic.next_id, state.input,
                        join(evaluation.reasons, "; "), strategy;
                        claim=h === nothing ? nothing : h.claim,
                        alternative=alternative)
    end

    emit!(state.trace, MEMORY_UPDATED; episode=agent.episodic.next_id,
          learned_concepts=memory_added, failure_stored=evaluation.punishment > 0f0)
    emit!(state.trace, EPISODE_STORED; episode=agent.episodic.next_id)
    success = evaluation.quality in (:EXCELLENT, :GOOD, :REQUESTED_REPETITION)
    ep = CognitiveEpisode(agent.episodic.next_id, state.input, state.goal, state.intent,
        copy(state.concepts), copy(state.retrieved_concepts), copy(state.phase_history),
        [hypothesis.id for hypothesis in state.hypotheses],
        [hypothesis.id for hypothesis in state.hypotheses if hypothesis.status == :rejected],
        state.current_hypothesis, copy(state.trace.events), output, evaluation, changes,
        memory_added, evaluation.reward, evaluation.punishment,
        h === nothing ? :none : h.strategy, state.budget_exhausted, success)
    store_episode!(agent, ep)

    agent.last_output = output
    agent.last_evaluation = evaluation
    agent.last_state = state
    metrics = agent.metrics
    metrics.episodes += 1
    metrics.quality_counts[evaluation.quality] = get(metrics.quality_counts, evaluation.quality, 0) + 1
    metrics.total_reward += evaluation.reward
    metrics.total_punishment += evaluation.punishment
    metrics.ternary_transitions += length(changes)
    push!(metrics.synapses_changed_per_episode, length(changes))
    push!(metrics.reasoning_steps, state.steps_used)
    push!(metrics.reward_curve, evaluation.reward)
    push!(metrics.punishment_curve, evaluation.punishment)
    push!(metrics.success_curve, ep.success)
    state.budget_exhausted && (metrics.budget_exhaustions += 1)
    metrics.contradictions_detected += length(state.contradictions)
    metrics.hypotheses_rejected += count(hx -> hx.status == :rejected, state.hypotheses)
    h !== nothing && h.origin in _NOVEL_ORIGINS && positive && (metrics.novel_recombinations += 1)
    return output, evaluation, ep
end

function respond!(agent::CognitiveAgent, input::String;
                  expected_output::Union{Nothing,String}=nothing,
                  expected_quality::Union{Nothing,Symbol}=nothing,
                  evaluator_mode::Symbol=:auto, domain::String="general")
    state = reason!(agent, input; domain=domain)
    result = complete_episode!(agent, state; expected_output=expected_output,
                               expected_quality=expected_quality, evaluator_mode=evaluator_mode)
    return result, state
end

# ===== END learning.jl =====


# ===== BEGIN diagnostics.jl =====

# diagnostics.jl — Machine-readable introspection and change auditing.
get_active_units(ai::CognitiveAgent) = active_unit_ids(ai.network)
get_active_synapses(ai::CognitiveAgent) = [(e.source,e.target,e.value,e.eligibility) for es in values(ai.network.outgoing) for e in es if e.eligibility > 0f0]
get_working_memory(ai::CognitiveAgent) = copy(ai.working.items)
get_retrieved_memories(ai::CognitiveAgent) = ai.last_state === nothing ? Int[] : copy(ai.last_state.retrieved_memories)
get_hypotheses(ai::CognitiveAgent) = ai.last_state === nothing ? CognitiveHypothesis[] : copy(ai.last_state.hypotheses)
get_current_goal(ai::CognitiveAgent) = ai.last_state === nothing ? "" : ai.last_state.goal
get_reasoning_state(ai::CognitiveAgent) = ai.last_state
get_last_evaluation(ai::CognitiveAgent) = ai.last_evaluation
get_last_reward(ai::CognitiveAgent) = ai.last_evaluation === nothing ? 0f0 : ai.last_evaluation.reward
get_last_punishment(ai::CognitiveAgent) = ai.last_evaluation === nothing ? 0f0 : ai.last_evaluation.punishment
get_last_synaptic_changes(ai::CognitiveAgent) = isempty(ai.episodic.episodes) ? SynapticChange[] : ai.episodic.episodes[end].synaptic_changes

synaptic_snapshot(ai::CognitiveAgent) = Dict((e.source,e.target)=>e.value for es in values(ai.network.outgoing) for e in es)
function synaptic_diff(before::Dict, after::CognitiveAgent)
    [SynapticChange(e.source,e.target,get(before,(e.source,e.target),Int8(0)),e.value,e.eligibility,credit_level(e),"snapshot diff")
     for es in values(after.network.outgoing) for e in es if get(before,(e.source,e.target),Int8(0)) != e.value]
end
memory_snapshot(ai::CognitiveAgent) = (concepts=Set(keys(ai.semantic.concepts)), relations=Set(ai.semantic.relations), episodes=length(ai.episodic.episodes))
function memory_diff(before, after::CognitiveAgent)
    (added_concepts=collect(setdiff(Set(keys(after.semantic.concepts)),before.concepts)),
     new_relations=collect(setdiff(Set(after.semantic.relations),before.relations)), new_episodes=length(after.episodic.episodes)-before.episodes)
end
function training_report(ai::CognitiveAgent)
    m=ai.metrics; d=ternary_distribution(ai.network)
    (episodes=m.episodes, quality=copy(m.quality_counts), total_reward=m.total_reward, total_punishment=m.total_punishment,
     synapses=d, average_changed=isempty(m.synapses_changed_per_episode) ? 0.0 : sum(m.synapses_changed_per_episode)/length(m.synapses_changed_per_episode),
     self_corrections=m.self_corrections, novelty=m.novel_recombinations, peak_bytes=m.peak_bytes, runtime_seconds=m.runtime_seconds)
end

# ===== END diagnostics.jl =====

# ===== END reasoning.jl =====


# ===== BEGIN app.jl =====

# app.jl -- consolidated bundle (auto-generated)
# Sources merged: json_util.jl, serialization.jl, training.jl, benchmarks.jl, cli.jl, gui.jl, web/index.html (embedded), data/*.txt (embedded)
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt

# NOTE: include order preserved from TernaryCognitiveAI.jl: json_util -> serialization -> training -> benchmarks -> cli -> gui

# ===== EMBEDDED ASSETS (web UI + datasets, replaces data/ and src/web/) =====

const EMBEDDED_INDEX_HTML = raw"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Ternary Cognitive AI — Dashboard</title>
  <style>
    :root {
      --bg: #0b1120;
      --card-bg: #1e293b;
      --card-border: #334155;
      --header-bg: #0f172a;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.15);
      --red: #f43f5e;
      --red-bg: rgba(244, 63, 94, 0.15);
      --purple: #8b5cf6;
      --cyan: #06b6d4;
      --amber: #f59e0b;
      --amber-bg: rgba(245, 158, 11, 0.15);
      --radius: 10px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }
    
    /* Top Navbar */
    header { background: var(--header-bg); border-bottom: 1px solid var(--card-border); padding: 12px 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon { font-size: 26px; }
    .brand-title { font-size: 19px; font-weight: 700; letter-spacing: -0.5px; }
    .brand-subtitle { font-size: 12px; color: var(--text-muted); }
    
    .nav-tabs { display: flex; gap: 8px; background: #0b1120; padding: 4px; border-radius: 8px; border: 1px solid var(--card-border); }
    .tab-btn { background: transparent; border: none; color: var(--text-muted); padding: 8px 18px; border-radius: 6px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 6px; }
    .tab-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.05); }
    .tab-btn.active { color: #fff; background: var(--accent); shadow: 0 2px 8px rgba(99, 102, 241, 0.4); }

    .header-pills { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .pill { background: #1e293b; border: 1px solid var(--card-border); padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; }
    .pill-green { color: var(--green); border-color: rgba(16, 185, 129, 0.4); }
    .btn-sm { background: #334155; border: 1px solid #475569; color: var(--text); padding: 5px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
    .btn-sm:hover { background: #475569; }

    /* Main Container */
    main { flex: 1; padding: 20px; max-width: 1400px; margin: 0 auto; width: 100%; display: flex; flex-direction: column; }
    .tab-content { display: none; height: 100%; }
    .tab-content.active { display: flex; flex-direction: column; flex: 1; }

    /* Page 1: Chatting Tab */
    .chat-layout { display: flex; flex-direction: column; flex: 1; gap: 14px; height: calc(100vh - 120px); }
    .chat-messages { flex: 1; overflow-y: auto; background: #0f172a; border: 1px solid var(--card-border); border-radius: var(--radius); padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    
    .msg { display: flex; flex-direction: column; max-width: 85%; animation: fadeIn 0.2s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .msg-user { align-self: flex-end; }
    .msg-ai { align-self: flex-start; }
    
    .msg-bubble { padding: 12px 18px; border-radius: 12px; font-size: 15px; line-height: 1.5; }
    .msg-user .msg-bubble { background: var(--accent); color: white; border-bottom-right-radius: 2px; }
    .msg-ai .msg-bubble { background: var(--card-bg); border: 1px solid var(--card-border); color: var(--text); border-bottom-left-radius: 2px; }

    .msg-meta { margin-top: 8px; display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px; align-items: center; }
    .badge { padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 11px; text-transform: uppercase; }
    .badge-excellent { background: var(--green-bg); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-good { background: rgba(6, 182, 212, 0.15); color: var(--cyan); border: 1px solid rgba(6, 182, 212, 0.4); }
    .badge-rep { background: rgba(139, 92, 246, 0.15); color: var(--purple); border: 1px solid rgba(139, 92, 246, 0.4); }
    .badge-partial { background: var(--amber-bg); color: var(--amber); border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-wrong { background: var(--red-bg); color: var(--red); border: 1px solid rgba(244, 63, 94, 0.4); }
    .badge-info { background: #334155; color: #cbd5e1; }

    .trace-accordion { margin-top: 8px; font-size: 12px; }
    .trace-summary { cursor: pointer; color: var(--text-muted); user-select: none; font-weight: 500; display: inline-flex; align-items: center; gap: 4px; }
    .trace-summary:hover { color: var(--text); }
    .trace-body { margin-top: 6px; background: #090d16; border: 1px solid #1e293b; border-radius: 6px; padding: 10px; font-family: monospace; font-size: 11px; max-height: 220px; overflow-y: auto; white-space: pre-wrap; color: #cbd5e1; }

    /* Quick Prompts Bar */
    .quick-prompts { display: flex; gap: 8px; overflow-x: auto; padding-bottom: 2px; }
    .prompt-chip { background: #1e293b; border: 1px solid var(--card-border); color: var(--text-muted); padding: 6px 12px; border-radius: 20px; font-size: 12px; white-space: nowrap; cursor: pointer; transition: all 0.15s; }
    .prompt-chip:hover { background: #334155; color: var(--text); border-color: var(--accent); }

    /* Input Bar */
    .chat-input-bar { display: flex; flex-direction: column; gap: 8px; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius); padding: 12px; }
    .input-row { display: flex; gap: 10px; }
    .chat-input { flex: 1; background: #0f172a; border: 1px solid var(--card-border); color: var(--text); border-radius: 8px; padding: 12px 16px; font-size: 15px; outline: none; transition: border-color 0.2s; }
    .chat-input:focus { border-color: var(--accent); }
    .btn-send { background: var(--accent); color: white; border: none; border-radius: 8px; padding: 0 24px; font-size: 15px; font-weight: 600; cursor: pointer; transition: background 0.15s; display: flex; align-items: center; gap: 6px; }
    .btn-send:hover { background: var(--accent-hover); }

    .input-options { display: flex; gap: 16px; font-size: 12px; color: var(--text-muted); align-items: center; }
    .option-label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
    .target-input { background: #0f172a; border: 1px solid var(--card-border); color: var(--text); border-radius: 6px; padding: 4px 8px; font-size: 12px; display: none; width: 260px; }

    /* Page 2: Training Tab */
    .training-grid { display: grid; grid-template-columns: 1.1fr 1fr; gap: 20px; flex: 1; }
    @media (max-width: 900px) { .training-grid { grid-template-columns: 1fr; } }
    
    .panel { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius); padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .panel-header { display: flex; justify-content: space-between; align-items: center; }
    .panel-title { font-size: 16px; font-weight: 700; display: flex; align-items: center; gap: 8px; }

    .dataset-select-row { display: flex; gap: 10px; align-items: center; }
    select, input[type="number"], input[type="text"] { background: #0f172a; border: 1px solid var(--card-border); color: var(--text); border-radius: 6px; padding: 8px 12px; font-size: 13px; outline: none; }
    select:focus, input:focus { border-color: var(--accent); }
    
    .train-editor { width: 100%; flex: 1; min-height: 220px; background: #090d16; border: 1px solid var(--card-border); color: #e2e8f0; border-radius: 6px; padding: 12px; font-family: monospace; font-size: 13px; resize: vertical; line-height: 1.4; outline: none; }
    .train-editor:focus { border-color: var(--accent); }
    
    .btn-train { background: var(--green); color: white; border: none; border-radius: 8px; padding: 12px 20px; font-size: 15px; font-weight: 700; cursor: pointer; transition: opacity 0.15s; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn-train:hover { opacity: 0.9; }
    .btn-train:disabled { opacity: 0.5; cursor: not-allowed; }

    /* KPI Metrics Cards */
    .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px; }
    .kpi-card { background: #0f172a; border: 1px solid var(--card-border); border-radius: 8px; padding: 12px; text-align: center; }
    .kpi-val { font-size: 20px; font-weight: 700; color: #fff; margin-top: 4px; }
    .kpi-lbl { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; }

    /* Synapse Distribution Visual Bar */
    .synapse-bar { height: 16px; border-radius: 8px; overflow: hidden; display: flex; background: #334155; margin-top: 6px; }
    .synapse-pos { background: var(--green); height: 100%; transition: width 0.3s; }
    .synapse-zero { background: #64748b; height: 100%; transition: width 0.3s; }
    .synapse-neg { background: var(--red); height: 100%; transition: width 0.3s; }

    .train-log { background: #090d16; border: 1px solid var(--card-border); border-radius: 6px; padding: 12px; font-family: monospace; font-size: 12px; height: 180px; overflow-y: auto; white-space: pre-wrap; color: #cbd5e1; }

    /* Modal for Concepts / Relations */
    .modal-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.7); align-items: center; justify-content: center; z-index: 100; }
    .modal-overlay.active { display: flex; }
    .modal-box { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: var(--radius); width: 680px; max-width: 90vw; max-height: 80vh; display: flex; flex-direction: column; overflow: hidden; }
    .modal-header { padding: 16px 20px; border-bottom: 1px solid var(--card-border); display: flex; justify-content: space-between; align-items: center; }
    .modal-body { padding: 20px; overflow-y: auto; font-size: 13px; }
  </style>
</head>
<body>

  <!-- Top Header -->
  <header>
    <div class="brand">
      <div class="brand-icon">🧠</div>
      <div>
        <div class="brand-title">Ternary Cognitive AI</div>
        <div class="brand-subtitle">Neuro-Symbolic Discrete {-1, 0, +1} Architecture</div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="switchTab('chat')">💬 Chatting</button>
      <button class="tab-btn" onclick="switchTab('train')">🎓 Training</button>
    </div>

    <!-- Live System Pills & Controls -->
    <div class="header-pills">
      <div class="pill pill-green"><span id="pill-status">● Ready</span></div>
      <div class="pill" id="pill-scale">Scale: small</div>
      <div class="pill" id="pill-synapses">Synapses: +0 | 0:0 | -0</div>
      <div class="pill" id="pill-concepts">Concepts: 0</div>
      <button class="btn-sm" onclick="showModal('concepts')">Concepts</button>
      <button class="btn-sm" onclick="showModal('relations')">Relations</button>
      <button class="btn-sm" onclick="resetAgent()">🔄 Reset</button>
    </div>
  </header>

  <main>
    <!-- TAB 1: CHATTING -->
    <div id="tab-chat" class="tab-content active">
      <div class="chat-layout">
        <!-- Message Feed -->
        <div class="chat-messages" id="chat-feed">
          <div class="msg msg-ai">
            <div class="msg-bubble">
              Hello! I am the <strong>Ternary Cognitive AI</strong> reasoning engine. You can teach me declarative facts (e.g. <code>sun causes heat</code>), ask deductive questions (e.g. <code>why sun causes expansion</code>), request repetitions, or enforce constraints like JSON structure.
            </div>
            <div class="msg-meta">
              <span class="badge badge-good">System Initialized</span>
              <span style="color: var(--text-muted);">Ready for input</span>
            </div>
          </div>
        </div>

        <!-- Quick Prompts Bar -->
        <div class="quick-prompts">
          <span style="font-size: 12px; color: var(--text-muted); align-self: center;">Try:</span>
          <div class="prompt-chip" onclick="sendQuickPrompt('sun causes heat')">☀️ sun causes heat</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('heat causes expansion')">🔥 heat causes expansion</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('why sun causes expansion')">❓ why sun causes expansion</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('rain causes wet ground')">🌧️ rain causes wet ground</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('why rain causes slippery pavement')">❓ why rain causes slippery pavement</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('repeat this')">🔁 repeat this</div>
          <div class="prompt-chip" onclick="sendQuickPrompt('sun causes heat in json')">📋 in json</div>
        </div>

        <!-- Chat Input Bar -->
        <div class="chat-input-bar">
          <div class="input-row">
            <input type="text" id="chat-input" class="chat-input" placeholder="Enter a statement, query, or command..." autocomplete="off">
            <button class="btn-send" id="btn-send" onclick="handleSendChat()">
              <span>Send</span> ➤
            </button>
          </div>
          <div class="input-options">
            <label class="option-label">
              <input type="checkbox" id="chk-supervised" onchange="toggleSupervised()">
              Supervised Target
            </label>
            <input type="text" id="target-input" class="target-input" placeholder="Expected output (e.g. 'sun causes heat')">
            <span style="margin-left: auto; cursor: pointer; color: var(--text-muted);" onclick="clearChat()">🗑️ Clear Chat</span>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: TRAINING -->
    <div id="tab-train" class="tab-content">
      <div class="training-grid">
        <!-- Left Panel: Data & Training -->
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">📚 Dataset & Training Controls</div>
          </div>

          <div class="dataset-select-row">
            <label style="font-size: 13px; color: var(--text-muted);">Preset Dataset:</label>
            <select id="dataset-select" onchange="onDatasetSelected()">
              <option value="data/train.txt">data/train.txt (Foundational)</option>
              <option value="data/science.txt">data/science.txt (Science & Physics)</option>
              <option value="data/logic.txt">data/logic.txt (Logic & Deduction)</option>
              <option value="custom">Custom Text Input</option>
            </select>
            <button class="btn-sm" onclick="loadDatasetToEditor()">Reload</button>
          </div>

          <textarea id="train-text" class="train-editor" placeholder="input | expected output | optional quality"></textarea>

          <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
            <div style="display: flex; gap: 8px; align-items: center;">
              <label style="font-size: 13px; color: var(--text-muted);">Epochs:</label>
              <input type="number" id="train-epochs" value="3" min="1" max="20" style="width: 70px;">
            </div>
            <button class="btn-train" id="btn-start-train" onclick="startTraining()" style="flex: 1;">
              <span>🚀 Start Training</span>
            </button>
          </div>

          <!-- Quick Teach Section -->
          <div style="border-top: 1px solid var(--card-border); padding-top: 14px; margin-top: 6px;">
            <div style="font-size: 13px; font-weight: 700; margin-bottom: 8px;">⚡ Single-Pair Quick Teach</div>
            <div style="display: flex; gap: 8px;">
              <input type="text" id="quick-teach-input" placeholder="Input (e.g. spark causes fire)" style="flex: 1;">
              <input type="text" id="quick-teach-expected" placeholder="Expected Target" style="flex: 1;">
              <button class="btn-sm" onclick="handleQuickTeach()">Teach</button>
            </div>
          </div>
        </div>

        <!-- Right Panel: Live Metrics & Feedback -->
        <div class="panel">
          <div class="panel-header">
            <div class="panel-title">📊 Cognitive Metrics & Synaptic State</div>
          </div>

          <!-- KPI Cards -->
          <div class="metrics-row">
            <div class="kpi-card">
              <div class="kpi-lbl">Episodes</div>
              <div class="kpi-val" id="kpi-episodes">0</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-lbl">Reward</div>
              <div class="kpi-val" style="color: var(--green);" id="kpi-reward">+0.00</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-lbl">Penalty</div>
              <div class="kpi-val" style="color: var(--red);" id="kpi-punish">-0.00</div>
            </div>
            <div class="kpi-card">
              <div class="kpi-lbl">Corrections</div>
              <div class="kpi-val" id="kpi-corrections">0</div>
            </div>
          </div>

          <!-- Synapse Distribution Bar -->
          <div>
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
              <span>Ternary Weights: <strong id="lbl-syn-pos">+1 (0)</strong> · <span id="lbl-syn-zero">0 (0)</span> · <strong id="lbl-syn-neg" style="color: var(--red);">-1 (0)</strong></span>
              <span id="lbl-syn-total">Total: 0</span>
            </div>
            <div class="synapse-bar">
              <div class="synapse-pos" id="bar-pos" style="width: 33%;"></div>
              <div class="synapse-zero" id="bar-zero" style="width: 34%;"></div>
              <div class="synapse-neg" id="bar-neg" style="width: 33%;"></div>
            </div>
          </div>

          <!-- Training Output Console -->
          <div style="display: flex; flex-direction: column; flex: 1;">
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">Live Training Output:</div>
            <div class="train-log" id="train-log">Awaiting training start...</div>
          </div>

          <!-- Model Management Row -->
          <div style="border-top: 1px solid var(--card-border); padding-top: 12px; display: flex; gap: 8px; align-items: center;">
            <input type="text" id="model-file-path" value="data/model.bin" style="flex: 1;" placeholder="data/model.bin">
            <button class="btn-sm" onclick="saveModel()">💾 Save</button>
            <button class="btn-sm" onclick="loadModel()">📂 Load</button>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- Modal for Concepts / Relations Inspection -->
  <div class="modal-overlay" id="modal-overlay" onclick="closeModal(event)">
    <div class="modal-box" onclick="event.stopPropagation()">
      <div class="modal-header">
        <div style="font-weight: 700; font-size: 16px;" id="modal-title">Memory Inspection</div>
        <button class="btn-sm" onclick="closeModal()">✕</button>
      </div>
      <div class="modal-body" id="modal-content">Loading...</div>
    </div>
  </div>

  <script>
    // Tab switching
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      if (tabId === 'chat') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('tab-chat').classList.add('active');
      } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('tab-train').classList.add('active');
      }
    }

    function toggleSupervised() {
      const chk = document.getElementById('chk-supervised').checked;
      document.getElementById('target-input').style.display = chk ? 'inline-block' : 'none';
    }

    // Refresh Agent Status
    async function refreshStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        document.getElementById('pill-scale').innerText = 'Scale: ' + data.scale;
        document.getElementById('pill-concepts').innerText = 'Concepts: ' + data.concepts;
        document.getElementById('pill-synapses').innerText = `Synapses: +${data.synapses.positive} | 0:${data.synapses.zero} | -${data.synapses.negative}`;
        
        // Update KPIs
        document.getElementById('kpi-episodes').innerText = data.metrics.episodes;
        document.getElementById('kpi-reward').innerText = '+' + data.metrics.total_reward.toFixed(2);
        document.getElementById('kpi-punish').innerText = '-' + data.metrics.total_punishment.toFixed(2);
        document.getElementById('kpi-corrections').innerText = data.metrics.self_corrections;

        // Update Synapse bar
        const total = data.synapses.total || 1;
        const pctPos = (data.synapses.positive / total) * 100;
        const pctZero = (data.synapses.zero / total) * 100;
        const pctNeg = (data.synapses.negative / total) * 100;
        document.getElementById('bar-pos').style.width = pctPos + '%';
        document.getElementById('bar-zero').style.width = pctZero + '%';
        document.getElementById('bar-neg').style.width = pctNeg + '%';
        document.getElementById('lbl-syn-pos').innerText = `+1 (${data.synapses.positive})`;
        document.getElementById('lbl-syn-zero').innerText = `0 (${data.synapses.zero})`;
        document.getElementById('lbl-syn-neg').innerText = `-1 (${data.synapses.negative})`;
        document.getElementById('lbl-syn-total').innerText = `Total: ${data.synapses.total}`;
      } catch (err) {
        console.error('Failed to fetch status:', err);
      }
    }

    // Send Chat
    async function handleSendChat() {
      const inputEl = document.getElementById('chat-input');
      const text = inputEl.value.trim();
      if (!text) return;

      const isSupervised = document.getElementById('chk-supervised').checked;
      const expected = isSupervised ? document.getElementById('target-input').value.trim() : null;

      // Add user message to UI
      appendUserMessage(text);
      inputEl.value = '';

      const btn = document.getElementById('btn-send');
      btn.disabled = true;

      try {
        const payload = { input: text };
        if (expected) payload.expected = expected;

        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        appendAIMessage(data);
        refreshStatus();
      } catch (err) {
        appendAIMessage({ output: 'Error communicating with reasoning engine: ' + err.message, quality: 'WRONG' });
      } finally {
        btn.disabled = false;
        inputEl.focus();
      }
    }

    function sendQuickPrompt(promptText) {
      document.getElementById('chat-input').value = promptText;
      handleSendChat();
    }

    function appendUserMessage(text) {
      const feed = document.getElementById('chat-feed');
      const div = document.createElement('div');
      div.className = 'msg msg-user';
      div.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
      feed.appendChild(div);
      feed.scrollTop = feed.scrollHeight;
    }

    function appendAIMessage(data) {
      const feed = document.getElementById('chat-feed');
      const div = document.createElement('div');
      div.className = 'msg msg-ai';

      let qualityClass = 'badge-info';
      if (data.quality === 'EXCELLENT') qualityClass = 'badge-excellent';
      else if (data.quality === 'GOOD') qualityClass = 'badge-good';
      else if (data.quality === 'REQUESTED_REPETITION') qualityClass = 'badge-rep';
      else if (data.quality === 'PARTIAL') qualityClass = 'badge-partial';
      else if (data.quality === 'WRONG' || (data.quality && data.quality.includes('FAILURE'))) qualityClass = 'badge-wrong';

      const sigStr = data.reward > 0 ? `+${data.reward.toFixed(2)} Reward` :
                     data.punishment > 0 ? `-${data.punishment.toFixed(2)} Penalty` : `0.00`;

      let traceHtml = '';
      if (data.trace && data.trace.length > 0) {
        traceHtml = `
          <details class="trace-accordion">
            <summary class="trace-summary">🔍 View Reasoning Trace (${data.trace.length} steps)</summary>
            <div class="trace-body">${escapeHtml(data.trace.join('\n'))}</div>
          </details>
        `;
      }

      div.innerHTML = `
        <div class="msg-bubble">${escapeHtml(data.output)}</div>
        <div class="msg-meta">
          <span class="badge ${qualityClass}">${data.quality || 'OUTPUT'}</span>
          <span class="badge badge-info">Conf: ${((data.confidence || 0) * 100).toFixed(0)}%</span>
          <span class="badge badge-info">${sigStr}</span>
          <span class="badge badge-info">${data.synaptic_changes || 0} transitions</span>
          ${data.contradiction ? '<span class="badge badge-wrong">Contradiction Backtracked</span>' : ''}
        </div>
        ${traceHtml}
      `;
      feed.appendChild(div);
      feed.scrollTop = feed.scrollHeight;
    }

    function clearChat() {
      document.getElementById('chat-feed').innerHTML = `
        <div class="msg msg-ai">
          <div class="msg-bubble">Chat cleared. Ready for your input.</div>
        </div>
      `;
    }

    // Dataset Management
    async function loadDatasetToEditor() {
      const select = document.getElementById('dataset-select');
      const filename = select.value;
      if (filename === 'custom') return;
      try {
        const res = await fetch('/api/dataset?name=' + encodeURIComponent(filename));
        const data = await res.json();
        document.getElementById('train-text').value = data.content || '';
      } catch (err) {
        alert('Failed to load dataset: ' + err.message);
      }
    }

    function onDatasetSelected() {
      loadDatasetToEditor();
    }

    // Training Process
    async function startTraining() {
      const text = document.getElementById('train-text').value.trim();
      if (!text) {
        alert('Please enter or load training data.');
        return;
      }
      const epochs = parseInt(document.getElementById('train-epochs').value) || 1;
      const btn = document.getElementById('btn-start-train');
      const logEl = document.getElementById('train-log');

      btn.disabled = true;
      btn.innerText = '⏳ Training...';
      logEl.innerText = `Starting training for ${epochs} epoch(s)...\n`;

      try {
        const res = await fetch('/api/train', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, epochs: epochs })
        });
        const data = await res.json();
        logEl.innerText = data.logs || 'Training completed successfully.';
        refreshStatus();
      } catch (err) {
        logEl.innerText += '\n[Error] Training failed: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.innerText = '🚀 Start Training';
      }
    }

    async function handleQuickTeach() {
      const inp = document.getElementById('quick-teach-input').value.trim();
      const exp = document.getElementById('quick-teach-expected').value.trim();
      if (!inp || !exp) {
        alert('Please provide both input and expected target.');
        return;
      }
      try {
        const res = await fetch('/api/teach', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ input: inp, expected: exp })
        });
        const data = await res.json();
        alert(`Learned: "${inp}" -> "${data.output}" (Quality: ${data.quality})`);
        document.getElementById('quick-teach-input').value = '';
        document.getElementById('quick-teach-expected').value = '';
        refreshStatus();
      } catch (err) {
        alert('Teach error: ' + err.message);
      }
    }

    async function resetAgent() {
      if (!confirm('Reset the agent to fresh state?')) return;
      try {
        await fetch('/api/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scale: 'small' }) });
        refreshStatus();
        clearChat();
        alert('Agent reinitialized.');
      } catch (err) {
        alert('Reset failed: ' + err.message);
      }
    }

    async function saveModel() {
      const path = document.getElementById('model-file-path').value.trim();
      if (!path) return;
      try {
        const res = await fetch('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path }) });
        const data = await res.json();
        alert(data.message || 'Saved successfully');
      } catch (err) {
        alert('Save error: ' + err.message);
      }
    }

    async function loadModel() {
      const path = document.getElementById('model-file-path').value.trim();
      if (!path) return;
      try {
        const res = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: path }) });
        const data = await res.json();
        alert(data.message || 'Loaded successfully');
        refreshStatus();
      } catch (err) {
        alert('Load error: ' + err.message);
      }
    }

    // Modal Inspection
    async function showModal(type) {
      const overlay = document.getElementById('modal-overlay');
      const title = document.getElementById('modal-title');
      const content = document.getElementById('modal-content');
      overlay.classList.add('active');
      content.innerHTML = 'Loading...';

      if (type === 'concepts') {
        title.innerText = 'Semantic Memory: Known Concepts';
        const res = await fetch('/api/concepts');
        const list = await res.json();
        if (!list || list.length === 0) {
          content.innerHTML = '<em>No concepts learned yet.</em>';
          return;
        }
        let html = '<table style="width:100%; border-collapse:collapse;">';
        html += '<tr style="border-bottom:1px solid #334155; text-align:left; color:#94a3b8;"><th>ID</th><th>Concept</th><th>Reliability</th><th>Successes</th><th>Failures</th><th>Freq</th></tr>';
        list.forEach(c => {
          html += `<tr style="border-bottom:1px solid #1e293b;"><td style="padding:6px;">${c.id}</td><td style="font-weight:600; color:#fff;">${escapeHtml(c.name)}</td><td>${c.reliability.toFixed(2)}</td><td>${c.successes}</td><td>${c.failures}</td><td>${c.frequency}</td></tr>`;
        });
        html += '</table>';
        content.innerHTML = html;
      } else if (type === 'relations') {
        title.innerText = 'Semantic Memory: Stored & Deduced Relations';
        const res = await fetch('/api/relations');
        const list = await res.json();
        if (!list || list.length === 0) {
          content.innerHTML = '<em>No relations stored yet.</em>';
          return;
        }
        let html = '<table style="width:100%; border-collapse:collapse;">';
        html += '<tr style="border-bottom:1px solid #334155; text-align:left; color:#94a3b8;"><th>Subject</th><th>Predicate</th><th>Object</th><th>Provenance</th></tr>';
        list.forEach(r => {
          html += `<tr style="border-bottom:1px solid #1e293b;"><td style="padding:6px; font-weight:600;">${escapeHtml(r.subject)}</td><td style="color:#6366f1;">${escapeHtml(r.predicate)}</td><td style="font-weight:600;">${escapeHtml(r.object)}</td><td style="color:#94a3b8;">${escapeHtml(r.provenance)}</td></tr>`;
        });
        html += '</table>';
        content.innerHTML = html;
      }
    }

    function closeModal() {
      document.getElementById('modal-overlay').classList.remove('active');
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Keyboard enter in chat
    document.getElementById('chat-input').addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleSendChat();
      }
    });

    // Initialize
    window.addEventListener('load', () => {
      refreshStatus();
      loadDatasetToEditor();
    });
  </script>
</body>
</html>
"""

const EMBEDDED_TRAIN_TXT = raw"""# Foundational Knowledge Training Data
# Format: input | expected output | optional quality

sun causes heat | sun causes heat
heat causes expansion | heat causes expansion
why sun causes expansion | sun causes expansion
rain causes wet ground | rain causes wet ground
wet ground causes slippery pavement | wet ground causes slippery pavement
why rain causes slippery pavement | rain causes slippery pavement
spark causes fire | spark causes fire
fire causes smoke | fire causes smoke
why spark causes smoke | spark causes smoke
water prevents fire | water prevents fire
"""

const EMBEDDED_LOGIC_TXT = raw"""# Logic and Deduction Dataset for Ternary Cognitive AI
# Format: input | expected output | optional quality

a causes b | a causes b
b causes c | b causes c
c causes d | c causes d
why a causes c | a causes c
why b causes d | b causes d
why a causes d | a causes d
x prevents y | x prevents y
y causes z | y causes z
repeat this | why a causes d | REQUESTED_REPETITION
"""

const EMBEDDED_SCIENCE_TXT = raw"""# Science Reasoning Dataset for Ternary Cognitive AI
# Format: input | expected output | optional quality

sun causes heat | sun causes heat
heat causes evaporation | heat causes evaporation
evaporation causes clouds | evaporation causes clouds
clouds cause rain | clouds cause rain
rain causes wet ground | rain causes wet ground
wet ground causes slippery pavement | wet ground causes slippery pavement
gravity causes attraction | gravity causes attraction
friction causes resistance | friction causes resistance
resistance causes deceleration | resistance causes deceleration
ice prevents heat | ice prevents heat
why sun causes evaporation | sun causes evaporation
why evaporation causes rain | evaporation causes rain
why rain causes slippery pavement | rain causes slippery pavement
why friction causes deceleration | friction causes deceleration
"""

const EMBEDDED_DATASETS = Dict{String,String}(
    "data/train.txt" => EMBEDDED_TRAIN_TXT,
    "train.txt" => EMBEDDED_TRAIN_TXT,
    "data/logic.txt" => EMBEDDED_LOGIC_TXT,
    "logic.txt" => EMBEDDED_LOGIC_TXT,
    "data/science.txt" => EMBEDDED_SCIENCE_TXT,
    "science.txt" => EMBEDDED_SCIENCE_TXT,
)


# ===== BEGIN json_util.jl =====

# Simple zero-dependency JSON encoder and parser for Julia stdlib

function to_json(val)
    if val === nothing
        return "null"
    elseif val isa Bool
        return val ? "true" : "false"
    elseif val isa Number
        return isfinite(val) ? string(val) : "null"
    elseif val isa Symbol
        return "\"" * _escape_json(string(val)) * "\""
    elseif val isa AbstractString
        return "\"" * _escape_json(String(val)) * "\""
    elseif val isa AbstractVector || val isa Tuple || val isa Set
        return "[" * join((to_json(x) for x in val), ",") * "]"
    elseif val isa AbstractDict
        items = String["\"" * _escape_json(string(k)) * "\":" * to_json(v) for (k, v) in val]
        return "{" * join(items, ",") * "}"
    elseif val isa NamedTuple
        items = String["\"" * _escape_json(string(k)) * "\":" * to_json(getfield(val, k)) for k in keys(val)]
        return "{" * join(items, ",") * "}"
    else
        # Fallback: if it has struct fields, serialize as dict
        fnames = fieldnames(typeof(val))
        if !isempty(fnames)
            items = String["\"" * _escape_json(string(k)) * "\":" * to_json(getfield(val, k)) for k in fnames]
            return "{" * join(items, ",") * "}"
        else
            return "\"" * _escape_json(string(val)) * "\""
        end
    end
end

function _escape_json(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            print(buf, "\\\"")
        elseif c == '\\'
            print(buf, "\\\\")
        elseif c == '\b'
            print(buf, "\\b")
        elseif c == '\f'
            print(buf, "\\f")
        elseif c == '\n'
            print(buf, "\\n")
        elseif c == '\r'
            print(buf, "\\r")
        elseif c == '\t'
            print(buf, "\\t")
        elseif Int(c) < 32
            print(buf, "\\u", string(Int(c); base=16, pad=4))
        else
            print(buf, c)
        end
    end
    return String(take!(buf))
end

struct JSONParser
    src::String
    len::Int
    idx::Ref{Int}
end

function from_json(s::AbstractString)
    p = JSONParser(String(s), lastindex(s), Ref(1))
    _skip_ws(p)
    res = _parse_val(p)
    return res
end

function _skip_ws(p::JSONParser)
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c in (' ', '\t', '\n', '\r')
            p.idx[] = nextind(p.src, p.idx[])
        else
            break
        end
    end
end

function _parse_val(p::JSONParser)
    _skip_ws(p)
    p.idx[] > p.len && return nothing
    c = p.src[p.idx[]]
    if c == '"'
        return _parse_string(p)
    elseif c == '{'
        return _parse_object(p)
    elseif c == '['
        return _parse_array(p)
    elseif c in ('t', 'f')
        return _parse_bool(p)
    elseif c == 'n'
        return _parse_null(p)
    elseif c in ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        return _parse_number(p)
    else
        error("Unexpected character in JSON: '$c' at position $(p.idx[])")
    end
end

function _parse_string(p::JSONParser)
    # skip opening quote
    p.idx[] = nextind(p.src, p.idx[])
    buf = IOBuffer()
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c == '"'
            p.idx[] = nextind(p.src, p.idx[])
            return String(take!(buf))
        elseif c == '\\'
            p.idx[] = nextind(p.src, p.idx[])
            p.idx[] > p.len && error("Unterminated escape sequence in JSON string")
            esc_c = p.src[p.idx[]]
            if esc_c == '"' || esc_c == '\\' || esc_c == '/'
                print(buf, esc_c)
            elseif esc_c == 'b'
                print(buf, '\b')
            elseif esc_c == 'f'
                print(buf, '\f')
            elseif esc_c == 'n'
                print(buf, '\n')
            elseif esc_c == 'r'
                print(buf, '\r')
            elseif esc_c == 't'
                print(buf, '\t')
            elseif esc_c == 'u'
                hex_str = p.src[p.idx[]+1 : min(p.len, p.idx[]+4)]
                print(buf, Char(parse(UInt32, hex_str; base=16)))
                p.idx[] += 4
            else
                print(buf, esc_c)
            end
        else
            print(buf, c)
        end
        p.idx[] = nextind(p.src, p.idx[])
    end
    error("Unterminated JSON string")
end

function _parse_object(p::JSONParser)
    p.idx[] = nextind(p.src, p.idx[]) # skip '{'
    dict = Dict{String, Any}()
    _skip_ws(p)
    if p.idx[] <= p.len && p.src[p.idx[]] == '}'
        p.idx[] = nextind(p.src, p.idx[])
        return dict
    end
    while p.idx[] <= p.len
        _skip_ws(p)
        p.src[p.idx[]] != '"' && error("Expected string key in JSON object at $(p.idx[])")
        key = _parse_string(p)
        _skip_ws(p)
        (p.idx[] > p.len || p.src[p.idx[]] != ':') && error("Expected ':' after key in JSON object at $(p.idx[])")
        p.idx[] = nextind(p.src, p.idx[]) # skip ':'
        val = _parse_val(p)
        dict[key] = val
        _skip_ws(p)
        if p.idx[] <= p.len && p.src[p.idx[]] == ','
            p.idx[] = nextind(p.src, p.idx[])
        elseif p.idx[] <= p.len && p.src[p.idx[]] == '}'
            p.idx[] = nextind(p.src, p.idx[])
            return dict
        else
            error("Expected ',' or '}' in JSON object at $(p.idx[])")
        end
    end
    error("Unterminated JSON object")
end

function _parse_array(p::JSONParser)
    p.idx[] = nextind(p.src, p.idx[]) # skip '['
    arr = Any[]
    _skip_ws(p)
    if p.idx[] <= p.len && p.src[p.idx[]] == ']'
        p.idx[] = nextind(p.src, p.idx[])
        return arr
    end
    while p.idx[] <= p.len
        val = _parse_val(p)
        push!(arr, val)
        _skip_ws(p)
        if p.idx[] <= p.len && p.src[p.idx[]] == ','
            p.idx[] = nextind(p.src, p.idx[])
        elseif p.idx[] <= p.len && p.src[p.idx[]] == ']'
            p.idx[] = nextind(p.src, p.idx[])
            return arr
        else
            error("Expected ',' or ']' in JSON array at $(p.idx[])")
        end
    end
    error("Unterminated JSON array")
end

function _parse_bool(p::JSONParser)
    if startswith(p.src[p.idx[]:end], "true")
        p.idx[] += 4
        return true
    elseif startswith(p.src[p.idx[]:end], "false")
        p.idx[] += 5
        return false
    else
        error("Invalid boolean in JSON at $(p.idx[])")
    end
end

function _parse_null(p::JSONParser)
    if startswith(p.src[p.idx[]:end], "null")
        p.idx[] += 4
        return nothing
    else
        error("Invalid null in JSON at $(p.idx[])")
    end
end

function _parse_number(p::JSONParser)
    start_pos = p.idx[]
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c in ('0','1','2','3','4','5','6','7','8','9','-','+','.','e','E')
            p.idx[] = nextind(p.src, p.idx[])
        else
            break
        end
    end
    num_str = p.src[start_pos : prevind(p.src, p.idx[])]
    if occursin('.', num_str) || occursin('e', num_str) || occursin('E', num_str)
        return parse(Float64, num_str)
    else
        return parse(Int, num_str)
    end
end

# ===== END json_util.jl =====


# ===== BEGIN serialization.jl =====

function save_model(ai::CognitiveAgent, path::String)
    open(path,"w") do io; serialize(io, (version=ai.serialization_version, agent=ai)); end; path
end
function load_model(path::String)
    data=open(deserialize,path); data.version == SERIALIZATION_VERSION || error("unsupported model serialization version")
    assert_network_invariants(data.agent.network); data.agent
end

# ===== END serialization.jl =====


# ===== BEGIN training.jl (patched: embedded-dataset fallback) =====

function parse_training_line(line::String, id::Int)
    s = strip(line)
    (isempty(s) || startswith(s, "#")) && return nothing
    p = split(s, " | "; limit=3)
    length(p) >= 2 || error("training line $id must be: input | expected output | optional quality")
    q = length(p) == 3 && !isempty(strip(p[3])) ? Symbol(strip(p[3])) : nothing
    return TrainingExample(id, strip(p[1]), strip(p[2]), q, nothing, String[], Symbol[], Dict{String,Any}())
end

function load_training_data(path::String)
    if !isfile(path)
        key = replace(path, "\\" => "/")
        for (k, v) in EMBEDDED_DATASETS
            endswith(key, k) && return load_training_data_from_string(v)
        end
        b = basename(path)
        haskey(EMBEDDED_DATASETS, b) && return load_training_data_from_string(EMBEDDED_DATASETS[b])
        error("training file not found: $path (no embedded fallback)")
    end
    out = TrainingExample[]
    for (i, l) in enumerate(readlines(path))
        x = parse_training_line(l, i)
        x !== nothing && push!(out, x)
    end
    return out
end


function load_training_data_from_string(text::String)
    out = TrainingExample[]
    i = 0
    for l in split(text, "\n")
        i += 1
        x = parse_training_line(String(l), i)
        x !== nothing && push!(out, x)
    end
    return out
end

function print_training_summary(ai::CognitiveAgent, report; io::IO=stdout)
    println(io, "\n" * "="^60)
    println(io, "                    TRAINING SUMMARY")
    println(io, "="^60)
    @printf(io, "  Episodes in Run:        %d\n", report.episodes)
    @printf(io, "  Total Reward:           +%.2f\n", report.total_reward)
    @printf(io, "  Total Punishment:       -%.2f\n", report.total_punishment)
    @printf(io, "  Avg Synapses Changed:   %.2f\n", report.average_changed)
    @printf(io, "  Self Corrections:       %d\n", report.self_corrections)
    @printf(io, "  Novel Recombinations:   %d\n", report.novelty)
    @printf(io, "  Runtime:                %.3f s\n", report.runtime_seconds)
    dist = report.synapses
    @printf(io, "  Ternary Synapses:       +1: %d (%.1f%%) | 0: %d (%.1f%%) | -1: %d (%.1f%%) [Total: %d]\n",
            dist.positive, dist.pct_positive, dist.zero, dist.pct_zero, dist.negative, dist.pct_negative, dist.total)
    println(io, "\n  Quality Distribution:")
    for (q, count) in sort(collect(report.quality); by=last, rev=true)
        @printf(io, "    %-22s: %d\n", string(q), count)
    end
    println(io, "="^60 * "\n")
end

function train!(ai::CognitiveAgent, data::Vector{TrainingExample}; epochs::Int=1, verbose::Bool=false, io::IO=stdout)
    start = time()
    total_examples = length(data)
    for epoch in 1:epochs
        verbose && println(io, "\n--- Epoch $epoch of $epochs ($total_examples examples) ---")
        for (i, x) in enumerate(data)
            (res, state) = respond!(ai, x.input; expected_output=x.expected_output, expected_quality=x.quality_target)
            output, eval_res, ep = res
            if verbose
                sig = eval_res.reward > 0f0 ? @sprintf("+%.2f", eval_res.reward) :
                      eval_res.punishment > 0f0 ? @sprintf("-%.2f", eval_res.punishment) : " 0.00"
                changes = length(ep.synaptic_changes)
                @printf(io, "  [%d/%d] \"%s\"\n         -> %s\n         [Quality: %s | Signal: %s | Transitions: %d]\n",
                        i, total_examples, x.input, output, string(eval_res.quality), sig, changes)
            end
        end
    end
    elapsed = time() - start
    ai.metrics.runtime_seconds += elapsed
    ai.metrics.peak_bytes = max(ai.metrics.peak_bytes, Base.gc_live_bytes())
    report = training_report(ai)
    verbose && print_training_summary(ai, report; io=io)
    return report
end

function train_file!(ai::CognitiveAgent, path::String; epochs::Int=1, verbose::Bool=true, io::IO=stdout)
    data = load_training_data(path)
    verbose && println(io, "[Training] Loaded $(length(data)) training examples from: $path")
    return train!(ai, data; epochs=epochs, verbose=verbose, io=io)
end

# ===== END training.jl =====


# ===== BEGIN benchmarks.jl =====

function run_core_demonstration(; io::IO=stdout)
    ai=cognitive_agent(:tiny)
    respond!(ai,"rain causes wet ground"; expected_output="rain causes wet ground")
    respond!(ai,"wet ground causes slippery pavement"; expected_output="wet ground causes slippery pavement")
    before=synaptic_snapshot(ai); (result,state)=respond!(ai,"why rain causes slippery pavement"; expected_output="rain causes slippery pavement")
    output,evaluation,episode=result
    println(io,"OUTPUT: $output\nQUALITY: $(evaluation.quality)\nHYPOTHESES: $(length(state.hypotheses))\nTRACE:")
    render_trace(state.trace;io=io); println(io,"SYNAPTIC_DIFF: ",synaptic_diff(before,ai)); return ai
end
function run_negative_demonstration(; io::IO=stdout)
    ai=cognitive_agent(:tiny); respond!(ai,"a causes b"; expected_output="a causes b")
    before=synaptic_snapshot(ai); (result,state)=respond!(ai,"why a causes b"; expected_output="a prevents b")
    println(io,"QUALITY: $(result[2].quality)\nPUNISHMENT: $(result[2].punishment)\nDIFF: ",synaptic_diff(before,ai)); ai
end
function run_ablation(; mode::Symbol=:no_reward)
    ai=cognitive_agent(:tiny); mode==:no_reward && (ai.config.reward_enabled=false); mode==:no_punishment && (ai.config.punishment_enabled=false)
    train!(ai,[TrainingExample(1,"x causes y","x causes y",nothing,nothing,String[],Symbol[],Dict{String,Any}())];epochs=3)
end

# ===== END benchmarks.jl =====


# ===== BEGIN cli.jl (patched: --test -> run_bundled_tests, +--legacy dispatch) =====

# cli.jl — Command-line interface, interactive chat REPL, and batch training.

function _print_cli_help(io::IO=stdout)
    println(io, raw"""
TernaryCognitiveAI — Discrete {-1, 0, +1} Reasoning Architecture

USAGE:
  julia --project=. main.jl [COMMAND / OPTIONS]
  run.bat [COMMAND / OPTIONS]

COMMANDS & MODES:
  (no arguments)              Launch the 2-page Web GUI (Chatting + Training)
  --gui                       Launch the 2-page Web GUI (Chatting + Training)
  --chat, --cli               Launch the interactive terminal chat REPL
  --train <file>              Train the agent on a dataset file
  --ask "<query>"             Ask a question or assert a statement and print answer
  --query "<query>"           Alias for --ask
  --demo                      Run the transitive deductive reasoning demonstration
  --negative-demo             Run the contradiction / self-correction demonstration
  --test                      Run the full test suite
  --help, -h                  Display this help documentation

OPTIONS:
  --port <N>                  Port for Web GUI server (default: 8080)
  --no-browser                Do not automatically open browser on startup
  --epochs <N>                Number of training epochs (default: 1)
  --scale <name>              Agent scale: tiny, small, medium, large (default: small)
  --load <model_path>         Load saved model state before running
  --save <model_path>         Save model state after training or query
  --expected "<text>"         Expected output for evaluation / supervised training
  --trace                     Display the full step-by-step reasoning trace
  --quiet                     Suppress verbose progress output during training

INTERACTIVE CHAT COMMANDS (inside REPL):
  :help                       Show help on chat commands and grammar
  :teach <in> | <target>      Teach the agent an input-target pair with immediate feedback
  :train <file> [epochs]      Train the agent on a dataset file
  :save <path>                Save the current agent state to disk
  :load <path>                Load a saved agent state from disk
  :stats, :metrics            Display agent metrics, memory counts, and ternary distribution
  :concepts                   List known concepts in semantic memory
  :relations                  List all learned and deduced relations
  :trace                      Show the full reasoning trace of the last response
  :memory                     Display working memory and episodic history
  :failures                   Display recorded reasoning failures
  :diff                       Display synaptic weight transitions from the last episode
  :reset [scale]              Reset the agent to a fresh state (tiny, small, medium, large)
  :clear                      Clear terminal screen
  :exit, :quit                Exit the interactive chat session
""")
end

function _print_chat_help(io::IO=stdout)
    println(io, raw"""
── Chat Commands ──────────────────────────────────────────────────────────
  :help                       Show this help guide
  :teach <in> | <target>      Teach input with expected target (e.g. :teach a causes b | a causes b)
  :train <path> [epochs]      Batch train from a text file (e.g. :train data/train.txt 3)
  :save <path>                Save agent state to a binary file (e.g. :save model.bin)
  :load <path>                Load agent state from file (e.g. :load model.bin)
  :stats                      View run metrics and ternary weight distribution
  :concepts                   List all concepts stored in semantic memory
  :relations                  List all stored & deduced relational triples
  :trace                      Show the authentic machine reasoning trace of the last turn
  :memory                     Inspect working memory and recalled episodes
  :failures                   Inspect failure memory records
  :diff                       Inspect synaptic changes from the last interaction
  :reset [scale]              Reinitialize agent (scales: :tiny, :small, :medium, :large)
  :clear                      Clear console screen
  :exit, :quit                Exit chat

── Natural Language Input Guide ───────────────────────────────────────────
  1. Declarative Facts:       sun causes heat
                              heat causes expansion
                              water leads to life
  2. Queries (Deductions):    why sun causes expansion
                              does sun causes heat
                              why rain causes slippery pavement
  3. Repetitions:             repeat this
                              repeat that 2 times
  4. Constraints:             sun causes heat in json
                              why sun causes expansion as a list
                              sun causes heat without expansion
                              sun causes heat in one sentence
───────────────────────────────────────────────────────────────────────────
""")
end

function _print_chat_banner(ai::CognitiveAgent, io::IO=stdout)
    d = ternary_distribution(ai.network)
    println(io, "╔════════════════════════════════════════════════════════════════════════╗")
    println(io, "║                    TERNARY COGNITIVE AI CHAT                           ║")
    println(io, "║     Discrete {-1, 0, +1} Sparse Synapses · Neuro-Symbolic Agent        ║")
    println(io, "╚════════════════════════════════════════════════════════════════════════╝")
    @printf(io, "  Scale: %-8s | Concepts: %-4d | Synapses: %-4d (+1:%d, 0:%d, -1:%d)\n",
            string(ai.config.scale), concept_count(ai), synapse_count(ai),
            d.positive, d.zero, d.negative)
    println(io, "  Type your statements or queries below. Type :help for commands, :exit to quit.\n")
end

"""Run an interactive chat REPL session with the CognitiveAgent."""
function run_interactive_chat(ai::CognitiveAgent=cognitive_agent(:small); io_in::IO=stdin, io_out::IO=stdout)
    _print_chat_banner(ai, io_out)

    while true
        print(io_out, "TernaryAI> ")
        flush(io_out)
        line = readline(io_in)
        if isempty(line) && eof(io_in)
            println(io_out, "\nExiting TernaryCognitiveAI.")
            break
        end

        raw = strip(line)
        isempty(raw) && continue

        # Handle meta-commands starting with ':' or '/'
        if startswith(raw, ":") || startswith(raw, "/")
            cmd_line = strip(raw[2:end])
            parts = split(cmd_line; limit=2)
            cmd = lowercase(first(parts))
            arg = length(parts) > 1 ? strip(parts[2]) : ""

            if cmd in ("exit", "quit", "q")
                println(io_out, "Goodbye! State preserved in memory.")
                break
            elseif cmd in ("help", "?")
                _print_chat_help(io_out)
            elseif cmd in ("clear", "cls")
                print(io_out, "\033[2J\033[1;1H")
            elseif cmd in ("stats", "metrics", "status")
                m = ai.metrics
                d = ternary_distribution(ai.network)
                println(io_out, "\n=== AGENT METRICS ===")
                @printf(io_out, "  Episodes:              %d\n", m.episodes)
                @printf(io_out, "  Total Reward:          +%.2f\n", m.total_reward)
                @printf(io_out, "  Total Punishment:      -%.2f\n", m.total_punishment)
                @printf(io_out, "  Self Corrections:      %d\n", m.self_corrections)
                @printf(io_out, "  Novel Recombinations:  %d\n", m.novel_recombinations)
                @printf(io_out, "  Contradictions Found:  %d\n", m.contradictions_detected)
                @printf(io_out, "  Hypotheses Rejected:   %d\n", m.hypotheses_rejected)
                @printf(io_out, "  Ternary Transitions:   %d\n", m.ternary_transitions)
                @printf(io_out, "  Concepts:              %d\n", concept_count(ai))
                @printf(io_out, "  Synapses:              +1: %d | 0: %d | -1: %d (Total: %d)\n",
                        d.positive, d.zero, d.negative, d.total)
                println(io_out, "=====================\n")
            elseif cmd in ("concepts", "concept")
                sm = ai.semantic
                if isempty(sm.concepts)
                    println(io_out, "No concepts learned yet.")
                else
                    println(io_out, "\n=== CONCEPTS ($(length(sm.concepts))) ===")
                    for (id, c) in sort(collect(sm.concepts); by=first)
                        rel = concept_reliability(c)
                        domains = join(c.domain_tags, ", ")
                        @printf(io_out, "  [%3d] %-20s | Domain: [%s] | Rel: %.2f (S:%d F:%d Freq:%d)\n",
                                id, c.name, domains, rel, c.successes, c.failures, c.frequency)
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd in ("relations", "relation", "facts")
                sm = ai.semantic
                if isempty(sm.relations)
                    println(io_out, "No relations stored in semantic memory.")
                else
                    println(io_out, "\n=== RELATIONS ($(length(sm.relations))) ===")
                    for (i, r) in enumerate(sm.relations)
                        s = concept_name(ai, r.subject)
                        o = concept_name(ai, r.object)
                        @printf(io_out, "  %3d. %-15s --[%-12s]--> %-15s (prov: %s)\n",
                                i, s, string(r.predicate), o, string(r.provenance))
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd == "trace"
                if ai.last_state === nothing || isempty(ai.last_state.trace.events)
                    println(io_out, "No reasoning trace recorded yet.")
                else
                    println(io_out, "\n=== REASONING TRACE (LAST TURN) ===")
                    render_trace(ai.last_state.trace; io=io_out)
                    println(io_out, "===================================\n")
                end
            elseif cmd in ("diff", "changes")
                changes = get_last_synaptic_changes(ai)
                if isempty(changes)
                    println(io_out, "No synaptic changes in the last episode.")
                else
                    println(io_out, "\n=== SYNAPTIC CHANGES ===")
                    for ch in changes
                        s = concept_name(ai, ch.source)
                        t = concept_name(ai, ch.target)
                        @printf(io_out, "  %s -> %s: %d -> %d | Credit: %s | Reason: %s\n",
                                s, t, ch.old_state, ch.new_state, string(ch.credit), ch.reason)
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd in ("memory", "mem")
                wm = get_working_memory(ai)
                println(io_out, "\n=== WORKING MEMORY ($(length(wm))) ===")
                for (i, it) in enumerate(wm)
                    @printf(io_out, "  %d. [%s] \"%s\" (rel: %.2f, step: %d)\n",
                            i, string(it.kind), it.text, it.relevance, it.step)
                end
                recalled = get_retrieved_memories(ai)
                println(io_out, "  Last Retrieved Past Episodes: ", isempty(recalled) ? "none" : string(recalled))
                println(io_out, "==============================\n")
            elseif cmd in ("failures", "failure")
                fails = ai.failures.records
                if isempty(fails)
                    println(io_out, "No failure records logged.")
                else
                    println(io_out, "\n=== RECORDED FAILURES ($(length(fails))) ===")
                    for (i, f) in enumerate(fails)
                        @printf(io_out, "  %d. Episode %d: \"%s\" (strategy: %s)\n     Reason: %s\n",
                                i, f.episode, f.context, string(f.strategy), f.reason)
                    end
                    println(io_out, "===============================\n")
                end
            elseif cmd in ("reset", "restart")
                scale = isempty(arg) ? ai.config.scale : Symbol(arg)
                ai = cognitive_agent(scale)
                println(io_out, "Agent reinitialized with scale: :$scale.")
            elseif cmd == "save"
                if isempty(arg)
                    println(io_out, "Usage: :save <path>")
                else
                    try
                        save_model(ai, String(arg))
                        println(io_out, "[OK] Model saved to: $arg")
                    catch e
                        println(io_out, "[Error] Failed to save: $e")
                    end
                end
            elseif cmd == "load"
                if isempty(arg)
                    println(io_out, "Usage: :load <path>")
                else
                    try
                        ai = load_model(String(arg))
                        println(io_out, "[OK] Model loaded from: $arg")
                        _print_chat_banner(ai, io_out)
                    catch e
                        println(io_out, "[Error] Failed to load: $e")
                    end
                end
            elseif cmd == "teach"
                if !occursin("|", arg)
                    println(io_out, "Usage: :teach <input> | <expected output> [| <quality>]")
                else
                    parts_teach = split(arg, "|")
                    inp = strip(parts_teach[1])
                    exp = strip(parts_teach[2])
                    q_tgt = length(parts_teach) >= 3 && !isempty(strip(parts_teach[3])) ? Symbol(strip(parts_teach[3])) : nothing
                    (res, state) = respond!(ai, inp; expected_output=exp, expected_quality=q_tgt)
                    out, ev, ep = res
                    sig = ev.reward > 0f0 ? @sprintf("+%.2f", ev.reward) :
                          ev.punishment > 0f0 ? @sprintf("-%.2f", ev.punishment) : "0.00"
                    println(io_out, "AI: ", out)
                    println(io_out, "  [Taught] Quality: $(ev.quality) | Signal: $sig | Synapses changed: $(length(ep.synaptic_changes))")
                end
            elseif cmd == "train"
                if isempty(arg)
                    println(io_out, "Usage: :train <dataset_path> [epochs]")
                else
                    train_args = split(arg)
                    t_path = String(train_args[1])
                    t_epochs = length(train_args) > 1 ? parse(Int, train_args[2]) : 1
                    try
                        train_file!(ai, t_path; epochs=t_epochs, verbose=true, io=io_out)
                    catch e
                        println(io_out, "[Error] Training failed: $e")
                    end
                end
            else
                println(io_out, "Unknown command: :$cmd. Type :help for commands.")
            end
            continue
        end

        # Normal reasoning response
        try
            (result, state) = respond!(ai, String(raw))
            output, evaluation, episode = result
            println(io_out, "\nAI: ", output)

            sig = evaluation.reward > 0f0 ? @sprintf("+%.2f", evaluation.reward) :
                  evaluation.punishment > 0f0 ? @sprintf("-%.2f", evaluation.punishment) : " 0.00"
            ch_cnt = length(episode.synaptic_changes)
            units_active = length(state.active_units)

            extra_notes = String[]
            if !isempty(state.contradictions)
                push!(extra_notes, "Contradiction self-corrected")
            end
            if evaluation.quality == :REQUESTED_REPETITION
                push!(extra_notes, "Repetition fulfilled")
            end
            if evaluation.punishment > 0f0
                push!(extra_notes, "Failure logged")
            end
            notes_str = isempty(extra_notes) ? "" : " | " * join(extra_notes, ", ")

            @printf(io_out, "  [Quality: %s | Conf: %.2f | Signal: %s | Steps: %d | Active: %d units | Changed: %d%s]\n\n",
                    string(evaluation.quality), state.confidence, sig, state.steps_used,
                    units_active, ch_cnt, notes_str)
        catch e
            println(io_out, "[Error] Processing failed: $e")
        end
    end
end

"""Command-line entry point with full argument dispatch."""
function run_cli(args=ARGS)
    # 0. Legacy ternary-reward baseline (bundled LegacyBaseline module)
    if "--legacy" in args
        legacy_args = filter(a -> a != "--legacy", collect(String, args))
        LegacyBaseline.run(Vector{String}(legacy_args))
        return 0
    end

    # 1. Help
    if "--help" in args || "-h" in args
        _print_cli_help()
        return 0
    end

    # 2. Existing demonstration & test modes
    if "--demo" in args
        run_core_demonstration()
        return 0
    end
    if "--negative-demo" in args
        run_negative_demonstration()
        return 0
    end
    if "--test" in args
        run_bundled_tests()
        return 0
    end

    # Parse scale
    scale = :small
    idx_scale = findfirst(a -> a == "--scale", args)
    if idx_scale !== nothing && idx_scale < length(args)
        scale = Symbol(args[idx_scale + 1])
    end

    # Create or load agent
    ai = cognitive_agent(scale)
    idx_load = findfirst(a -> a == "--load", args)
    if idx_load !== nothing && idx_load < length(args)
        load_path = args[idx_load + 1]
        println("[Init] Loading model from: $load_path")
        ai = load_model(load_path)
    end

    # 3. Batch training mode
    idx_train = findfirst(a -> a == "--train", args)
    if idx_train !== nothing && idx_train < length(args)
        train_path = args[idx_train + 1]
        epochs = 1
        idx_epochs = findfirst(a -> a == "--epochs", args)
        if idx_epochs !== nothing && idx_epochs < length(args)
            epochs = parse(Int, args[idx_epochs + 1])
        end
        verbose = !("--quiet" in args)
        train_file!(ai, train_path; epochs=epochs, verbose=verbose)

        idx_save = findfirst(a -> a == "--save", args)
        if idx_save !== nothing && idx_save < length(args)
            save_path = args[idx_save + 1]
            save_model(ai, save_path)
            println("[Saved] Model saved to: $save_path")
        end

        # If --chat was not explicitly passed, exit after training
        if !("--chat" in args)
            return 0
        end
    end

    # 4. Single-shot query / ask mode
    idx_ask = findfirst(a -> a in ("--ask", "--query"), args)
    if idx_ask !== nothing && idx_ask < length(args)
        query_text = args[idx_ask + 1]
        exp_text = nothing
        idx_exp = findfirst(a -> a == "--expected", args)
        if idx_exp !== nothing && idx_exp < length(args)
            exp_text = args[idx_exp + 1]
        end

        (result, state) = respond!(ai, query_text; expected_output=exp_text)
        output, evaluation, ep = result

        println("\nINPUT:      $query_text")
        println("OUTPUT:     $output")
        println("QUALITY:    $(evaluation.quality)")
        @printf("CONFIDENCE: %.3f\n", state.confidence)
        if evaluation.reward > 0f0
            @printf("REWARD:     +%.2f\n", evaluation.reward)
        elseif evaluation.punishment > 0f0
            @printf("PUNISHMENT: -%.2f\n", evaluation.punishment)
        end
        println("REASONS:    ", join(evaluation.reasons, "; "))

        if "--trace" in args
            println("\n--- TRACE ---")
            render_trace(state.trace)
        end

        idx_save = findfirst(a -> a == "--save", args)
        if idx_save !== nothing && idx_save < length(args)
            save_path = args[idx_save + 1]
            save_model(ai, save_path)
            println("[Saved] Model saved to: $save_path")
        end
        return 0
    end

    # 5. Port and browser options for GUI
    port = 8080
    idx_port = findfirst(a -> a == "--port", args)
    if idx_port !== nothing && idx_port < length(args)
        port = parse(Int, args[idx_port + 1])
    end
    open_browser = !("--no-browser" in args)

    # 6. Terminal interactive chat REPL if --chat or --cli passed
    if "--chat" in args || "--cli" in args
        run_interactive_chat(ai)
        return 0
    end

    # 7. Default mode: Launch 2-page Web GUI (Chatting + Training)
    return run_gui(; port=port, open_browser=open_browser, scale=scale,
                   load_path=idx_load !== nothing && idx_load < length(args) ? args[idx_load + 1] : nothing)
end

# ===== END cli.jl =====


# ===== BEGIN gui.jl (patched: flat paths, embedded HTML/datasets) =====

# gui.jl — Built-in HTTP server and Web GUI for Chatting & Training.
# Zero external packages; uses standard library Sockets and json_util.jl.

using Sockets

# (bundled) json_util.jl already included above — include removed for flat layout

function _url_decode(s::AbstractString)
    s = replace(s, "+" => " ")
    out = IOBuffer()
    i = 1
    len = lastindex(s)
    while i <= len
        c = s[i]
        if c == '%' && i + 2 <= len
            hex_str = s[i+1:i+2]
            try
                b = parse(UInt8, hex_str; base=16)
                write(out, b)
                i += 3
                continue
            catch
            end
        end
        write(out, c)
        i = nextind(s, i)
    end
    return String(take!(out))
end

function _parse_query_params(query_str::AbstractString)
    params = Dict{String, String}()
    isempty(query_str) && return params
    for pair in split(query_str, "&")
        parts = split(pair, "="; limit=2)
        if length(parts) == 2
            params[_url_decode(parts[1])] = _url_decode(parts[2])
        elseif length(parts) == 1
            params[_url_decode(parts[1])] = ""
        end
    end
    return params
end

function _format_trace_step(e::ReasoningEvent)
    buf = IOBuffer()
    print(buf, string(e.kind))
    attrs = String[]
    for (k, v) in e.attributes
        push!(attrs, "$k=$v")
    end
    if !isempty(attrs)
        print(buf, " (", join(attrs, ", "), ")")
    end
    return String(take!(buf))
end

"""Start the built-in Web GUI server serving the 2-page Chatting & Training dashboard."""
function start_gui_server(initial_agent::CognitiveAgent=cognitive_agent(:small); port::Int=8080, open_browser::Bool=true)
    ai_ref = Ref{CognitiveAgent}(initial_agent)

    # Listen on IPv4 localhost with fallback port hunting
    local server = nothing
    local actual_port = port
    for p in port:(port + 15)
        try
            server = listen(IPv4(127, 0, 0, 1), p)
            actual_port = p
            break
        catch
            # Try next port
        end
    end

    if server === nothing
        error("Could not bind GUI server on any port between $port and $(port + 15)")
    end

    html_path = joinpath(@__DIR__, "index.html")
    html_content = isfile(html_path) ? read(html_path, String) : EMBEDDED_INDEX_HTML

    url = "http://127.0.0.1:$actual_port"
    println("="^68)
    println("   🧠 TERNARY COGNITIVE AI — WEB DASHBOARD RUNNING")
    println("   URL: $url")
    println("   Features: 💬 Chatting Tab  |  🎓 Training Tab")
    println("   Press Ctrl+C in this console to stop the server.")
    println("="^68)

    if open_browser
        try
            run(`cmd.exe /c start $url`)
        catch
        end
    end

    @async try
        while isopen(server)
            client = accept(server)
            @async try
                _handle_client(client, ai_ref, html_content)
            catch
            finally
                close(client)
            end
        end
    catch e
        if !(e isa Base.IOError)
            @warn "GUI server loop error: $e"
        end
    end

    return server, actual_port
end

function _send_response(io::IO, status_code::Int, content_type::String, body::Union{String, Vector{UInt8}})
    bytes = body isa String ? codeunits(body) : body
    status_text = status_code == 200 ? "OK" : status_code == 404 ? "Not Found" : "Internal Server Error"
    write(io, "HTTP/1.1 $status_code $status_text\r\n")
    write(io, "Content-Type: $content_type\r\n")
    write(io, "Content-Length: $(length(bytes))\r\n")
    write(io, "Access-Control-Allow-Origin: *\r\n")
    write(io, "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n")
    write(io, "Access-Control-Allow-Headers: Content-Type\r\n")
    write(io, "Connection: close\r\n")
    write(io, "\r\n")
    write(io, bytes)
    flush(io)
end

function _handle_client(io::IO, ai_ref::Ref{CognitiveAgent}, html_content::String)
    line = readline(io)
    isempty(line) && return
    parts = split(line)
    length(parts) >= 2 || return
    method = parts[1]
    raw_uri = parts[2]

    # Preflight OPTIONS
    if method == "OPTIONS"
        _send_response(io, 200, "text/plain", "")
        return
    end

    # Read Headers
    headers = Dict{String, String}()
    while true
        hline = readline(io)
        (isempty(hline) || hline == "\r") && break
        hparts = split(hline, ":"; limit=2)
        if length(hparts) == 2
            headers[lowercase(strip(hparts[1]))] = strip(hparts[2])
        end
    end

    # Read Body if present
    content_len = parse(Int, get(headers, "content-length", "0"))
    body_str = ""
    if content_len > 0
        body_bytes = read(io, content_len)
        body_str = String(body_bytes)
    end

    # Route request
    uri_parts = split(raw_uri, "?"; limit=2)
    path = uri_parts[1]
    query_str = length(uri_parts) > 1 ? uri_parts[2] : ""
    params = _parse_query_params(query_str)

    ai = ai_ref[]

    if method == "GET" && path == "/"
        _send_response(io, 200, "text/html; charset=utf-8", html_content)

    elseif method == "GET" && path == "/api/status"
        m = ai.metrics
        d = ternary_distribution(ai.network)
        data = Dict(
            "scale" => string(ai.config.scale),
            "concepts" => concept_count(ai),
            "synapses" => Dict("positive" => d.positive, "zero" => d.zero, "negative" => d.negative, "total" => d.total),
            "metrics" => Dict(
                "episodes" => m.episodes,
                "total_reward" => m.total_reward,
                "total_punishment" => m.total_punishment,
                "self_corrections" => m.self_corrections,
                "novelty" => m.novel_recombinations,
                "contradictions" => m.contradictions_detected,
                "rejected" => m.hypotheses_rejected
            )
        )
        _send_response(io, 200, "application/json", to_json(data))

    elseif method == "POST" && path == "/api/chat"
        req = from_json(body_str)
        inp = string(get(req, "input", ""))
        expected = haskey(req, "expected") && !isempty(req["expected"]) ? string(req["expected"]) : nothing

        (result, state) = respond!(ai, inp; expected_output=expected)
        output, evaluation, ep = result

        trace_steps = String[_format_trace_step(e) for e in state.trace.events]
        has_contra = !isempty(state.contradictions)

        resp = Dict(
            "output" => output,
            "quality" => string(evaluation.quality),
            "confidence" => state.confidence,
            "reward" => evaluation.reward,
            "punishment" => evaluation.punishment,
            "synaptic_changes" => length(ep.synaptic_changes),
            "contradiction" => has_contra,
            "trace" => trace_steps
        )
        _send_response(io, 200, "application/json", to_json(resp))

    elseif method == "POST" && path == "/api/teach"
        req = from_json(body_str)
        inp = string(get(req, "input", ""))
        exp = string(get(req, "expected", ""))
        (result, state) = respond!(ai, inp; expected_output=exp)
        output, evaluation, ep = result
        resp = Dict("output" => output, "quality" => string(evaluation.quality), "reward" => evaluation.reward, "punishment" => evaluation.punishment)
        _send_response(io, 200, "application/json", to_json(resp))

    elseif method == "GET" && path == "/api/dataset"
        dname = get(params, "name", "data/train.txt")
        # Sanitize relative path
        norm_path = normpath(joinpath(@__DIR__, dname))
        content = isfile(norm_path) ? read(norm_path, String) : get(EMBEDDED_DATASETS, dname, get(EMBEDDED_DATASETS, basename(dname), ""))
        resp = Dict("name" => dname, "content" => content)
        _send_response(io, 200, "application/json", to_json(resp))

    elseif method == "POST" && path == "/api/train"
        req = from_json(body_str)
        text = string(get(req, "text", ""))
        epochs = parse(Int, string(get(req, "epochs", "1")))

        lines = split(text, "\n")
        examples = TrainingExample[]
        for (i, l) in enumerate(lines)
            ex = parse_training_line(String(l), i)
            ex !== nothing && push!(examples, ex)
        end

        log_buf = IOBuffer()
        println(log_buf, "Parsed $(length(examples)) training examples.")
        report = train!(ai, examples; epochs=epochs, verbose=true, io=log_buf)
        logs = String(take!(log_buf))

        resp = Dict(
            "success" => true,
            "logs" => logs,
            "episodes" => report.episodes,
            "total_reward" => report.total_reward,
            "total_punishment" => report.total_punishment
        )
        _send_response(io, 200, "application/json", to_json(resp))

    elseif method == "POST" && path == "/api/reset"
        req = isempty(body_str) ? Dict() : from_json(body_str)
        scale = Symbol(get(req, "scale", "small"))
        ai_ref[] = cognitive_agent(scale)
        _send_response(io, 200, "application/json", to_json(Dict("success" => true, "scale" => string(scale))))

    elseif method == "POST" && path == "/api/save"
        req = from_json(body_str)
        save_path = normpath(joinpath(@__DIR__, get(req, "path", "model.bin")))
        save_model(ai, String(save_path))
        _send_response(io, 200, "application/json", to_json(Dict("success" => true, "message" => "Model saved to $save_path")))

    elseif method == "POST" && path == "/api/load"
        req = from_json(body_str)
        load_path = normpath(joinpath(@__DIR__, get(req, "path", "model.bin")))
        if isfile(load_path)
            ai_ref[] = load_model(String(load_path))
            _send_response(io, 200, "application/json", to_json(Dict("success" => true, "message" => "Model loaded from $load_path")))
        else
            _send_response(io, 404, "application/json", to_json(Dict("success" => false, "message" => "File not found: $load_path")))
        end

    elseif method == "GET" && path == "/api/concepts"
        concepts_list = Any[]
        for (id, c) in sort(collect(ai.semantic.concepts); by=first)
            push!(concepts_list, Dict(
                "id" => id,
                "name" => c.name,
                "reliability" => Float64(concept_reliability(c)),
                "successes" => c.successes,
                "failures" => c.failures,
                "frequency" => c.frequency
            ))
        end
        _send_response(io, 200, "application/json", to_json(concepts_list))

    elseif method == "GET" && path == "/api/relations"
        rels_list = Any[]
        for r in ai.semantic.relations
            push!(rels_list, Dict(
                "subject" => concept_name(ai, r.subject),
                "predicate" => string(r.predicate),
                "object" => concept_name(ai, r.object),
                "provenance" => string(r.provenance)
            ))
        end
        _send_response(io, 200, "application/json", to_json(rels_list))

    else
        _send_response(io, 404, "text/plain", "Not Found: $path")
    end
end

"""Run the Web GUI server and block until interrupted (e.g. via Ctrl+C)."""
function run_gui(; port::Int=8080, open_browser::Bool=true, scale::Symbol=:small, load_path::Union{Nothing,String}=nothing)
    ai = load_path !== nothing && isfile(load_path) ? load_model(load_path) : cognitive_agent(scale)
    server, actual_port = start_gui_server(ai; port=port, open_browser=open_browser)
    try
        while isopen(server)
            sleep(0.5)
        end
    catch e
        if e isa InterruptException
            println("\n[Shutdown] Stopping GUI server...")
            close(server)
        else
            rethrow(e)
        end
    end
    return 0
end

# ===== END gui.jl =====

# ===== END app.jl =====
