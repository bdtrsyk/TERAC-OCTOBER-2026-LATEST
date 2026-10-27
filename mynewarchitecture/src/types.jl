
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

