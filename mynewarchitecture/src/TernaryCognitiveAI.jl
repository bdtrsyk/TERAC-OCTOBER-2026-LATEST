"""
    TernaryCognitiveAI

Research-grade discrete ternary cognitive architecture.

Persistent learned synapses are strictly ∈ {-1, 0, +1}.
Learning uses episode → evaluation → reward/punishment → eligibility → credit → ternary transition.
No next-token prediction, no gradient-based updates to the cognitive network, no hidden float weights.
"""
module TernaryCognitiveAI

using Random
using Dates
using Printf
using Serialization
using SHA

# ── Core configuration & typed events ──────────────────────────────────────
include("config.jl")
include("events.jl")
include("types.jl")

# ── Ternary network ────────────────────────────────────────────────────────
include("synapses.jl")
include("units.jl")
include("pathways.jl")
include("network.jl")
include("activation.jl")

# ── Concepts & language ────────────────────────────────────────────────────
include("concepts.jl")
include("relations.jl")
include("language.jl")
include("intent.jl")

# ── Memory systems ─────────────────────────────────────────────────────────
include("memory_systems.jl")
include("working_memory.jl")
include("semantic_memory.jl")
include("episodic_memory.jl")
include("strategy_memory.jl")

# ── Reasoning engine ───────────────────────────────────────────────────────
include("hypotheses.jl")
include("contradiction.jl")
include("self_correction.jl")
include("meta_reasoning.jl")
include("confidence.jl")
include("novelty.jl")
include("agent.jl")
include("reasoning.jl")
include("output_generation.jl")

# ── Evaluation & learning ──────────────────────────────────────────────────
include("evaluator.jl")
include("rewards.jl")
include("punishment.jl")
include("eligibility.jl")
include("credit_assignment.jl")
include("plasticity.jl")
include("learning.jl")

# ── Diagnostics, I/O, training ─────────────────────────────────────────────
include("diagnostics.jl")
include("audit.jl")
include("embedded_assets.jl")
include("json_util.jl")
include("serialization.jl")
include("training.jl")
include("benchmarks.jl")
include("experiments.jl")
include("bridge.jl")
include("cli.jl")

export
    # Config / construction
    DebugLevel, QUIET, NORMAL, VERBOSE, TRACE, AUDIT,
    ReasoningBudget, EligibilityConfig, PlasticityConfig,
    CognitiveConfig, config_for_scale, default_quality_signals,
    CognitiveAgent, cognitive_agent, soft_reset!,
    # Core types
    CognitiveUnit, SparseSynapse, SparseTernaryNetwork,
    CognitiveConcept, CognitiveRelation, UserIntent,
    CognitiveHypothesis, HypothesisScore, CognitiveState, CognitiveEpisode,
    TrainingExample, EvaluationResult, CreditAssignment, SynapticChange,
    CreditLevel, STRONG_CREDIT, MODERATE_CREDIT, WEAK_CREDIT, NEUTRAL_CREDIT, NEGATIVE_CREDIT,
    ReasoningPhase, EventKind, ReasoningEvent, TraceBuffer,
    # Operations
    reason!, respond!, render_output, evaluate_output,
    assign_credit, apply_plasticity!, complete_episode!,
    train!, train_file!, load_training_data, save_model, load_model,
    verify_integrity, assert_network_invariants, ternary_distribution,
    # Introspection
    get_active_units, get_active_synapses, get_working_memory, get_retrieved_memories,
    get_hypotheses, get_current_goal, get_reasoning_state, get_last_evaluation,
    get_last_reward, get_last_punishment, get_last_synaptic_changes,
    synaptic_snapshot, synaptic_diff, memory_snapshot, memory_diff,
    training_report, print_training_summary, render_trace,
    concept_count, synapse_count, get_concept_id, episode_count,
    reward!, punish!, learn_from_feedback!,
    # Audit / stats
    run_architecture_audit, run_resource_stats, static_source_audit,
    floating_learned_weight_count,
    # Demonstrations / CLI
    run_core_demonstration, run_negative_demonstration, run_repetition_demonstration,
    run_novelty_demonstration, run_self_correction_demonstration,
    run_cli, run_interactive_chat, run_experiment,
    # Python bridge (JSON-lines)
    run_json_bridge

end # module
