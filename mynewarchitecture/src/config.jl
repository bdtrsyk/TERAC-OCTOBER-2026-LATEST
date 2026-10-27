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
Base.@kwdef mutable struct PlasticityConfig
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

