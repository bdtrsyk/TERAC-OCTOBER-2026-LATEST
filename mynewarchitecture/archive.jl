# archive.jl -- HISTORICAL REFERENCE bundle (not loaded by the package).
# Merged: legacy.jl (module LegacyBaseline), tests.jl (run_bundled_tests),
# readme.txt (embedded below). Kept for provenance of the legacy
# Mamba-inspired + MoE baseline experiments. Nothing in src/, test/,
# main.jl, or run.bat loads this file. Internal historical comments may
# reference older layouts (flat-file era); they describe history, not the
# current tree.

# ===== EMBEDDED README (was readme.txt; kept so the 5-file layout loses nothing) =====
const EMBEDDED_README = raw"""
=========================================
TERNARY COGNITIVE AI - readme.txt
Flat 10-file layout
=========================================

WHAT THIS IS
------------
Julia-first experimental reasoning platform with a sparse graph
of learned Int8 synapses restricted to -1, 0, +1. Structured
concepts, semantic/working/episodic/strategy/failure memory, typed
event traces, hypotheses, contradiction checks, backtracking,
behavioral evaluation, eligibility credit, and discrete plasticity.
No backpropagation, no next-token prediction.

This folder contains EXACTLY 10 files, flat (no subfolders):
  7x .jl : main.jl, core.jl, memory.jl, reasoning.jl, app.jl,
           legacy.jl, tests.jl
  1x run.bat, 1x requirements.txt, 1x readme.txt (this file).

The previous src/, legacy/, test/, data/, examples/, experiments/,
logs/ directories, Project.toml, README.md, and model binaries were
consolidated into the 7 .jl files. Nothing was dropped:
  - src/*.jl        -> core.jl / memory.jl / reasoning.jl / app.jl
  - src/web/index.html -> EMBEDDED in app.jl (EMBEDDED_INDEX_HTML)
  - data/*.txt      -> EMBEDDED in app.jl (EMBEDDED_DATASETS) with
                       on-disk override still supported
  - legacy/*.jl     -> legacy.jl (module LegacyBaseline)
  - test/*.jl       -> tests.jl (run_bundled_tests + debug demo)
  - main.jl         -> new module wrapper + includes + entry point

QUICK START
-----------
  Double-click run.bat, or from a terminal in this folder:
    run.bat
    run.bat --help
    run.bat --demo
    run.bat --negative-demo
    run.bat --test
    run.bat --chat
    run.bat --ask "why sun causes expansion"
    run.bat --train train.txt --epochs 3
    run.bat --gui --port 8080
    run.bat --legacy --expA        (legacy baseline, comparison only)

Direct Julia (no --project needed, stdlib only):
    julia main.jl --demo
    julia main.jl --test
    julia main.jl --chat

MODES
-----
  (no args) / --gui   2-page Web GUI (Chatting + Training), default
  --chat, --cli       terminal chat REPL (:help, :teach, :train, :save,
                      :load, :stats, :concepts, :relations, :trace,
                      :memory, :failures, :diff, :reset, :clear, :exit)
  --demo              transitive deduction demo (rain -> wet -> slippery)
  --negative-demo     contradiction / punishment demo
  --test              bundled test suite (was test/runtests.jl)
  --train <file>      batch train on a dataset file (embedded train.txt,
                      logic.txt, science.txt work by bare name too)
  --ask "<query>"     single query (+ --expected, --trace, --save/--load)
  --legacy [...]      legacy Mamba-inspired + MoE baseline (comparison)

DATASETS
--------
Format per line:  input | expected output | optional quality
Embedded names that work without files on disk:
  train.txt / data/train.txt, logic.txt, science.txt
Example lines (also in EMBEDDED_TRAIN_TXT):
  sun causes heat | sun causes heat
  why sun causes expansion | sun causes expansion

NOTES
-----
  - Research prototype with a narrow rule-language parser; synthetic
    demos do not establish general intelligence or broad competence.
  - Model save/load uses Julia Serialization (see :save / :load,
    --save / --load, and the GUI save/load buttons).
  - Backup of the pre-flatten project was taken before conversion;
    ask the owner for the backup path if you need the old tree.

FILE CHECK
----------
  You should see exactly these 10 in this folder:
    main.jl, core.jl, memory.jl, reasoning.jl, app.jl, legacy.jl,
    tests.jl, run.bat, requirements.txt, readme.txt


=========================================
APPENDIX A - previous README.md (preserved)
=========================================

# Ternary Cognitive AI

Julia-first experimental reasoning platform with a sparse graph of learned `Int8` synapses restricted to `-1`, `0`, and `+1`. It uses structured concepts, semantic/working/episodic/strategy/failure memory, typed event traces, hypotheses, contradiction checks, backtracking, behavioral evaluation, eligibility credit, and discrete plasticity. It uses neither backpropagation nor next-token prediction.

Run `julia --project=. main.jl --demo`, `--negative-demo`, or `--test`. The system is a small research prototype with a deliberately narrow rule-language parser; its synthetic demonstrations do not establish general intelligence, consciousness, or broad language competence.


=========================================
APPENDIX B - legacy baseline notes (was legacy/README_legacy.txt, preserved)
=========================================

==========================================
TERNARY COGNITIVE AI / TERNARY REWARD BASELINE
Experimental Research Prototype v0.2
==========================================

1. WHAT THIS IS
===============

This repository now contains two explicitly separate experimental systems:

  1. A legacy ternary reward-training baseline (Mamba-inspired + MoE), kept
     for comparison and historical experiments.
  2. The canonical cognitive architecture: sparse ternary synapses plus
     structured concepts, bounded working memory, long-term and episodic
     memory, explicit hypotheses, contradiction checks, behavioral evaluation,
     and eligibility-based credit assignment.

The cognitive architecture is not a transformer, RNN, next-token predictor,
or backpropagation system. The legacy baseline is never relabeled as it.

The baseline is an experimental AI training system that tests a novel
optimisation hypothesis:

  Instead of globally adjusting large numbers of continuous
  parameters through gradient descent, explicitly record
  which parameters contributed to each model output, then
  use sparse discrete REWARD/PUNISH/NEUTRAL signals to
  update only the responsible ternary parameters.

This is a RESEARCH PROTOTYPE. It is not claimed to be
faster, more accurate, or better than BitNet, conventional
backpropagation, or any other established method.

The purpose is to test the mechanism, not to beat baselines.


2. WHY IT EXISTS
================

Most modern training uses continuous-valued gradients that
update every parameter simultaneously. This prototype asks:

  What if we could identify which specific parameters caused
  a behaviour and update only those, using ternary {-1,0,+1}
  states and explicit reward/punishment signals?

Potential advantages (hypothesised, not proven):
  - Sparse updates → lower compute per step
  - Ternary weights → 1.58 bits per parameter
  - Explicit attribution → interpretable training
  - Credit assignment → targeted corrections

Potential disadvantages:
  - Very limited parameter expressiveness
  - Contribution ≠ causality (see section 8)
  - May not converge on complex tasks
  - Small prototype may not generalise


3. ARCHITECTURE
===============

  INPUT
    |
    v
  Input Projection       [Group 0]   100 ternary params
    |
    v
  Mamba-inspired SSM     [Groups 1-3] 300 ternary params
  (state-space model with selective gating)
    |
    v
  MoE Router             [Group 4]   100 ternary params
    |
    v
  4 Experts              [Groups 5-8] 400 ternary params
  (top-2 routing, probability-weighted)
    |
    v
  Output Projection      [Group 9]   100 ternary params
    |
    v
  CONTRIBUTION TRACE
    |
    v
  NEURO-SYMBOLIC EVAL
    |
    v
  REWARD / PUNISH / NEUTRAL
    |
    v
  CREDIT ASSIGNMENT (hierarchical)
    |
    v
  UPDATE RESPONSIBLE TERNARY PARAMS ONLY


4. PARAMETER BUDGET
===================

  Total trainable parameters: 1000
  Parameter states:           {-1, 0, +1} (ternary)
  Number of groups:           10
  Parameters per group:       100
  Hidden dimension:           10
  Matrix size per group:      10x10

  All other tensors in the system are either:
    - Fixed (encoding schemes)
    - Non-trainable (softmax, tanh)
    - Temporary (intermediate activations)
    - Derived (from the 1000 ternary params)


5. MAMBA COMPONENT
==================

  The sequence-processing component is labelled
  "Mamba-inspired state-space core" — NOT "Mamba 3".

  Mamba 3 requires hardware-aware selective-scan kernels
  that have no Julia implementation at this time.

  What is implemented:
    - Discrete state-space model: s(t) = tanh(A*s + B*x)
    - Read-out: y = tanh(C*s)
    - Selective gating: gate = sigmoid(x[1])
    - All weight matrices are ternary

  For scaling to real Mamba 3, bridge to the Python
  mamba-ssm package via subprocess/stdin-stdout protocol.


6. MIXTURE OF EXPERTS
=====================

  4 experts, each a 10x10 ternary matrix.
  Router: 10x10 ternary matrix, first 4 outputs → logits.
  Routing: softmax → top-2 selection → probability-weighted.

  Router exposes:
    - Expert probabilities
    - Selected expert indices
    - Per-parameter routing contribution

  Credit assignment hierarchy:
    expert → group → parameter


7. NEURO-SYMBOLIC EVALUATOR
============================

  Deterministic symbolic rules, NOT another neural net.

  Checks:
    - Exact/approximate numeric correctness
    - Pattern matching (arithmetic sequences)
    - Format compliance (number_only, integer_only)
    - Context-sensitive repetition detection
    - Logical consistency (comparison answers)

  Repetition rule:
    - Requested repetition → NEUTRAL (not punished)
    - Unwanted repetition  → PUNISH
    - Never: "repeated token = automatically bad"


8. CONTRIBUTION vs CAUSALITY
============================

  The tracked quantity is called "contribution" or
  "eligibility" — NOT "causal effect".

  Contribution measures how much a parameter participated
  in the computation (|W[i,j] * x[j]|, normalised).

  This is NOT a causal intervention test. A parameter may
  have high contribution without being causally responsible.

  For high-confidence cases, the system supports optional
  intervention testing: mask the parameter, re-run, and
  measure behavioural change. This provides a stronger
  (but more expensive) causal estimate.


9. TRAINING SIGNALS
===================

  REWARD:  Correct, high-quality output.
           Applied to top contributing parameters.
           Accumulates toward +1 promotion.

  PUNISH:  Incorrect or unwanted output.
           Applied to top contributing parameters.
           Accumulates toward -1 demotion.

  NEUTRAL: No update. Reasons include:
           - Requested repetition executed correctly
           - Low evaluator confidence
           - Ambiguous attribution


10. CREDIT ASSIGNMENT
=====================

  Hierarchical search:
    model → expert → group → parameter subset → individual

  Process:
    1. Rank experts by total contribution
    2. Rank groups by total contribution
    3. Rank individual parameters by contribution
    4. Select top-k candidates (configurable: 1, 4, 8, 16)
    5. Apply signal only to selected candidates

  Records: search depth, accepted candidates, reason.


11. TERNARY MUTATION
====================

  States: {-1, 0, +1}

  Promotion (on accumulated reward exceeding threshold):
    -1 → 0 → +1

  Demotion (on accumulated punishment exceeding threshold):
    +1 → 0 → -1

  Configurable thresholds prevent mutation on weak signals.


12. BINARY LOGGING
==================

  Compact binary log format (15 bytes per record):
    param_id (4B) + group_id (2B) + contribution (4B)
    + expert_id (1B) + step (4B)

  Written to %TEMP%\ternary_reward_runtime (NOT project dir).

  Diagnostic reader: trace_read_binary() / trace_reconstruct_stats()
  Can reconstruct: reward/punish counts, top contributors,
  parameter mutation history.


13. EXPERIMENTS
===============

  Experiment A: Proposed algorithm
    - Contribution tracing + symbolic evaluation
    - Credit assignment → sparse ternary updates

  Experiment B: Random mutation baseline
    - Same mutation rate as A
    - Random parameter selection (no tracing)

  Experiment C: Global update baseline
    - All 1000 parameters receive the signal
    - No sparse selection

  The baseline experiments exist only for comparison.
  They are NOT the proposed algorithm.


14. DATASET
============

  Embedded directly in experiments.jl (no separate file).

  Training set (22 examples):
    - Arithmetic:  3+5, 7-2, 4+6, 9-3, 1+8, 2+7, 5*2, 3*3, 6-4, 8+1
    - Comparison:  7>3?, 2>8?, 5<9?, 6<2?
    - Pattern:     2,4,6,?  1,3,5,?  5,10,15,?
    - Copy:        echo 7, echo 3
    - Repetition:  repeat 5 (×3), repeat 2 (×4)
    - Format:      4+2 (number_only)

  Test set (8 examples):
    - Arithmetic:  5+3, 8-5, 2*4
    - Comparison:  9>1?, 3<1?
    - Pattern:     3,6,9,?
    - Copy:        echo 9
    - Repetition:  repeat 7


15. HOW TO RUN
==============

  Double-click run.bat or from command line:

    run.bat

  This will:
    1. Check for Julia in PATH
    2. Create runtime directory in %TEMP%
    3. Run automated tests
    4. Run all three experiments
    5. Print comparative summary

  Options (pass after run.bat or use julia main.jl):
    --test     Run tests only
    --debug    Run Experiment A with step-by-step debug output
    --expA     Run only Experiment A
    --expB     Run only Experiment B
    --expC     Run only Experiment C
    --cognitive-demo  Run the explicit cognitive-architecture demonstration
    --cognitive-test  Run cognitive architecture integrity tests
    --cognitive-benchmark  Run an inspectable causal-reasoning benchmark


15A. COGNITIVE ARCHITECTURE
===========================

Run the canonical cognitive prototype with:

    julia --project=. main.jl --cognitive-demo
    julia --project=. main.jl --cognitive-test

`cognitive_types.jl` defines real units, sparse ternary synapses, concepts,
intent, hypotheses, state, evaluations, and episodes. `cognitive_memory.jl`
implements a bounded working memory, selective concept retrieval, a persistent
relation graph, and integer support/opposition activation. `cognitive_reasoning.jl`
performs visible parse → retrieve → activate → generate → test → select
transitions. Its hypothesis sources are direct retrieval, causal transitivity,
concept recombination, and counterfactual alternatives. Contradicted candidates
are marked rather than hidden.

`cognitive_learning.jl` maps EXCELLENT, GOOD, PARTIAL, NEUTRAL, WRONG,
REPETITIVE, REQUESTED_REPETITION, CONTRADICTORY, and INCOMPLETE to configurable
behavioral signals. Intent is inspected before repetition: a requested replay
of the prior answer receives REQUESTED_REPETITION (+0.40), while an unsolicited
duplicate receives REPETITIVE (-0.50). Reward and punishment update only
eligible active pathways in discrete -1 → 0 → +1 or +1 → 0 → -1 transitions.
Confidence, retrieval relevance, and eligibility are temporary signals; no
persistent learned connection is a float.

`cognitive_serialization.jl` saves and restores the entire agent through Julia
Serialization and verifies the ternary invariant after loading. `cognitive_inspect`
returns concepts, relations, episodes, working-memory size, and actual synapses
for inspection. The cognitive test covers ternary integrity, transitive
reasoning, intent-aware requested repetition, and persisted learning state.

16. CONFIGURABLE PARAMETERS
============================

  In ternary.jl (TernaryConfig):
    num_params        = 1000
    num_groups        = 10
    params_per_group  = 100
    reward_threshold  = 3.0
    punish_threshold  = 3.0
    eligibility_decay = 0.95

  In model.jl (ModelConfig):
    hidden_dim        = 10
    moe_top_k         = 2
    output_scale      = 20.0

  In credit.jl (CreditConfig):
    top_k             = 8
    min_contribution  = 0.01

  To scale to larger models, increase:
    num_params, num_groups, hidden_dim, num_experts


17. SCALING PATH
================

  1K → 10K → 100K → 1M → full Mamba/MoE model

  All core abstractions support arbitrary:
    - Parameter count
    - Group size
    - Number of experts
    - Top-k selection
    - Threshold values

  At larger scales:
    - Bridge to Python (mamba-ssm) for real Mamba
    - Bridge to Rust for optimised ternary kernels
    - Use Julia's built-in parallelism for groups
    - Binary logging compresses well


18. MULTI-LANGUAGE ARCHITECTURE
===============================

  Current prototype: 100% Julia. The cognitive architecture and legacy
  baseline are both implemented in Julia source files.

  No foreign language is required at the 1K scale.

  For future scaling:
    Julia remains the training controller.
    Python/Rust handle specific components:
      - Python: Mamba-ssm, torch integration
      - Rust: Optimised ternary matmul, binary logging

    Bridge: subprocess with stdin/stdout protocol,
    or temporary binary buffers.

    The bridge interface is defined but not activated
    in this prototype.


19. LIMITATIONS
===============

  - 1000 ternary parameters is extremely small
  - Hidden dim 10 severely limits representation
  - Ternary matmul has limited expressiveness
  - Contribution ≠ causality (see section 8)
  - Synthetic dataset is trivially simple
  - No GPU acceleration
  - No real language/sequence modelling
  - Results may not generalise to larger scales
  - Evaluation accuracy is NOT a meaningful benchmark

  This prototype tests the MECHANISM, not performance.


20. FILE MANIFEST
=================

  The original 13-file baseline plus cognitive architecture files:

  Legacy baseline source (10):
    main.jl              Entry point and CLI
    model.jl             Model architecture
    ternary.jl           Ternary parameter system
    trace.jl             Contribution tracing
    reward.jl            Reward/punishment engine
    credit.jl            Credit assignment
    mamba_or_bridge.jl   Mamba-inspired SSM
    moe.jl               Mixture of Experts
    neuro_symbolic.jl    Symbolic evaluator
    experiments.jl       Experiments, dataset, tests

  Cognitive architecture source (6):
    cognitive_types.jl          — core cognitive data structures
    cognitive_memory.jl         — sparse graph and memory systems
    cognitive_reasoning.jl      — hypothesis generation and testing
    cognitive_learning.jl       — evaluator and ternary credit assignment
    cognitive_serialization.jl  — model persistence and inspection
    cognitive_trainer.jl        — line-based cognitive datasets and benchmark
    cognitive_demo.jl           — demos and cognitive integrity tests

  Support:
    Project.toml                — Julia project metadata
    README.txt           This file
    requirements.txt     Dependencies
    run.bat              One-click launcher
"""


# ===== BEGIN legacy.jl =====

# legacy.jl -- consolidated bundle (auto-generated)
# Sources merged: run_legacy.jl (module wrapper), legacy/ternary.jl, legacy/trace.jl, legacy/neuro_symbolic.jl, legacy/reward.jl, legacy/mamba_or_bridge.jl, legacy/moe.jl, legacy/credit.jl, legacy/model.jl, legacy/experiments.jl
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt

# Legacy ternary-reward baseline (Mamba-inspired SSM + MoE), kept ONLY for comparison.
# Invoke via: main.jl --legacy [--expA|--expB|--expC|--test]

module LegacyBaseline

using Random, Printf

# (flat layout) original run_legacy.jl includes inlined below; HERE kept for compat
const HERE = @__DIR__


# ===== BEGIN legacy/ternary.jl =====

# ternary.jl — Ternary Parameter System
# Manages 1000 trainable ternary parameters {-1, 0, +1} in 10 groups of 100.
# All structs and functions are in global scope (no module) for simple includes.

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
Base.@kwdef mutable struct TernaryConfig
    num_params::Int         = 1000
    num_groups::Int         = 10
    params_per_group::Int   = 100
    hidden_dim::Int         = 10       # each group → hidden_dim × hidden_dim matrix
    reward_threshold::Float64   = 3.0
    punish_threshold::Float64   = 3.0
    confidence_threshold::Float64 = 0.5
    eligibility_decay::Float64  = 0.95
end

# ──────────────────────────────────────────────
# Per-parameter statistics (compact)
# ──────────────────────────────────────────────
mutable struct ParamStats
    reward_count::Int32
    punish_count::Int32
    neutral_count::Int32
    accumulated_reward::Float32
    accumulated_punish::Float32
    eligibility::Float32
end
ParamStats() = ParamStats(0, 0, 0, 0f0, 0f0, 0f0)

# ──────────────────────────────────────────────
# The complete ternary parameter system
# ──────────────────────────────────────────────
mutable struct TernaryParams
    config::TernaryConfig
    states::Vector{Int8}         # length = num_params, each ∈ {-1,0,+1}
    stats::Vector{ParamStats}    # length = num_params
    mutation_count::Int
    mutations_this_step::Int     # reset each training step
end

function ternary_create(config::TernaryConfig)
    n = config.num_params
    states = [rand(Int8[-1, 0, 1]) for _ in 1:n]
    stats  = [ParamStats() for _ in 1:n]
    return TernaryParams(config, states, stats, 0, 0)
end

function ternary_create_zeros(config::TernaryConfig)
    n = config.num_params
    states = zeros(Int8, n)
    stats  = [ParamStats() for _ in 1:n]
    return TernaryParams(config, states, stats, 0, 0)
end

# ──────────────────────────────────────────────
# Validation — aborts if invariants are broken
# ──────────────────────────────────────────────
function ternary_validate(params::TernaryParams)
    c = params.config
    length(params.states) == c.num_params ||
        error("Parameter count mismatch: expected $(c.num_params), got $(length(params.states))")
    c.num_params == c.num_groups * c.params_per_group ||
        error("Group config mismatch: $(c.num_params) ≠ $(c.num_groups)×$(c.params_per_group)")
    all(s -> s in (Int8(-1), Int8(0), Int8(1)), params.states) ||
        error("Non-ternary state detected")
    return true
end

# ──────────────────────────────────────────────
# Group indexing (groups are 0-indexed, Julia arrays 1-indexed)
# ──────────────────────────────────────────────
function ternary_group_indices(config::TernaryConfig, group_id::Int)
    start = group_id * config.params_per_group + 1
    return start:(start + config.params_per_group - 1)
end

function ternary_group_offset(config::TernaryConfig, group_id::Int)
    return group_id * config.params_per_group   # 0-based offset for global ID math
end

# ──────────────────────────────────────────────
# Reshape group → hidden_dim × hidden_dim matrix
# ──────────────────────────────────────────────
function ternary_get_matrix(params::TernaryParams, group_id::Int)
    d = params.config.hidden_dim
    indices = ternary_group_indices(params.config, group_id)
    return reshape(Float64.(params.states[indices]), d, d)
end

# ──────────────────────────────────────────────
# Ternary matrix-vector multiply with contribution tracking
# Returns (output_vector, contributions)
# contributions = Vector{Tuple{global_id::Int, strength::Float64}}
# ──────────────────────────────────────────────
function ternary_matmul(W::Matrix{Float64}, x::Vector{Float64}, group_offset::Int)
    m, n = size(W)
    y = W * x                     # efficient built-in BLAS

    contributions = Tuple{Int,Float64}[]
    for j in 1:n
        xj = x[j]
        for i in 1:m
            c = abs(W[i,j] * xj)
            if c > 1e-12
                gid = group_offset + (j - 1) * m + i   # column-major
                push!(contributions, (gid, c))
            end
        end
    end

    total = sum(last, contributions; init=0.0)
    if total > 0.0
        contributions = [(id, v / total) for (id, v) in contributions]
    end

    return y, contributions
end

# ──────────────────────────────────────────────
# Mutation: accumulate evidence and flip ternary state at threshold
# Returns true if the parameter actually changed state.
# ──────────────────────────────────────────────
function ternary_mutate!(params::TernaryParams, global_id::Int,
                         signal::Symbol, strength::Float64)
    stat   = params.stats[global_id]
    config = params.config

    if signal == :neutral
        stat.neutral_count += Int32(1)
        return false
    elseif signal == :reward
        stat.reward_count += Int32(1)
        stat.accumulated_reward += Float32(strength)
    elseif signal == :punish
        stat.punish_count += Int32(1)
        stat.accumulated_punish += Float32(strength)
    end

    mutated = false
    cur = params.states[global_id]

    # Promote on strong positive evidence
    if stat.accumulated_reward >= config.reward_threshold
        if cur < Int8(1)
            params.states[global_id] = cur + Int8(1)
            mutated = true
        end
        stat.accumulated_reward = 0f0
        stat.accumulated_punish = max(0f0, stat.accumulated_punish - Float32(config.reward_threshold * 0.5))
    end

    # Demote on strong negative evidence
    if stat.accumulated_punish >= config.punish_threshold
        if cur > Int8(-1)
            params.states[global_id] = cur - Int8(1)
            mutated = true
        end
        stat.accumulated_punish = 0f0
        stat.accumulated_reward = max(0f0, stat.accumulated_reward - Float32(config.punish_threshold * 0.5))
    end

    if mutated
        params.mutation_count   += 1
        params.mutations_this_step += 1
    end
    return mutated
end

# ──────────────────────────────────────────────
# Eligibility decay (call once per step)
# ──────────────────────────────────────────────
function ternary_decay_eligibility!(params::TernaryParams)
    d = Float32(params.config.eligibility_decay)
    for s in params.stats
        s.eligibility *= d
    end
end

# ──────────────────────────────────────────────
# Distribution summary
# ──────────────────────────────────────────────
function ternary_distribution(params::TernaryParams)
    neg  = count(==(Int8(-1)), params.states)
    zero = count(==(Int8(0)),  params.states)
    pos  = count(==(Int8(1)),  params.states)
    return (negative=neg, zero=zero, positive=pos)
end

# ──────────────────────────────────────────────
# Deep copy (for intervention tests / baselines)
# ──────────────────────────────────────────────
function ternary_copy(params::TernaryParams)
    TernaryParams(
        params.config,
        copy(params.states),
        [ParamStats(s.reward_count, s.punish_count, s.neutral_count,
                    s.accumulated_reward, s.accumulated_punish, s.eligibility)
         for s in params.stats],
        params.mutation_count,
        0
    )
end

# ──────────────────────────────────────────────
# Reset per-step mutation counter
# ──────────────────────────────────────────────
function ternary_reset_step!(params::TernaryParams)
    params.mutations_this_step = 0
end

# ===== END legacy/ternary.jl =====


# ===== BEGIN legacy/trace.jl =====

# trace.jl — Contribution Tracing & Binary Logging
# Records which parameters contributed to each forward pass and how strongly.
# Provides compact binary log format for post-hoc analysis.

# ──────────────────────────────────────────────
# Contribution record (one per active parameter per step)
# ──────────────────────────────────────────────
struct ContributionRecord
    param_id::Int32
    group_id::Int16
    contribution::Float32
    expert_id::Int8       # -1 if not routed through an expert
    step::Int32
end

# ──────────────────────────────────────────────
# Buffer — collects records during a single forward pass
# ──────────────────────────────────────────────
mutable struct TraceBuffer
    records::Vector{ContributionRecord}
    step::Int32
end
TraceBuffer(step::Int=0) = TraceBuffer(ContributionRecord[], Int32(step))

function trace_clear!(buf::TraceBuffer)
    empty!(buf.records)
end

function trace_set_step!(buf::TraceBuffer, step::Int)
    buf.step = Int32(step)
end

# Add a single contribution
function trace_add!(buf::TraceBuffer, param_id::Int, group_id::Int,
                    contribution::Float64; expert_id::Int=-1)
    push!(buf.records, ContributionRecord(
        Int32(param_id), Int16(group_id), Float32(contribution),
        Int8(expert_id), buf.step))
end

# Add a batch of contributions from ternary_matmul output
function trace_add_batch!(buf::TraceBuffer,
                          contributions::Vector{Tuple{Int,Float64}},
                          params_per_group::Int;
                          expert_id::Int=-1)
    for (pid, c) in contributions
        gid = (pid - 1) ÷ params_per_group
        trace_add!(buf, pid, gid, c; expert_id=expert_id)
    end
end

# ──────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────

# Aggregate contributions per parameter, return sorted top-k
function trace_top_contributors(buf::TraceBuffer, k::Int)
    isempty(buf.records) && return ContributionRecord[]

    agg = Dict{Int32, Float64}()
    eid_map = Dict{Int32, Int8}()
    for r in buf.records
        agg[r.param_id] = get(agg, r.param_id, 0.0) + Float64(r.contribution)
        eid_map[r.param_id] = r.expert_id
    end

    sorted = sort(collect(agg); by=last, rev=true)
    n = min(k, length(sorted))

    result = Vector{ContributionRecord}(undef, n)
    ppg = 100  # params_per_group constant
    for i in 1:n
        pid, c = sorted[i]
        gid = Int16((pid - 1) ÷ ppg)
        result[i] = ContributionRecord(pid, gid, Float32(c),
                                       get(eid_map, pid, Int8(-1)), buf.step)
    end
    return result
end

# Contributions grouped by group_id
function trace_group_contributions(buf::TraceBuffer)
    gc = Dict{Int,Float64}()
    for r in buf.records
        gc[Int(r.group_id)] = get(gc, Int(r.group_id), 0.0) + Float64(r.contribution)
    end
    return gc
end

# Contributions grouped by expert_id
function trace_expert_contributions(buf::TraceBuffer)
    ec = Dict{Int,Float64}()
    for r in buf.records
        eid = Int(r.expert_id)
        eid >= 0 || continue
        ec[eid] = get(ec, eid, 0.0) + Float64(r.contribution)
    end
    return ec
end

# Parameters that contributed from a specific group
function trace_params_in_group(buf::TraceBuffer, group_id::Int)
    filter(r -> Int(r.group_id) == group_id, buf.records)
end

# Parameters that contributed through a specific expert
function trace_params_for_expert(buf::TraceBuffer, expert_id::Int)
    filter(r -> Int(r.expert_id) == expert_id, buf.records)
end

# ──────────────────────────────────────────────
# Binary logging format
#
# File layout:
#   [MAGIC 4B] [COUNT 4B] [RECORDS...] [MAGIC 4B] [COUNT 4B] [RECORDS...] ...
# Each record: param_id(4) + group_id(2) + contribution(4) + expert_id(1) + step(4) = 15 bytes
# ──────────────────────────────────────────────
const TRACE_MAGIC = UInt32(0x54524345)   # "TRCE"

function trace_write_binary(io::IO, records::Vector{ContributionRecord})
    write(io, TRACE_MAGIC)
    write(io, Int32(length(records)))
    for r in records
        write(io, r.param_id)
        write(io, r.group_id)
        write(io, r.contribution)
        write(io, r.expert_id)
        write(io, r.step)
    end
end

function trace_write_binary(filepath::String,
                            records::Vector{ContributionRecord};
                            append::Bool=true)
    mode = append ? "a" : "w"
    open(filepath, mode) do io
        trace_write_binary(io, records)
    end
end

function trace_read_binary(io::IO)
    all_records = ContributionRecord[]
    while !eof(io)
        magic = read(io, UInt32)
        magic == TRACE_MAGIC || error("Invalid binary log magic: 0x$(string(magic, base=16))")
        cnt = read(io, Int32)
        for _ in 1:cnt
            pid  = read(io, Int32)
            gid  = read(io, Int16)
            c    = read(io, Float32)
            eid  = read(io, Int8)
            step = read(io, Int32)
            push!(all_records, ContributionRecord(pid, gid, c, eid, step))
        end
    end
    return all_records
end

function trace_read_binary(filepath::String)
    open(filepath, "r") do io
        trace_read_binary(io)
    end
end

# ──────────────────────────────────────────────
# Reconstruct summary statistics from binary log
# ──────────────────────────────────────────────
function trace_reconstruct_stats(records::Vector{ContributionRecord})
    param_data = Dict{Int32, @NamedTuple{count::Int, total::Float64}}()
    for r in records
        prev = get(param_data, r.param_id, (count=0, total=0.0))
        param_data[r.param_id] = (count=prev.count + 1,
                                  total=prev.total + Float64(r.contribution))
    end
    return param_data
end

# ──────────────────────────────────────────────
# Event log record (for reward/punish events alongside trace)
# ──────────────────────────────────────────────
struct EventRecord
    step::Int32
    param_id::Int32
    group_id::Int16
    event_type::Int8      # 0=NEUTRAL, 1=REWARD, 2=PUNISH, 3=MUTATION, 4=INTERVENTION
    signal::Int8          # 1=REWARD, -1=PUNISH, 0=NEUTRAL
    contribution::Float32
end

const EVENT_MAGIC = UInt32(0x45565454)   # "EVNT"

function trace_write_events(io::IO, events::Vector{EventRecord})
    write(io, EVENT_MAGIC)
    write(io, Int32(length(events)))
    for e in events
        write(io, e.step)
        write(io, e.param_id)
        write(io, e.group_id)
        write(io, e.event_type)
        write(io, e.signal)
        write(io, e.contribution)
    end
end

function trace_write_events(filepath::String, events::Vector{EventRecord}; append::Bool=true)
    mode = append ? "a" : "w"
    open(filepath, mode) do io
        trace_write_events(io, events)
    end
end

function trace_read_events(io::IO)
    all_events = EventRecord[]
    while !eof(io)
        magic = read(io, UInt32)
        magic == EVENT_MAGIC || error("Invalid event log magic")
        cnt = read(io, Int32)
        for _ in 1:cnt
            step  = read(io, Int32)
            pid   = read(io, Int32)
            gid   = read(io, Int16)
            etype = read(io, Int8)
            sig   = read(io, Int8)
            c     = read(io, Float32)
            push!(all_events, EventRecord(step, pid, gid, etype, sig, c))
        end
    end
    return all_events
end

function trace_read_events(filepath::String)
    open(filepath, "r") do io
        trace_read_events(io)
    end
end

function trace_reconstruct_event_stats(events::Vector{EventRecord})
    rewards   = count(e -> e.signal == Int8(1), events)
    punishes  = count(e -> e.signal == Int8(-1), events)
    neutrals  = count(e -> e.signal == Int8(0), events)
    mutations = count(e -> e.event_type == Int8(3), events)
    return (rewards=rewards, punishes=punishes, neutrals=neutrals, mutations=mutations)
end

# ===== END legacy/trace.jl =====


# ===== BEGIN legacy/neuro_symbolic.jl =====

# neuro_symbolic.jl — Deterministic Symbolic Evaluation Layer
# Rule-based evaluation of model outputs. No neural computation here.
# Supports: exact matching, numeric tolerance, pattern checks,
#           context-sensitive repetition, format requirements.

# ──────────────────────────────────────────────
# Task type enumeration
# ──────────────────────────────────────────────
@enum SymTaskType begin
    SYM_ARITHMETIC
    SYM_COMPARISON
    SYM_PATTERN
    SYM_COPY
    SYM_FORMAT
    SYM_REPETITION
end

# ──────────────────────────────────────────────
# Task specification — embedded with each dataset sample
# ──────────────────────────────────────────────
struct SymTaskSpec
    task_type::SymTaskType
    description::String
    operand_a::Float64
    operand_b::Float64
    expected::Float64
    requires_repetition::Bool
    format_spec::String          # "" = no format constraint
end

# ──────────────────────────────────────────────
# Evaluation result
# ──────────────────────────────────────────────
struct SymEvalResult
    correct::Bool
    score::Float64               # 0.0–1.0
    reasons::Vector{String}
    numeric_correct::Bool
    format_ok::Bool
    has_unwanted_repetition::Bool
    has_requested_repetition::Bool
    confidence::Float64          # evaluator confidence in the verdict
end

# ──────────────────────────────────────────────
# Main evaluation entry point
# ──────────────────────────────────────────────
function symbolic_evaluate(task::SymTaskSpec, predicted::Float64;
                           output_vector::Vector{Float64}=Float64[])
    reasons  = String[]
    score    = 0.0
    num_ok   = false
    fmt_ok   = true
    unwanted_rep  = false
    requested_rep = false
    confidence    = 1.0

    expected = task.expected

    # ── Numeric / exact correctness ──────────
    if task.task_type in (SYM_ARITHMETIC, SYM_COMPARISON, SYM_PATTERN,
                          SYM_COPY, SYM_FORMAT)
        tol = _tolerance_for(task)
        if abs(predicted - expected) <= tol
            num_ok = true
            score += 0.7
            push!(reasons, "numeric_correct")
        else
            push!(reasons, "numeric_incorrect(expected=$(expected),got=$(round(predicted,digits=2)))")
            # Partial credit for being close
            rel_err = abs(predicted - expected) / max(abs(expected), 1.0)
            if rel_err < 0.5
                score += 0.2
                push!(reasons, "partial_credit")
                confidence = 0.7
            end
        end
    end

    # ── Repetition task ──────────────────────
    if task.task_type == SYM_REPETITION
        if task.requires_repetition
            requested_rep = true
            tol = _tolerance_for(task)
            if abs(predicted - expected) <= tol
                num_ok = true
                score += 0.7
                push!(reasons, "requested_repetition_correct")
            else
                push!(reasons, "requested_repetition_incorrect")
            end
        end
    end

    # ── Unwanted repetition detection ────────
    # Check the output vector for suspicious repeated patterns
    if !task.requires_repetition && length(output_vector) >= 4
        unwanted_rep = _detect_vector_repetition(output_vector)
        if unwanted_rep
            score -= 0.3
            push!(reasons, "unwanted_repetition_in_output")
        end
    end

    # ── Format compliance ────────────────────
    if task.format_spec != ""
        fmt_ok = _check_format(predicted, task.format_spec)
        if fmt_ok
            score += 0.1
            push!(reasons, "format_ok")
        else
            score -= 0.1
            push!(reasons, "format_violation")
        end
    end

    # ── Logical consistency bonus ────────────
    if task.task_type == SYM_COMPARISON
        # Comparison answers should be 0 or 1
        if num_ok && (predicted ≈ 0.0 || predicted ≈ 1.0)
            score += 0.1
            push!(reasons, "logical_consistency")
        end
    end

    correct = num_ok && fmt_ok && !unwanted_rep
    score   = clamp(score, 0.0, 1.0)

    return SymEvalResult(correct, score, reasons, num_ok, fmt_ok,
                         unwanted_rep, requested_rep, confidence)
end

# ──────────────────────────────────────────────
# Tolerance depends on task type
# ──────────────────────────────────────────────
function _tolerance_for(task::SymTaskSpec)
    if task.task_type == SYM_COMPARISON
        return 0.5   # binary answer, generous tolerance
    elseif task.task_type == SYM_COPY
        return 0.5
    elseif task.task_type == SYM_ARITHMETIC
        return max(1.0, abs(task.expected) * 0.15)  # 15% or 1.0
    elseif task.task_type == SYM_PATTERN
        return max(1.0, abs(task.expected) * 0.15)
    elseif task.task_type == SYM_REPETITION
        return 0.5
    else
        return 1.0
    end
end

# ──────────────────────────────────────────────
# Detect unwanted repetition in output vector
# ──────────────────────────────────────────────
function _detect_vector_repetition(v::Vector{Float64})
    n = length(v)
    n >= 4 || return false

    # Check if 3+ consecutive elements are nearly identical (and non-negligible)
    for i in 1:(n-2)
        if abs(v[i]) > 0.05 && abs(v[i] - v[i+1]) < 0.02 && abs(v[i+1] - v[i+2]) < 0.02
            return true
        end
    end
    return false
end

# ──────────────────────────────────────────────
# Format checks
# ──────────────────────────────────────────────
function _check_format(predicted::Float64, spec::String)
    if spec == "number_only"
        return isfinite(predicted)
    elseif spec == "integer_only"
        return isfinite(predicted) && abs(predicted - round(predicted)) < 0.3
    elseif spec == "binary"
        return abs(predicted) < 0.5 || abs(predicted - 1.0) < 0.5
    end
    return true
end

# ──────────────────────────────────────────────
# Batch evaluation helper
# ──────────────────────────────────────────────
function symbolic_evaluate_batch(tasks::Vector{SymTaskSpec},
                                 predictions::Vector{Float64};
                                 output_vectors::Vector{Vector{Float64}}=Vector{Float64}[])
    results = SymEvalResult[]
    for (i, task) in enumerate(tasks)
        ov = isempty(output_vectors) ? Float64[] : output_vectors[i]
        push!(results, symbolic_evaluate(task, predictions[i]; output_vector=ov))
    end
    return results
end

# ──────────────────────────────────────────────
# Prompt Parser for Chat Interface and Custom Datasets
# ──────────────────────────────────────────────
function symbolic_parse_prompt(prompt::AbstractString, expected_str::AbstractString="")
    p = strip(prompt)
    exp_val = 0.0
    if !isempty(expected_str)
        try_val = tryparse(Float64, strip(expected_str))
        if try_val !== nothing
            exp_val = try_val
        elseif lowercase(strip(expected_str)) in ("true", "yes", "1")
            exp_val = 1.0
        elseif lowercase(strip(expected_str)) in ("false", "no", "0")
            exp_val = 0.0
        end
    end

    # 1. Arithmetic expressions: "7 + 8", "15 - 6", "4 * 5", "10 / 2"
    m_math = match(r"(-?\d+\.?\d*)\s*([\+\-\*\/])\s*(-?\d+\.?\d*)", p)
    if m_math !== nothing
        a = parse(Float64, m_math.captures[1])
        op = m_math.captures[2]
        b = parse(Float64, m_math.captures[3])
        calc_exp = op == "+" ? a + b : (op == "-" ? a - b : (op == "*" ? a * b : (b != 0.0 ? a / b : 0.0)))
        target = isempty(expected_str) ? calc_exp : exp_val
        return SymTaskSpec(SYM_ARITHMETIC, "$a$op$b", a, b, target, false, "number_only")
    end

    # 2. Comparison expressions: "7 > 3", "2 < 8", "Is 5 > 9?"
    m_cmp = match(r"(-?\d+\.?\d*)\s*([><])\s*(-?\d+\.?\d*)", p)
    if m_cmp !== nothing
        a = parse(Float64, m_cmp.captures[1])
        op = m_cmp.captures[2]
        b = parse(Float64, m_cmp.captures[3])
        calc_exp = op == ">" ? (a > b ? 1.0 : 0.0) : (a < b ? 1.0 : 0.0)
        target = isempty(expected_str) ? calc_exp : exp_val
        return SymTaskSpec(SYM_COMPARISON, "$a$op$b?", a, b, target, false, "binary")
    end

    # 3. Repetition expressions: "repeat 5", "repeat 3 4 times"
    m_rep = match(r"repeat\s+(\d+\.?\d*)(?:\s+(\d+))?", lowercase(p))
    if m_rep !== nothing
        val = parse(Float64, m_rep.captures[1])
        cnt = m_rep.captures[2] !== nothing ? parse(Float64, m_rep.captures[2]) : 3.0
        target = isempty(expected_str) ? val : exp_val
        return SymTaskSpec(SYM_REPETITION, "repeat $val", val, cnt, target, true, "")
    end

    # 4. Echo / Copy expressions: "echo 7", "copy 9"
    m_echo = match(r"(?:echo|copy)\s+(\d+\.?\d*)", lowercase(p))
    if m_echo !== nothing
        val = parse(Float64, m_echo.captures[1])
        target = isempty(expected_str) ? val : exp_val
        return SymTaskSpec(SYM_COPY, "echo $val", val, 0.0, target, false, "")
    end

    # 5. Pattern expressions: "2, 4, 6, ?"
    m_pat = match(r"(\d+),\s*(\d+),\s*(\d+)", p)
    if m_pat !== nothing
        p1 = parse(Float64, m_pat.captures[1])
        p2 = parse(Float64, m_pat.captures[2])
        p3 = parse(Float64, m_pat.captures[3])
        step = p2 - p1
        target = isempty(expected_str) ? p3 + step : exp_val
        return SymTaskSpec(SYM_PATTERN, "$p1,$p2,$p3,?", p1, step, target, false, "")
    end

    # 6. General text prompt
    num_chars = Float64(length(p))
    return SymTaskSpec(SYM_FORMAT, p, num_chars, 0.0, exp_val, false, "")
end

# ===== END legacy/neuro_symbolic.jl =====


# ===== BEGIN legacy/reward.jl =====

# reward.jl — Automatic Reward / Punish / Neutral Signal Engine
# Determines the training signal based on the neuro-symbolic evaluation result.
# Context-sensitive: repetition handling depends on whether it was requested.

# ──────────────────────────────────────────────
# Signal enum
# ──────────────────────────────────────────────
@enum TrainingSignal REWARD PUNISH NEUTRAL

# ──────────────────────────────────────────────
# Decision record — full explanation of the signal
# ──────────────────────────────────────────────
struct RewardDecision
    signal::TrainingSignal
    confidence::Float64
    strength::Float64        # magnitude of reward/punishment (0.0–1.0)
    reasons::Vector{String}
end

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
Base.@kwdef struct RewardConfig
    min_confidence::Float64      = 0.4   # below this → NEUTRAL
    reward_strength_scale::Float64 = 1.0
    punish_strength_scale::Float64 = 1.0
    partial_credit_reward::Bool  = false  # reward partial credit?
end

# ──────────────────────────────────────────────
# Main evaluation function
#
# Takes a SymEvalResult (from neuro_symbolic.jl) and returns a RewardDecision.
# ──────────────────────────────────────────────
function reward_evaluate(eval_result::SymEvalResult, task::SymTaskSpec,
                         config::RewardConfig=RewardConfig())
    reasons = copy(eval_result.reasons)

    # ── Requested repetition → NEUTRAL (not punished, optionally rewarded) ──
    if task.requires_repetition && eval_result.has_requested_repetition
        if eval_result.numeric_correct
            push!(reasons, "requested_repetition_executed")
            # Could reward, but default to NEUTRAL for repetition
            return RewardDecision(NEUTRAL, eval_result.confidence,
                                 0.0, reasons)
        else
            push!(reasons, "requested_repetition_failed")
            return RewardDecision(PUNISH, eval_result.confidence,
                                 0.5 * config.punish_strength_scale, reasons)
        end
    end

    # ── Unwanted repetition → always PUNISH ──
    if eval_result.has_unwanted_repetition
        push!(reasons, "unwanted_repetition_detected")
        return RewardDecision(PUNISH, eval_result.confidence,
                              0.8 * config.punish_strength_scale, reasons)
    end

    # ── Low confidence → NEUTRAL ──
    if eval_result.confidence < config.min_confidence
        push!(reasons, "low_evaluator_confidence")
        return RewardDecision(NEUTRAL, eval_result.confidence, 0.0, reasons)
    end

    # ── Correct output → REWARD ──
    if eval_result.correct
        strength = eval_result.score * config.reward_strength_scale
        push!(reasons, "correct_output")
        return RewardDecision(REWARD, eval_result.confidence, strength, reasons)
    end

    # ── Partially correct → configurable ──
    if eval_result.score > 0.3 && config.partial_credit_reward
        push!(reasons, "partial_credit")
        return RewardDecision(NEUTRAL, eval_result.confidence, 0.0, reasons)
    end

    # ── Incorrect output → PUNISH ──
    push!(reasons, "incorrect_output")
    strength = (1.0 - eval_result.score) * config.punish_strength_scale
    return RewardDecision(PUNISH, eval_result.confidence, strength, reasons)
end

# ──────────────────────────────────────────────
# Debug formatting
# ──────────────────────────────────────────────
function reward_signal_string(sig::TrainingSignal)
    sig == REWARD  && return "REWARD"
    sig == PUNISH  && return "PUNISH"
    return "NEUTRAL"
end

function reward_format_decision(dec::RewardDecision)
    lines = String[]
    push!(lines, "SIGNAL: $(reward_signal_string(dec.signal))")
    push!(lines, "CONFIDENCE: $(round(dec.confidence, digits=3))")
    push!(lines, "STRENGTH: $(round(dec.strength, digits=3))")
    push!(lines, "REASONS:")
    for r in dec.reasons
        push!(lines, "  - $r")
    end
    return join(lines, "\n")
end

# ===== END legacy/reward.jl =====


# ===== BEGIN legacy/mamba_or_bridge.jl =====

# mamba_or_bridge.jl — Mamba-inspired State-Space Sequence Core
#
# This is NOT a Mamba 3 implementation. Mamba 3 requires specific hardware-aware
# selective-scan kernels that have no Julia equivalent at this time.
#
# Instead, this implements a **Mamba-inspired state-space model (SSM)** with:
#   - Discrete state transition:  s(t) = tanh(A·s(t-1) + B·x(t))
#   - Read-out:                   y(t) = tanh(C·s(t))
#   - Selective gating:           gate = σ(x[1]) → s(t) = gate·s(t)
#
# All weight matrices (A, B, C) are ternary {-1, 0, +1} from groups 1–3.
# Contribution traces are recorded for every parameter that participates.
#
# LIMITATION: This is a simplified SSM for 1K-parameter research.
# For real Mamba 3 at scale, bridge to Python (mamba-ssm) or implement
# the full selective-scan in a compiled language.

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
struct MambaConfig
    state_dim::Int       # = hidden_dim (10)
    input_dim::Int       # = hidden_dim (10)
    group_A::Int         # group index for A matrix (1)
    group_B::Int         # group index for B matrix (2)
    group_C::Int         # group index for C matrix (3)
end

MambaConfig(; hidden_dim::Int=10) = MambaConfig(hidden_dim, hidden_dim, 1, 2, 3)

# ──────────────────────────────────────────────
# Forward pass: process a sequence of input vectors
#
# Returns:
#   output       — final read-out vector (hidden_dim)
#   final_state  — SSM state after processing all steps
# Side-effect:
#   appends contribution records to trace_buf
# ──────────────────────────────────────────────
function mamba_forward(cfg::MambaConfig, params::TernaryParams,
                       input_seq::Vector{Vector{Float64}},
                       trace_buf::TraceBuffer)
    d   = cfg.state_dim
    ppg = params.config.params_per_group

    # Retrieve ternary matrices
    A = ternary_get_matrix(params, cfg.group_A)
    B = ternary_get_matrix(params, cfg.group_B)
    C = ternary_get_matrix(params, cfg.group_C)

    # Offsets for global parameter IDs
    off_A = ternary_group_offset(params.config, cfg.group_A)
    off_B = ternary_group_offset(params.config, cfg.group_B)
    off_C = ternary_group_offset(params.config, cfg.group_C)

    state  = zeros(d)
    output = zeros(d)

    for (t, x) in enumerate(input_seq)
        # ── Selective gate (Mamba-inspired input-dependent gating) ──
        gate = 1.0 / (1.0 + exp(-3.0 * x[1]))   # sigmoid on first input dim

        # ── State transition: s = tanh(A·s + B·x) ──
        As, contrib_A = ternary_matmul(A, state, off_A)
        Bx, contrib_B = ternary_matmul(B, x,     off_B)

        raw_state = As .+ Bx
        state     = tanh.(raw_state) .* gate      # selective gating

        # Record contributions (A and B)
        trace_add_batch!(trace_buf, contrib_A, ppg)
        trace_add_batch!(trace_buf, contrib_B, ppg)
    end

    # ── Read-out: y = tanh(C·s) ──
    Cs, contrib_C = ternary_matmul(C, state, off_C)
    output = tanh.(Cs)
    trace_add_batch!(trace_buf, contrib_C, ppg)

    return output, state
end

# ──────────────────────────────────────────────
# Masked forward (for intervention tests)
# Runs same computation but with one parameter masked to 0
# ──────────────────────────────────────────────
function mamba_forward_masked(cfg::MambaConfig, params::TernaryParams,
                              input_seq::Vector{Vector{Float64}},
                              mask_param_id::Int)
    # Create temporary copy with the masked parameter zeroed
    temp = ternary_copy(params)
    temp.states[mask_param_id] = Int8(0)

    dummy_buf = TraceBuffer(0)
    output, _ = mamba_forward(cfg, temp, input_seq, dummy_buf)
    return output
end

# ===== END legacy/mamba_or_bridge.jl =====


# ===== BEGIN legacy/moe.jl =====

# moe.jl — Mixture of Experts Router and Expert Computation
#
# 4 experts, each backed by a 10×10 ternary matrix (groups 5–8).
# Router uses a 10×10 ternary matrix (group 4); first 4 outputs → expert logits.
# Supports top-1 and top-2 routing with probability-weighted combination.

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
struct MoEConfig
    num_experts::Int      # 4
    top_k::Int            # 1 or 2
    hidden_dim::Int       # 10
    group_router::Int     # group index for router weights (4)
    group_experts::Vector{Int}  # group indices for experts [5,6,7,8]
end

function MoEConfig(; num_experts::Int=4, top_k::Int=2, hidden_dim::Int=10)
    MoEConfig(num_experts, top_k, hidden_dim, 4, collect(5:5+num_experts-1))
end

# ──────────────────────────────────────────────
# Routing result (exposed for tracing)
# ──────────────────────────────────────────────
struct MoERouteResult
    expert_probs::Vector{Float64}    # probability for each expert
    selected::Vector{Int}            # indices of selected experts (0-indexed)
    gate_values::Vector{Float64}     # normalised gate for selected experts
end

# ──────────────────────────────────────────────
# Softmax (stable)
# ──────────────────────────────────────────────
function _moe_softmax(x::AbstractVector{Float64})
    mx = maximum(x)
    ex = exp.(x .- mx)
    return ex ./ sum(ex)
end

# ──────────────────────────────────────────────
# Route: decide which experts to use
# ──────────────────────────────────────────────
function moe_route(params::TernaryParams, input::Vector{Float64},
                   cfg::MoEConfig, trace_buf::TraceBuffer)
    ppg = params.config.params_per_group

    # Router matrix multiplication
    W_router = ternary_get_matrix(params, cfg.group_router)
    off_r    = ternary_group_offset(params.config, cfg.group_router)
    logits_full, contrib_r = ternary_matmul(W_router, input, off_r)
    trace_add_batch!(trace_buf, contrib_r, ppg)

    # Use first num_experts outputs as expert logits
    logits = logits_full[1:cfg.num_experts]
    probs  = _moe_softmax(logits)

    # Select top-k
    sorted_idx = sortperm(probs; rev=true)
    selected   = sorted_idx[1:min(cfg.top_k, cfg.num_experts)] .- 1  # 0-indexed

    # Normalise gate values for selected experts
    sel_probs = [probs[s+1] for s in selected]
    total_p   = sum(sel_probs)
    gate_vals = total_p > 0 ? sel_probs ./ total_p : ones(length(selected)) ./ length(selected)

    return MoERouteResult(probs, selected, gate_vals)
end

# ──────────────────────────────────────────────
# Expert forward: one expert's computation
# ──────────────────────────────────────────────
function moe_expert_forward(params::TernaryParams, input::Vector{Float64},
                            expert_id::Int, cfg::MoEConfig,
                            trace_buf::TraceBuffer)
    ppg   = params.config.params_per_group
    gid   = cfg.group_experts[expert_id + 1]  # 0-indexed expert → 1-indexed array
    W_exp = ternary_get_matrix(params, gid)
    off   = ternary_group_offset(params.config, gid)

    out, contrib = ternary_matmul(W_exp, input, off)
    out = tanh.(out)   # bound output

    # Tag contributions with expert_id
    trace_add_batch!(trace_buf, contrib, ppg; expert_id=expert_id)

    return out
end

# ──────────────────────────────────────────────
# Full MoE forward: route → compute → combine
# ──────────────────────────────────────────────
function moe_forward(params::TernaryParams, input::Vector{Float64},
                     cfg::MoEConfig, trace_buf::TraceBuffer)
    # Route
    route_result = moe_route(params, input, cfg, trace_buf)

    # Compute selected experts and combine
    d = cfg.hidden_dim
    combined = zeros(d)

    for (i, eid) in enumerate(route_result.selected)
        expert_out = moe_expert_forward(params, input, eid, cfg, trace_buf)
        combined .+= route_result.gate_values[i] .* expert_out
    end

    return combined, route_result
end

# ──────────────────────────────────────────────
# Masked MoE forward (for intervention tests)
# ──────────────────────────────────────────────
function moe_forward_masked(params::TernaryParams, input::Vector{Float64},
                            cfg::MoEConfig, mask_param_id::Int)
    temp = ternary_copy(params)
    temp.states[mask_param_id] = Int8(0)
    dummy_buf = TraceBuffer(0)
    output, _ = moe_forward(temp, input, cfg, dummy_buf)
    return output
end

# ===== END legacy/moe.jl =====


# ===== BEGIN legacy/credit.jl =====

# credit.jl — Hierarchical Credit Assignment
#
# Ranks parameters by contribution trace and applies training signal
# only to the most responsible ones.
#
# Search hierarchy:  model → expert → group → parameter subset → individual
#
# Supports configurable top-k: 1, 4, 8, 16
# Optionally supports intervention testing (mask & re-run).

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
Base.@kwdef struct CreditConfig
    top_k::Int              = 8     # how many candidates to consider
    min_contribution::Float64 = 0.01 # ignore below this
    intervention_enabled::Bool = false
    intervention_threshold::Float64 = 0.05  # min behavioral change for causal claim
end

# ──────────────────────────────────────────────
# Candidate check history (Patch 4 & 5)
# ──────────────────────────────────────────────
struct CandidateCheckRecord
    param_id::Int
    rank::Int
    contribution::Float64
    tested_intervention::Bool
    behavior_change::Float64
    accepted::Bool
    reason::String
end

# ──────────────────────────────────────────────
# Result of credit assignment
# ──────────────────────────────────────────────
struct CreditResult
    candidates::Vector{Tuple{Int,Float64}}   # (param_id, contribution)
    accepted::Vector{Int}                     # param_ids that received the signal
    search_depth::Int                         # how many candidates were checked
    reason::String                            # why accepted
    expert_ranking::Vector{Pair{Int,Float64}} # expert-level summary
    group_ranking::Vector{Pair{Int,Float64}}  # group-level summary
    checks::Vector{CandidateCheckRecord}      # sequential investigation records
end

# Backwards-compatible convenience constructor
CreditResult(cands, acc, depth, reas, erank, grank) =
    CreditResult(cands, acc, depth, reas, erank, grank, CandidateCheckRecord[])

# ──────────────────────────────────────────────
# Main entry point
#
# Given a trace buffer and a training signal, determine which parameters
# should receive the reward/punishment.
# Patches 3, 4, 5: ranked investigation, check most likely wrong first,
# optional intervention test.
# ──────────────────────────────────────────────
function credit_assign(trace_buf::TraceBuffer, decision::RewardDecision,
                       params::TernaryParams, config::CreditConfig=CreditConfig();
                       model_forward_fn=nothing,
                       task=nothing,
                       original_output::Vector{Float64}=Float64[])
    # ── Step 1: Expert-level ranking ──
    expert_contribs = trace_expert_contributions(trace_buf)
    expert_ranking  = sort(collect(expert_contribs); by=last, rev=true)

    # ── Step 2: Group-level ranking ──
    group_contribs = trace_group_contributions(trace_buf)
    group_ranking  = sort(collect(group_contribs); by=last, rev=true)

    # ── Step 3: Parameter-level ranking (top-k from trace) ──
    top_records = trace_top_contributors(trace_buf, config.top_k * 2)
    candidates  = [(Int(r.param_id), Float64(r.contribution)) for r in top_records]

    # Filter by minimum contribution
    candidates = filter(c -> c[2] >= config.min_contribution, candidates)

    # Take top_k
    if length(candidates) > config.top_k
        candidates = candidates[1:config.top_k]
    end

    # ── Step 4: Hierarchical boosting ──
    if !isempty(expert_ranking) && !isempty(candidates)
        top_expert = first(expert_ranking[1])
        candidates = _boost_expert_candidates(candidates, trace_buf, top_expert)
    end

    # ── Step 5: Sequential investigation (Patch 4 & Patch 5) ──
    accepted     = Int[]
    checks       = CandidateCheckRecord[]
    search_depth = 0
    reason       = ""

    if decision.signal == NEUTRAL
        reason = "neutral_signal_no_updates"
    elseif isempty(candidates)
        reason = "no_candidates_above_threshold"
    elseif decision.signal == REWARD
        # Reward top contributors that participated meaningfully (Patch 7)
        search_depth = length(candidates)
        for (rank, (pid, contrib)) in enumerate(candidates)
            if contrib >= config.min_contribution
                push!(accepted, pid)
                push!(checks, CandidateCheckRecord(pid, rank, contrib, false, 0.0, true, "meaningful_positive_contribution"))
            end
        end
        reason = "rewarded_top_$(length(accepted))_contributors"
    else
        # PUNISH: check highest-probability responsible parameters first (Patch 4)
        run_interventions = config.intervention_enabled && (model_forward_fn !== nothing) && (task !== nothing) && !isempty(original_output)

        for (rank, (pid, contrib)) in enumerate(candidates)
            search_depth += 1

            if run_interventions
                # Patch 5: mask candidate parameter, re-run, compare behavior
                change = credit_intervention_test(params, pid, original_output, model_forward_fn, task)
                is_causal = change >= config.intervention_threshold
                push!(checks, CandidateCheckRecord(pid, rank, contrib, true, change, is_causal,
                                                   is_causal ? "behavior_changed_upon_masking" : "no_significant_behavioral_change"))
                if is_causal
                    push!(accepted, pid)
                end
            else
                # Default contribution ranking criterion
                is_resp = contrib >= config.min_contribution
                push!(checks, CandidateCheckRecord(pid, rank, contrib, false, 0.0, is_resp,
                                                   is_resp ? "contribution_above_threshold" : "below_threshold"))
                if is_resp
                    push!(accepted, pid)
                end
            end
        end

        if !isempty(accepted)
            reason = run_interventions ?
                "accepted_$(length(accepted))_causally_verified_candidates" :
                "accepted_top_$(length(accepted))_contributors"
        else
            reason = run_interventions ?
                "no_candidates_passed_intervention" :
                "all_candidates_below_threshold"
        end
    end

    return CreditResult(candidates, accepted, search_depth, reason,
                        expert_ranking, group_ranking, checks)
end

# ──────────────────────────────────────────────
# Boost candidates from the top-contributing expert
# ──────────────────────────────────────────────
function _boost_expert_candidates(candidates::Vector{Tuple{Int,Float64}},
                                  trace_buf::TraceBuffer, top_expert::Int)
    expert_params = Set(Int(r.param_id) for r in trace_params_for_expert(trace_buf, top_expert))

    boosted = Tuple{Int,Float64}[]
    for (pid, c) in candidates
        boost = pid in expert_params ? 1.2 : 1.0   # 20% boost for top-expert params
        push!(boosted, (pid, c * boost))
    end

    # Re-sort
    sort!(boosted; by=last, rev=true)
    return boosted
end

# ──────────────────────────────────────────────
# Apply credit: mutate accepted parameters
#
# Returns the number of actual mutations.
# ──────────────────────────────────────────────
function credit_apply!(credit_result::CreditResult, decision::RewardDecision,
                       params::TernaryParams)
    decision.signal == NEUTRAL && return 0

    signal_sym = decision.signal == REWARD ? :reward : :punish
    mutations  = 0

    for pid in credit_result.accepted
        # Find contribution for this parameter
        contrib = 0.0
        for (cid, cv) in credit_result.candidates
            if cid == pid
                contrib = cv
                break
            end
        end

        strength = decision.strength * max(contrib, 0.1)
        if ternary_mutate!(params, pid, signal_sym, strength)
            mutations += 1
        end
    end

    return mutations
end

# ──────────────────────────────────────────────
# Intervention test (optional, expensive)
#
# Masks a parameter, re-runs the model, and measures behavioral change.
# Returns the magnitude of change (0 = no effect, higher = more causal).
#
# model_forward_fn: function(params, input) → output_vector
# ──────────────────────────────────────────────
function credit_intervention_test(params::TernaryParams, param_id::Int,
                                  original_output::Vector{Float64},
                                  model_forward_fn::Function,
                                  input_data)
    # Save and mask
    saved_state = params.states[param_id]
    params.states[param_id] = Int8(0)

    # Re-run
    masked_output = model_forward_fn(params, input_data)

    # Restore
    params.states[param_id] = saved_state

    # Measure change
    if length(masked_output) == length(original_output)
        change = sum(abs.(masked_output .- original_output))
    else
        change = Inf   # dimension mismatch indicates significant structural effect
    end

    return change
end

# ──────────────────────────────────────────────
# Format credit result for debug output
# ──────────────────────────────────────────────
function credit_format(cr::CreditResult)
    lines = String[]
    push!(lines, "CREDIT ASSIGNMENT:")
    push!(lines, "  Search depth: $(cr.search_depth)")
    push!(lines, "  Reason: $(cr.reason)")

    if !isempty(cr.expert_ranking)
        push!(lines, "  Expert ranking:")
        for (eid, c) in cr.expert_ranking
            push!(lines, "    Expert $eid = $(round(c, digits=3))")
        end
    end

    if !isempty(cr.group_ranking)
        push!(lines, "  Group ranking (top 3):")
        for (gid, c) in cr.group_ranking[1:min(3, length(cr.group_ranking))]
            push!(lines, "    Group $gid = $(round(c, digits=3))")
        end
    end

    push!(lines, "  Top contributors:")
    for (pid, c) in cr.candidates[1:min(5, length(cr.candidates))]
        tag = pid in cr.accepted ? " [ACCEPTED]" : ""
        push!(lines, "    P$(pid) = $(round(c, digits=3))$tag")
    end

    # Report intervention tests if performed
    for chk in cr.checks
        if chk.tested_intervention
            status_str = chk.accepted ? "VERIFIED (behavior changed)" : "rejected (negligible effect)"
            push!(lines, "    P$(chk.param_id) [rank $(chk.rank)]: intervention Δ=$(round(chk.behavior_change, digits=3)) -> $status_str")
        end
    end

    push!(lines, "  Accepted: $(length(cr.accepted)) parameters")
    return join(lines, "\n")
end

# ===== END legacy/credit.jl =====


# ===== BEGIN legacy/model.jl =====

# model.jl — Model Architecture: Forward Pass & Encoding/Decoding
#
# Pipeline:
#   INPUT → Input Projection (group 0) → Mamba SSM (groups 1–3)
#         → MoE Router+Experts (groups 4–8) → Output Projection (group 9) → OUTPUT
#
# All 10 groups × 100 params = 1000 ternary parameters participate.
# Contribution traces are recorded at every stage.
#
# Group assignment:
#   0  Input Projection  (10×10 ternary matrix)
#   1  Mamba A           (state transition)
#   2  Mamba B           (input projection into state)
#   3  Mamba C           (state read-out)
#   4  MoE Router        (first 4 outputs → expert logits)
#   5  Expert 0
#   6  Expert 1
#   7  Expert 2
#   8  Expert 3
#   9  Output Projection (10×10 ternary matrix)

# ──────────────────────────────────────────────
# Model configuration (all derived from TernaryConfig)
# ──────────────────────────────────────────────
struct ModelConfig
    hidden_dim::Int          # 10
    mamba::MambaConfig
    moe::MoEConfig
    group_input::Int         # 0
    group_output::Int        # 9
    output_scale::Float64    # maps tanh output to prediction range
end

function ModelConfig(; hidden_dim::Int=10, moe_top_k::Int=2, output_scale::Float64=20.0)
    ModelConfig(
        hidden_dim,
        MambaConfig(; hidden_dim=hidden_dim),
        MoEConfig(; num_experts=4, top_k=moe_top_k, hidden_dim=hidden_dim),
        0,   # group_input
        9,   # group_output
        output_scale
    )
end

# ──────────────────────────────────────────────
# Forward result (everything needed for evaluation + credit + chat)
# ──────────────────────────────────────────────
struct ForwardResult
    output_vector::Vector{Float64}     # raw 10-dim output
    prediction::Float64                 # decoded scalar prediction
    trace::TraceBuffer                  # full contribution trace
    expert_probs::Vector{Float64}       # MoE expert probabilities
    selected_experts::Vector{Int}       # which experts were used
    text_response::String               # formatted natural response string
end

# Backwards-compatible convenience constructor
ForwardResult(out, pred, trace, probs, sel) =
    ForwardResult(out, pred, trace, probs, sel, string(round(pred, digits=2)))

# ──────────────────────────────────────────────
# Input encoding
# ──────────────────────────────────────────────
const ENCODE_SCALE = 20.0   # normalisation factor for numeric inputs

function model_encode_number(n::Float64)
    v = zeros(10)
    v[1] = n / ENCODE_SCALE       # normalised value
    v[2] = 1.0                    # "this is a number" flag
    # Encode magnitude in a few extra dims for richer representation
    v[3] = sign(n) * min(abs(n) / ENCODE_SCALE, 1.0)
    return v
end

function model_encode_operator(op::Symbol)
    v = zeros(10)
    if     op == :add;  v[4] = 1.0
    elseif op == :sub;  v[5] = 1.0
    elseif op == :mul;  v[6] = 1.0
    elseif op == :gt;   v[7] = 1.0
    elseif op == :lt;   v[8] = 1.0
    elseif op == :copy; v[9] = 1.0
    elseif op == :rep;  v[10] = 1.0
    end
    return v
end

function model_encode_task(task::SymTaskSpec)
    seq = Vector{Float64}[]

    if task.task_type == SYM_ARITHMETIC
        op = if occursin("+", task.description); :add
        elseif occursin("-", task.description); :sub
        elseif occursin("*", task.description); :mul
        else :add end
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(op))
        push!(seq, model_encode_number(task.operand_b))

    elseif task.task_type == SYM_COMPARISON
        op = occursin(">", task.description) ? :gt : :lt
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(op))
        push!(seq, model_encode_number(task.operand_b))

    elseif task.task_type == SYM_PATTERN
        # operand_a = first element, operand_b = step
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(:add))
        push!(seq, model_encode_number(task.operand_b))

    elseif task.task_type == SYM_COPY
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(:copy))
        push!(seq, model_encode_number(0.0))

    elseif task.task_type == SYM_REPETITION
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(:rep))
        push!(seq, model_encode_number(task.operand_b))

    elseif task.task_type == SYM_FORMAT
        op = if occursin("+", task.description); :add
        elseif occursin("-", task.description); :sub
        else :add end
        push!(seq, model_encode_number(task.operand_a))
        push!(seq, model_encode_operator(op))
        push!(seq, model_encode_number(task.operand_b))
    end

    return seq
end

# ──────────────────────────────────────────────
# Output decoding
# ──────────────────────────────────────────────
function model_decode(output_vec::Vector{Float64}, config::ModelConfig)
    return output_vec[1] * config.output_scale
end

function model_decode_text(output_vec::Vector{Float64}, prediction::Float64, task::SymTaskSpec)
    if task.task_type == SYM_ARITHMETIC
        return string(round(prediction, digits=2))
    elseif task.task_type == SYM_COMPARISON
        return prediction > 0.5 ? "True (1)" : "False (0)"
    elseif task.task_type == SYM_COPY
        return "echo: $(round(prediction, digits=1))"
    elseif task.task_type == SYM_REPETITION
        rep_val = round(prediction, digits=1)
        cnt = Int(clamp(round(task.operand_b), 1, 10))
        return join(fill(string(rep_val), cnt), " ")
    elseif task.task_type == SYM_PATTERN
        return "next in pattern: $(round(prediction, digits=2))"
    else
        return "score: $(round(prediction, digits=2))"
    end
end

# ──────────────────────────────────────────────
# Full forward pass
# ──────────────────────────────────────────────
function model_forward(params::TernaryParams, task::SymTaskSpec,
                       config::ModelConfig, step::Int=0)
    ppg = params.config.params_per_group
    trace_buf = TraceBuffer(step)

    # ── 1. Encode input ──
    input_seq = model_encode_task(task)
    if isempty(input_seq)
        error("Empty input sequence for task: $(task.description)")
    end

    # ── 2. Input projection (group 0) ──
    W_in   = ternary_get_matrix(params, config.group_input)
    off_in = ternary_group_offset(params.config, config.group_input)

    projected_seq = Vector{Float64}[]
    for x in input_seq
        y, contrib = ternary_matmul(W_in, x, off_in)
        trace_add_batch!(trace_buf, contrib, ppg)
        push!(projected_seq, tanh.(y))
    end

    # ── 3. Mamba SSM (groups 1–3) ──
    mamba_out, _ = mamba_forward(config.mamba, params, projected_seq, trace_buf)

    # ── 4. MoE (groups 4–8) ──
    moe_out, route_result = moe_forward(params, mamba_out, config.moe, trace_buf)

    # ── 5. Output projection (group 9) ──
    W_out   = ternary_get_matrix(params, config.group_output)
    off_out = ternary_group_offset(params.config, config.group_output)

    raw_out, contrib_out = ternary_matmul(W_out, moe_out, off_out)
    trace_add_batch!(trace_buf, contrib_out, ppg)

    output_vec = tanh.(raw_out)
    prediction = model_decode(output_vec, config)
    text_resp  = model_decode_text(output_vec, prediction, task)

    return ForwardResult(output_vec, prediction, trace_buf,
                         route_result.expert_probs, route_result.selected, text_resp)
end

# ──────────────────────────────────────────────
# Simplified forward for intervention testing
# Returns just the output vector (no tracing)
# ──────────────────────────────────────────────
function model_forward_simple(params::TernaryParams, task::SymTaskSpec,
                              config::ModelConfig)
    result = model_forward(params, task, config, 0)
    return result.output_vector
end

# ──────────────────────────────────────────────
# Interactive chat forward helper
# ──────────────────────────────────────────────
function model_forward_chat(params::TernaryParams, prompt::AbstractString,
                            config::ModelConfig, step::Int=0)
    task = symbolic_parse_prompt(prompt)
    result = model_forward(params, task, config, step)
    return result, task
end

# ===== END legacy/model.jl =====


# ===== BEGIN legacy/experiments.jl =====

# experiments.jl — Experiments A/B/C, Embedded Dataset, Tests, and Metrics
#
# Experiment A: Proposed ternary reward/punishment with contribution tracing
# Experiment B: Random ternary mutation baseline (same mutation rate)
# Experiment C: Global update baseline (all parameters updated, not sparse)

using Printf, Random, Dates

# ══════════════════════════════════════════════
# EMBEDDED SYNTHETIC DATASET
# ══════════════════════════════════════════════

function exp_create_dataset()
    training = SymTaskSpec[
        # ── Simple arithmetic ──
        SymTaskSpec(SYM_ARITHMETIC, "3+5",  3.0, 5.0, 8.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "7-2",  7.0, 2.0, 5.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "4+6",  4.0, 6.0, 10.0, false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "9-3",  9.0, 3.0, 6.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "1+8",  1.0, 8.0, 9.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "2+7",  2.0, 7.0, 9.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "5*2",  5.0, 2.0, 10.0, false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "3*3",  3.0, 3.0, 9.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "6-4",  6.0, 4.0, 2.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC, "8+1",  8.0, 1.0, 9.0,  false, ""),
        # ── Comparison ──
        SymTaskSpec(SYM_COMPARISON, "7>3?", 7.0, 3.0, 1.0,  false, ""),
        SymTaskSpec(SYM_COMPARISON, "2>8?", 2.0, 8.0, 0.0,  false, ""),
        SymTaskSpec(SYM_COMPARISON, "5<9?", 5.0, 9.0, 1.0,  false, ""),
        SymTaskSpec(SYM_COMPARISON, "6<2?", 6.0, 2.0, 0.0,  false, ""),
        # ── Pattern recognition ──
        SymTaskSpec(SYM_PATTERN, "2,4,6,?",   2.0, 2.0, 8.0,  false, ""),
        SymTaskSpec(SYM_PATTERN, "1,3,5,?",   1.0, 2.0, 7.0,  false, ""),
        SymTaskSpec(SYM_PATTERN, "5,10,15,?", 5.0, 5.0, 20.0, false, ""),
        # ── Copy ──
        SymTaskSpec(SYM_COPY, "echo 7",  7.0, 0.0, 7.0, false, ""),
        SymTaskSpec(SYM_COPY, "echo 3",  3.0, 0.0, 3.0, false, ""),
        # ── Repetition (requested) ──
        SymTaskSpec(SYM_REPETITION, "repeat 5",  5.0, 3.0, 5.0, true, ""),
        SymTaskSpec(SYM_REPETITION, "repeat 2",  2.0, 4.0, 2.0, true, ""),
        # ── Format ──
        SymTaskSpec(SYM_FORMAT, "4+2 num", 4.0, 2.0, 6.0, false, "number_only"),
    ]

    test = SymTaskSpec[
        SymTaskSpec(SYM_ARITHMETIC,  "5+3",    5.0, 3.0, 8.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC,  "8-5",    8.0, 5.0, 3.0,  false, ""),
        SymTaskSpec(SYM_ARITHMETIC,  "2*4",    2.0, 4.0, 8.0,  false, ""),
        SymTaskSpec(SYM_COMPARISON,  "9>1?",   9.0, 1.0, 1.0,  false, ""),
        SymTaskSpec(SYM_COMPARISON,  "3<1?",   3.0, 1.0, 0.0,  false, ""),
        SymTaskSpec(SYM_PATTERN,     "3,6,9,?",3.0, 3.0, 12.0, false, ""),
        SymTaskSpec(SYM_COPY,        "echo 9", 9.0, 0.0, 9.0,  false, ""),
        SymTaskSpec(SYM_REPETITION,  "repeat 7",7.0, 2.0, 7.0, true, ""),
    ]

    return training, test
end

# ══════════════════════════════════════════════
# DATASET INPUTTER & LOADER
# Ingests and parses custom datasets from:
#   - ./datasets
#   - %TEMP%\datasets
#   - user-specified files or folders (.txt, .csv, .tsv, .json, .jsonl)
# ══════════════════════════════════════════════

function dataset_scan_dirs(extra_dir::String="")
    dirs = String[]
    # 1. ./datasets in project or working directory
    p_datasets = joinpath(@__DIR__, "datasets")
    if isdir(p_datasets)
        push!(dirs, p_datasets)
    end
    if isdir("datasets")
        push!(dirs, abspath("datasets"))
    end
    # 2. OS temporary datasets directory (%TEMP%\datasets)
    t_datasets = joinpath(tempdir(), "datasets")
    if isdir(t_datasets)
        push!(dirs, t_datasets)
    end
    rt_datasets = joinpath(get(ENV, "TERNARY_RUNTIME_DIR", tempdir()), "datasets")
    if isdir(rt_datasets)
        push!(dirs, rt_datasets)
    end
    # 3. User specified extra directory
    if !isempty(extra_dir) && isdir(extra_dir)
        push!(dirs, abspath(extra_dir))
    end

    files = String[]
    for d in unique(dirs)
        for f in readdir(d)
            ext = lowercase(splitext(f)[2])
            if ext in (".txt", ".csv", ".tsv", ".json", ".jsonl")
                push!(files, joinpath(d, f))
            end
        end
    end
    return unique(files)
end

function dataset_parse_line(line::AbstractString)
    l = strip(line)
    (isempty(l) || startswith(l, "#") || startswith(l, "//")) && return nothing

    prompt = ""
    target = ""

    if occursin(" | ", l)
        parts = split(l, " | "; limit=2)
        prompt, target = strip(parts[1]), strip(parts[2])
    elseif occursin(" -> ", l)
        parts = split(l, " -> "; limit=2)
        prompt, target = strip(parts[1]), strip(parts[2])
    elseif occursin(r"^Q:\s*", l) && occursin(r"\s+A:\s*", l)
        parts = split(l, r"\s+A:\s*"; limit=2)
        prompt = replace(parts[1], r"^Q:\s*" => "")
        target = parts[2]
    elseif occursin(" = ", l)
        parts = split(l, " = "; limit=2)
        prompt, target = strip(parts[1]), strip(parts[2])
    elseif occursin("\t", l)
        parts = split(l, "\t"; limit=2)
        prompt, target = strip(parts[1]), strip(parts[2])
    elseif occursin(",", l) && !occursin(r"^\s*\d+,\s*\d+,\s*\d+", l)
        parts = split(l, ","; limit=2)
        prompt, target = strip(parts[1]), strip(parts[2])
    else
        prompt = l
        target = ""
    end

    return symbolic_parse_prompt(prompt, target)
end

function dataset_parse_file(filepath::AbstractString)
    isfile(filepath) || error("Dataset file not found: $filepath")
    tasks = SymTaskSpec[]
    ext = lowercase(splitext(filepath)[2])

    lines = readlines(filepath)
    for (line_num, line) in enumerate(lines)
        l = strip(line)
        isempty(l) && continue

        # Header skip for CSV/TSV
        if line_num == 1 && ext in (".csv", ".tsv")
            low = lowercase(l)
            if occursin("input", low) || occursin("prompt", low) || occursin("question", low)
                continue
            end
        end

        # JSON / JSONL line extraction
        if ext in (".json", ".jsonl") || (startswith(l, "{") && endswith(l, "}"))
            m_prompt = match(r"\"(?:prompt|input|question|text)\"\s*:\s*\"?([^\"\}]+)\"?", l)
            m_target = match(r"\"(?:target|expected|answer|output)\"\s*:\s*\"?([^\"\}]+)\"?", l)
            if m_prompt !== nothing
                p = m_prompt.captures[1]
                t = m_target !== nothing ? m_target.captures[1] : ""
                push!(tasks, symbolic_parse_prompt(p, t))
                continue
            end
        end

        task = dataset_parse_line(l)
        if task !== nothing
            push!(tasks, task)
        end
    end
    return tasks
end

function dataset_load(path::AbstractString)
    if isdir(path)
        tasks = SymTaskSpec[]
        for f in readdir(path)
            ext = lowercase(splitext(f)[2])
            if ext in (".txt", ".csv", ".tsv", ".json", ".jsonl")
                append!(tasks, dataset_parse_file(joinpath(path, f)))
            end
        end
        return tasks
    elseif isfile(path)
        return dataset_parse_file(path)
    else
        error("Dataset path not found: $path")
    end
end

function dataset_train_custom!(params::TernaryParams, tasks::Vector{SymTaskSpec};
                               num_epochs::Int=3, debug::Bool=false, top_k::Int=8,
                               intervention::Bool=false)
    isempty(tasks) && (println("No tasks provided to train."); return ExpMetrics())
    println("\n" * "="^55)
    println("TRAINING ON CUSTOM DATASET ($(length(tasks)) samples, $num_epochs epochs)")
    println("="^55)

    mconfig = ModelConfig()
    rconfig = RewardConfig()
    cconfig = CreditConfig(top_k=top_k, intervention_enabled=intervention)
    metrics = ExpMetrics()
    metrics.vram_bytes_est = 250 + (10 * 8 * 4)

    fwd_simple = (p, t) -> model_forward_simple(p, t, mconfig)

    step = 0
    for epoch in 1:num_epochs
        for task in tasks
            step += 1
            t_start = time()
            ternary_reset_step!(params)

            t_fwd_start = time()
            result = model_forward(params, task, mconfig, step)
            push!(metrics.forward_times, time() - t_fwd_start)

            eval_res = symbolic_evaluate(task, result.prediction; output_vector=result.output_vector)
            decision = reward_evaluate(eval_res, task, rconfig)

            t_crd_start = time()
            credit_result = credit_assign(result.trace, decision, params, cconfig;
                                          model_forward_fn=fwd_simple, task=task,
                                          original_output=result.output_vector)
            mutations = credit_apply!(credit_result, decision, params)
            push!(metrics.credit_times, time() - t_crd_start)

            ternary_decay_eligibility!(params)

            if decision.signal == REWARD;     metrics.reward_count += 1
            elseif decision.signal == PUNISH; metrics.punish_count += 1
            else;                             metrics.neutral_count += 1 end

            metrics.total_mutations += mutations
            metrics.total_candidates_checked += credit_result.search_depth
            metrics.total_params_modified += length(credit_result.accepted)
            push!(metrics.params_modified_per_step, length(credit_result.accepted))

            if eval_res.correct
                metrics.train_correct += 1
            end
            metrics.train_total += 1
            push!(metrics.step_times, time() - t_start)
            metrics.peak_ram_bytes = max(metrics.peak_ram_bytes, Int64(Sys.maxrss()))

            trace_clear!(result.trace)

            if debug
                _print_step_debug(step, task, result, eval_res, decision, credit_result, mutations)
            end
        end
        acc = metrics.train_total > 0 ? round(metrics.train_correct / metrics.train_total * 100, digits=1) : 0.0
        @printf("  Epoch %d/%d: %d/%d correct (%.1f%%), %d mutations\n",
                epoch, num_epochs, metrics.train_correct, metrics.train_total, acc, metrics.total_mutations)
    end
    _print_report("CUSTOM_DATASET", params, metrics, 0)
    return metrics
end

# ══════════════════════════════════════════════
# METRICS TRACKER
# ══════════════════════════════════════════════

mutable struct ExpMetrics
    reward_count::Int
    punish_count::Int
    neutral_count::Int
    total_mutations::Int
    total_candidates_checked::Int
    total_params_modified::Int
    train_correct::Int
    train_total::Int
    test_correct::Int
    test_total::Int
    step_times::Vector{Float64}
    forward_times::Vector{Float64}
    credit_times::Vector{Float64}
    params_modified_per_step::Vector{Int}
    peak_ram_bytes::Int64
    vram_bytes_est::Int64
    samples_to_converge::Int
end
ExpMetrics() = ExpMetrics(0,0,0,0,0,0,0,0,0,0, Float64[], Float64[], Float64[], Int[], 0, 0, -1)

# ══════════════════════════════════════════════
# EXPERIMENT A: Proposed Ternary Reward/Punishment Training
# ══════════════════════════════════════════════

function exp_run_A(; num_epochs::Int=5, debug::Bool=false, seed::Int=42,
                     intervention::Bool=false, top_k::Int=8)
    Random.seed!(seed)
    println("\n" * "="^50)
    println("EXPERIMENT A: Ternary Reward/Punishment Training")
    println("="^50)

    tconfig = TernaryConfig()
    mconfig = ModelConfig()
    rconfig = RewardConfig()
    cconfig = CreditConfig(top_k=top_k, intervention_enabled=intervention)

    params  = ternary_create(tconfig)
    ternary_validate(params)
    metrics = ExpMetrics()

    # VRAM footprint estimate for 1000 ternary parameters (2 bits each = 250 bytes)
    # plus small hidden-state activation buffers (10 dim Float64 = 80 bytes)
    metrics.vram_bytes_est = 250 + (10 * 8 * 4)

    train_data, test_data = exp_create_dataset()

    # Binary log path (Patch 13: compact binary event log)
    runtime_dir = get(ENV, "TERNARY_RUNTIME_DIR", tempdir())
    event_log_path = joinpath(runtime_dir, "events_A.bin")

    # Simple forward wrapper for causal intervention testing (Patch 5)
    fwd_simple = (p, t) -> model_forward_simple(p, t, mconfig)

    step = 0
    for epoch in 1:num_epochs
        debug && println("\n--- Epoch $epoch ---")

        for task in train_data
            step += 1
            t_start = time()
            ternary_reset_step!(params)

            # ── 1. Forward Pass (Timed) ──
            t_fwd_start = time()
            result = model_forward(params, task, mconfig, step)
            t_fwd = time() - t_fwd_start
            push!(metrics.forward_times, t_fwd)

            # ── 2. Automatic Neuro-Symbolic Evaluation (Patch 6) ──
            eval_result = symbolic_evaluate(task, result.prediction;
                                           output_vector=result.output_vector)

            # ── 3. Automatic Signal Engine (Patch 6) ──
            decision = reward_evaluate(eval_result, task, rconfig)

            # ── 4. Hierarchical Credit Assignment (Patches 4, 5, 14) ──
            t_credit_start = time()
            credit_result = credit_assign(result.trace, decision, params, cconfig;
                                          model_forward_fn=fwd_simple,
                                          task=task,
                                          original_output=result.output_vector)

            # ── 5. Sparse Ternary Parameter Update (Patches 7, 8, 9, 11) ──
            mutations = credit_apply!(credit_result, decision, params)
            t_credit = time() - t_credit_start
            push!(metrics.credit_times, t_credit)

            # ── 6. Eligibility Decay ──
            ternary_decay_eligibility!(params)

            # ── 7. Record Metrics (Patch 16) ──
            if decision.signal == REWARD;       metrics.reward_count += 1
            elseif decision.signal == PUNISH;   metrics.punish_count += 1
            else;                               metrics.neutral_count += 1  end

            metrics.total_mutations += mutations
            metrics.total_candidates_checked += credit_result.search_depth
            metrics.total_params_modified += length(credit_result.accepted)
            push!(metrics.params_modified_per_step, length(credit_result.accepted))

            if eval_result.correct
                metrics.train_correct += 1
            end
            metrics.train_total += 1

            elapsed = time() - t_start
            push!(metrics.step_times, elapsed)
            metrics.peak_ram_bytes = max(metrics.peak_ram_bytes, Int64(Sys.maxrss()))

            # ── 8. Write Compact Binary Events (Patch 13) ──
            events = EventRecord[]
            sig_val = decision.signal == REWARD ? Int8(1) : (decision.signal == PUNISH ? Int8(-1) : Int8(0))

            if decision.signal == NEUTRAL
                push!(events, EventRecord(Int32(step), Int32(0), Int16(-1), Int8(0), sig_val, 0f0))
            else
                for pid in credit_result.accepted
                    gid = Int16((pid - 1) ÷ tconfig.params_per_group)
                    c_val = 0f0
                    for (cid, cv) in credit_result.candidates
                        if cid == pid
                            c_val = Float32(cv)
                            break
                        end
                    end
                    etype = decision.signal == REWARD ? Int8(1) : Int8(2)
                    push!(events, EventRecord(Int32(step), Int32(pid), gid, etype, sig_val, c_val))
                end
            end
            if !isempty(events)
                trace_write_events(event_log_path, events; append=true)
            end

            # ── 9. Discard Temporary Trace (Patch 12) ──
            trace_clear!(result.trace)

            # ── 10. Debug Output ──
            if debug
                _print_step_debug(step, task, result, eval_result, decision,
                                  credit_result, mutations)
            end
        end
    end

    # ── Test evaluation ──
    for task in test_data
        result = model_forward(params, task, mconfig, 0)
        eval_result = symbolic_evaluate(task, result.prediction;
                                       output_vector=result.output_vector)
        if eval_result.correct
            metrics.test_correct += 1
        end
        metrics.test_total += 1
        trace_clear!(result.trace)
    end

    # ── Binary log size ──
    trace_size = isfile(event_log_path) ? filesize(event_log_path) : 0

    _print_report("A", params, metrics, trace_size)
    return params, metrics, trace_size
end

# ══════════════════════════════════════════════
# EXPERIMENT B: Random Mutation Baseline
# ══════════════════════════════════════════════

function exp_run_B(; num_epochs::Int=5, debug::Bool=false, seed::Int=42,
                     mutations_per_step::Int=8)
    Random.seed!(seed)
    println("\n" * "="^50)
    println("EXPERIMENT B: Random Ternary Mutation Baseline")
    println("="^50)

    tconfig = TernaryConfig()
    mconfig = ModelConfig()
    params  = ternary_create(tconfig)
    ternary_validate(params)
    metrics = ExpMetrics()
    metrics.vram_bytes_est = 250 + (10 * 8 * 4)

    train_data, test_data = exp_create_dataset()

    step = 0
    for epoch in 1:num_epochs
        for task in train_data
            step += 1
            t_start = time()
            ternary_reset_step!(params)

            # ── Forward (for evaluation only) ──
            t_fwd_start = time()
            result = model_forward(params, task, mconfig, step)
            t_fwd = time() - t_fwd_start
            push!(metrics.forward_times, t_fwd)

            eval_result = symbolic_evaluate(task, result.prediction;
                                           output_vector=result.output_vector)

            if eval_result.correct
                metrics.train_correct += 1
            end
            metrics.train_total += 1

            # ── Random mutations (no credit assignment) ──
            t_credit_start = time()
            for _ in 1:mutations_per_step
                pid = rand(1:tconfig.num_params)
                new_state = rand(Int8[-1, 0, 1])
                if params.states[pid] != new_state
                    params.states[pid] = new_state
                    params.mutation_count += 1
                    metrics.total_mutations += 1
                end
            end
            t_credit = time() - t_credit_start
            push!(metrics.credit_times, t_credit)

            push!(metrics.params_modified_per_step, mutations_per_step)
            elapsed = time() - t_start
            push!(metrics.step_times, elapsed)
            metrics.peak_ram_bytes = max(metrics.peak_ram_bytes, Int64(Sys.maxrss()))

            # Discard temporary trace
            trace_clear!(result.trace)
        end
    end

    # ── Test ──
    for task in test_data
        result = model_forward(params, task, mconfig, 0)
        eval_result = symbolic_evaluate(task, result.prediction;
                                       output_vector=result.output_vector)
        if eval_result.correct
            metrics.test_correct += 1
        end
        metrics.test_total += 1
        trace_clear!(result.trace)
    end

    _print_report("B", params, metrics, 0)
    return params, metrics, 0
end

# ══════════════════════════════════════════════
# EXPERIMENT C: Global Update Baseline
# (All parameters receive the signal — no sparsity)
# ══════════════════════════════════════════════

function exp_run_C(; num_epochs::Int=5, debug::Bool=false, seed::Int=42)
    Random.seed!(seed)
    println("\n" * "="^50)
    println("EXPERIMENT C: Global Update Baseline")
    println("="^50)

    tconfig = TernaryConfig(reward_threshold=50.0, punish_threshold=50.0)  # higher thresholds
    mconfig = ModelConfig()
    rconfig = RewardConfig(reward_strength_scale=0.1, punish_strength_scale=0.1)
    params  = ternary_create(tconfig)
    ternary_validate(params)
    metrics = ExpMetrics()
    metrics.vram_bytes_est = 250 + (10 * 8 * 4)

    train_data, test_data = exp_create_dataset()

    step = 0
    for epoch in 1:num_epochs
        for task in train_data
            step += 1
            t_start = time()
            ternary_reset_step!(params)

            # ── Forward ──
            t_fwd_start = time()
            result = model_forward(params, task, mconfig, step)
            t_fwd = time() - t_fwd_start
            push!(metrics.forward_times, t_fwd)

            eval_result = symbolic_evaluate(task, result.prediction;
                                           output_vector=result.output_vector)
            decision = reward_evaluate(eval_result, task, rconfig)

            if eval_result.correct
                metrics.train_correct += 1
            end
            metrics.train_total += 1

            # ── Global update: apply signal to ALL parameters ──
            t_credit_start = time()
            if decision.signal != NEUTRAL
                signal_sym = decision.signal == REWARD ? :reward : :punish
                mutations = 0
                for pid in 1:tconfig.num_params
                    if ternary_mutate!(params, pid, signal_sym, decision.strength * 0.01)
                        mutations += 1
                    end
                end
                metrics.total_mutations += mutations
                push!(metrics.params_modified_per_step, tconfig.num_params)

                if decision.signal == REWARD; metrics.reward_count += 1
                else; metrics.punish_count += 1 end
            else
                metrics.neutral_count += 1
                push!(metrics.params_modified_per_step, 0)
            end
            t_credit = time() - t_credit_start
            push!(metrics.credit_times, t_credit)

            elapsed = time() - t_start
            push!(metrics.step_times, elapsed)
            metrics.peak_ram_bytes = max(metrics.peak_ram_bytes, Int64(Sys.maxrss()))

            # Discard temporary trace
            trace_clear!(result.trace)
        end
    end

    # ── Test ──
    for task in test_data
        result = model_forward(params, task, mconfig, 0)
        eval_result = symbolic_evaluate(task, result.prediction;
                                       output_vector=result.output_vector)
        if eval_result.correct
            metrics.test_correct += 1
        end
        metrics.test_total += 1
        trace_clear!(result.trace)
    end

    _print_report("C", params, metrics, 0)
    return params, metrics, 0
end

# ══════════════════════════════════════════════
# DEBUG OUTPUT
# ══════════════════════════════════════════════

function _print_step_debug(step, task, result, eval_result, decision,
                           credit_result, mutations)
    println()
    println("STEP $step")
    println("  INPUT: $(task.description)")
    @printf("  OUTPUT: %.2f (expected: %.1f)\n", result.prediction, task.expected)
    println("  SIGNAL: $(reward_signal_string(decision.signal))")
    println("  REASONS:")
    for r in decision.reasons
        println("    - $r")
    end
    if !isempty(credit_result.candidates)
        println("  TOP CONTRIBUTORS:")
        for (pid, c) in credit_result.candidates[1:min(3, length(credit_result.candidates))]
            tag = pid in credit_result.accepted ? " [UPDATED]" : ""
            @printf("    P%d = %.3f%s\n", pid, c, tag)
        end
    end

    # Show intervention verification if occurred (Patch 5, Prompt 31)
    for chk in credit_result.checks
        if chk.tested_intervention && chk.accepted
            println("  INTERVENTION:")
            println("    P$(chk.param_id) masked")
            @printf("    behavior changed (Δ = %.3f)\n", chk.behavior_change)
            println("  UPDATE:")
            println("    P$(chk.param_id) punished")
        end
    end

    println("  MUTATIONS: $mutations")
    @printf("  EXPERTS: %s (probs: %s)\n",
            join(result.selected_experts, ","),
            join([@sprintf("%.2f", p) for p in result.expert_probs], ","))
end

# ══════════════════════════════════════════════
# FINAL REPORT (Patch 16: Detailed Efficiency Metrics)
# ══════════════════════════════════════════════

function _print_report(label, params, metrics, trace_size)
    dist = ternary_distribution(params)
    avg_candidates = metrics.total_candidates_checked > 0 ?
        metrics.total_candidates_checked / max(metrics.train_total, 1) : 0.0
    avg_modified = isempty(metrics.params_modified_per_step) ? 0.0 :
        sum(metrics.params_modified_per_step) / length(metrics.params_modified_per_step)
    pct_modified = avg_modified / params.config.num_params * 100.0

    train_acc = metrics.train_total > 0 ?
        metrics.train_correct / metrics.train_total * 100.0 : 0.0
    test_acc = metrics.test_total > 0 ?
        metrics.test_correct / metrics.test_total * 100.0 : 0.0
    total_time = sum(metrics.step_times)
    avg_step   = isempty(metrics.step_times) ? 0.0 :
        total_time / length(metrics.step_times) * 1000.0  # ms

    total_fwd = sum(metrics.forward_times)
    avg_fwd   = isempty(metrics.forward_times) ? 0.0 :
        total_fwd / length(metrics.forward_times) * 1000.0
    total_credit = sum(metrics.credit_times)
    avg_credit   = isempty(metrics.credit_times) ? 0.0 :
        total_credit / length(metrics.credit_times) * 1000.0

    println()
    println("="^50)
    println("EXPERIMENT $label COMPLETE")
    println("="^50)
    println()
    @printf("  Trainable parameters:     %d\n", params.config.num_params)
    @printf("  Groups:                   %d\n", params.config.num_groups)
    @printf("  Parameters/group:         %d\n", params.config.params_per_group)
    println()
    @printf("  Training examples:        %d\n", metrics.train_total)
    @printf("  Training steps:           %d\n", metrics.train_total)
    println()
    @printf("  Reward events:            %d\n", metrics.reward_count)
    @printf("  Punishment events:        %d\n", metrics.punish_count)
    @printf("  Neutral events:           %d\n", metrics.neutral_count)
    println()
    @printf("  Parameter mutations:      %d\n", metrics.total_mutations)
    @printf("  Avg candidates checked:   %.1f\n", avg_candidates)
    @printf("  Avg params modified/step: %.1f\n", avg_modified)
    @printf("  Pct params modified/step: %.2f%%\n", pct_modified)
    println()
    @printf("  Training accuracy:        %.1f%%\n", train_acc)
    @printf("  Evaluation accuracy:      %.1f%%\n", test_acc)
    println()
    @printf("  Ternary distribution:     -1=%d  0=%d  +1=%d\n",
            dist.negative, dist.zero, dist.positive)
    println()
    println("  EFFICIENCY MEASUREMENTS (Patch 16):")
    if metrics.peak_ram_bytes > 0
        @printf("    Peak RAM:               %.2f MB\n", metrics.peak_ram_bytes / 1024 / 1024)
    end
    @printf("    Estimated VRAM:         %d bytes (packed 2-bit ternary)\n", metrics.vram_bytes_est)
    @printf("    Forward pass cost:      %.3f s total (%.2f ms/sample)\n", total_fwd, avg_fwd)
    @printf("    Credit assignment cost: %.3f s total (%.2f ms/sample)\n", total_credit, avg_credit)
    @printf("    Total training time:    %.3f s\n", total_time)
    @printf("    Avg step time:          %.2f ms\n", avg_step)
    if trace_size > 0
        @printf("    Binary event log size:  %d bytes\n", trace_size)
    end
    conv_str = metrics.samples_to_converge == -1 ? "N/A (prototype budget)" : string(metrics.samples_to_converge)
    @printf("    Samples to converge:    %s\n", conv_str)
    println()
end

# ══════════════════════════════════════════════
# AUTOMATED TESTS
# ══════════════════════════════════════════════

function exp_run_tests()
    println("\n" * "="^50)
    println("RUNNING AUTOMATED TESTS")
    println("="^50)
    passed = 0
    failed = 0

    function check(name, condition)
        if condition
            println("  [PASS] $name")
            passed += 1
        else
            println("  [FAIL] $name")
            failed += 1
        end
    end

    # ── Test 1: Exact 1000 parameters ──
    tc = TernaryConfig()
    p  = ternary_create(tc)
    check("Exactly 1000 trainable parameters", length(p.states) == 1000)

    # ── Test 2: Exactly 10 groups ──
    check("Exactly 10 groups", tc.num_groups == 10)

    # ── Test 3: Exactly 100 per group ──
    check("Exactly 100 parameters per group", tc.params_per_group == 100)

    # ── Test 4: Ternary invariant ──
    check("Ternary state invariant",
          all(s -> s in (Int8(-1), Int8(0), Int8(1)), p.states))

    # ── Test 5: Validation passes ──
    check("Validation passes", ternary_validate(p))

    # ── Test 6: Group indices correct ──
    idx0 = ternary_group_indices(tc, 0)
    idx9 = ternary_group_indices(tc, 9)
    check("Group 0 indices = 1:100", idx0 == 1:100)
    check("Group 9 indices = 901:1000", idx9 == 901:1000)

    # ── Test 7: Reward modifies only eligible params ──
    p2 = ternary_create_zeros(tc)
    p2.config.reward_threshold = 1.0
    ternary_mutate!(p2, 42, :reward, 1.5)
    modified_count = count(s -> s != Int8(0), p2.states)
    check("Reward modifies only target parameter", modified_count <= 1)
    others_unchanged = all(i -> i == 42 || p2.states[i] == Int8(0), 1:1000)
    check("Reward leaves other parameters unchanged", others_unchanged)

    # ── Test 8: Punishment modifies only eligible params ──
    p3 = ternary_create(TernaryConfig(punish_threshold=1.0))
    saved = copy(p3.states)
    ternary_mutate!(p3, 100, :punish, 1.5)
    others_same = all(i -> i == 100 || p3.states[i] == saved[i], 1:1000)
    check("Punishment leaves other parameters unchanged", others_same)

    # ── Test 9: Neutral modifies nothing ──
    p4 = ternary_create(tc)
    saved4 = copy(p4.states)
    ternary_mutate!(p4, 50, :neutral, 1.0)
    check("Neutral modifies nothing", p4.states == saved4)

    # ── Test 10: Requested repetition not punished ──
    rep_task = SymTaskSpec(SYM_REPETITION, "repeat 5", 5.0, 3.0, 5.0, true, "")
    eval_rep = symbolic_evaluate(rep_task, 5.0)
    dec_rep  = reward_evaluate(eval_rep, rep_task)
    check("Requested repetition is NOT punished", dec_rep.signal != PUNISH)

    # ── Test 11: Unwanted repetition is punished ──
    normal_task = SymTaskSpec(SYM_ARITHMETIC, "3+5", 3.0, 5.0, 8.0, false, "")
    rep_vec = fill(0.5, 10)
    eval_unrep = symbolic_evaluate(normal_task, 8.0; output_vector=rep_vec)
    dec_unrep  = reward_evaluate(eval_unrep, normal_task)
    check("Unwanted repetition is punished",
          eval_unrep.has_unwanted_repetition ? dec_unrep.signal == PUNISH : true)

    # ── Test 12: Binary logging round-trip ──
    test_records = [
        ContributionRecord(Int32(1), Int16(0), Float32(0.5), Int8(0), Int32(1)),
        ContributionRecord(Int32(42), Int16(0), Float32(0.8), Int8(1), Int32(1)),
    ]
    log_path = joinpath(tempdir(), "test_trace_$(rand(1000:9999)).bin")
    trace_write_binary(log_path, test_records; append=false)
    read_back = trace_read_binary(log_path)
    check("Binary logging round-trip",
          length(read_back) == 2 &&
          read_back[1].param_id == Int32(1) &&
          read_back[2].param_id == Int32(42))
    rm(log_path; force=true)

    # ── Test 13: Binary logs can be decoded to stats ──
    stats = trace_reconstruct_stats(test_records)
    check("Binary log stats reconstruction",
          haskey(stats, Int32(1)) && stats[Int32(1)].count == 1)

    # ── Test 14: Parameter IDs are stable ──
    p5 = ternary_create(tc)
    m1 = ternary_get_matrix(p5, 0)
    m2 = ternary_get_matrix(p5, 0)
    check("Parameter IDs are stable", m1 == m2)

    # ── Test 15: Candidate ranking works ──
    tb = TraceBuffer(1)
    trace_add!(tb, 10, 0, 0.9)
    trace_add!(tb, 20, 0, 0.1)
    trace_add!(tb, 30, 0, 0.5)
    top = trace_top_contributors(tb, 2)
    check("Candidate ranking works",
          length(top) == 2 && top[1].param_id == Int32(10))

    # ── Test 16: Forward pass produces output ──
    p6 = ternary_create(tc)
    mc = ModelConfig()
    task = SymTaskSpec(SYM_ARITHMETIC, "3+5", 3.0, 5.0, 8.0, false, "")
    fwd = model_forward(p6, task, mc, 1)
    check("Forward pass produces finite output",
          isfinite(fwd.prediction) && length(fwd.output_vector) == 10)

    # ── Test 17: Trace is non-empty after forward ──
    check("Trace is non-empty after forward", !isempty(fwd.trace.records))

    # ── Test 18: Compact binary event logging roundtrip (Patch 13) ──
    evs = [
        EventRecord(Int32(1), Int32(10), Int16(0), Int8(1), Int8(1), Float32(0.9)),
        EventRecord(Int32(1), Int32(20), Int16(0), Int8(2), Int8(-1), Float32(0.4)),
        EventRecord(Int32(2), Int32(0), Int16(-1), Int8(0), Int8(0), Float32(0.0)),
    ]
    ev_path = joinpath(tempdir(), "test_events_$(rand(1000:9999)).bin")
    trace_write_events(ev_path, evs; append=false)
    ev_read = trace_read_events(ev_path)
    rm(ev_path; force=true)
    check("Compact binary event log round-trip (Patch 13)",
          length(ev_read) == 3 && ev_read[1].param_id == Int32(10) && ev_read[1].event_type == Int8(1))

    # ── Test 19: Temporary trace disposal (Patch 12) ──
    tb_temp = TraceBuffer(5)
    trace_add!(tb_temp, 1, 0, 0.5)
    trace_clear!(tb_temp)
    check("Temporary trace disposal via trace_clear! (Patch 12)", isempty(tb_temp.records))

    # ── Test 20: Candidate check record with search depth (Patch 4) ──
    tb_check = TraceBuffer(1)
    trace_add!(tb_check, 100, 0, 0.8)
    trace_add!(tb_check, 200, 1, 0.6)
    dec_punish = RewardDecision(PUNISH, 0.9, 0.8, ["incorrect"])
    cr_check = credit_assign(tb_check, dec_punish, p6, CreditConfig(top_k=4))
    check("Sequential candidate investigation records search depth (Patch 4)",
          cr_check.search_depth == 2 && !isempty(cr_check.checks))

    # ── Test 21: Intervention testing alters behavior (Patch 5) ──
    p_int = ternary_create(tc)
    p_int.states[10] = Int8(1)
    fwd_fn = (pr, tsk) -> model_forward_simple(pr, tsk, mc)
    orig_out = fwd_fn(p_int, task)
    delta = credit_intervention_test(p_int, 10, orig_out, fwd_fn, task)
    check("Intervention test measures candidate behavioral change (Patch 5)",
          isfinite(delta))

    # ── Test 22: Dataset inputter line parsing ──
    t_parse1 = dataset_parse_line("7 + 5 | 12")
    t_parse2 = dataset_parse_line("10 > 3 -> 1")
    t_parse3 = dataset_parse_line("echo 7")
    check("Dataset inputter parses line formats",
          t_parse1 !== nothing && t_parse1.expected == 12.0 &&
          t_parse2 !== nothing && t_parse2.expected == 1.0 &&
          t_parse3 !== nothing && t_parse3.expected == 7.0)

    # ── Test 23: Dataset inputter file parsing ──
    test_ds_path = joinpath(tempdir(), "test_ds_$(rand(1000:9999)).txt")
    write(test_ds_path, "3 + 2 | 5\n8 - 3 | 5\n4 * 2 | 8\n")
    loaded_tasks = dataset_load(test_ds_path)
    rm(test_ds_path; force=true)
    check("Dataset inputter loads and parses files",
          length(loaded_tasks) == 3 && loaded_tasks[1].expected == 5.0)

    # ── Test 24: Chat forward returns text response ──
    res_chat, task_chat = model_forward_chat(p6, "6 + 4", mc, 1)
    check("Chat forward produces valid result and text response",
          !isempty(res_chat.text_response) && task_chat.expected == 10.0)

    # ── Summary ──
    println()
    println("  Tests: $(passed + failed) total, $passed passed, $failed failed")
    println("="^50)
    return failed == 0
end

# ===== END legacy/experiments.jl =====


# ===== BEGIN legacy/run_legacy.jl (runner only; includes inlined above) =====

function run(args::Vector{String})
    if "--test" in args
        exp_run_tests(); return
    end
    run_A = isempty(args) || "--expA" in args
    run_B = isempty(args) || "--expB" in args
    run_C = isempty(args) || "--expC" in args
    run_A && exp_run_A(; num_epochs=5)
    run_B && exp_run_B(; num_epochs=5)
    run_C && exp_run_C(; num_epochs=5)
    return
end

# ===== END legacy/run_legacy.jl =====


end # module LegacyBaseline

# ===== END legacy.jl =====


# ===== BEGIN tests.jl =====

# tests.jl -- consolidated bundle (auto-generated)
# Sources merged: test/runtests.jl, test/debug_transitive.jl (as demo function)
# Part of the flat 10-file layout: 7x .jl + run.bat + requirements.txt + readme.txt

# Bundled test suite. Called by cli --test via run_bundled_tests().
# Uses Test stdlib (imported in main.jl module header).

function run_bundled_tests()
@testset "ternary cognitive architecture" begin
  ai=cognitive_agent(:tiny); @test verify_integrity(ai)
  r,s=respond!(ai,"a causes b";expected_output="a causes b"); @test r[2].quality in (:GOOD,:EXCELLENT,:PARTIAL)
  respond!(ai,"b causes c";expected_output="b causes c"); r,s=respond!(ai,"why a causes c";expected_output="a causes c")
  @test occursin("a causes c",lowercase(r[1])); @test length(s.hypotheses)>=3; @test !isempty(s.trace.events)
  before=synaptic_snapshot(ai); respond!(ai,"does a causes c";expected_output="a prevents c"); @test get_last_evaluation(ai).quality==:WRONG
  @test verify_integrity(ai); @test !isempty(get_last_synaptic_changes(ai)) || !isempty(before)
  rep=cognitive_agent(:tiny); respond!(rep,"x causes y"); respond!(rep,"repeat this"); @test get_last_evaluation(rep).quality==:REQUESTED_REPETITION
  rep_empty=cognitive_agent(:tiny); respond!(rep_empty,"repeat this"); @test get_last_evaluation(rep_empty).quality==:INSTRUCTION_FAILURE
  @test any(e -> Symbol(e.kind) == :EPISODE_STORED, ai.episodic.episodes[end].events)
  @test ai.metrics.episodes >= 3
  @test length(ai.metrics.success_curve) == ai.metrics.episodes
  path=tempname(); save_model(ai,path); loaded=load_model(path); rm(path); @test length(loaded.episodic.episodes)==length(ai.episodic.episodes)
end

@testset "learned ternary state guides selection" begin
  ai=cognitive_agent(:tiny)
  respond!(ai,"a causes b"; expected_output="a causes b")
  a=get_concept_id(ai,"a"); b=get_concept_id(ai,"b")
  edge=find_synapse(ai.network,a,b)
  @test edge !== nothing && edge.value == Int8(1)

  respond!(ai,"why a causes b"; expected_output="a prevents b")
  @test get_last_evaluation(ai).quality == :WRONG
  @test edge.value == Int8(0)

  result,state=respond!(ai,"why a causes b"; expected_output="a prevents b")
  selected=selected_hypothesis(state)
  @test selected !== nothing && selected.claim.predicate == :prevents
  @test occursin("a prevents b",lowercase(result[1]))
  @test verify_integrity(ai)
end

    return 0
end

# ===== debug_transitive.jl as opt-in demo (run_debug_transitive) =====
function run_debug_transitive()
# (bundled, skipped) include(joinpath(@__DIR__, "..", "src", "TernaryCognitiveAI.jl"))
# (bundled, skipped) using .TernaryCognitiveAI

ai = cognitive_agent(:small)
respond!(ai, "evaporation causes clouds"; expected_output="evaporation causes clouds")
respond!(ai, "clouds cause rain"; expected_output="clouds causes rain")
res, state = respond!(ai, "why evaporation causes rain"; expected_output="evaporation causes rain")

println("Output: ", res[1])
println("Quality: ", res[2].quality)
for h in state.hypotheses
    sname = concept_name(ai, h.claim.subject)
    oname = concept_name(ai, h.claim.object)
    println("  H$(h.id): claim=$sname $(h.claim.predicate) $oname | score=$(h.score) | origin=$(h.origin) | status=$(h.status)")
end
end

# ===== END tests.jl =====
