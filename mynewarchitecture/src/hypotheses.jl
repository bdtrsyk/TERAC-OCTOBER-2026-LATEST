
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

