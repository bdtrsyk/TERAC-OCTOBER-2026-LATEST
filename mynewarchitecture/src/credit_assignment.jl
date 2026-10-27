# credit_assignment.jl — Structured pathway credit (spec §12, §44).
#
# Credit is local to activated pathways with temporal + causal relevance.
# Never indiscriminately reward/punish the whole network.

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
