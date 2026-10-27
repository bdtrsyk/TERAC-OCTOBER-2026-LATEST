# semantic_memory.jl — Concepts and relations (spec §21–§23).
# Construction/retrieval live in concepts.jl and memory_systems.jl.

semantic_concept_count(agent::CognitiveAgent) = length(agent.semantic.concepts)
semantic_relation_count(agent::CognitiveAgent) = length(agent.semantic.relations)

function list_concept_names(agent::CognitiveAgent)
    return sort([c.name for c in values(agent.semantic.concepts)])
end
