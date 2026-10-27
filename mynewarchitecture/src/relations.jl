# relations.jl — Relation helpers shared by concept graph and novelty operators.
# Core mutation APIs live in concepts.jl (`add_relation!`, `relation_exists`, …).

function relation_triple_string(agent::CognitiveAgent, r::CognitiveRelation)
    return "$(concept_name(agent, r.subject)) $(r.predicate) $(concept_name(agent, r.object))"
end

function opposing_relation(predicate::Symbol)
    predicate in OPPOSING_PREDICATES && return predicate
    return get(Dict(
        :causes => :prevents,
        :supports => :contradicts,
        :enables => :blocks,
        :requires => :prevents,
    ), predicate, :contradicts)
end
