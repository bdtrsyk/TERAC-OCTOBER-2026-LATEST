
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

