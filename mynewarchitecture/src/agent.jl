
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

