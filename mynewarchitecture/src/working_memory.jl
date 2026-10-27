# working_memory.jl — Bounded structured working memory (spec §22).
# Mutation API: `remember!` in memory_systems.jl.

working_memory_size(agent::CognitiveAgent) = length(agent.working.items)
working_memory_capacity(agent::CognitiveAgent) = agent.working.capacity

function clear_working_memory!(agent::CognitiveAgent)
    empty!(agent.working.items)
    return agent
end
