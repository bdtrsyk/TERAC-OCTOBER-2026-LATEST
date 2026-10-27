# episodic_memory.jl — Past experience store (spec §22, §39, §117).

episodic_count(agent::CognitiveAgent) = length(agent.episodic.episodes)

function last_episode(agent::CognitiveAgent)
    isempty(agent.episodic.episodes) && return nothing
    return agent.episodic.episodes[end]
end

function episodes_with_quality(agent::CognitiveAgent, quality::Symbol)
    return [ep for ep in agent.episodic.episodes if ep.evaluation !== nothing && ep.evaluation.quality == quality]
end
