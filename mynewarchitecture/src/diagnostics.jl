
# diagnostics.jl — Machine-readable introspection and change auditing.
get_active_units(ai::CognitiveAgent) = active_unit_ids(ai.network)
get_active_synapses(ai::CognitiveAgent) = [(e.source,e.target,e.value,e.eligibility) for es in values(ai.network.outgoing) for e in es if e.eligibility > 0f0]
get_working_memory(ai::CognitiveAgent) = copy(ai.working.items)
get_retrieved_memories(ai::CognitiveAgent) = ai.last_state === nothing ? Int[] : copy(ai.last_state.retrieved_memories)
get_hypotheses(ai::CognitiveAgent) = ai.last_state === nothing ? CognitiveHypothesis[] : copy(ai.last_state.hypotheses)
get_current_goal(ai::CognitiveAgent) = ai.last_state === nothing ? "" : ai.last_state.goal
get_reasoning_state(ai::CognitiveAgent) = ai.last_state
get_last_evaluation(ai::CognitiveAgent) = ai.last_evaluation
get_last_reward(ai::CognitiveAgent) = ai.last_evaluation === nothing ? 0f0 : ai.last_evaluation.reward
get_last_punishment(ai::CognitiveAgent) = ai.last_evaluation === nothing ? 0f0 : ai.last_evaluation.punishment
get_last_synaptic_changes(ai::CognitiveAgent) = isempty(ai.episodic.episodes) ? SynapticChange[] : ai.episodic.episodes[end].synaptic_changes

synaptic_snapshot(ai::CognitiveAgent) = Dict((e.source,e.target)=>e.value for es in values(ai.network.outgoing) for e in es)
function synaptic_diff(before::Dict, after::CognitiveAgent)
    [SynapticChange(e.source,e.target,get(before,(e.source,e.target),Int8(0)),e.value,e.eligibility,credit_level(e),"snapshot diff")
     for es in values(after.network.outgoing) for e in es if get(before,(e.source,e.target),Int8(0)) != e.value]
end
memory_snapshot(ai::CognitiveAgent) = (concepts=Set(keys(ai.semantic.concepts)), relations=Set(ai.semantic.relations), episodes=length(ai.episodic.episodes))
function memory_diff(before, after::CognitiveAgent)
    (added_concepts=collect(setdiff(Set(keys(after.semantic.concepts)),before.concepts)),
     new_relations=collect(setdiff(Set(after.semantic.relations),before.relations)), new_episodes=length(after.episodic.episodes)-before.episodes)
end
function training_report(ai::CognitiveAgent)
    m=ai.metrics; d=ternary_distribution(ai.network)
    (episodes=m.episodes, quality=copy(m.quality_counts), total_reward=m.total_reward, total_punishment=m.total_punishment,
     synapses=d, average_changed=isempty(m.synapses_changed_per_episode) ? 0.0 : sum(m.synapses_changed_per_episode)/length(m.synapses_changed_per_episode),
     self_corrections=m.self_corrections, novelty=m.novel_recombinations, peak_bytes=m.peak_bytes, runtime_seconds=m.runtime_seconds)
end

