# plasticity.jl — Discrete ternary synaptic transitions (spec §10, §42, §44).
#
# Synapses transition in {-1,0,+1} under eligibility-gated reward/punishment.
# There is no float shadow weight and no gradient step.

function _step_toward(value::Int8, target::Int8)
    @assert is_ternary(value) && is_ternary(target) "non-ternary plasticity input: $value → $target"
    value == target ? value : Int8(Int(value) + sign(Int(target) - Int(value)))
end

"""Apply one-step ternary transitions to eligible pathways; never mutate a float weight."""
function apply_plasticity!(agent::CognitiveAgent, state::CognitiveState,
                           evaluation::EvaluationResult)
    assignments = assign_credit(agent, state, evaluation)
    signal = evaluation.reward > 0f0 ? evaluation.reward : -evaluation.punishment
    changes = SynapticChange[]
    cfg = agent.config.plasticity
    selected = selected_hypothesis(state)

    if signal != 0f0 && agent.config.ternary_enabled
        for assignment in assignments
            assignment.credit == NEUTRAL_CREDIT && continue
            edge = find_synapse(agent.network, assignment.source, assignment.target)
            edge === nothing && continue
            old = edge.value
            desired = relation_polarity(agent, edge, selected)

            signal > 0f0 ? (edge.times_rewarded += 1) : (edge.times_punished += 1)
            target = signal > 0f0 ? desired : old == 0 ? Int8(-desired) : Int8(0)
            if old == target
                edge.pressure = 0f0
                signal > 0f0 && (edge.stability = min(cfg.max_stability,
                    edge.stability + cfg.stability_gain * abs(signal) * assignment.eligibility))
                continue
            end

            sign(edge.pressure) != sign(signal) && (edge.pressure = 0f0)
            multiplier = 0.75f0 + 0.75f0 * assignment.causal_relevance +
                         0.25f0 * assignment.temporal_relevance
            edge.pressure += signal * assignment.eligibility * multiplier
            resistance = old == 0 ? 0f0 : edge.stability
            threshold = (signal > 0f0 ? cfg.promote_threshold : cfg.demote_threshold) *
                        (1f0 + resistance)
            explored = evaluation.quality == :PARTIAL && old == 0 &&
                       abs(edge.pressure) >= threshold / 2f0 && rand(agent.rng) < cfg.exploration_prob
            if abs(edge.pressure) >= threshold || explored
                edge.value = assert_ternary(_step_toward(old, target), "plasticity transition")
                edge.pressure = 0f0
                edge.last_changed_episode = Int32(agent.episodic.next_id)
                if signal > 0f0
                    edge.stability = min(cfg.max_stability,
                        edge.stability + cfg.stability_gain * abs(signal) * assignment.eligibility)
                else
                    edge.stability = max(0f0,
                        edge.stability - cfg.stability_decay * abs(signal) * assignment.eligibility)
                end
                reason = explored ? "bounded exploration toward context-compatible state" :
                         signal > 0f0 ? "reward reinforced the successful ternary configuration" :
                                        "punishment moved the responsible pathway away from its failed state"
                change = SynapticChange(edge.source, edge.target, old, edge.value,
                                        assignment.eligibility, assignment.credit, reason)
                push!(changes, change)
                emit!(state.trace, SYNAPSE_CHANGED; source=edge.source, target=edge.target,
                      old=old, new=edge.value, eligibility=edge.eligibility,
                      credit=assignment.credit, reason=reason)
            end
        end
    end

    decay_eligibility!(agent.network, agent.config.eligibility)
    agent.config.debug_invariants && assert_network_invariants(agent.network)
    return changes
end

function complete_episode!(agent::CognitiveAgent, state::CognitiveState;
                           expected_output::Union{Nothing,String}=nothing,
                           expected_quality::Union{Nothing,Symbol}=nothing,
                           evaluator_mode::Symbol=:auto)
    output = render_output(agent, state)
    emit!(state.trace, OUTPUT_GENERATED; output=output)
    evaluation = evaluate_output(agent, state, output; expected_output=expected_output,
                                 expected_quality=expected_quality, evaluator_mode=evaluator_mode)
    emit!(state.trace, EVALUATION_COMPLETED; quality=evaluation.quality,
          expected_quality=evaluation.expected_quality, mode=evaluation.evaluator_mode,
          reasons=evaluation.reasons)
    evaluation.reward > 0f0 && emit!(state.trace, REWARD_ISSUED; amount=evaluation.reward)
    evaluation.punishment > 0f0 && emit!(state.trace, PUNISHMENT_ISSUED; amount=evaluation.punishment)
    changes = apply_plasticity!(agent, state, evaluation)

    h = selected_hypothesis(state)
    positive = evaluation.reward > 0f0 && evaluation.reward > evaluation.punishment
    nonneutral = evaluation.reward > 0f0 || evaluation.punishment > 0f0
    memory_added = Int32[]
    if h !== nothing
        nonneutral && credit_concepts!(agent, h.supporting_concepts, positive)
        nonneutral && note_strategy_outcome!(agent, h.strategy, positive;
                                             episode=Int32(agent.episodic.next_id))
        if positive && h.origin != :direct && !relation_exists(agent, h.claim)
            add_relation_ids!(agent, h.claim.subject, h.claim.predicate, h.claim.object;
                              provenance=:deduced, episode=Int32(agent.episodic.next_id))
            append!(memory_added, Int32[h.claim.subject, h.claim.object])
        end
    end
    if evaluation.punishment > 0f0
        strategy = h === nothing ? :none : h.strategy
        alternative = evaluation.quality == :WRONG ? expected_output : nothing
        record_failure!(agent, agent.episodic.next_id, state.input,
                        join(evaluation.reasons, "; "), strategy;
                        claim=h === nothing ? nothing : h.claim,
                        alternative=alternative)
    end

    emit!(state.trace, MEMORY_UPDATED; episode=agent.episodic.next_id,
          learned_concepts=memory_added, failure_stored=evaluation.punishment > 0f0)
    emit!(state.trace, EPISODE_STORED; episode=agent.episodic.next_id)
    success = evaluation.quality in (:EXCELLENT, :GOOD, :REQUESTED_REPETITION)
    ep = CognitiveEpisode(agent.episodic.next_id, state.input, state.goal, state.intent,
        copy(state.concepts), copy(state.retrieved_concepts), copy(state.phase_history),
        [hypothesis.id for hypothesis in state.hypotheses],
        [hypothesis.id for hypothesis in state.hypotheses if hypothesis.status == :rejected],
        state.current_hypothesis, copy(state.trace.events), output, evaluation, changes,
        memory_added, evaluation.reward, evaluation.punishment,
        h === nothing ? :none : h.strategy, state.budget_exhausted, success)
    store_episode!(agent, ep)

    agent.last_output = output
    agent.last_evaluation = evaluation
    agent.last_state = state
    metrics = agent.metrics
    metrics.episodes += 1
    metrics.quality_counts[evaluation.quality] = get(metrics.quality_counts, evaluation.quality, 0) + 1
    metrics.total_reward += evaluation.reward
    metrics.total_punishment += evaluation.punishment
    metrics.ternary_transitions += length(changes)
    push!(metrics.synapses_changed_per_episode, length(changes))
    push!(metrics.reasoning_steps, state.steps_used)
    push!(metrics.reward_curve, evaluation.reward)
    push!(metrics.punishment_curve, evaluation.punishment)
    push!(metrics.success_curve, ep.success)
    state.budget_exhausted && (metrics.budget_exhaustions += 1)
    metrics.contradictions_detected += length(state.contradictions)
    metrics.hypotheses_rejected += count(hx -> hx.status == :rejected, state.hypotheses)
    h !== nothing && h.origin in _NOVEL_ORIGINS && positive && (metrics.novel_recombinations += 1)
    return output, evaluation, ep
end

function respond!(agent::CognitiveAgent, input::String;
                  expected_output::Union{Nothing,String}=nothing,
                  expected_quality::Union{Nothing,Symbol}=nothing,
                  evaluator_mode::Symbol=:auto, domain::String="general")
    state = reason!(agent, input; domain=domain)
    result = complete_episode!(agent, state; expected_output=expected_output,
                               expected_quality=expected_quality, evaluator_mode=evaluator_mode)
    return result, state
end
