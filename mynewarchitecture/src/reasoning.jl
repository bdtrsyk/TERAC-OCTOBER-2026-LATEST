
# reasoning.jl — The explicit reasoning state machine (spec §24, §33, §34).
#
# This is the controller that ties the engine together: it walks the named
# ReasoningPhase states, activating the ternary network, retrieving memory,
# generating and testing hypotheses, self-correcting on contradiction, and
# selecting a conclusion under the reasoning budget. It performs NO learning and
# NO output rendering — those are separate concerns (evaluator/learning/output).
# The only "weights" touched are transient eligibility traces set by activation.

"""Build a fresh per-episode reasoning state (spec §23 working state)."""
function fresh_state(agent::CognitiveAgent, input::String, intent::UserIntent, goal::String)
    CognitiveState(input, goal, intent,
                   Int32[], Int32[], Tuple{Int32,Int32}[], Int32[], Int[],
                   CognitiveHypothesis[], nothing, copy(intent.constraints), String[],
                   INITIALIZE, ReasoningPhase[], Symbol[], Symbol[],
                   0f0, 0, TraceBuffer(true),
                   0, 0, 0, 0, 0, false,
                   Set{UInt64}(), Set{UInt64}())
end

"""State the explicit goal for this episode from the parsed input (spec §24 FORM_GOAL)."""
function form_goal(agent::CognitiveAgent, state::CognitiveState, parsed)
    if parsed.query_target !== nothing
        s, p, o = parsed.query_target
        return "establish whether $(concept_name(agent,s)) $p $(concept_name(agent,o))"
    elseif parsed.learned_relation !== nothing
        r = parsed.learned_relation
        return "assert $(concept_name(agent,r.subject)) $(r.predicate) $(concept_name(agent,r.object))"
    elseif state.intent.goal == :repeat
        return "repeat the previous conclusion"
    else
        return "interpret and respond to: $(state.input)"
    end
end

"""Broaden retrieval when no viable hypothesis was found (spec §34 retrieve-more)."""
function expand_retrieval!(agent::CognitiveAgent, state::CognitiveState)
    seeds = vcat(state.concepts, state.retrieved_concepts, collect(state.active_units))
    for id in retrieve_concepts(agent, seeds; limit=agent.config.retrieval_limit, trace=state.trace)
        id in state.retrieved_concepts && continue
        id in state.concepts && continue
        push!(state.retrieved_concepts, id)
    end
    return state
end

"""Commit to the best viable hypothesis using evidence, experience, and goal fit."""
function select_hypothesis!(agent::CognitiveAgent, state::CognitiveState;
                            target::Union{Nothing,Tuple{Int32,Symbol,Int32}}=nothing)
    viable = [h for h in state.hypotheses if h.status in (:supported, :active, :weak)]
    isempty(viable) && return nothing
    target_claim = target === nothing ? nothing : CognitiveRelation(target[1], target[2], target[3], :goal)
    matches(h) = target_claim !== nothing && same_claim(h.claim, target_claim)
    goal_ids = Set(state.concepts)
    function selection_score(h)
        involved = isempty(goal_ids) ? 2 : ((h.claim.subject in goal_ids) + (h.claim.object in goal_ids))
        off_topic_penalty = involved == 0 ? 1.0f0 : 0.0f0
        target_bonus = matches(h) && h.scores.experience >= 0f0 ? 0.85f0 : 0f0
        support_bonus = h.status == :supported ? 0.15f0 : 0f0
        direct_bonus = h.origin == :direct ? 0.10f0 : 0f0
        strategy_bias = 0.10f0 * (strategy_success_rate(agent, h.strategy) - 0.5f0)
        return h.score + target_bonus + support_bonus + direct_bonus + strategy_bias - off_topic_penalty
    end
    sort!(viable; by=selection_score, rev=true)
    best = viable[1]
    state.current_hypothesis = best.id
    best.status = :selected
    emit!(state.trace, HYPOTHESIS_SELECTED; hypothesis=best.id,
          score=round(best.score; digits=3), selection_score=round(selection_score(best); digits=3),
          confidence=round(best.confidence; digits=3), matches_goal=matches(best),
          synaptic_support=best.scores.synaptic_support, experience=best.scores.experience)
    return best
end

function mark_selected_pathways!(agent::CognitiveAgent, state::CognitiveState,
                                  selected::CognitiveHypothesis)
    claim = canonical_claim(selected.claim)
    output_pathway = (claim.subject, claim.object)
    supporting = Set(selected.supporting_concepts)
    pathways = Tuple{Int32,Int32}[]
    for relation in agent.semantic.relations
        canonical = canonical_claim(relation)
        pathway = (canonical.subject, canonical.object)
        pathway == output_pathway && continue
        canonical.subject in supporting && canonical.object in supporting || continue
        find_synapse(agent.network, pathway[1], pathway[2]) === nothing || push!(pathways, pathway)
    end
    output_edge = find_synapse(agent.network, output_pathway[1], output_pathway[2])
    output_edge === nothing && (output_edge = connect!(agent.network, output_pathway[1], output_pathway[2], Int8(0);
                                                   episode=Int32(agent.episodic.next_id)))
    push!(pathways, output_pathway)

    for pathway in unique(pathways)
        edge = find_synapse(agent.network, pathway[1], pathway[2])
        edge === nothing && continue
        edge.eligibility = min(agent.config.eligibility.cap,
                               edge.eligibility + agent.config.eligibility.gain)
        edge.times_activated += 1
        push!(state.active_pathways, pathway)
    end
    return state
end

"""
Internal self-evaluation (spec §37): the agent's own judgement of whether it
produced a confident, committed conclusion. This is distinct from the external
reward evaluator; it only reflects on the reasoning state and emits an event.
"""
function self_reflect!(agent::CognitiveAgent, state::CognitiveState)
    sel = state.current_hypothesis
    committed = sel !== nothing && state.confidence >= agent.config.confidence_select_threshold
    emit!(state.trace, SELF_EVALUATED; committed=committed,
          confidence=round(state.confidence; digits=3),
          viable=count(h -> h.status in (:supported, :active, :weak), state.hypotheses),
          rejected=count(h -> h.status == :rejected, state.hypotheses))
    return committed
end

"""
Run one reasoning episode over `input` and return the resulting `CognitiveState`
(spec §24). Performs the full phase walk under the reasoning budget; requested
repetition is short-circuited to the CONCLUDE phase because repeating is the
correct behaviour (intent-aware, spec §16/§17). No synapse learning happens here.
"""
function reason!(agent::CognitiveAgent, input::String; domain::String="general")
    soft_reset!(agent)
    episode = agent.episodic.next_id

    # ── INITIALIZE ──────────────────────────────────────────────────────────
    intent = parse_intent(input)
    state = fresh_state(agent, input, intent, "")
    emit!(state.trace, EPISODE_STARTED; episode=episode, input=input)
    emit!(state.trace, INTENT_DETECTED; goal=intent.goal,
          requested_repetition=intent.requested_repetition,
          repetition_count=intent.requested_repetition_count)
    transition!(state, INITIALIZE)

    # ── PARSE + REPRESENT ─────────────────────────────────────────────────────
    transition!(state, PARSE)
    parsed = parse_input!(agent, input; episode=Int32(episode), domain=domain)
    state.intent = parsed.intent
    state.constraints = copy(parsed.intent.constraints)
    state.concepts = collect(parsed.seed_ids)
    emit!(state.trace, PARSED_INPUT; concepts=state.concepts, is_query=parsed.is_query,
          learned=parsed.learned_relation !== nothing)
    remember!(agent, input, state.concepts; step=state.steps_used,
              kind = parsed.is_query ? :goal : :observation)

    transition!(state, REPRESENT)
    state.active_units, state.active_pathways =
        activate!(agent.network, state.concepts, agent.config.eligibility; hops=2, trace=state.trace)

    # ── RETRIEVE ──────────────────────────────────────────────────────────────
    transition!(state, RETRIEVE)
    state.retrieved_concepts = retrieve_concepts(agent, state.concepts; trace=state.trace)
    state.retrievals_used += 1
    recalled = retrieve_episodes(agent, state.concepts; limit=3)
    state.retrieved_memories = [ep.id for ep in recalled]
    for ep in recalled
        emit!(state.trace, MEMORY_RETRIEVED; episode=ep.id, goal=ep.goal)
    end

    # ── FORM_GOAL ─────────────────────────────────────────────────────────────
    transition!(state, FORM_GOAL)
    state.goal = form_goal(agent, state, parsed)
    emit!(state.trace, GOAL_CREATED; goal=state.goal)

    # Requested repetition: repeating is the intended, correct action (spec §16).
    if intent.requested_repetition
        transition!(state, CONCLUDE)
        state.confidence = 1f0
        transition!(state, FINISH)
        return state
    end

    # ── Reasoning cycle: GENERATE → TEST → (backtrack / retrieve-more) ──────────
    for _ in 1:4
        transition!(state, GENERATE_HYPOTHESES)
        generate_hypotheses!(agent, state)

        transition!(state, TEST)
        test_hypotheses!(agent, state)

        decision = meta_decide(agent, state)
        if decision === :backtrack
            transition!(state, DETECT_CONTRADICTION)
            transition!(state, BACKTRACK)
            self_correct!(agent, state)
            any_viable(state) && break
        elseif decision === :retrieve_more
            transition!(state, RETRIEVE)
            expand_retrieval!(agent, state)
            state.retrievals_used += 1
        elseif decision === :switch_strategy
            transition!(state, REFINE)
            expand_retrieval!(agent, state)
            state.retrievals_used += 1
        else                    # :select or :stop
            break
        end
    end

    # ── VERIFY (continuous ternary invariant, spec §4) ──────────────────────────
    transition!(state, VERIFY)
    agent.config.debug_invariants && verify_integrity(agent)

    # ── SELECT ──────────────────────────────────────────────────────────────────
    transition!(state, SELECT)
    selection_target = parsed.query_target
    if selection_target === nothing && parsed.learned_relation !== nothing
        relation = parsed.learned_relation
        selection_target = (relation.subject, relation.predicate, relation.object)
    end
    selected = select_hypothesis!(agent, state; target=selection_target)
    selected !== nothing && mark_selected_pathways!(agent, state, selected)

    # ── CONCLUDE ──────────────────────────────────────────────────────────────
    transition!(state, CONCLUDE)
    state.confidence = state_confidence(state)

    # ── SELF_EVALUATE ───────────────────────────────────────────────────────────
    transition!(state, SELF_EVALUATE)
    self_reflect!(agent, state)

    transition!(state, FINISH)
    agent.last_state = state
    return state
end

