# learning.jl — Learning pipeline orchestration (spec §57).
#
# Episode learning is implemented across:
#   evaluator.jl          → quality classification
#   rewards.jl            → positive signal gating
#   punishment.jl         → negative signal gating
#   eligibility.jl        → temporal traces
#   credit_assignment.jl  → pathway credit
#   plasticity.jl         → ternary transitions + episode completion
#
# This file intentionally contains no alternate learner and no optimizer.

"""Human / interactive feedback enters the same reward/punishment pipeline (spec §60)."""
function learn_from_feedback!(agent::CognitiveAgent, quality::Symbol;
                              correction::Union{Nothing,String}=nothing)
    quality in QUALITY_CLASSES || error("invalid evaluator label: $quality")
    state = agent.last_state
    state === nothing && error("no prior episode to reinforce; run respond! first")
    output = isempty(agent.last_output) ? render_output(agent, state) : agent.last_output
    evaluation = evaluate_output(agent, state, output;
                                 expected_output=correction,
                                 expected_quality=quality,
                                 evaluator_mode=:human_feedback)
    changes = apply_plasticity!(agent, state, evaluation)
    agent.last_evaluation = evaluation
    if !isempty(agent.episodic.episodes)
        ep = agent.episodic.episodes[end]
        ep.evaluation = evaluation
        ep.reward = evaluation.reward
        ep.punishment = evaluation.punishment
        ep.synaptic_changes = changes
        ep.success = evaluation.quality in (:EXCELLENT, :GOOD, :REQUESTED_REPETITION)
    end
    return evaluation, changes
end

reward!(agent::CognitiveAgent; quality::Symbol=:GOOD) =
    learn_from_feedback!(agent, quality)

punish!(agent::CognitiveAgent; quality::Symbol=:WRONG, correction=nothing) =
    learn_from_feedback!(agent, quality; correction=correction)
