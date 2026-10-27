
# evaluator.jl — Behavioral, intent-aware, multi-mode outcome evaluation.

_canonical_text(text::AbstractString) = join(split(lowercase(replace(strip(String(text)), r"[^a-z0-9_\-]+" => " "))), " ")

function _expected_relation(text::String)
    candidate = replace(normalize(text), r"^(conclusion\s*:\s*|yes\s*\.\s*)" => "")
    first_sentence = first(split(candidate, '.'; limit=2))
    return match_relation(strip(first_sentence))
end

function repetition_detected(agent::CognitiveAgent, state::CognitiveState, output::String)
    isempty(agent.last_output) && return false
    if state.intent.requested_repetition
        count = max(1, state.intent.requested_repetition_count)
        expected = join(fill(agent.last_output, count), "\n")
        return _canonical_text(output) == _canonical_text(expected)
    end

    segments = filter(!isempty, _canonical_text.(segment_sentences(replace(output, '\n' => '.'))))
    internally_repeated = length(segments) > 1 && length(unique(segments)) < length(segments)
    same_input = !isempty(agent.episodic.episodes) &&
                 _canonical_text(agent.episodic.episodes[end].input) == _canonical_text(state.input)
    repeated_previous = same_input && _canonical_text(output) == _canonical_text(agent.last_output)
    return internally_repeated || repeated_previous
end

function _structural_correctness(agent::CognitiveAgent, state::CognitiveState, expected::String)
    h = selected_hypothesis(state)
    h === nothing && return 0f0
    relation = _expected_relation(expected)
    relation === nothing && return 0f0
    predicate, subject, object = relation
    return (_concept_key(subject) == concept_name(agent, h.claim.subject) &&
            predicate == h.claim.predicate &&
            _concept_key(object) == concept_name(agent, h.claim.object)) ? 1f0 : 0f0
end

function _textual_correctness(output::String, expected::String; exact::Bool=false)
    actual = _canonical_text(output)
    target = _canonical_text(expected)
    exact && return actual == target ? 1f0 : 0f0
    occursin(target, actual) && return 1f0
    expected_tokens = Set(split(target))
    isempty(expected_tokens) && return 0f0
    output_tokens = Set(split(actual))
    return Float32(length(intersect(expected_tokens, output_tokens)) / length(expected_tokens))
end

function _constraint_score(state::CognitiveState, output::String)
    checks = Bool[]
    reasons = String[]
    if state.intent.requested_structure == "json"
        valid = startswith(strip(output), "{") && endswith(strip(output), "}")
        push!(checks, valid); !valid && push!(reasons, "requested JSON structure was not produced")
    elseif state.intent.requested_structure == "list"
        lines = filter(!isempty, strip.(split(output, '\n')))
        valid = !isempty(lines) && all(startswith(line, "- ") for line in lines)
        push!(checks, valid); !valid && push!(reasons, "requested list structure was not produced")
    end
    for exclusion in state.intent.exclusions
        valid = !occursin(_canonical_text(exclusion), _canonical_text(output))
        push!(checks, valid); !valid && push!(reasons, "excluded content appeared: $exclusion")
    end
    if "concise" in state.intent.constraints || "concise" in state.constraints
        valid = length(segment_sentences(output)) <= 1
        push!(checks, valid); !valid && push!(reasons, "requested concise form was exceeded")
    end
    score = isempty(checks) ? 1f0 : Float32(count(identity, checks) / length(checks))
    return score, reasons
end

function _evaluation_mode(mode::Symbol, expected_output::Union{Nothing,String}, state::Union{Nothing,CognitiveState}=nothing)
    canonical = Symbol(replace(string(mode), "-" => "_"))
    canonical != :auto && return canonical
    if expected_output === nothing
        if state !== nothing && (!isempty(state.intent.constraints) || !isempty(state.intent.exclusions) || !isempty(state.intent.requested_structure))
            return :constraint_based
        end
        return :rule_based
    end
    return _expected_relation(expected_output) === nothing ? :rule_based : :structural
end

"""Evaluate behavior and retain both actual and expected quality (spec §13–§19, §59, §103)."""
function evaluate_output(agent::CognitiveAgent, state::CognitiveState, output::String;
                         expected_output::Union{Nothing,String}=nothing,
                         expected_quality::Union{Nothing,Symbol}=nothing,
                         evaluator_mode::Symbol=:auto)
    expected_quality !== nothing && !(expected_quality in QUALITY_CLASSES) &&
        error("unknown expected quality: $expected_quality")
    mode = _evaluation_mode(evaluator_mode, expected_output, state)
    mode in (:exact_match, :rule_based, :constraint_based, :structural,
             :semantic_structure, :human_feedback) || error("unknown evaluator mode: $evaluator_mode")

    requested = state.intent.requested_repetition
    repeated = repetition_detected(agent, state, output)
    h = selected_hypothesis(state)
    selected_consistent = requested || (h !== nothing && h.status ∉ (:rejected, :contradicted))
    consistency = selected_consistent && isempty(state.contradictions) ? 1f0 :
                  selected_consistent ? 0.7f0 : 0f0
    relevance = requested ? 1f0 : h === nothing ? 0f0 : h.scores.goal_relevance
    constraint, constraint_reasons = _constraint_score(state, output)

    correctness = if requested
        repeated ? 1f0 : 0f0
    elseif expected_output === nothing
        h === nothing ? 0f0 : clamp(state.confidence, 0f0, 1f0)
    elseif mode == :exact_match
        _textual_correctness(output, expected_output; exact=true)
    elseif mode in (:structural, :semantic_structure)
        structural = _structural_correctness(agent, state, expected_output)
        max(structural, _textual_correctness(output, expected_output))
    else
        _textual_correctness(output, expected_output)
    end

    instruction_ok = state.intent.goal != :transform &&
                     !(state.intent.requested_repetition && isempty(agent.last_output))
    quality = if mode == :human_feedback
        expected_quality === nothing && error("human-feedback evaluation requires expected_quality")
        expected_quality
    elseif requested && repeated
        :REQUESTED_REPETITION
    elseif expected_output !== nothing && correctness < 0.999f0
        # Explicit feedback mismatch outranks repetition: punishment without
        # the correction would discard the teaching signal (plasticity uses
        # expected_output as the correction alternative only on :WRONG).
        :WRONG
    elseif repeated
        :REPETITIVE
    elseif !instruction_ok
        :INSTRUCTION_FAILURE
    elseif constraint < 1f0
        :CONSTRAINT_FAILURE
    elseif !isempty(state.contradictions) && !selected_consistent
        :CONTRADICTORY
    elseif h === nothing
        :INCOMPLETE
    elseif relevance < 0.25f0
        :IRRELEVANT
    elseif correctness >= 0.999f0 && state.confidence >= 0.80f0
        :EXCELLENT
    elseif correctness >= 0.65f0 && state.confidence >= 0.45f0
        :GOOD
    elseif correctness > 0f0 || state.confidence > 0f0
        :PARTIAL
    else
        :NEUTRAL
    end

    signal = get(agent.config.quality_signals, quality, 0f0)
    reward = signal > 0f0 && agent.config.reward_enabled ? signal : 0f0
    punishment = signal < 0f0 && agent.config.punishment_enabled ? -signal : 0f0
    reasons = String["actual quality=$quality", "intent=$(state.intent.goal)", "evaluator mode=$mode"]
    append!(reasons, constraint_reasons)
    repeated && push!(reasons, requested ? "repetition explicitly requested and fulfilled" :
                                       "semantically equivalent content repeated without a repetition instruction")
    expected_output !== nothing && push!(reasons, correctness >= 0.999f0 ?
        "behavioral target matched" : "behavioral target did not match")
    !selected_consistent && push!(reasons, "no consistent selected hypothesis")
    signal > 0f0 && !agent.config.reward_enabled && push!(reasons, "reward suppressed by config (reward_enabled=false)")
    signal < 0f0 && !agent.config.punishment_enabled && push!(reasons, "punishment suppressed by config (punishment_enabled=false)")
    if expected_quality !== nothing
        push!(reasons, expected_quality == quality ? "actual quality matched expected quality" :
              "quality discrepancy: expected $expected_quality, actual $quality")
    end

    return EvaluationResult(quality, state.intent.goal, reward, punishment, repeated, requested,
                            correctness, relevance, consistency, constraint, mode,
                            expected_quality, reasons)
end

