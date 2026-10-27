
# output_generation.jl — Rule-driven language generated from structured state.


predicate_text(predicate::Symbol) = replace(string(predicate), "_" => " ")

function _json_escape(text::String)
    replace(text, '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n", '\r' => "\\r", '\t' => "\\t")
end

function _format_conclusion(text::String, state::CognitiveState)
    if state.intent.requested_structure == "json"
        return "{\"conclusion\":\"$(_json_escape(text))\",\"confidence\":$(round(state.confidence; digits=3))}"
    elseif state.intent.requested_structure == "list"
        return "- $text"
    end
    return text
end

"""Render an answer solely from the selected structured hypothesis and intent."""
function render_output(agent::CognitiveAgent, state::CognitiveState)
    if state.intent.requested_repetition
        isempty(agent.last_output) && return "There is no previous conclusion to repeat."
        count = max(1, state.intent.requested_repetition_count)
        return join(fill(agent.last_output, count), "\n")
    end

    h = selected_hypothesis(state)
    h === nothing && return "Insufficient supported evidence; more information is required."

    subject = concept_name(agent, h.claim.subject)
    object = concept_name(agent, h.claim.object)
    relation = predicate_text(h.claim.predicate)
    conclusion = "$subject $relation $object"

    if state.intent.goal == :verify
        conclusion = "Yes. $conclusion."
    else
        conclusion = "Conclusion: $conclusion."
    end

    if h.claim.predicate == :causes && h.origin == :deduction
        supported, bridge = transitive_causal_support(agent, h.claim)
        if supported
            middle = concept_name(agent, bridge)
            conclusion *= " Evidence: $subject causes $middle; $middle causes $object."
        end
    elseif !isempty(h.assumptions) && state.intent.goal == :explain
        conclusion *= " Basis: $(join(h.assumptions, "; "))."
    end

    return _format_conclusion(conclusion, state)
end

