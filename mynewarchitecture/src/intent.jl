# intent.jl — Explicit user-intent representation (spec §17).
# Parsing lives in language.jl (`parse_intent`, `detect_repetition_request`).

function intent_requests_repetition(intent::UserIntent)
    return intent.requested_repetition || intent.goal == :repeat
end

function intent_summary(intent::UserIntent)
    return Dict{Symbol,Any}(
        :goal => intent.goal,
        :requested_repetition => intent.requested_repetition,
        :requested_repetition_count => intent.requested_repetition_count,
        :requested_transformation => intent.requested_transformation,
        :requested_structure => intent.requested_structure,
        :constraints => copy(intent.constraints),
        :requested_quantity => intent.requested_quantity,
        :exclusions => copy(intent.exclusions),
    )
end
