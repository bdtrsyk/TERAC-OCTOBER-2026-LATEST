# punishment.jl — Behavioral punishment signals (spec §13–§16).
#
# Unrequested repetition and WRONG outcomes produce punishment.
# Explicitly requested repetition must NOT be punished.

function punishment_from_quality(config::CognitiveConfig, quality::Symbol)
    signal = get(config.quality_signals, quality, 0f0)
    return config.punishment_enabled ? max(0f0, -signal) : 0f0
end

function is_negative_quality(quality::Symbol)
    return quality in (:WRONG, :REPETITIVE, :IRRELEVANT, :CONTRADICTORY,
                       :INCOMPLETE, :CONSTRAINT_FAILURE, :INSTRUCTION_FAILURE)
end

function apply_punishment_gate(config::CognitiveConfig, punishment::Float32)
    return config.punishment_enabled ? punishment : 0f0
end
