# rewards.jl — Behavioral reward signals from quality taxonomy (spec §13–§15).
#
# Rewards are never gradients. They are outcome-quality magnitudes that later
# drive eligibility-gated ternary plasticity.

function reward_from_quality(config::CognitiveConfig, quality::Symbol)
    signal = get(config.quality_signals, quality, 0f0)
    return config.reward_enabled ? max(0f0, signal) : 0f0
end

function is_positive_quality(quality::Symbol)
    return quality in (:EXCELLENT, :GOOD, :PARTIAL, :REQUESTED_REPETITION)
end

function apply_reward_gate(config::CognitiveConfig, reward::Float32)
    return config.reward_enabled ? reward : 0f0
end
