module TriTierAdapter

"""Stable Julia math boundary for TriTier.

The Rust runtime owns all block lifetime and versioning. Julia receives an
immutable copied view of a block and produces a candidate delta; it cannot
access the persistent baseline or GPU allocator directly.
"""

export evaluate_candidate, ternary_delta, tritier_abi_version

const tritier_abi_version = v"1.0.0"

"""Return a discrete objective score for a candidate delta.
This deliberately uses integer arithmetic to preserve the existing model's
ternary learning semantics; replace the objective body with TernaryCognitiveAI
evaluation when wiring a trained model."""
function evaluate_candidate(before::Vector{Int8}, delta::Vector{Int8})::Int64
    length(before) == length(delta) || throw(ArgumentError("length mismatch"))
    score::Int64 = 0
    @inbounds for i in eachindex(before, delta)
        score += abs(Int64(delta[i])) <= 1 ? 1 : -1000
    end
    return score
end

"""Generate a bounded ternary delta without mutating the source parameter view."""
function ternary_delta(source::Vector{Int8}, signal::Int8)::Vector{Int8}
    signal in Int8[-1, 0, 1] || throw(ArgumentError("signal must be ternary"))
    result = Vector{Int8}(undef, length(source))
    @inbounds for i in eachindex(source)
        result[i] = signal == 0 ? Int8(0) : (source[i] == 0 ? signal : Int8(0))
    end
    return result
end

end
