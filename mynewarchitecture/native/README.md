# native/README.md
# Optional native acceleration backends for TernaryCognitiveAI.
#
# Julia remains the architectural source of truth.
# Native code may accelerate packed ternary ops, eligibility updates, or
# graph traversal, but must not alter the cognitive model or introduce
# floating-point learned weights.

See per-language README files for ABI notes.
