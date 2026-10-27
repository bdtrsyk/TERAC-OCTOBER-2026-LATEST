@testset "compositional generalization" begin
    ai = cognitive_agent(:tiny)
    # Train A-B and B-C, test A-C (never seen as a direct assertion target during first two steps)
    respond!(ai, "a causes b"; expected_output="a causes b")
    respond!(ai, "b causes c"; expected_output="b causes c")
    (result, state) = respond!(ai, "why a causes c"; expected_output="a causes c")
    @test occursin("a causes c", lowercase(result[1])) ||
          any(h -> h.claim.predicate == :causes &&
                   TernaryCognitiveAI.concept_name(ai, h.claim.subject) == "a" &&
                   TernaryCognitiveAI.concept_name(ai, h.claim.object) == "c",
              state.hypotheses)
end
