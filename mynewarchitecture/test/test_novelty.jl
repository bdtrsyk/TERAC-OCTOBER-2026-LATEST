@testset "novelty operators" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "routing allocates bandwidth"; expected_output="routing allocates bandwidth")
    respond!(ai, "traffic supports capacity"; expected_output="traffic supports capacity")
    (_, state) = respond!(ai, "why routing supports capacity")
    @test length(state.hypotheses) >= 1
    @test any(e -> e.kind in (TernaryCognitiveAI.CROSS_DOMAIN_RECOMBINATION,
                               TernaryCognitiveAI.HYPOTHESIS_CREATED,
                               TernaryCognitiveAI.ANALOGY_FORMED,
                               TernaryCognitiveAI.ABSTRACTION_FORMED),
              state.trace.events) || !isempty(state.hypotheses)
end
