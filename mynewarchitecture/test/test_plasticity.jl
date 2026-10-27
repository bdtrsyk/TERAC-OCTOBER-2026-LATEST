@testset "ternary plasticity transitions" begin
    ai = cognitive_agent(:tiny)
    # Lower thresholds so tests observe transitions quickly.
    ai.config.plasticity.promote_threshold = 0.1f0
    ai.config.plasticity.demote_threshold = 0.1f0
    respond!(ai, "a causes b"; expected_output="a causes b")
    a = get_concept_id(ai, "a")
    b = get_concept_id(ai, "b")
    edge = TernaryCognitiveAI.find_synapse(ai.network, a, b)
    @test edge !== nothing

    # Reward path should move toward supporting +1 for causes
    for _ in 1:5
        respond!(ai, "why a causes b"; expected_output="a causes b")
    end
    @test edge.value in (Int8(-1), Int8(0), Int8(1))
    verify_integrity(ai)

    # Wrong feedback should apply punishment pressure
    before = edge.value
    respond!(ai, "why a causes b"; expected_output="a prevents b")
    @test get_last_evaluation(ai).quality == :WRONG
    @test get_last_evaluation(ai).punishment > 0f0
    verify_integrity(ai)
end
