@testset "sparse ternary network" begin
    ai = cognitive_agent(:tiny)
    a = TernaryCognitiveAI.concept!(ai, "alpha")
    b = TernaryCognitiveAI.concept!(ai, "beta")
    e = TernaryCognitiveAI.connect!(ai.network, a, b, 0)
    @test e.value == Int8(0)
    e.value = Int8(1)
    verify_integrity(ai)
    active, pathways = TernaryCognitiveAI.activate!(ai.network, Int32[a], ai.config.eligibility)
    @test a in active
    @test !isempty(pathways) || e.eligibility >= 0f0
    @test floating_learned_weight_count(ai) == 0
end
