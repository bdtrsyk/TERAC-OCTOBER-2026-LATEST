@testset "architectural integrity" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "p causes q"; expected_output="p causes q")
    @test verify_integrity(ai)
    @test TernaryCognitiveAI.floating_learned_weight_count(ai) == 0

    # No forbidden float learned field on synapses
    for edges in values(ai.network.outgoing), e in edges
        @test e.value isa Int8
        @test e.value in (Int8(-1), Int8(0), Int8(1))
    end

    ok, findings = TernaryCognitiveAI.static_source_audit()
    @test ok || isempty(filter(f -> occursin(r"\bAdam\b|\bFlux\b|cross.?entropy|OpenAI", f), findings))

    # Audit mode must pass on a fresh agent
    @test run_architecture_audit(ai; io=devnull) == 0

    # Learned ternary state guides selection after punishment (from legacy suite)
    a = get_concept_id(ai, "p")
    b = get_concept_id(ai, "q")
    edge = TernaryCognitiveAI.find_synapse(ai.network, a, b)
    @test edge !== nothing
    ai.config.plasticity.promote_threshold = 0.05f0
    ai.config.plasticity.demote_threshold = 0.05f0
    for _ in 1:4
        respond!(ai, "why p causes q"; expected_output="p causes q")
    end
    respond!(ai, "why p causes q"; expected_output="p prevents q")
    @test get_last_evaluation(ai).quality == :WRONG
end
