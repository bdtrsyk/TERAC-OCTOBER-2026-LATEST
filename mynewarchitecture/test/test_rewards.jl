@testset "reward and punishment signals" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "x causes y"; expected_output="x causes y")
    ev = get_last_evaluation(ai)
    @test ev.reward > 0f0 || ev.quality in (:GOOD, :EXCELLENT, :PARTIAL)
    before = synaptic_snapshot(ai)
    respond!(ai, "why x causes y"; expected_output="x prevents y")
    ev2 = get_last_evaluation(ai)
    @test ev2.quality == :WRONG
    @test ev2.punishment > 0f0
    @test verify_integrity(ai)
end
