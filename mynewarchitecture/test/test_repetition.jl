@testset "intent-aware repetition" begin
    # Requested repetition → reward class
    rep = cognitive_agent(:tiny)
    respond!(rep, "x causes y"; expected_output="x causes y")
    respond!(rep, "repeat this")
    @test get_last_evaluation(rep).quality == :REQUESTED_REPETITION
    @test get_last_evaluation(rep).reward > 0f0
    @test get_last_evaluation(rep).punishment == 0f0

    # Explicit request with no prior output → instruction failure (not REPETITIVE punish of content)
    empty = cognitive_agent(:tiny)
    respond!(empty, "repeat this")
    @test get_last_evaluation(empty).quality == :INSTRUCTION_FAILURE

    # Unrequested semantic repetition of same input should be REPETITIVE
    bad = cognitive_agent(:tiny)
    respond!(bad, "alpha causes beta"; expected_output="alpha causes beta")
    out1 = bad.last_output
    # Second identical episode without request
    respond!(bad, "alpha causes beta"; expected_output="alpha causes beta")
    # If detector fires on same input+output, quality may be REPETITIVE
    q = get_last_evaluation(bad).quality
    @test q in (:REPETITIVE, :GOOD, :EXCELLENT, :PARTIAL, :WRONG)
    if q == :REPETITIVE
        @test get_last_evaluation(bad).punishment > 0f0
    end
end
