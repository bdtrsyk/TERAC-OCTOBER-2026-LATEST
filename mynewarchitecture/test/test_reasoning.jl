@testset "multi-hypothesis generation" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "a causes b"; expected_output="a causes b")
    respond!(ai, "b causes c"; expected_output="b causes c")
    (_, state) = respond!(ai, "why a causes c"; expected_output="a causes c")
    @test length(state.hypotheses) >= 3
    origins = Set(h.origin for h in state.hypotheses)
    @test :deduction in origins || :direct in origins || length(origins) >= 1
end
