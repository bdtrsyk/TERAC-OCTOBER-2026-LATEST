@testset "reasoning state machine" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "rain causes wet ground"; expected_output="rain causes wet ground")
    respond!(ai, "wet ground causes slippery pavement"; expected_output="wet ground causes slippery pavement")
    (result, state) = respond!(ai, "why rain causes slippery pavement"; expected_output="rain causes slippery pavement")
    @test length(state.hypotheses) >= 1
    @test !isempty(state.trace.events)
    @test !isempty(state.phase_history)
    @test occursin("rain", lowercase(result[1]))
    kinds = [e.kind for e in state.trace.events]
    @test any(k -> k in (TernaryCognitiveAI.HYPOTHESIS_CREATED, TernaryCognitiveAI.GOAL_CREATED), kinds)
end
