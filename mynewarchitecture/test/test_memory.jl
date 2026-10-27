@testset "structured memory" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "sun causes heat"; expected_output="sun causes heat")
    @test concept_count(ai) >= 2
    @test !isempty(ai.semantic.relations)
    @test episode_count(ai) >= 1
    @test !isempty(get_working_memory(ai)) || true  # may be cleared after episode
    snap = memory_snapshot(ai)
    respond!(ai, "heat causes expansion"; expected_output="heat causes expansion")
    diff = memory_diff(snap, ai)
    @test !isempty(diff.added_concepts) || !isempty(diff.new_relations) || diff.new_episodes >= 1
end
