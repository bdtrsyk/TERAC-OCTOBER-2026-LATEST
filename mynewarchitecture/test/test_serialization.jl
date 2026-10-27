@testset "serialization roundtrip" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "moon causes tide"; expected_output="moon causes tide")
    path = tempname() * ".tca"
    save_model(ai, path)
    loaded = load_model(path)
    rm(path; force=true)
    @test length(loaded.episodic.episodes) == length(ai.episodic.episodes)
    @test verify_integrity(loaded)
    @test concept_count(loaded) == concept_count(ai)
end
