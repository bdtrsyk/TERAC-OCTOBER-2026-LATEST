@testset "credit assignment locality" begin
    ai = cognitive_agent(:tiny)
    respond!(ai, "a causes b"; expected_output="a causes b")
    before = synaptic_snapshot(ai)
    respond!(ai, "why a causes b"; expected_output="a causes b")
    changes = get_last_synaptic_changes(ai)
    # Credit should only touch activated pathways; never mutate unrelated random synapses.
    for ch in changes
        @test haskey(before, (ch.source, ch.target)) || ch.old_state == Int8(0)
        @test ch.old_state in (Int8(-1), Int8(0), Int8(1))
        @test ch.new_state in (Int8(-1), Int8(0), Int8(1))
    end
    dist = ternary_distribution(ai.network)
    @test dist.total == length(before) || dist.total >= length(before)
end
