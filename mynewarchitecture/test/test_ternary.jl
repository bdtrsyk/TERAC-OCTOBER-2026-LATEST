@testset "ternary invariant" begin
    @test TernaryCognitiveAI.is_ternary(-1)
    @test TernaryCognitiveAI.is_ternary(0)
    @test TernaryCognitiveAI.is_ternary(1)
    @test !TernaryCognitiveAI.is_ternary(2)
    @test TernaryCognitiveAI.assert_ternary(Int8(1)) == Int8(1)
    @test_throws ErrorException TernaryCognitiveAI.assert_ternary(Int8(2))

    packed = TernaryCognitiveAI.PackedTernaryVector(10)
    packed[1] = -1
    packed[2] = 0
    packed[3] = 1
    @test packed[1] == -1
    @test packed[2] == 0
    @test packed[3] == 1
    @test TernaryCognitiveAI.packed_roundtrip(Int8[-1,0,1,1,0,-1]) == Int8[-1,0,1,1,0,-1]
end
