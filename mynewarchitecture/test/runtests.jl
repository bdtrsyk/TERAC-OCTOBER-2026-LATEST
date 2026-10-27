using Test
using TernaryCognitiveAI

@testset "TernaryCognitiveAI" begin
    include("test_ternary.jl")
    include("test_network.jl")
    include("test_reasoning.jl")
    include("test_hypotheses.jl")
    include("test_novelty.jl")
    include("test_memory.jl")
    include("test_rewards.jl")
    include("test_repetition.jl")
    include("test_credit.jl")
    include("test_plasticity.jl")
    include("test_serialization.jl")
    include("test_generalization.jl")
    include("test_integrity.jl")
end
