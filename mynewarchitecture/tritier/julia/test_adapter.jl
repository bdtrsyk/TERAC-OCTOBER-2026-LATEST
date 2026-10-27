include("TriTierAdapter.jl")
using .TriTierAdapter

println("TriTierAdapter ABI: ", TriTierAdapter.tritier_abi_version)

before = Int8[1, 0, -1, 0, 1]
delta = Int8[0, 1, 0, -1, 0]
score = TriTierAdapter.evaluate_candidate(before, delta)
println("Candidate delta score: ", score)

td = TriTierAdapter.ternary_delta(before, Int8(1))
println("Ternary delta output: ", td)
@assert length(td) == length(before)
@assert all(x -> x in (-1, 0, 1), td)

println("[OK] Julia TriTierAdapter verified successfully!")
