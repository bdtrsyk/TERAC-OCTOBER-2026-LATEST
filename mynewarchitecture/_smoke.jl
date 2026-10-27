using Pkg
Pkg.instantiate()
using TernaryCognitiveAI
println("LOADED OK")
ai = cognitive_agent(:tiny)
println("integrity=", verify_integrity(ai))
r = respond!(ai, "a causes b"; expected_output="a causes b")
println("quality=", r[1][2].quality)
println("output=", r[1][1])
