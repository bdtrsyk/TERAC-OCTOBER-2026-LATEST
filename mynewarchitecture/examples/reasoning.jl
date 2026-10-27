using TernaryCognitiveAI
(result, state) = respond!(cognitive_agent(:tiny), "sun causes heat"; expected_output="sun causes heat")
println(result[1])
render_trace(state.trace)
