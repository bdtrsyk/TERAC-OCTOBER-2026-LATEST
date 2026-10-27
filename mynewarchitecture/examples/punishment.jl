using TernaryCognitiveAI
ai = cognitive_agent(:tiny)
respond!(ai, "a causes b"; expected_output="a causes b")
respond!(ai, "why a causes b"; expected_output="a prevents b")
println("quality=", get_last_evaluation(ai).quality, " punishment=", get_last_punishment(ai))
