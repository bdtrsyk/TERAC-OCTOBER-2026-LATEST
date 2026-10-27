using TernaryCognitiveAI
ai = cognitive_agent(:tiny)
respond!(ai, "a causes b"; expected_output="a causes b")
println("quality=", get_last_evaluation(ai).quality, " reward=", get_last_reward(ai))
