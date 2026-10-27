
function run_core_demonstration(; io::IO=stdout)
    ai=cognitive_agent(:tiny)
    respond!(ai,"rain causes wet ground"; expected_output="rain causes wet ground")
    respond!(ai,"wet ground causes slippery pavement"; expected_output="wet ground causes slippery pavement")
    before=synaptic_snapshot(ai); (result,state)=respond!(ai,"why rain causes slippery pavement"; expected_output="rain causes slippery pavement")
    output,evaluation,episode=result
    println(io,"OUTPUT: $output\nQUALITY: $(evaluation.quality)\nHYPOTHESES: $(length(state.hypotheses))\nTRACE:")
    render_trace(state.trace;io=io); println(io,"SYNAPTIC_DIFF: ",synaptic_diff(before,ai)); return ai
end
function run_negative_demonstration(; io::IO=stdout)
    ai=cognitive_agent(:tiny); respond!(ai,"a causes b"; expected_output="a causes b")
    before=synaptic_snapshot(ai); (result,state)=respond!(ai,"why a causes b"; expected_output="a prevents b")
    println(io,"QUALITY: $(result[2].quality)\nPUNISHMENT: $(result[2].punishment)\nDIFF: ",synaptic_diff(before,ai)); ai
end
function run_ablation(; mode::Symbol=:no_reward)
    ai=cognitive_agent(:tiny); mode==:no_reward && (ai.config.reward_enabled=false); mode==:no_punishment && (ai.config.punishment_enabled=false)
    train!(ai,[TrainingExample(1,"x causes y","x causes y",nothing,nothing,String[],Symbol[],Dict{String,Any}())];epochs=3)
end

