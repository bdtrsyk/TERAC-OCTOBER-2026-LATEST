# experiments.jl — Configurable experiment runner (spec §86).

function run_repetition_demonstration(; io::IO=stdout)
    println(io, "=== REPETITION DEMONSTRATION ===")
    println(io, "CASE A: repetition NOT requested → REPETITIVE → punishment")
    a = cognitive_agent(:tiny)
    respond!(a, "heat causes expansion"; expected_output="heat causes expansion")
    before_a = synaptic_snapshot(a)
    # Force unrequested repetition by answering the same query twice with same output path
    respond!(a, "heat causes expansion"; expected_output="heat causes expansion")
    (res_a, _) = respond!(a, "heat causes expansion")
    qa = get_last_evaluation(a)
    println(io, "  quality=$(qa.quality) reward=$(qa.reward) punishment=$(qa.punishment)")
    println(io, "  diff=", synaptic_diff(before_a, a))

    println(io, "\nCASE B: repetition explicitly requested → REQUESTED_REPETITION → reward")
    b = cognitive_agent(:tiny)
    respond!(b, "heat causes expansion"; expected_output="heat causes expansion")
    before_b = synaptic_snapshot(b)
    (res_b, _) = respond!(b, "repeat this")
    qb = get_last_evaluation(b)
    println(io, "  output=$(res_b[1])")
    println(io, "  quality=$(qb.quality) reward=$(qb.reward) punishment=$(qb.punishment)")
    println(io, "  diff=", synaptic_diff(before_b, b))
    return (case_a=qa, case_b=qb)
end

function run_novelty_demonstration(; io::IO=stdout)
    println(io, "=== NOVELTY / RECOMBINATION DEMONSTRATION ===")
    ai = cognitive_agent(:tiny)
    # Train A and B separately; never present A+B composition during training.
    respond!(ai, "routing allocates bandwidth"; expected_output="routing allocates bandwidth")
    respond!(ai, "traffic management allocates capacity"; expected_output="traffic management allocates capacity")
    before = synaptic_snapshot(ai)
    (result, state) = respond!(ai, "why routing supports capacity")
    println(io, "OUTPUT: ", result[1])
    println(io, "QUALITY: ", result[2].quality)
    println(io, "HYPOTHESES: ", length(state.hypotheses))
    for h in state.hypotheses
        println(io, "  H$(h.id) origin=$(h.origin) status=$(h.status) claim=",
                relation_triple_string(ai, h.claim))
    end
    println(io, "TRACE:")
    render_trace(state.trace; io=io)
    println(io, "SYNAPTIC_DIFF: ", synaptic_diff(before, ai))
    return ai, state
end

function run_self_correction_demonstration(; io::IO=stdout)
    println(io, "=== SELF-CORRECTION DEMONSTRATION ===")
    ai = cognitive_agent(:tiny)
    respond!(ai, "pressure causes flow"; expected_output="pressure causes flow")
    # Attractive wrong answer first
    (bad, state1) = respond!(ai, "why pressure causes flow"; expected_output="pressure prevents flow")
    println(io, "Initial quality: ", bad[2].quality, " punishment=", bad[2].punishment)
    # Provide contradictory evidence via opposing fact, then re-ask
    respond!(ai, "pressure causes flow"; expected_output="pressure causes flow")
    (good, state2) = respond!(ai, "why pressure causes flow"; expected_output="pressure causes flow")
    println(io, "Corrected quality: ", good[2].quality)
    println(io, "Rejected hypotheses: ", count(h -> h.status == :rejected, state2.hypotheses))
    println(io, "Contradictions: ", state2.contradictions)
    println(io, "TRACE:")
    render_trace(state2.trace; io=io)
    return ai, state2
end

function run_reward_ablation_experiment(; io::IO=stdout)
    println(io, "=== REWARD ABLATION ===")
    results = Dict{Symbol,Any}()
    for (mode, cfg!) in (
        (:both_enabled, ai -> (ai.config.reward_enabled = true; ai.config.punishment_enabled = true)),
        (:no_reward, ai -> (ai.config.reward_enabled = false)),
        (:no_punishment, ai -> (ai.config.punishment_enabled = false)),
    )
        ai = cognitive_agent(:tiny)
        cfg!(ai)
        data = [
            TrainingExample(1, "a causes b", "a causes b", nothing, nothing, String[], Symbol[], Dict{String,Any}()),
            TrainingExample(2, "b causes c", "b causes c", nothing, nothing, String[], Symbol[], Dict{String,Any}()),
            TrainingExample(3, "why a causes c", "a causes c", nothing, nothing, String[], Symbol[], Dict{String,Any}()),
        ]
        report = train!(ai, data; epochs=2)
        results[mode] = (reward=report.total_reward, punishment=report.total_punishment,
                         transitions=ai.metrics.ternary_transitions, success=sum(ai.metrics.success_curve))
        println(io, mode, " => ", results[mode])
    end
    return results
end

"""Dispatch named experiments (spec §86)."""
function run_experiment(name::AbstractString; io::IO=stdout)
    key = Symbol(replace(lowercase(strip(name)), "-" => "_"))
    # Accept "<name>_experiment" too (matches experiments/*.jl script names).
    key_str = String(key)
    endswith(key_str, "_experiment") && (key = Symbol(key_str[1:end-11]))
    if key == :reasoning_depth || key == :core || key == :demo
        return run_core_demonstration(; io=io)
    elseif key == :reward_ablation
        return run_reward_ablation_experiment(; io=io)
    elseif key == :punishment_ablation
        return run_ablation(; mode=:no_punishment)
    elseif key == :repetition_learning || key == :repetition
        return run_repetition_demonstration(; io=io)
    elseif key == :self_correction
        return run_self_correction_demonstration(; io=io)
    elseif key == :novel_recombination || key == :novelty
        return run_novelty_demonstration(; io=io)
    elseif key == :negative
        return run_negative_demonstration(; io=io)
    elseif key == :cross_domain
        return run_novelty_demonstration(; io=io)
    elseif key == :memory_generalization
        return run_core_demonstration(; io=io)
    else
        error("unknown experiment: $name")
    end
end
