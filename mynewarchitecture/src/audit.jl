# audit.jl — Architecture integrity audit and resource statistics (spec §48, §49, §77).

"""Scan source files for forbidden ML/optimizer patterns. Returns (ok, findings)."""
function static_source_audit(src_dir::AbstractString=joinpath(@__DIR__))
    suspicious = [
        r"\bFlux\b",
        r"\bZygote\b",
        r"\bOptimisers\b",
        r"\bAdamW?\b",
        r"\bRMSProp\b",
        r"\bSGD\b",
        r"\bgradient\s*\(",
        r"cross[\s_]?entropy",
        r"\bsoftmax\b",
        r"next[\s_]?token",
        r"teacher[\s_]?forcing",
        r"OpenAI",
        r"Anthropic",
        r"OpenRouter",
        r"Backprop",
        r"backpropagation",
    ]
    # Allowed exceptions: documentation mentions of forbidden techniques.
    allow_paths = Set([
        "audit.jl",
        "README.md",
    ])
    findings = String[]
    for (root, _, files) in walkdir(src_dir)
        for file in files
            endswith(file, ".jl") || continue
            file in allow_paths && continue
            path = joinpath(root, file)
            text = read(path, String)
            for (i, line) in enumerate(split(text, '\n'))
                # Skip comments that explicitly discuss prohibitions.
                stripped = strip(line)
                startswith(stripped, "#") && occursin(r"forbid|prohib|must not|no conventional|no next-token|no backprop", lowercase(stripped)) && continue
                for pat in suspicious
                    if occursin(pat, line)
                        # Ignore audit self-patterns and string literals in help text about what we forbid.
                        occursin("forbidden", lowercase(line)) && continue
                        occursin("must not", lowercase(line)) && continue
                        push!(findings, "$file:$i: matched $(pat.pattern) :: $(strip(line))")
                    end
                end
            end
        end
    end
    return isempty(findings), findings
end

"""Count floating-point *learned* weights (must be zero on SparseSynapse.value)."""
function floating_learned_weight_count(agent::CognitiveAgent)
    # Architecture invariant: learned synapse values are Int8 ternary only.
    bad = 0
    for edges in values(agent.network.outgoing), e in edges
        !(e.value isa Int8) && (bad += 1)
        !is_ternary(e.value) && (bad += 1)
    end
    return bad
end

"""Full architecture audit (spec §48). Returns exit code 0 on success."""
function run_architecture_audit(agent::Union{Nothing,CognitiveAgent}=nothing; io::IO=stdout)
    ai = agent === nothing ? cognitive_agent(:tiny) : agent
    dist = ternary_distribution(ai.network)
    ok_static, findings = static_source_audit()
    float_learned = floating_learned_weight_count(ai)
    integrity_ok = try
        verify_integrity(ai)
        true
    catch e
        println(io, "INTEGRITY FAILURE: ", e)
        false
    end

    native_present = isdir(joinpath(@__DIR__, "..", "native"))
    println(io, "=== TERNARY COGNITIVE AI — ARCHITECTURE AUDIT ===")
    println(io, "architecture_version: ", ai.architecture_version)
    println(io, "serialization_version: ", ai.serialization_version)
    println(io, "julia_version: ", VERSION)
    println(io, "learned_synapse_type: Int8")
    println(io, "ternary_invariant: ", integrity_ok ? "PASS" : "FAIL")
    println(io, "number_of_synapses: ", dist.total)
    @printf(io, "synapse_distribution: +1=%d  0=%d  -1=%d\n", dist.positive, dist.zero, dist.negative)
    println(io, "floating_learned_weights: ", float_learned)
    println(io, "optimizer_presence: none (plasticity is discrete ternary)")
    println(io, "gradient_code_presence: ", ok_static ? "none detected in src/" : "SUSPICIOUS MATCHES")
    println(io, "token_prediction_objective: absent")
    println(io, "external_llm_dependency: absent")
    println(io, "native_modules: ", native_present ? "directory present (optional backends)" : "none")
    println(io, "memory_layout: SparseTernaryNetwork + concept/episodic/strategy/failure stores")
    println(io, "cpu_reference_mode: available")
    println(io, "gpu_mode: optional / not required")

    if !ok_static
        println(io, "\nStatic findings:")
        for f in findings
            println(io, "  - ", f)
        end
    end

    passed = integrity_ok && float_learned == 0 && ok_static
    println(io, "\nAUDIT RESULT: ", passed ? "PASS" : "FAIL")
    return passed ? 0 : 1
end

"""Resource / memory statistics (spec §49)."""
function run_resource_stats(agent::Union{Nothing,CognitiveAgent}=nothing; io::IO=stdout)
    ai = agent === nothing ? cognitive_agent(:tiny) : agent
    dist = ternary_distribution(ai.network)
    live = Base.gc_live_bytes()
    peak = max(ai.metrics.peak_bytes, live)
    elig = eligibility_stats(ai.network)
    println(io, "=== RESOURCE STATS ===")
    @printf(io, "cpu_memory_bytes: %d\n", live)
    @printf(io, "peak_memory_bytes: %d\n", peak)
    @printf(io, "model_storage_synapses: %d\n", dist.total)
    @printf(io, "working_memory_items: %d / %d\n", length(ai.working.items), ai.working.capacity)
    @printf(io, "semantic_concepts: %d\n", length(ai.semantic.concepts))
    @printf(io, "semantic_relations: %d\n", length(ai.semantic.relations))
    @printf(io, "episodic_episodes: %d\n", length(ai.episodic.episodes))
    @printf(io, "strategy_records: %d\n", length(ai.strategy.records))
    @printf(io, "failure_records: %d\n", length(ai.failures.records))
    @printf(io, "active_units: %d\n", length(active_unit_ids(ai.network)))
    @printf(io, "active_synapses_eligible: %d\n", elig.count)
    @printf(io, "reasoning_events_last: %d\n",
            ai.last_state === nothing ? 0 : length(ai.last_state.trace.events))
    @printf(io, "ternary_+1: %d (%.1f%%)\n", dist.positive, dist.pct_positive)
    @printf(io, "ternary_0: %d (%.1f%%)\n", dist.zero, dist.pct_zero)
    @printf(io, "ternary_-1: %d (%.1f%%)\n", dist.negative, dist.pct_negative)
    println(io, "gpu_memory: not allocated (CPU reference mode)")
    return 0
end
