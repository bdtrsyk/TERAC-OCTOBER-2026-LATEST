
# cli.jl — Command-line interface, interactive chat REPL, and batch training.

function _print_cli_help(io::IO=stdout)
    println(io, raw"""
TernaryCognitiveAI — Discrete {-1, 0, +1} Reasoning Architecture

USAGE:
  julia --project=. main.jl [COMMAND / OPTIONS]
  run.bat                     Launch the unified Julia+Python PyQt6 desktop GUI

COMMANDS & MODES:
  --chat, --cli               Launch the interactive terminal chat REPL
  --bridge                    JSON-lines bridge for Python (tcai_bridge.py)
  --train <file>              Train the agent on a dataset file
  --ask "<query>"             Ask a question or assert a statement and print answer
  --query "<query>"           Alias for --ask
  --demo                      Run the transitive deductive reasoning demonstration
  --negative-demo             Run the contradiction / self-correction demonstration
  --test                      Run the full test suite
  --help, -h                  Display this help documentation

OPTIONS:
  --epochs <N>                Number of training epochs (default: 1)
  --scale <name>              Agent scale: tiny, small, medium, large (default: small)
  --load <model_path>         Load saved model state before running
  --save <model_path>         Save model state after training or query
  --expected "<text>"         Expected output for evaluation / supervised training
  --trace                     Display the full step-by-step reasoning trace
  --quiet                     Suppress verbose progress output during training

INTERACTIVE CHAT COMMANDS (inside REPL):
  :help                       Show help on chat commands and grammar
  :teach <in> | <target>      Teach the agent an input-target pair with immediate feedback
  :train <file> [epochs]      Train the agent on a dataset file
  :save <path>                Save the current agent state to disk
  :load <path>                Load a saved agent state from disk
  :stats, :metrics            Display agent metrics, memory counts, and ternary distribution
  :concepts                   List known concepts in semantic memory
  :relations                  List all learned and deduced relations
  :trace                      Show the full reasoning trace of the last response
  :memory                     Display working memory and episodic history
  :failures                   Display recorded reasoning failures
  :diff                       Display synaptic weight transitions from the last episode
  :reset [scale]              Reset the agent to a fresh state (tiny, small, medium, large)
  :clear                      Clear terminal screen
  :exit, :quit                Exit the interactive chat session
""")
end

function _print_chat_help(io::IO=stdout)
    println(io, raw"""
── Chat Commands ──────────────────────────────────────────────────────────
  :help                       Show this help guide
  :teach <in> | <target>      Teach input with expected target (e.g. :teach a causes b | a causes b)
  :train <path> [epochs]      Batch train from a text file (e.g. :train data/train.txt 3)
  :save <path>                Save agent state to a binary file (e.g. :save model.bin)
  :load <path>                Load agent state from file (e.g. :load model.bin)
  :stats                      View run metrics and ternary weight distribution
  :concepts                   List all concepts stored in semantic memory
  :relations                  List all stored & deduced relational triples
  :trace                      Show the authentic machine reasoning trace of the last turn
  :memory                     Inspect working memory and recalled episodes
  :failures                   Inspect failure memory records
  :diff                       Inspect synaptic changes from the last interaction
  :reset [scale]              Reinitialize agent (scales: :tiny, :small, :medium, :large)
  :clear                      Clear console screen
  :exit, :quit                Exit chat

── Natural Language Input Guide ───────────────────────────────────────────
  1. Declarative Facts:       sun causes heat
                              heat causes expansion
                              water leads to life
  2. Queries (Deductions):    why sun causes expansion
                              does sun causes heat
                              why rain causes slippery pavement
  3. Repetitions:             repeat this
                              repeat that 2 times
  4. Constraints:             sun causes heat in json
                              why sun causes expansion as a list
                              sun causes heat without expansion
                              sun causes heat in one sentence
───────────────────────────────────────────────────────────────────────────
""")
end

function _print_chat_banner(ai::CognitiveAgent, io::IO=stdout)
    d = ternary_distribution(ai.network)
    println(io, "╔════════════════════════════════════════════════════════════════════════╗")
    println(io, "║                    TERNARY COGNITIVE AI CHAT                           ║")
    println(io, "║     Discrete {-1, 0, +1} Sparse Synapses · Neuro-Symbolic Agent        ║")
    println(io, "╚════════════════════════════════════════════════════════════════════════╝")
    @printf(io, "  Scale: %-8s | Concepts: %-4d | Synapses: %-4d (+1:%d, 0:%d, -1:%d)\n",
            string(ai.config.scale), concept_count(ai), synapse_count(ai),
            d.positive, d.zero, d.negative)
    println(io, "  Type your statements or queries below. Type :help for commands, :exit to quit.\n")
end

"""Run an interactive chat REPL session with the CognitiveAgent."""
function run_interactive_chat(ai::CognitiveAgent=cognitive_agent(:small); io_in::IO=stdin, io_out::IO=stdout)
    _print_chat_banner(ai, io_out)

    while true
        print(io_out, "TernaryAI> ")
        flush(io_out)
        line = readline(io_in)
        if isempty(line) && eof(io_in)
            println(io_out, "\nExiting TernaryCognitiveAI.")
            break
        end

        raw = strip(line)
        isempty(raw) && continue

        # Handle meta-commands starting with ':' or '/'
        if startswith(raw, ":") || startswith(raw, "/")
            cmd_line = strip(raw[2:end])
            parts = split(cmd_line; limit=2)
            cmd = lowercase(first(parts))
            arg = length(parts) > 1 ? strip(parts[2]) : ""

            if cmd in ("exit", "quit", "q")
                println(io_out, "Goodbye! State preserved in memory.")
                break
            elseif cmd in ("help", "?")
                _print_chat_help(io_out)
            elseif cmd in ("clear", "cls")
                print(io_out, "\033[2J\033[1;1H")
            elseif cmd in ("stats", "metrics", "status")
                m = ai.metrics
                d = ternary_distribution(ai.network)
                println(io_out, "\n=== AGENT METRICS ===")
                @printf(io_out, "  Episodes:              %d\n", m.episodes)
                @printf(io_out, "  Total Reward:          +%.2f\n", m.total_reward)
                @printf(io_out, "  Total Punishment:      -%.2f\n", m.total_punishment)
                @printf(io_out, "  Self Corrections:      %d\n", m.self_corrections)
                @printf(io_out, "  Novel Recombinations:  %d\n", m.novel_recombinations)
                @printf(io_out, "  Contradictions Found:  %d\n", m.contradictions_detected)
                @printf(io_out, "  Hypotheses Rejected:   %d\n", m.hypotheses_rejected)
                @printf(io_out, "  Ternary Transitions:   %d\n", m.ternary_transitions)
                @printf(io_out, "  Concepts:              %d\n", concept_count(ai))
                @printf(io_out, "  Synapses:              +1: %d | 0: %d | -1: %d (Total: %d)\n",
                        d.positive, d.zero, d.negative, d.total)
                println(io_out, "=====================\n")
            elseif cmd in ("concepts", "concept")
                sm = ai.semantic
                if isempty(sm.concepts)
                    println(io_out, "No concepts learned yet.")
                else
                    println(io_out, "\n=== CONCEPTS ($(length(sm.concepts))) ===")
                    for (id, c) in sort(collect(sm.concepts); by=first)
                        rel = concept_reliability(c)
                        domains = join(c.domain_tags, ", ")
                        @printf(io_out, "  [%3d] %-20s | Domain: [%s] | Rel: %.2f (S:%d F:%d Freq:%d)\n",
                                id, c.name, domains, rel, c.successes, c.failures, c.frequency)
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd in ("relations", "relation", "facts")
                sm = ai.semantic
                if isempty(sm.relations)
                    println(io_out, "No relations stored in semantic memory.")
                else
                    println(io_out, "\n=== RELATIONS ($(length(sm.relations))) ===")
                    for (i, r) in enumerate(sm.relations)
                        s = concept_name(ai, r.subject)
                        o = concept_name(ai, r.object)
                        @printf(io_out, "  %3d. %-15s --[%-12s]--> %-15s (prov: %s)\n",
                                i, s, string(r.predicate), o, string(r.provenance))
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd == "trace"
                if ai.last_state === nothing || isempty(ai.last_state.trace.events)
                    println(io_out, "No reasoning trace recorded yet.")
                else
                    println(io_out, "\n=== REASONING TRACE (LAST TURN) ===")
                    render_trace(ai.last_state.trace; io=io_out)
                    println(io_out, "===================================\n")
                end
            elseif cmd in ("diff", "changes")
                changes = get_last_synaptic_changes(ai)
                if isempty(changes)
                    println(io_out, "No synaptic changes in the last episode.")
                else
                    println(io_out, "\n=== SYNAPTIC CHANGES ===")
                    for ch in changes
                        s = concept_name(ai, ch.source)
                        t = concept_name(ai, ch.target)
                        @printf(io_out, "  %s -> %s: %d -> %d | Credit: %s | Reason: %s\n",
                                s, t, ch.old_state, ch.new_state, string(ch.credit), ch.reason)
                    end
                    println(io_out, "========================\n")
                end
            elseif cmd in ("memory", "mem")
                wm = get_working_memory(ai)
                println(io_out, "\n=== WORKING MEMORY ($(length(wm))) ===")
                for (i, it) in enumerate(wm)
                    @printf(io_out, "  %d. [%s] \"%s\" (rel: %.2f, step: %d)\n",
                            i, string(it.kind), it.text, it.relevance, it.step)
                end
                recalled = get_retrieved_memories(ai)
                println(io_out, "  Last Retrieved Past Episodes: ", isempty(recalled) ? "none" : string(recalled))
                println(io_out, "==============================\n")
            elseif cmd in ("failures", "failure")
                fails = ai.failures.records
                if isempty(fails)
                    println(io_out, "No failure records logged.")
                else
                    println(io_out, "\n=== RECORDED FAILURES ($(length(fails))) ===")
                    for (i, f) in enumerate(fails)
                        @printf(io_out, "  %d. Episode %d: \"%s\" (strategy: %s)\n     Reason: %s\n",
                                i, f.episode, f.context, string(f.strategy), f.reason)
                    end
                    println(io_out, "===============================\n")
                end
            elseif cmd in ("reset", "restart")
                scale = isempty(arg) ? ai.config.scale : Symbol(arg)
                ai = cognitive_agent(scale)
                println(io_out, "Agent reinitialized with scale: :$scale.")
            elseif cmd == "save"
                if isempty(arg)
                    println(io_out, "Usage: :save <path>")
                else
                    try
                        save_model(ai, String(arg))
                        println(io_out, "[OK] Model saved to: $arg")
                    catch e
                        println(io_out, "[Error] Failed to save: $e")
                    end
                end
            elseif cmd == "load"
                if isempty(arg)
                    println(io_out, "Usage: :load <path>")
                else
                    try
                        ai = load_model(String(arg))
                        println(io_out, "[OK] Model loaded from: $arg")
                        _print_chat_banner(ai, io_out)
                    catch e
                        println(io_out, "[Error] Failed to load: $e")
                    end
                end
            elseif cmd == "teach"
                if !occursin("|", arg)
                    println(io_out, "Usage: :teach <input> | <expected output> [| <quality>]")
                else
                    parts_teach = split(arg, "|")
                    inp = strip(parts_teach[1])
                    exp = strip(parts_teach[2])
                    q_tgt = length(parts_teach) >= 3 && !isempty(strip(parts_teach[3])) ? Symbol(strip(parts_teach[3])) : nothing
                    (res, state) = respond!(ai, inp; expected_output=exp, expected_quality=q_tgt)
                    out, ev, ep = res
                    sig = ev.reward > 0f0 ? @sprintf("+%.2f", ev.reward) :
                          ev.punishment > 0f0 ? @sprintf("-%.2f", ev.punishment) : "0.00"
                    println(io_out, "AI: ", out)
                    println(io_out, "  [Taught] Quality: $(ev.quality) | Signal: $sig | Synapses changed: $(length(ep.synaptic_changes))")
                end
            elseif cmd == "train"
                if isempty(arg)
                    println(io_out, "Usage: :train <dataset_path> [epochs]")
                else
                    train_args = split(arg)
                    t_path = String(train_args[1])
                    t_epochs = length(train_args) > 1 ? parse(Int, train_args[2]) : 1
                    try
                        train_file!(ai, t_path; epochs=t_epochs, verbose=true, io=io_out)
                    catch e
                        println(io_out, "[Error] Training failed: $e")
                    end
                end
            else
                println(io_out, "Unknown command: :$cmd. Type :help for commands.")
            end
            continue
        end

        # Normal reasoning response
        try
            (result, state) = respond!(ai, String(raw))
            output, evaluation, episode = result
            println(io_out, "\nAI: ", output)

            sig = evaluation.reward > 0f0 ? @sprintf("+%.2f", evaluation.reward) :
                  evaluation.punishment > 0f0 ? @sprintf("-%.2f", evaluation.punishment) : " 0.00"
            ch_cnt = length(episode.synaptic_changes)
            units_active = length(state.active_units)

            extra_notes = String[]
            if !isempty(state.contradictions)
                push!(extra_notes, "Contradiction self-corrected")
            end
            if evaluation.quality == :REQUESTED_REPETITION
                push!(extra_notes, "Repetition fulfilled")
            end
            if evaluation.punishment > 0f0
                push!(extra_notes, "Failure logged")
            end
            notes_str = isempty(extra_notes) ? "" : " | " * join(extra_notes, ", ")

            @printf(io_out, "  [Quality: %s | Conf: %.2f | Signal: %s | Steps: %d | Active: %d units | Changed: %d%s]\n\n",
                    string(evaluation.quality), state.confidence, sig, state.steps_used,
                    units_active, ch_cnt, notes_str)
        catch e
            println(io_out, "[Error] Processing failed: $e")
        end
    end
end

"""Command-line entry point with full argument dispatch."""
function run_cli(args=ARGS)
    args = collect(String, args)

    # Positional subcommands: audit / stats / help / demo / ...
    # Option values must be skipped, otherwise `--save demo` (or an expected
    # output literally named "demo") would dispatch the demo command.
    value_opts = Set(["--epochs", "--scale", "--load", "--save",
                      "--expected", "--train", "--ask", "--query"])
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a in value_opts
            i += 2
            continue
        elseif startswith(a, "-")
            i += 1
            continue
        else
            push!(positional, a)
            i += 1
        end
    end
    if !isempty(positional)
        cmd = lowercase(positional[1])
        if cmd == "audit"
            return run_architecture_audit()
        elseif cmd == "stats"
            return run_resource_stats()
        elseif cmd == "help"
            _print_cli_help()
            return 0
        elseif cmd == "demo"
            run_core_demonstration()
            return 0
        elseif cmd == "experiment" && length(positional) >= 2
            run_experiment(positional[2])
            return 0
        end
    end

    # 1. Help
    if "--help" in args || "-h" in args
        _print_cli_help()
        return 0
    end

    # CPU reference mode is the default; --gpu is acknowledged but optional.
    if "--gpu" in args
        println("[note] GPU backend not required; running CPU reference ternary architecture.")
    end
    # --cpu is explicit no-op affirmation of reference mode
    _ = "--cpu" in args

    # Parse scale early (needed by --bridge)
    scale = :small
    idx_scale = findfirst(a -> a == "--scale", args)
    if idx_scale !== nothing && idx_scale < length(args)
        scale = Symbol(args[idx_scale + 1])
    end

    # JSON-lines bridge for Python interop (must be quiet on stderr-free stdout)
    if "--bridge" in args
        load_path = nothing
        idx_load_b = findfirst(a -> a == "--load", args)
        if idx_load_b !== nothing && idx_load_b < length(args)
            load_path = args[idx_load_b + 1]
        end
        return run_json_bridge(; scale=scale, load_path=load_path)
    end

    # 2. Demonstration, audit, stats, test modes
    if "--demo" in args
        run_core_demonstration()
        return 0
    end
    if "--negative-demo" in args
        run_negative_demonstration()
        return 0
    end
    if "--repetition-demo" in args
        run_repetition_demonstration()
        return 0
    end
    if "--novelty-demo" in args
        run_novelty_demonstration()
        return 0
    end
    if "--self-correction-demo" in args
        run_self_correction_demonstration()
        return 0
    end
    if "--audit" in args
        return run_architecture_audit()
    end
    if "--stats" in args
        return run_resource_stats()
    end
    if "--test" in args
        project = dirname(@__DIR__)
        testfile = joinpath(project, "test", "runtests.jl")
        isfile(testfile) || error("test/runtests.jl not found")
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        cmd = `$julia --project=$project $testfile`
        println("[test] ", cmd)
        return success(run(cmd)) ? 0 : 1
    end

    # Create or load agent
    ai = cognitive_agent(scale)
    idx_load = findfirst(a -> a == "--load", args)
    if idx_load !== nothing && idx_load < length(args)
        load_path = args[idx_load + 1]
        println("[Init] Loading model from: $load_path")
        ai = load_model(load_path)
    end

    # 3. Batch training mode
    idx_train = findfirst(a -> a == "--train", args)
    if idx_train !== nothing && idx_train < length(args)
        train_path = args[idx_train + 1]
        epochs = 1
        idx_epochs = findfirst(a -> a == "--epochs", args)
        if idx_epochs !== nothing && idx_epochs < length(args)
            epochs = parse(Int, args[idx_epochs + 1])
        end
        verbose = !("--quiet" in args)
        train_file!(ai, train_path; epochs=epochs, verbose=verbose)

        idx_save = findfirst(a -> a == "--save", args)
        if idx_save !== nothing && idx_save < length(args)
            save_path = args[idx_save + 1]
            save_model(ai, save_path)
            println("[Saved] Model saved to: $save_path")
        end

        # If --chat was not explicitly passed, exit after training
        if !("--chat" in args)
            return 0
        end
    end

    # 4. Single-shot query / ask mode
    idx_ask = findfirst(a -> a in ("--ask", "--query"), args)
    if idx_ask !== nothing && idx_ask < length(args)
        query_text = args[idx_ask + 1]
        exp_text = nothing
        idx_exp = findfirst(a -> a == "--expected", args)
        if idx_exp !== nothing && idx_exp < length(args)
            exp_text = args[idx_exp + 1]
        end

        (result, state) = respond!(ai, query_text; expected_output=exp_text)
        output, evaluation, ep = result

        println("\nINPUT:      $query_text")
        println("OUTPUT:     $output")
        println("QUALITY:    $(evaluation.quality)")
        @printf("CONFIDENCE: %.3f\n", state.confidence)
        if evaluation.reward > 0f0
            @printf("REWARD:     +%.2f\n", evaluation.reward)
        elseif evaluation.punishment > 0f0
            @printf("PUNISHMENT: -%.2f\n", evaluation.punishment)
        end
        println("REASONS:    ", join(evaluation.reasons, "; "))

        if "--trace" in args
            println("\n--- TRACE ---")
            render_trace(state.trace)
        end

        # Interactive feedback hooks
        if "--reward" in args
            reward!(ai)
            println("[feedback] reward applied")
        elseif "--punish" in args
            punish!(ai)
            println("[feedback] punishment applied")
        end

        idx_save = findfirst(a -> a == "--save", args)
        if idx_save !== nothing && idx_save < length(args)
            save_path = args[idx_save + 1]
            save_model(ai, save_path)
            println("[Saved] Model saved to: $save_path")
        end
        return 0
    end

    # 5. Terminal interactive chat REPL if --chat or --cli passed
    if "--chat" in args || "--cli" in args
        run_interactive_chat(ai)
        return 0
    end

    # 6. No desktop GUI here on the Julia side: the unified PyQt6 GUI lives in
    # Python (gui.py) and drives this package through the JSON bridge.
    # Bare `run.bat` launches it. Bare `julia main.jl` shows the help.
    _print_cli_help()
    return 0
end

