
function parse_training_line(line::String, id::Int)
    s = strip(line)
    (isempty(s) || startswith(s, "#")) && return nothing
    p = split(s, " | "; limit=3)
    length(p) >= 2 || error("training line $id must be: input | expected output | optional quality")
    q = length(p) == 3 && !isempty(strip(p[3])) ? Symbol(strip(p[3])) : nothing
    return TrainingExample(id, strip(p[1]), strip(p[2]), q, nothing, String[], Symbol[], Dict{String,Any}())
end

function parse_training_json_obj(obj::Dict, id::Int)
    input = string(get(obj, "input", get(obj, :input, "")))
    isempty(input) && error("training record $id missing input")
    expected = get(obj, "expected_output", get(obj, :expected_output, nothing))
    expected_s = expected === nothing ? nothing : string(expected)
    qt = get(obj, "quality_target", get(obj, :quality_target, nothing))
    qt_s = qt === nothing || qt === nothing ? nothing : Symbol(string(qt))
    tags_raw = get(obj, "tags", get(obj, :tags, Any[]))
    tags = Symbol[Symbol(string(t)) for t in tags_raw]
    constraints_raw = get(obj, "constraints", get(obj, :constraints, Any[]))
    constraints = String[string(c) for c in constraints_raw]
    return TrainingExample(id, input, expected_s, qt_s, nothing, constraints, tags, Dict{String,Any}(string(k)=>v for (k,v) in obj))
end

function load_training_data(path::String)
    if !isfile(path)
        key = replace(path, "\\" => "/")
        for (k, v) in EMBEDDED_DATASETS
            endswith(key, k) && return load_training_data_from_string(v)
        end
        b = basename(path)
        haskey(EMBEDDED_DATASETS, b) && return load_training_data_from_string(EMBEDDED_DATASETS[b])
        error("training file not found: $path (no embedded fallback)")
    end
    if endswith(lowercase(path), ".jsonl")
        out = TrainingExample[]
        for (i, l) in enumerate(readlines(path))
            s = strip(l)
            isempty(s) && continue
            obj = from_json(s)
            obj isa Dict || error("JSONL line $i must be an object")
            push!(out, parse_training_json_obj(obj, i))
        end
        return out
    elseif endswith(lowercase(path), ".json")
        data = from_json(read(path, String))
        data isa AbstractVector || error("JSON training file must be an array of objects")
        return [parse_training_json_obj(obj, i) for (i, obj) in enumerate(data)]
    end
    out = TrainingExample[]
    for (i, l) in enumerate(readlines(path))
        x = parse_training_line(l, i)
        x !== nothing && push!(out, x)
    end
    return out
end


function load_training_data_from_string(text::String)
    out = TrainingExample[]
    i = 0
    for l in split(text, "\n")
        i += 1
        x = parse_training_line(String(l), i)
        x !== nothing && push!(out, x)
    end
    return out
end

function print_training_summary(ai::CognitiveAgent, report; io::IO=stdout)
    println(io, "\n" * "="^60)
    println(io, "                    TRAINING SUMMARY")
    println(io, "="^60)
    @printf(io, "  Episodes in Run:        %d\n", report.episodes)
    @printf(io, "  Total Reward:           +%.2f\n", report.total_reward)
    @printf(io, "  Total Punishment:       -%.2f\n", report.total_punishment)
    @printf(io, "  Avg Synapses Changed:   %.2f\n", report.average_changed)
    @printf(io, "  Self Corrections:       %d\n", report.self_corrections)
    @printf(io, "  Novel Recombinations:   %d\n", report.novelty)
    @printf(io, "  Runtime:                %.3f s\n", report.runtime_seconds)
    dist = report.synapses
    @printf(io, "  Ternary Synapses:       +1: %d (%.1f%%) | 0: %d (%.1f%%) | -1: %d (%.1f%%) [Total: %d]\n",
            dist.positive, dist.pct_positive, dist.zero, dist.pct_zero, dist.negative, dist.pct_negative, dist.total)
    println(io, "\n  Quality Distribution:")
    for (q, count) in sort(collect(report.quality); by=last, rev=true)
        @printf(io, "    %-22s: %d\n", string(q), count)
    end
    println(io, "="^60 * "\n")
end

function train!(ai::CognitiveAgent, data::Vector{TrainingExample}; epochs::Int=1, verbose::Bool=false, io::IO=stdout)
    start = time()
    total_examples = length(data)
    for epoch in 1:epochs
        verbose && println(io, "\n--- Epoch $epoch of $epochs ($total_examples examples) ---")
        for (i, x) in enumerate(data)
            (res, state) = respond!(ai, x.input; expected_output=x.expected_output, expected_quality=x.quality_target)
            output, eval_res, ep = res
            if verbose
                sig = eval_res.reward > 0f0 ? @sprintf("+%.2f", eval_res.reward) :
                      eval_res.punishment > 0f0 ? @sprintf("-%.2f", eval_res.punishment) : " 0.00"
                changes = length(ep.synaptic_changes)
                @printf(io, "  [%d/%d] \"%s\"\n         -> %s\n         [Quality: %s | Signal: %s | Transitions: %d]\n",
                        i, total_examples, x.input, output, string(eval_res.quality), sig, changes)
            end
        end
    end
    elapsed = time() - start
    ai.metrics.runtime_seconds += elapsed
    ai.metrics.peak_bytes = max(ai.metrics.peak_bytes, Base.gc_live_bytes())
    report = training_report(ai)
    verbose && print_training_summary(ai, report; io=io)
    return report
end

function train_file!(ai::CognitiveAgent, path::String; epochs::Int=1, verbose::Bool=true, io::IO=stdout)
    data = load_training_data(path)
    verbose && println(io, "[Training] Loaded $(length(data)) training examples from: $path")
    return train!(ai, data; epochs=epochs, verbose=verbose, io=io)
end

