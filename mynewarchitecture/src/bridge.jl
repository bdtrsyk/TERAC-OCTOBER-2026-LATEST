# bridge.jl — JSON-lines protocol shared with Python (tcai_bridge.py).
#
# Wire format: one JSON object per line on stdin; one JSON object per line on stdout.
# Keeps Julia TernaryCognitiveAI and Python LumenMAX on the same contract
# (shared/contract.json): ternary domain, quality classes, ask/teach/audit ops.

const BRIDGE_PROTOCOL_VERSION = 1

function _contract_path()
    return joinpath(dirname(@__DIR__), "shared", "contract.json")
end

function _load_shared_contract()
    path = _contract_path()
    if isfile(path)
        try
            return from_json(read(path, String))
        catch
            # fall through to embedded defaults
        end
    end
    return Dict{String,Any}(
        "contract_version" => 1,
        "ternary_domain" => [-1, 0, 1],
        "eos_token" => "<eos>",
        "engines" => Dict(
            "julia" => "TernaryCognitiveAI cognitive reasoner",
            "python" => "LumenMAX ternary sequence substrate",
        ),
    )
end

function _bridge_ok(payload::AbstractDict)
    out = Dict{String,Any}("ok" => true, "engine" => "julia", "ternary" => true)
    for (k, v) in payload
        out[string(k)] = v
    end
    return out
end

function _bridge_err(msg::AbstractString; code::String="error")
    return Dict{String,Any}(
        "ok" => false,
        "engine" => "julia",
        "ternary" => true,
        "error" => String(msg),
        "code" => code,
    )
end

function _bridge_handle(ai::CognitiveAgent, req::AbstractDict)
    op = lowercase(string(get(req, "op", "")))
    isempty(op) && return _bridge_err("missing op", code="bad_request")

    if op == "ping"
        return _bridge_ok(Dict(
            "protocol_version" => BRIDGE_PROTOCOL_VERSION,
            "architecture_version" => string(ARCHITECTURE_VERSION),
            "message" => "pong",
        ))
    elseif op == "contract"
        return _bridge_ok(Dict("contract" => _load_shared_contract()))
    elseif op == "ask" || op == "respond"
        text = string(get(req, "text", get(req, "input", "")))
        isempty(strip(text)) && return _bridge_err("empty text", code="bad_request")
        expected = get(req, "expected", nothing)
        expected_str = expected === nothing ? nothing : string(expected)
        (result, state) = respond!(ai, text; expected_output=expected_str)
        output, evaluation, _ep = result
        return _bridge_ok(Dict(
            "output" => String(output),
            "quality" => string(evaluation.quality),
            "confidence" => Float64(state.confidence),
            "stop_reason" => "complete",
            "reasons" => collect(String, evaluation.reasons),
            "reward" => Float64(evaluation.reward),
            "punishment" => Float64(evaluation.punishment),
            "steps" => Int(state.steps_used),
            "concepts" => concept_count(ai),
            "synapses" => synapse_count(ai),
        ))
    elseif op == "teach"
        text = string(get(req, "text", get(req, "input", "")))
        expected = string(get(req, "expected", get(req, "target", "")))
        (isempty(strip(text)) || isempty(strip(expected))) &&
            return _bridge_err("teach requires text and expected", code="bad_request")
        (result, state) = respond!(ai, text; expected_output=expected)
        output, evaluation, _ep = result
        return _bridge_ok(Dict(
            "output" => String(output),
            "quality" => string(evaluation.quality),
            "confidence" => Float64(state.confidence),
            "stop_reason" => "complete",
            "reasons" => collect(String, evaluation.reasons),
            "taught" => true,
        ))
    elseif op == "audit"
        dist = ternary_distribution(ai.network)
        integrity = try
            verify_integrity(ai)
            true
        catch
            false
        end
        return _bridge_ok(Dict(
            "integrity" => integrity,
            "ternary_distribution" => Dict(
                "negative" => dist.negative,
                "zero" => dist.zero,
                "positive" => dist.positive,
                "total" => dist.total,
                "pct_negative" => dist.pct_negative,
                "pct_zero" => dist.pct_zero,
                "pct_positive" => dist.pct_positive,
            ),
            "pure_ternary" => integrity,
        ))
    elseif op == "stats"
        return _bridge_ok(Dict(
            "concepts" => concept_count(ai),
            "synapses" => synapse_count(ai),
            "episodes" => episode_count(ai),
            "scale" => string(ai.config.scale),
        ))
    elseif op == "reset"
        scale = Symbol(string(get(req, "scale", "small")))
        return _bridge_ok(Dict("reset" => true, "scale" => string(scale),
                               "note" => "agent replaced in session loop"))
    elseif op == "shutdown"
        return _bridge_ok(Dict("shutdown" => true))
    else
        return _bridge_err("unknown op: $op", code="unknown_op")
    end
end

"""Run a single JSON-lines bridge session on stdin/stdout (Python-facing)."""
function run_json_bridge(; scale::Symbol=:small, load_path::Union{Nothing,String}=nothing,
                         io_in::IO=stdin, io_out::IO=stdout)
    ai = load_path === nothing ? cognitive_agent(scale) : load_model(load_path)
    println(io_out, to_json(_bridge_ok(Dict(
        "op" => "ready",
        "protocol_version" => BRIDGE_PROTOCOL_VERSION,
        "architecture_version" => string(ARCHITECTURE_VERSION),
        "scale" => string(ai.config.scale),
    ))))
    flush(io_out)

    while true
        eof(io_in) && break
        local line
        try
            line = readline(io_in)
        catch e
            # Pipe closed between the eof check and the read.
            e isa EOFError && break
            rethrow(e)
        end
        isempty(strip(line)) && continue
        local req
        try
            req = from_json(line)
            !(req isa AbstractDict) && error("request must be a JSON object")
        catch e
            println(io_out, to_json(_bridge_err("invalid JSON: $e", code="bad_json")))
            flush(io_out)
            continue
        end

        op = lowercase(string(get(req, "op", "")))
        if op == "reset"
            scale_s = Symbol(string(get(req, "scale", string(ai.config.scale))))
            try
                ai = cognitive_agent(scale_s)
                println(io_out, to_json(_bridge_ok(Dict("reset" => true, "scale" => string(scale_s)))))
            catch e
                println(io_out, to_json(_bridge_err("reset failed: $e", code="internal")))
            end
            flush(io_out)
            continue
        end

        resp = try
            _bridge_handle(ai, req)
        catch e
            _bridge_err("handler failed: $e", code="internal")
        end
        println(io_out, to_json(resp))
        flush(io_out)
        op == "shutdown" && break
    end
    return 0
end
