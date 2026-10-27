# strategy_memory.jl — Procedural strategy statistics (spec §40).

function strategy_stats(agent::CognitiveAgent)
    out = Dict{Symbol,NamedTuple}()
    for (name, rec) in agent.strategy.records
        out[name] = (successes=rec.successes, failures=rec.failures,
                     last_used_episode=rec.last_used_episode)
    end
    return out
end

function strategy_success_rate(agent::CognitiveAgent, name::Symbol)
    rec = get(agent.strategy.records, name, nothing)
    rec === nothing && return 0.5
    total = rec.successes + rec.failures
    total == 0 && return 0.5
    return rec.successes / total
end
