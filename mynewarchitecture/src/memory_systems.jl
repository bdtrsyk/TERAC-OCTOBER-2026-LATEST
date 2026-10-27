
# memory.jl — Working, semantic, episodic, strategy, and failure memory.
#
# Memory is structured, not a text log (spec §23). Retrieval is selective and the
# stores interact: semantic retrieval is biased by concept reliability, episodic
# recall surfaces prior successes, and failure memory lets the engine avoid
# repeating known-bad reasoning (spec §62, §63).

# ── Working memory (bounded, spec §22) ──────────────────────────────────────
function remember!(agent::CognitiveAgent, text::String, ids::Vector{Int32};
                   relevance::Float32=1f0, step::Int=0, kind::Symbol=:observation)
    wm = agent.working
    push!(wm.items, MemoryItem(text, copy(ids), relevance, Int32(step), kind))
    if length(wm.items) > wm.capacity
        # Evict the least-relevant item (structured eviction, not FIFO of text).
        idx = argmin([it.relevance for it in wm.items])
        deleteat!(wm.items, idx)
    end
    return wm.items[end]
end

working_memory_texts(agent::CognitiveAgent) = [it.text for it in agent.working.items]
# NOTE: clear_working_memory! lives in working_memory.jl (single definition).

# ── Semantic retrieval (selective, spec §23, §28) ───────────────────────────
"""
Score and return the most relevant concept ids for a query. Scoring combines
lexical identity, graph connectivity (relations linking to the query), learned
reliability, and cross-domain bridging so an exact keyword match is not required.
"""
function retrieve_concepts(agent::CognitiveAgent, query_ids::Vector{Int32};
                           limit::Int=agent.config.retrieval_limit,
                           trace::Union{Nothing,TraceBuffer}=nothing)
    agent.config.memory_enabled || return Int32[]
    query = Set(query_ids)
    query_domains = Set{String}()
    for q in query_ids
        union!(query_domains, concept_domains(agent, q))
    end
    scored = Tuple{Int32,Float32}[]
    for (id, concept) in agent.semantic.concepts
        lexical = id in query ? 2f0 : 0f0
        connected = 0
        for r in agent.semantic.relations
            ((r.subject == id && r.object in query) || (r.object == id && r.subject in query)) && (connected += 1)
        end
        # cross-domain bridge: a shared domain tag counts even without a direct edge
        bridge = isempty(intersect(concept.domain_tags, query_domains)) ? 0f0 : 0.5f0
        reliability = concept_reliability(concept)
        score = lexical + Float32(connected) + bridge + reliability
        score > 0f0 && push!(scored, (id, score))
    end
    sort!(scored; by=last, rev=true)
    ids = Int32[first(x) for x in scored[1:min(limit, length(scored))]]
    if trace !== nothing
        for id in ids
            emit!(trace, CONCEPT_RETRIEVED; concept=id, name=concept_name(agent, id))
        end
    end
    return ids
end

# ── Episodic recall (spec §63) ──────────────────────────────────────────────
"""Recall past episodes whose representation overlaps the query concepts."""
function retrieve_episodes(agent::CognitiveAgent, query_ids::Vector{Int32}; limit::Int=5, successful_only::Bool=false)
    query = Set(query_ids)
    scored = Tuple{Int,Int}[]
    for (i, ep) in enumerate(agent.episodic.episodes)
        successful_only && !ep.success && continue
        overlap = count(id -> id in query, ep.representation)
        overlap > 0 && push!(scored, (i, overlap))
    end
    sort!(scored; by=last, rev=true)
    return [agent.episodic.episodes[first(x)] for x in scored[1:min(limit, length(scored))]]
end

function store_episode!(agent::CognitiveAgent, ep::CognitiveEpisode)
    push!(agent.episodic.episodes, ep)
    agent.episodic.next_id = ep.id + 1
    return ep
end

# ── Procedural strategy memory (spec §40) ───────────────────────────────────
function strategy_record!(agent::CognitiveAgent, name::Symbol)
    get!(agent.strategy.records, name, StrategyRecord(name))
end

function note_strategy_outcome!(agent::CognitiveAgent, name::Symbol, success::Bool; episode::Int32=Int32(0))
    rec = strategy_record!(agent, name)
    success ? (rec.successes += 1) : (rec.failures += 1)
    rec.last_used_episode = episode
    return rec
end

strategy_success_rate(rec::StrategyRecord) = (rec.successes + 1f0) / (rec.successes + rec.failures + 2f0)
# NOTE: strategy_success_rate(agent, name) lives in strategy_memory.jl (single
# definition; side-effect-free with 0.5 prior for unknown strategies).

"""Strategies ordered by historical success (used to bias selection, spec §40)."""
function preferred_strategies(agent::CognitiveAgent)
    recs = collect(values(agent.strategy.records))
    sort!(recs; by=strategy_success_rate, rev=true)
    return [r.name for r in recs]
end

# ── Failure memory (spec §62) ───────────────────────────────────────────────
function record_failure!(agent::CognitiveAgent, episode::Int, context::String, reason::String,
                         strategy::Symbol; claim::Union{Nothing,CognitiveRelation}=nothing,
                         alternative::Union{Nothing,String}=nothing)
    push!(agent.failures.records, FailureRecord(episode, context, reason, strategy, claim, alternative))
end

"""Recall prior failures relevant to the current context (avoid repeating them)."""
function recall_failures(agent::CognitiveAgent, context::AbstractString)
    key = lowercase(strip(String(context)))
    return [f for f in agent.failures.records if occursin(lowercase(f.context), key) || occursin(key, lowercase(f.context))]
end

