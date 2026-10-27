
# network.jl — Sparse ternary network: units, synapses, discrete activation.
#
# Activation is genuinely computational: integer support/opposition over ternary
# synapses determines discrete unit states. There is no floating-point matmul and
# no learned float anywhere in this file.

function ensure_unit!(net::SparseTernaryNetwork, id::Int32, concept_ids::Vector{Int32}=Int32[id])
    if !haskey(net.units, id)
        net.units[id] = CognitiveUnit(id, concept_ids)
    end
    return net.units[id]
end

find_synapse(net::SparseTernaryNetwork, s::Int32, t::Int32) =
    (edges = get(net.outgoing, s, nothing); edges === nothing ? nothing :
     (i = findfirst(e -> e.target == t, edges); i === nothing ? nothing : edges[i]))

"""Create a ternary synapse without overwriting previously learned state."""
function connect!(net::SparseTernaryNetwork, s::Int32, t::Int32, value::Integer; episode::Int32=Int32(0))
    assert_ternary(value, "synapse")
    ensure_unit!(net, s); ensure_unit!(net, t)
    existing = find_synapse(net, s, t)
    existing !== nothing && return existing
    edge = SparseSynapse(s, t, Int8(value), episode)
    push!(get!(net.outgoing, s, SparseSynapse[]), edge)
    push!(get!(net.incoming, t, SparseSynapse[]), edge)
    net.synapse_count += 1
    return edge
end

out_edges(net::SparseTernaryNetwork, id::Int32) = get(net.outgoing, id, SparseSynapse[])
in_edges(net::SparseTernaryNetwork, id::Int32)  = get(net.incoming, id, SparseSynapse[])

function reset_activation!(net::SparseTernaryNetwork)
    for u in values(net.units)
        u.state = Int8(0)
    end
    return net
end

"""
Sparse discrete activation (spec §113). Seeds are set to +1; support is
propagated over `hops` passes. Each traversed synapse gains eligibility and its
`times_activated` counter increments. Returns `(active_ids, pathways)`.
"""
function activate!(net::SparseTernaryNetwork, seed_ids::Vector{Int32}, elig::EligibilityConfig;
                   hops::Int=2, trace::Union{Nothing,TraceBuffer}=nothing)
    reset_activation!(net)
    for id in seed_ids
        haskey(net.units, id) && (net.units[id].state = Int8(1))
    end
    active = Set{Int32}(filter(id -> haskey(net.units, id), seed_ids))
    pathways = Tuple{Int32,Int32}[]
    frontier = collect(active)
    for _ in 1:max(hops, 1)
        support = Dict{Int32,Int}()
        for src in frontier
            src_state = net.units[src].state
            src_state == 0 && continue
            for e in out_edges(net, src)
                # integer contribution: source polarity × synapse polarity
                support[e.target] = get(support, e.target, 0) + Int(src_state) * Int(e.value)
                e.eligibility = min(elig.cap, e.eligibility + elig.gain)
                e.times_activated += 1
                push!(pathways, (e.source, e.target))
            end
        end
        newly = Int32[]
        for (tgt, total) in support
            state = total > 0 ? Int8(1) : total < 0 ? Int8(-1) : Int8(0)
            u = net.units[tgt]
            if u.state != state
                u.state = state
            end
            push!(u.history, state)
            if state != 0 && !(tgt in active)
                push!(active, tgt); push!(newly, tgt)
                trace !== nothing && emit!(trace, UNIT_ACTIVATED; unit=tgt, state=state)
            end
        end
        isempty(newly) && break
        frontier = newly
    end
    return collect(active), pathways
end

active_unit_ids(net::SparseTernaryNetwork) = Int32[id for (id,u) in net.units if u.state != 0]
active_synapse_count(net::SparseTernaryNetwork) =
    sum(e -> e.eligibility > 0f0 ? 1 : 0, (e for edges in values(net.outgoing) for e in edges); init=0)

"""Decay all eligibility traces one step; clear those below the floor (spec §11)."""
function decay_eligibility!(net::SparseTernaryNetwork, elig::EligibilityConfig)
    for edges in values(net.outgoing), e in edges
        e.eligibility *= elig.decay
        e.eligibility < elig.floor && (e.eligibility = 0f0)
    end
    return net
end

