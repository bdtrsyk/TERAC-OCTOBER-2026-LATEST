# eligibility.jl — Temporal credit eligibility traces (spec §11).
#
# Eligibility is continuous and transient. Synaptic *state* remains ternary.
# Core decay is `decay_eligibility!` in network.jl; activation increments live
# inside `activate!`.

function eligibility_stats(net::SparseTernaryNetwork)
    vals = Float32[e.eligibility for edges in values(net.outgoing) for e in edges if e.eligibility > 0f0]
    isempty(vals) && return (count=0, mean=0f0, max=0f0, sum=0f0)
    return (count=length(vals), mean=sum(vals)/length(vals), max=maximum(vals), sum=sum(vals))
end

function clear_eligibility!(net::SparseTernaryNetwork)
    for edges in values(net.outgoing), e in edges
        e.eligibility = 0f0
    end
    return net
end

function eligible_synapses(net::SparseTernaryNetwork; floor::Float32=0f0)
    return [(e.source, e.target, e.value, e.eligibility)
            for edges in values(net.outgoing) for e in edges if e.eligibility > floor]
end
