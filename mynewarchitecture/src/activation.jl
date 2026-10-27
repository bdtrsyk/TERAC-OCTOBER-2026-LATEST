# activation.jl — Discrete sparse activation over ternary synapses (spec §113).
# The canonical implementation is `activate!` in network.jl.
# This file exposes activation statistics used by diagnostics and audits.

function activation_support(net::SparseTernaryNetwork, id::Int32)
    total = 0
    for e in in_edges(net, id)
        src = get(net.units, e.source, nothing)
        src === nothing && continue
        total += Int(src.state) * Int(e.value)
    end
    return total
end
