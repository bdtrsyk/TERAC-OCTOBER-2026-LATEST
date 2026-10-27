# units.jl — Computational unit helpers (spec §5).
# Unit state is discrete. Synapses are separate ternary learned parameters.
# Core type: CognitiveUnit in types.jl. Construction/activation live in network.jl.

unit_state(net::SparseTernaryNetwork, id::Int32) =
    haskey(net.units, id) ? net.units[id].state : Int8(0)

unit_count(net::SparseTernaryNetwork) = length(net.units)
