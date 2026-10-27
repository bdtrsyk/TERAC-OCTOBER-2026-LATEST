
# ternary.jl — Ternary value domain, invariant checks, packed storage, stats.
#
# The learned state domain is strictly {-1, 0, +1}. This file provides the
# canonical Int8 checks plus an OPTIONAL packed representation that must remain
# behaviorally equivalent to the Int8 reference (spec §50).

const TERNARY_VALUES = (Int8(-1), Int8(0), Int8(1))

@inline is_ternary(x::Integer) = x == -1 || x == 0 || x == 1

"""Throw an architecture-integrity error unless `x` is a valid ternary state."""
@inline function assert_ternary(x::Integer, where::AbstractString="synapse")
    is_ternary(x) || error("ArchitectureIntegrityError: non-ternary $where value = $x")
    return Int8(x)
end

"""Continuously verify the ternary invariant across the whole network (spec §4)."""
function assert_network_invariants(net::SparseTernaryNetwork)
    for edges in values(net.outgoing), e in edges
        is_ternary(e.value) || error("ArchitectureIntegrityError: synapse $(e.source)->$(e.target) has non-ternary value $(e.value)")
    end
    for u in values(net.units)
        is_ternary(u.state) || error("ArchitectureIntegrityError: unit $(u.id) has non-discrete state $(u.state)")
    end
    return true
end

"""Ternary distribution over all learned synapses (spec §89)."""
function ternary_distribution(net::SparseTernaryNetwork)
    neg = zero = pos = 0
    for edges in values(net.outgoing), e in edges
        e.value < 0 ? (neg += 1) : e.value > 0 ? (pos += 1) : (zero += 1)
    end
    total = neg + zero + pos
    pct(x) = total == 0 ? 0.0 : round(100 * x / total; digits=2)
    return (negative=neg, zero=zero, positive=pos, total=total,
            pct_negative=pct(neg), pct_zero=pct(zero), pct_positive=pct(pos))
end

# ── Optional packed ternary vector (2 bits per trit, 32 trits per UInt64) ────
# Encoding: 0b00 = 0, 0b01 = +1, 0b10 = -1. This is the compact native-style
# representation referenced in spec §50/§51; it is an optimization, never a
# replacement for the architecture.
const _TRITS_PER_WORD = 32

mutable struct PackedTernaryVector
    words::Vector{UInt64}
    len::Int
end

function PackedTernaryVector(n::Int)
    nwords = cld(n, _TRITS_PER_WORD)
    PackedTernaryVector(zeros(UInt64, nwords), n)
end

Base.length(p::PackedTernaryVector) = p.len

@inline function _encode(v::Integer)
    v == 0 && return UInt64(0)
    v == 1 && return UInt64(1)
    v == -1 && return UInt64(2)
    error("ArchitectureIntegrityError: cannot pack non-ternary value $v")
end

@inline _decode(bits::UInt64) = bits == 0 ? Int8(0) : bits == 1 ? Int8(1) : Int8(-1)

function Base.getindex(p::PackedTernaryVector, i::Int)
    (1 <= i <= p.len) || throw(BoundsError(p, i))
    w = (i - 1) ÷ _TRITS_PER_WORD + 1
    shift = ((i - 1) % _TRITS_PER_WORD) * 2
    bits = (p.words[w] >> shift) & UInt64(0b11)
    return _decode(bits)
end

function Base.setindex!(p::PackedTernaryVector, v::Integer, i::Int)
    (1 <= i <= p.len) || throw(BoundsError(p, i))
    w = (i - 1) ÷ _TRITS_PER_WORD + 1
    shift = ((i - 1) % _TRITS_PER_WORD) * 2
    mask = UInt64(0b11) << shift
    p.words[w] = (p.words[w] & ~mask) | (_encode(v) << shift)
    return v
end

"""Round-trip an Int8 vector through the packed store and confirm equivalence."""
function packed_roundtrip(values::AbstractVector{<:Integer})
    p = PackedTernaryVector(length(values))
    @inbounds for i in eachindex(values)
        p[i] = values[i]
    end
    out = Vector{Int8}(undef, length(values))
    @inbounds for i in eachindex(values)
        out[i] = p[i]
    end
    return out
end

"""Bytes used by the packed representation of `n` synapses vs the Int8 reference."""
function packed_storage_bytes(n::Int)
    packed = cld(n, _TRITS_PER_WORD) * 8
    reference = n                     # 1 byte per Int8
    naive_f64 = n * 8                 # if someone stored trits as Float64
    return (packed_bytes=packed, int8_bytes=reference, float64_bytes=naive_f64,
            bits_per_synapse=n == 0 ? 0.0 : round(packed * 8 / n; digits=3))
end

