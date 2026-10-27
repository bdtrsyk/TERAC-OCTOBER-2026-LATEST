# pathways.jl — Pathway tracking for structured credit assignment (spec §12, §101).
# Active pathways are recorded during `activate!` and stored on CognitiveState.

pathway_key(source::Int32, target::Int32) = (source, target)

function unique_pathways(pathways::Vector{Tuple{Int32,Int32}})
    seen = Set{Tuple{Int32,Int32}}()
    out = Tuple{Int32,Int32}[]
    for p in pathways
        p in seen && continue
        push!(seen, p)
        push!(out, p)
    end
    return out
end

function pathway_overlap(a::Vector{Tuple{Int32,Int32}}, b::Vector{Tuple{Int32,Int32}})
    sa = Set(a)
    return count(p -> p in sa, b)
end
