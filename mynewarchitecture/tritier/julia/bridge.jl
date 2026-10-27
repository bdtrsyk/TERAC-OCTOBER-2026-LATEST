#!/usr/bin/env julia
include("TriTierAdapter.jl")
using .TriTierAdapter

# A newline-delimited, deliberately small protocol for out-of-process use:
# "score|<comma-separated Int8>|<comma-separated Int8>" -> integer score.
# Production Rust integration should launch this executable and maintain the
# process, never transfer entire model blocks across the boundary.
for line in eachline(stdin)
    fields = split(chomp(line), '|')
    if length(fields) == 3 && fields[1] == "score"
        before = Int8.(parse.(Int, split(fields[2], ',')))
        delta = Int8.(parse.(Int, split(fields[3], ',')))
        println(TriTierAdapter.evaluate_candidate(before, delta))
    else
        println("error|invalid request")
    end
    flush(stdout)
end
