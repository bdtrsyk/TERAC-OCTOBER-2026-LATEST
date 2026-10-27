#!/usr/bin/env julia
# experiments/run_experiment.jl
#
# Usage:
#   julia --project=. experiments/run_experiment.jl reasoning_depth
#   julia --project=. experiments/run_experiment.jl reward_ablation
#   julia --project=. experiments/run_experiment.jl repetition_learning
#   julia --project=. experiments/run_experiment.jl self_correction
#   julia --project=. experiments/run_experiment.jl novel_recombination

using TernaryCognitiveAI
using Dates
using SHA

function dataset_hash(path::AbstractString)
    isfile(path) || return "missing"
    return bytes2hex(sha256(read(path)))
end

function main(args)
    isempty(args) && (println("usage: run_experiment.jl <name>"); return 1)
    name = args[1]
    println("experiment: ", name)
    println("seed: 42")
    println("timestamp: ", Dates.now())
    println("julia: ", VERSION)
    println("dataset_hash(train.txt): ", dataset_hash(joinpath(@__DIR__, "..", "data", "train.txt")))
    println("dataset_hash(train.jsonl): ", dataset_hash(joinpath(@__DIR__, "..", "data", "train.jsonl")))
    run_experiment(name)
    return 0
end

exit(main(ARGS))
