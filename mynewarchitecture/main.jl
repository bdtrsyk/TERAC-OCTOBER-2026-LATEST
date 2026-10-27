#!/usr/bin/env julia
# main.jl — TernaryCognitiveAI entry point
#
# Examples:
#   julia --project=. main.jl --cpu --demo
#   julia --project=. main.jl audit
#   julia --project=. main.jl stats
#   julia --project=. main.jl --chat
#   julia --project=. main.jl --bridge          # JSON-lines for Python tcai_bridge.py
#   julia --project=. main.jl --test

using TernaryCognitiveAI

exit(run_cli(ARGS))
