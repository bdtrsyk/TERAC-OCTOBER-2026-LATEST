#!/usr/bin/env julia
# build.jl — Optional native acceleration build (spec §81).
#
# The reference cognitive architecture is pure Julia and requires no native
# build. This script verifies optional native directories and documents the ABI
# expectation. A missing native library does not fail the Julia CPU reference.

println("TernaryCognitiveAI build")
println("Julia version: ", VERSION)

native_root = joinpath(@__DIR__, "native")
for sub in ("c", "cpp", "rust", "cuda")
    path = joinpath(native_root, sub)
    readme = joinpath(path, "README.md")
    if isdir(path)
        println("[native/$sub] present")
        if isfile(readme)
            println("  README: ok")
        else
            println("  README: missing")
        end
    else
        println("[native/$sub] absent (optional)")
    end
end

println()
println("Julia-only reference implementation: ALWAYS AVAILABLE")
println("No native library is required for CPU reasoning / learning.")
println("Build complete.")
