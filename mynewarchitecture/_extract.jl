# _extract.jl — ONE-SHOT historical splitter (do not run casually).
#
# This script originally split the flat engine.jl bundle into per-module files.
# That split already happened (see src/). The new modules created afterwards
# (units, pathways, activation, intent, *_memory, rewards, punishment,
# eligibility, credit_assignment, plasticity, audit, experiments) have no
# counterpart in engine.jl, so re-running a split cannot reproduce src/.
#
# To prevent accidental overwrites of the canonical src/ tree, output goes to
# a scratch directory (extract_out/) and nothing under src/ is touched.
const root = @__DIR__
engine = read(joinpath(root, "engine.jl"), String)
lines = split(engine, "\n"; keepempty=true)

function write_range(path, start_line, end_line)
    open(path, "w") do io
        for i in start_line:end_line
            println(io, lines[i])
        end
    end
    println("wrote ", basename(path), " lines ", start_line, "-", end_line)
end

src = joinpath(root, "extract_out")
mkpath(src)

write_range(joinpath(src, "embedded_assets.jl"), 2638, 3346)
write_range(joinpath(src, "memory_systems.jl"), 928, 1044)
write_range(joinpath(src, "reasoning.jl"), 1907, 2136)

sections = [
    ("config.jl", 15, 130),
    ("events.jl", 135, 229),
    ("types.jl", 234, 609),
    ("synapses.jl", 614, 719),
    ("network.jl", 724, 823),
    ("concepts.jl", 828, 914),
    ("language.jl", 1049, 1189),
    ("hypotheses.jl", 1194, 1365),
    ("contradiction.jl", 1370, 1457),
    ("self_correction.jl", 1462, 1529),
    ("meta_reasoning.jl", 1534, 1616),
    ("confidence.jl", 1621, 1659),
    ("novelty.jl", 1664, 1843),
    ("agent.jl", 1857, 1902),
    ("output_generation.jl", 2141, 2194),
    ("evaluator.jl", 2199, 2368),
    ("learning.jl", 2373, 2587),
    ("diagnostics.jl", 2592, 2622),
    ("json_util.jl", 3350, 3587),
    ("serialization.jl", 3592, 3600),
    ("benchmarks.jl", 3701, 3720),
]
for (name, a, b) in sections
    write_range(joinpath(src, name), a, b)
end

write_range(joinpath(src, "training.jl"), 3605, 3696)
write_range(joinpath(src, "cli.jl"), 3725, 4163)
write_range(joinpath(src, "gui.jl"), 4168, 4499)

println("done")
