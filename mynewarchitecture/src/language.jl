
# language.jl — Replaceable local language interface (spec §38).
#
# Deliberately a transparent symbolic grammar (no LLM, no learned tokenizer).
# Produces structured representations; the reasoning engine never operates on raw
# text alone. Intent (spec §16, §17) is first-class.

const _NUMBER_WORDS = Dict("zero"=>0,"one"=>1,"two"=>2,"three"=>3,"four"=>4,"five"=>5,
                           "six"=>6,"seven"=>7,"eight"=>8,"nine"=>9,"ten"=>10)

# Ordered longest-first so multi-word predicates match before single words.
const _PREDICATE_PHRASES = [
    ("is opposite of", :opposite_of), ("is similar to", :similar_to),
    ("results in", :causes), ("leads to", :causes), ("related to", :related_to),
    ("part of", :part_of), ("is a", :is_a),
    ("causes", :causes), ("cause", :causes),
    ("prevents", :prevents), ("prevent", :prevents),
    ("requires", :requires), ("require", :requires),
    ("supports", :supports), ("support", :supports),
    ("contradicts", :contradicts), ("contradict", :contradicts),
    ("inhibits", :inhibits), ("inhibit", :inhibits),
    ("blocks", :blocks), ("block", :blocks),
    ("reduces", :reduces), ("reduce", :reduces),
    ("enables", :enables), ("enable", :enables),
    ("affects", :affects), ("affect", :affects),
]

const _STOPWORDS = Set(["the","a","an","of","to","in","on","at","and","or","that","this",
                        "then","with","for","does","do","is","are","can","why","what","how",
                        "will","would","it","its","as","by","be"])

tokenize(text::AbstractString) = split(lowercase(strip(String(text))))
normalize(text::AbstractString) = strip(replace(lowercase(String(text)), r"[?.!,]" => ""))
segment_sentences(text::AbstractString) = filter(!isempty, strip.(split(String(text), r"[.!?]+")))

function detect_repetition_request(text::AbstractString)
    lower = lowercase(String(text))
    requested = occursin(r"\b(repeat|say (it|that|this) again|say again|again exactly|reiterate)\b", lower)
    count = 1
    m = match(r"(\d+)\s*times?", lower)
    if m !== nothing
        count = parse(Int, m.captures[1])
    else
        for (w, n) in _NUMBER_WORDS
            if occursin(Regex("\\b$w\\s+times?\\b"), lower)
                count = n; break
            end
        end
    end
    return requested, count
end

detect_instruction(text::AbstractString) =
    occursin(r"\b(repeat|explain|list|convert|transform|rewrite|summariz|compare|solve|show)\b", lowercase(String(text)))

function extract_constraints(text::AbstractString)
    lower = lowercase(String(text))
    constraints = String[]
    structure = ""
    exclusions = String[]
    occursin(r"\bin json\b", lower)      && (structure = "json")
    occursin(r"\bas a list\b", lower)    && (structure = "list")
    occursin(r"\bone (word|sentence)\b", lower) && push!(constraints, "concise")
    occursin(r"\bonly\b", lower)         && push!(constraints, "only")
    m = match(r"\bwithout ([a-z ]+)", lower)
    m !== nothing && push!(exclusions, strip(m.captures[1]))
    me = match(r"\bexclude ([a-z ]+)", lower)
    me !== nothing && push!(exclusions, strip(me.captures[1]))
    return constraints, structure, exclusions
end

"""Parse explicit user intent (spec §17)."""
function parse_intent(text::AbstractString)
    lower = lowercase(strip(String(text)))
    requested, count = detect_repetition_request(text)
    goal = if requested
        :repeat
    elseif startswith(lower, "why")
        :explain
    elseif occursin(r"^(does|is|are|can|do|will|would)\b", lower)
        :verify
    elseif occursin(r"\b(transform|convert|rewrite|translate)\b", lower)
        :transform
    else
        :assert
    end
    constraints, structure, exclusions = extract_constraints(text)
    intent = UserIntent(goal)
    intent.requested_repetition = requested
    intent.requested_repetition_count = count
    intent.requested_structure = structure
    intent.constraints = constraints
    intent.exclusions = exclusions
    intent.requested_quantity = count
    return intent
end

"""Return `(predicate::Symbol, subject::String, object::String)` or `nothing`."""
function match_relation(normalized::AbstractString)
    for (phrase, pred) in _PREDICATE_PHRASES
        m = match(Regex("^(.+?)\\s+" * phrase * "\\s+(.+)\$"), normalized)
        m !== nothing && return (pred, strip(m.captures[1]), strip(m.captures[2]))
    end
    return nothing
end

content_words(normalized::AbstractString) =
    filter(w -> length(w) > 2 && !(w in _STOPWORDS), split(normalized))

"""
Full structured parse used by the reasoning engine. Creates concepts, asserts a
relation when the input is a declarative fact, and detects a query target when
the input is interrogative.
"""
function parse_input!(agent::CognitiveAgent, text::String; episode::Int32=Int32(0), domain::String="general")
    intent = parse_intent(text)
    normalized = normalize(text)
    is_query = intent.goal in (:verify, :explain)
    body = is_query ? strip(replace(normalized, r"^(why|does|is|are|can|do|will|would)\s+" => "")) : normalized
    body = strip(replace(body, r"\b(in json|as a list|in one (word|sentence)|one (word|sentence)|only)\b" => ""))
    body = strip(replace(body, r"\b(without|exclude)\s+[a-z0-9_\- ]+\b" => ""))

    matched = match_relation(body)
    if matched !== nothing
        pred, subj, obj = matched
        if is_query
            s = concept!(agent, subj; domain=domain); o = concept!(agent, obj; domain=domain)
            return (seed_ids=Int32[s,o], learned_relation=nothing,
                    query_target=(s, pred, o), intent=intent, is_query=true)
        else
            rel = add_relation!(agent, subj, pred, obj; provenance=:input, domain=domain, episode=episode)
            return (seed_ids=Int32[rel.subject, rel.object], learned_relation=rel,
                    query_target=nothing, intent=intent, is_query=false)
        end
    end

    words = content_words(body)
    ids = Int32[concept!(agent, w; domain=domain) for w in words]
    return (seed_ids=ids, learned_relation=nothing, query_target=nothing, intent=intent, is_query=is_query)
end

