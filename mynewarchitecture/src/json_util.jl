
# Simple zero-dependency JSON encoder and parser for Julia stdlib

function to_json(val)
    if val === nothing
        return "null"
    elseif val isa Bool
        return val ? "true" : "false"
    elseif val isa Number
        return isfinite(val) ? string(val) : "null"
    elseif val isa Symbol
        return "\"" * _escape_json(string(val)) * "\""
    elseif val isa AbstractString
        return "\"" * _escape_json(String(val)) * "\""
    elseif val isa AbstractVector || val isa Tuple || val isa Set
        return "[" * join((to_json(x) for x in val), ",") * "]"
    elseif val isa AbstractDict
        items = String["\"" * _escape_json(string(k)) * "\":" * to_json(v) for (k, v) in val]
        return "{" * join(items, ",") * "}"
    elseif val isa NamedTuple
        items = String["\"" * _escape_json(string(k)) * "\":" * to_json(getfield(val, k)) for k in keys(val)]
        return "{" * join(items, ",") * "}"
    else
        # Fallback: if it has struct fields, serialize as dict
        fnames = fieldnames(typeof(val))
        if !isempty(fnames)
            items = String["\"" * _escape_json(string(k)) * "\":" * to_json(getfield(val, k)) for k in fnames]
            return "{" * join(items, ",") * "}"
        else
            return "\"" * _escape_json(string(val)) * "\""
        end
    end
end

function _escape_json(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            print(buf, "\\\"")
        elseif c == '\\'
            print(buf, "\\\\")
        elseif c == '\b'
            print(buf, "\\b")
        elseif c == '\f'
            print(buf, "\\f")
        elseif c == '\n'
            print(buf, "\\n")
        elseif c == '\r'
            print(buf, "\\r")
        elseif c == '\t'
            print(buf, "\\t")
        elseif Int(c) < 32
            print(buf, "\\u", string(Int(c); base=16, pad=4))
        else
            print(buf, c)
        end
    end
    return String(take!(buf))
end

struct JSONParser
    src::String
    len::Int
    idx::Ref{Int}
end

function from_json(s::AbstractString)
    p = JSONParser(String(s), lastindex(s), Ref(1))
    _skip_ws(p)
    res = _parse_val(p)
    return res
end

function _skip_ws(p::JSONParser)
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c in (' ', '\t', '\n', '\r')
            p.idx[] = nextind(p.src, p.idx[])
        else
            break
        end
    end
end

function _parse_val(p::JSONParser)
    _skip_ws(p)
    p.idx[] > p.len && return nothing
    c = p.src[p.idx[]]
    if c == '"'
        return _parse_string(p)
    elseif c == '{'
        return _parse_object(p)
    elseif c == '['
        return _parse_array(p)
    elseif c in ('t', 'f')
        return _parse_bool(p)
    elseif c == 'n'
        return _parse_null(p)
    elseif c in ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
        return _parse_number(p)
    else
        error("Unexpected character in JSON: '$c' at position $(p.idx[])")
    end
end

function _parse_string(p::JSONParser)
    # skip opening quote
    p.idx[] = nextind(p.src, p.idx[])
    buf = IOBuffer()
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c == '"'
            p.idx[] = nextind(p.src, p.idx[])
            return String(take!(buf))
        elseif c == '\\'
            p.idx[] = nextind(p.src, p.idx[])
            p.idx[] > p.len && error("Unterminated escape sequence in JSON string")
            esc_c = p.src[p.idx[]]
            if esc_c == '"' || esc_c == '\\' || esc_c == '/'
                print(buf, esc_c)
            elseif esc_c == 'b'
                print(buf, '\b')
            elseif esc_c == 'f'
                print(buf, '\f')
            elseif esc_c == 'n'
                print(buf, '\n')
            elseif esc_c == 'r'
                print(buf, '\r')
            elseif esc_c == 't'
                print(buf, '\t')
            elseif esc_c == 'u'
                hex_str = p.src[p.idx[]+1 : min(p.len, p.idx[]+4)]
                print(buf, Char(parse(UInt32, hex_str; base=16)))
                p.idx[] += 4
            else
                print(buf, esc_c)
            end
        else
            print(buf, c)
        end
        p.idx[] = nextind(p.src, p.idx[])
    end
    error("Unterminated JSON string")
end

function _parse_object(p::JSONParser)
    p.idx[] = nextind(p.src, p.idx[]) # skip '{'
    dict = Dict{String, Any}()
    _skip_ws(p)
    if p.idx[] <= p.len && p.src[p.idx[]] == '}'
        p.idx[] = nextind(p.src, p.idx[])
        return dict
    end
    while p.idx[] <= p.len
        _skip_ws(p)
        p.src[p.idx[]] != '"' && error("Expected string key in JSON object at $(p.idx[])")
        key = _parse_string(p)
        _skip_ws(p)
        (p.idx[] > p.len || p.src[p.idx[]] != ':') && error("Expected ':' after key in JSON object at $(p.idx[])")
        p.idx[] = nextind(p.src, p.idx[]) # skip ':'
        val = _parse_val(p)
        dict[key] = val
        _skip_ws(p)
        if p.idx[] <= p.len && p.src[p.idx[]] == ','
            p.idx[] = nextind(p.src, p.idx[])
        elseif p.idx[] <= p.len && p.src[p.idx[]] == '}'
            p.idx[] = nextind(p.src, p.idx[])
            return dict
        else
            error("Expected ',' or '}' in JSON object at $(p.idx[])")
        end
    end
    error("Unterminated JSON object")
end

function _parse_array(p::JSONParser)
    p.idx[] = nextind(p.src, p.idx[]) # skip '['
    arr = Any[]
    _skip_ws(p)
    if p.idx[] <= p.len && p.src[p.idx[]] == ']'
        p.idx[] = nextind(p.src, p.idx[])
        return arr
    end
    while p.idx[] <= p.len
        val = _parse_val(p)
        push!(arr, val)
        _skip_ws(p)
        if p.idx[] <= p.len && p.src[p.idx[]] == ','
            p.idx[] = nextind(p.src, p.idx[])
        elseif p.idx[] <= p.len && p.src[p.idx[]] == ']'
            p.idx[] = nextind(p.src, p.idx[])
            return arr
        else
            error("Expected ',' or ']' in JSON array at $(p.idx[])")
        end
    end
    error("Unterminated JSON array")
end

function _parse_bool(p::JSONParser)
    if startswith(p.src[p.idx[]:end], "true")
        p.idx[] += 4
        return true
    elseif startswith(p.src[p.idx[]:end], "false")
        p.idx[] += 5
        return false
    else
        error("Invalid boolean in JSON at $(p.idx[])")
    end
end

function _parse_null(p::JSONParser)
    if startswith(p.src[p.idx[]:end], "null")
        p.idx[] += 4
        return nothing
    else
        error("Invalid null in JSON at $(p.idx[])")
    end
end

function _parse_number(p::JSONParser)
    start_pos = p.idx[]
    while p.idx[] <= p.len
        c = p.src[p.idx[]]
        if c in ('0','1','2','3','4','5','6','7','8','9','-','+','.','e','E')
            p.idx[] = nextind(p.src, p.idx[])
        else
            break
        end
    end
    num_str = p.src[start_pos : prevind(p.src, p.idx[])]
    if occursin('.', num_str) || occursin('e', num_str) || occursin('E', num_str)
        return parse(Float64, num_str)
    else
        return parse(Int, num_str)
    end
end

