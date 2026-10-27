from tcai_bridge import UnifiedEngine

u = UnifiedEngine(engine="unified")
m = u.python

q_toks = [t.lower() for t in m._tokenize("does sun cause heat?") if t not in ("?", "!", ".", ",")]
q_set = set(q_toks)
print("q_set:", q_set)

for s in m.sequences:
    toks = s["tokens"]
    if m.sep_token in toks:
        idx = toks.index(m.sep_token)
        p_toks = [t.lower() for t in toks[:idx] if t not in ("?", "!", ".", ",")]
        p_set = set(p_toks)
        overlap = len(q_set & p_set)
        if overlap > 0:
            score = overlap * 25 - abs(len(q_set) - len(p_set)) * 4
            if q_set == p_set: score += 60
            elif q_toks and p_toks and q_toks[0] == p_toks[0]: score += 15
            print(f"Overlap {overlap} on {' '.join(p_toks)}: score={score}")
