from pathlib import Path
from lumenmax import LumenMAX, tproj, assert_ternary
import numpy as np

# Prototype Semantic Associative Grounding in LumenMAX
m = LumenMAX(seed=42)
dataset_txt = Path("dataset.txt").read_text(encoding="utf-8")
m.ingest_dataset_text(dataset_txt)

print(f"Loaded {len(m.dataset)} items and {len(m.sequences)} sequences.")

# Compute ternary semantic fingerprint for a query
def query_fingerprint(tokens):
    accum = np.zeros(m.N, dtype=np.int32)
    for tok in tokens:
        _, x = m.symbols.encode(tok)
        accum += x.astype(np.int32)
    return tproj(accum, theta=1)

# Test matching across sequences
def grounded_generate(prompt: str, max_tokens: int = 40):
    q_toks = m._tokenize(prompt.lower())
    q_fp = query_fingerprint(q_toks)

    best_seq = None
    best_score = -999999

    for seq_item in m.sequences:
        toks = seq_item["tokens"]
        if m.sep_token in toks:
            sep_idx = toks.index(m.sep_token)
            p_toks = toks[:sep_idx]
            p_fp = query_fingerprint(p_toks)

            # Discrete ternary dot product between prompt fingerprint and sequence prompt fingerprint
            score = int(q_fp.astype(np.int32) @ p_fp.astype(np.int32))

            # Exact keyword overlap bonus (integer)
            overlap = len(set(q_toks) & set(p_toks))
            score += overlap * 10

            if score > best_score:
                best_score = score
                best_seq = toks[sep_idx + 1:]

    if best_seq:
        # Run state transitions through the matched sequence to update internal ternary registers!
        m.reset_state()
        m.prompt_context = q_fp
        gen_tokens = []
        for tok in best_seq:
            if tok in (m.eos_token, "<eos>"):
                break
            info = m.forward_step(tok, is_generating_answer=True)
            gen_tokens.append(tok)
            if len(gen_tokens) >= max_tokens:
                break

        # Format text cleanly
        text = ""
        for tok in gen_tokens:
            if tok in (".", ",", "!", "?", ";", ":", "'", ")", "]", "}"):
                text = text.rstrip() + tok + " "
            elif tok in ("(", "[", "{"):
                text += tok
            else:
                text += tok + " "
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in (".", "!", "?"):
                text += "."
        return text, best_score

    return "", 0

test_prompts = [
    "hello",
    "how are you",
    "who are you",
    "what is default parameters?",
    "what is searching?",
    "how would you describe if else to a beginner?",
]

print("\n=== TEST GROUNDED GENERATION ===")
for p in test_prompts:
    res, sc = grounded_generate(p)
    print(f"PROMPT: '{p}' (score={sc})")
    print(f"OUTPUT: '{res}'")
    print("-" * 50)
