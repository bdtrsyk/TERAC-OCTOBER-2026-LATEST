from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)

def get_readout_for(prompt, seq_tokens):
    m.reset_state()
    p_accum = np.zeros(m.N, dtype=np.int32)
    for tok in m._tokenize(prompt):
        _, px = m.symbols.encode(tok)
        p_accum += px.astype(np.int32)
        m.forward_step(tok, is_generating_answer=False)
    m.prompt_context = tproj(p_accum, theta=1)

    m.forward_step(m.sep_token, is_generating_answer=True)
    for tok in seq_tokens:
        m.forward_step(tok, is_generating_answer=True)
    return m._readout_state().copy()

r1 = get_readout_for("how are you", ["i", "am"])
r2 = get_readout_for("who are you", ["i", "am"])
r3 = get_readout_for("what are you", ["i", "am"])

h12 = np.sum(r1 != r2)
h13 = np.sum(r1 != r3)
h23 = np.sum(r2 != r3)

print(f"Readout distance at 'am':")
print(f"  'how are you' vs 'who are you' : dist={h12}/64")
print(f"  'how are you' vs 'what are you': dist={h13}/64")
print(f"  'who are you' vs 'what are you': dist={h23}/64")
