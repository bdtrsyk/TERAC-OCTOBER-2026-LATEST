from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
]

for ep in range(10):
    for toks, dom in seqs:
        m.reset_state()
        sep_idx = toks.index(m.sep_token)
        p_accum = np.zeros(m.N, dtype=np.int32)
        for ptok in toks[:sep_idx]:
            _, px = m.symbols.encode(ptok, domain=dom)
            p_accum += px.astype(np.int32)
            m.forward_step(ptok, domain=dom, is_generating_answer=False)
        m.prompt_context = tproj(p_accum, theta=1)
        m.train_step(m.sep_token, toks[sep_idx+1], domain=dom, keep_state=True)
        for i in range(sep_idx+1, len(toks)-1):
            m.train_step(toks[i], toks[i+1], domain=dom, keep_state=True)

# Now check scores at <sep> for "how are you"
m.reset_state()
p_accum = np.zeros(m.N, dtype=np.int32)
for ptok in ["how", "are", "you"]:
    _, px = m.symbols.encode(ptok)
    p_accum += px.astype(np.int32)
    m.forward_step(ptok, is_generating_answer=False)
m.prompt_context = tproj(p_accum, theta=1)
m.forward_step(m.sep_token, is_generating_answer=True)
s = m._readout_state()
scores = m.W_out.W.astype(np.int32) @ s.astype(np.int32)

for tok in ["i", "doing", "am", "well", "lumenmax"]:
    tid = m.symbols.tok2id[tok]
    print(f"Token '{tok:10s}' score at <sep>: {scores[tid]:3d} | W_out row={m.W_out.W[tid][:10]}")
