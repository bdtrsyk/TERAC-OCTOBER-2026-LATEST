from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()
for ep in range(15):
    m.train_epoch()

prompt = "how are you"
tokens = m._tokenize(prompt)
m.reset_state()
p_accum = np.zeros(m.N, dtype=np.int32)
for tok in tokens:
    tid, x = m.symbols.encode(tok, domain="chat")
    p_accum += x.astype(np.int32)
    m.forward_step(tok, domain="chat", prefer_domain="chat", is_generating_answer=False)
m.prompt_context = tproj(p_accum, theta=1)

# Now transition on <sep>
sep_info = m.forward_step(m.sep_token, domain="chat", prefer_domain="chat", is_generating_answer=True)
readout = m._readout_state()
scores = m.W_out.W.astype(np.int32) @ readout.astype(np.int32)

# Look at top scoring REGISTERED tokens
registered = list(m.symbols.id2tok.keys())
reg_scores = [(rid, m.symbols.id2tok[rid], scores[rid]) for rid in registered]
reg_scores.sort(key=lambda x: x[2], reverse=True)

print(f"Total registered tokens: {len(registered)}")
print("Top 15 REGISTERED tokens after <sep>:")
for rid, tok, sc in reg_scores[:15]:
    print(f"  [{rid:3d}] {tok:20s}: score={sc:3d}")

# What was the score of "i"?
i_id = m.symbols.tok2id.get("i")
print(f"\nScore of 'i' (id={i_id}): {scores[i_id] if i_id is not None else 'NOT FOUND'}")

