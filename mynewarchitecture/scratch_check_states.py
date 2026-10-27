from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
tokens = ["hello", "how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"]

m.reset_state()
states = []
for tok in tokens:
    tok_id, x = m.symbols.encode(tok)
    new_state, gate, exp = m._transition(x)
    m.state = new_state
    states.append((tok, new_state.copy()))

print("State norms and Hamming distances between consecutive tokens:")
for i in range(len(states) - 1):
    t1, s1 = states[i]
    t2, s2 = states[i+1]
    h = np.sum(s1 != s2)
    print(f"  {t1:6s} -> {t2:6s} : dist={h:2d}/64 | norm1={np.sum(np.abs(s1)):2d} | norm2={np.sum(np.abs(s2)):2d}")
