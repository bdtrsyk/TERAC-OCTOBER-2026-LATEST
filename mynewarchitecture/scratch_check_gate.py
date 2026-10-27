from lumenmax import LumenMAX
import numpy as np

m = LumenMAX(seed=42)
tokens = ["hello", "how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"]

m.reset_state()
for tok in tokens:
    tok_id, x = m.symbols.encode(tok)
    expert_id = m.router.route(m.state, x)
    gate_inp = np.concatenate([m.state, x])
    gate = m.W_gate.forward(gate_inp)
    print(f"Token '{tok:6s}': gate sum: -1={np.sum(gate==-1):2d}, 0={np.sum(gate==0):2d}, +1={np.sum(gate==1):2d}")
    new_state, _, _ = m._transition(x)
    m.state = new_state
