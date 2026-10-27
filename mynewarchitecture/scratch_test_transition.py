from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)

# Test candidate-driven state transition with balanced gating
def test_new_transition(x, state, prompt_ctx):
    s_in = m.W_in.forward(x)
    s_rec = m.W_rec.forward(state)
    expert_id = m.router.route(state, x)
    s_exp = m.experts[expert_id].forward(state)
    s_ctx = m.W_rec.forward(prompt_ctx) if np.any(prompt_ctx) else np.zeros(m.N, dtype=np.int8)

    # Candidate combines input driving force, recurrence, expert, and context
    # Input has strong weight (2x) so input token always shifts state!
    cand = tproj(
        s_in.astype(np.int32) * 2
        + s_rec.astype(np.int32)
        + s_exp.astype(np.int32)
        + s_ctx.astype(np.int32),
        theta=2
    )

    # Gate balances between:
    # 0 -> adopt cand (fresh input-driven update)
    # 1 -> retain previous state (recurrent continuity)
    # -1 -> clear to 0 (sparse reset)
    gate_inp = np.concatenate([state, x])
    # Compute gate with scaled theta to produce real balanced ternary distribution
    gate_raw = m.W_gate.W.astype(np.int32) @ gate_inp.astype(np.int32)
    # With 128 inputs, std is ~4-5. theta=4 yields ~35% zeros!
    gate = tproj(gate_raw, theta=4)

    new_state = np.zeros(m.N, dtype=np.int8)
    for i in range(m.N):
        if gate[i] == 0:
            new_state[i] = cand[i]
        elif gate[i] == 1:
            new_state[i] = state[i] if state[i] != 0 else cand[i]
        else:
            new_state[i] = 0

    if np.sum(np.abs(new_state)) < 12:
        new_state = cand

    return new_state, gate

tokens = ["hello", "how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"]
state = np.zeros(m.N, dtype=np.int8)
p_accum = np.zeros(m.N, dtype=np.int32)
states = []

for tok in tokens:
    _, x = m.symbols.encode(tok)
    p_accum += x.astype(np.int32)
    state, gate = test_new_transition(x, state, tproj(p_accum, theta=1))
    states.append((tok, state.copy()))

print("New transition Hamming distances between consecutive tokens:")
for i in range(len(states) - 1):
    t1, s1 = states[i]
    t2, s2 = states[i+1]
    h = np.sum(s1 != s2)
    print(f"  {t1:6s} -> {t2:6s} : dist={h:2d}/64 | norm1={np.sum(np.abs(s1)):2d} | norm2={np.sum(np.abs(s2)):2d}")
