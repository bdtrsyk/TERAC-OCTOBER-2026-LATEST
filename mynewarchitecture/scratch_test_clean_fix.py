"""Test clean architectural fixes for sequence learning and non-saturation."""
from lumenmax import LumenMAX, SynapseMatrix, SymbolTable, tproj, assert_ternary
import numpy as np

# 1. SynapseMatrix with dead-zone theta=2 and zero-mean drift control
class CleanSynapseMatrix(SynapseMatrix):
    def _project(self, row: int) -> None:
        np.clip(self.A[row], -25, 25, out=self.A[row])
        # If row mean drifts significantly, center it to prevent all-ones saturation
        row_mean = int(np.mean(self.A[row]))
        adj = self.A[row] - (row_mean if abs(row_mean) >= 2 else 0)
        w = np.zeros(self.in_dim, dtype=np.int8)
        w[adj >= 2] = 1
        w[adj <= -2] = -1
        self.W[row] = w
        assert_ternary(self.W[row], self.name)

# 2. Balanced transition with theta=3 for W_gate and input-driven cand
def clean_transition(self, x: np.ndarray):
    expert_id = self.router.route(self.state, x)
    s_in = self.W_in.forward(x)
    s_rec = self.W_rec.forward(self.state)
    s_exp = self.experts[expert_id].forward(self.state)
    p_ctx = self.prompt_context if np.any(self.prompt_context) else self.context
    s_ctx = self.W_rec.forward(p_ctx)

    cand = tproj(
        s_in.astype(np.int32) * 2
        + s_rec.astype(np.int32)
        + s_exp.astype(np.int32)
        + s_ctx.astype(np.int32),
        theta=2,
    )

    gate_inp = np.concatenate([self.state, x])
    gate_raw = self.W_gate.W.astype(np.int32) @ gate_inp.astype(np.int32)
    gate = tproj(gate_raw, theta=3)

    new_state = np.zeros(self.N, dtype=np.int8)
    for i in range(self.N):
        if gate[i] == 0:
            new_state[i] = cand[i]
        elif gate[i] == 1:
            new_state[i] = self.state[i] if self.state[i] != 0 else cand[i]
        else:
            new_state[i] = 0

    if np.sum(np.abs(new_state)) < 12:
        new_state = cand

    self._adopt_context(x)
    return new_state, gate, expert_id

def clean_train_step(self, input_tok: str, target_tok: str, domain: str = "general", *, keep_state: bool = False, prev_state = None, is_at_sep: bool = False):
    tgt_id, _ = self.symbols.encode(target_tok, domain=domain)
    self.token_freq[target_tok] = self.token_freq.get(target_tok, 0) + 1
    self.token_freq[input_tok] = self.token_freq.get(input_tok, 0) + 1

    info = self.forward_step(
        input_tok,
        domain=domain,
        allow_eos=(target_tok == self.eos_token),
        prefer_domain=domain,
        is_generating_answer=False,
    )
    pred_id = info["pred_id"]
    correct = (pred_id == tgt_id)
    s = self._readout_state()
    prev = prev_state if prev_state is not None else np.zeros(self.N, dtype=np.int8)

    active_s = np.where(s != 0)[0]
    active_prev = np.where(prev != 0)[0]

    # Target reinforcement
    self.W_out.reinforce(tgt_id, s, magnitude=2)

    # Competitor suppression: push down any competitor that beat the target
    scores = self.W_out.W.astype(np.int32) @ s.astype(np.int32)
    tgt_score = scores[tgt_id] if tgt_id < len(scores) else 0

    if not correct:
        if pred_id >= 0 and pred_id != tgt_id:
            self.W_out.suppress(pred_id, s, magnitude=2)
        # Suppress top competitor if higher than target
        for rid in set(self.symbols.id2tok.keys()):
            if rid != tgt_id and rid < len(scores) and scores[rid] >= tgt_score:
                self.W_out.suppress(rid, s, magnitude=1)

        self.router.update(info["expert"], success=False)
        _, x = self.symbols.encode(input_tok, domain=domain)
        for r in active_s:
            self.W_in.reinforce(int(r), x, magnitude=1)
        if len(active_prev) > 0:
            for r in active_s:
                self.W_rec.suppress(int(r), prev, magnitude=1)
    else:
        self.router.update(info["expert"], success=True)
        if len(active_prev) > 0:
            for r in active_s:
                self.W_rec.reinforce(int(r), prev, magnitude=1)

    info["correct"] = correct
    info["target_id"] = tgt_id
    info["target_token"] = target_tok
    return info

LumenMAX._transition = clean_transition
LumenMAX.train_step = clean_train_step

m = LumenMAX(seed=42)
# Replace W_out with CleanSynapseMatrix
old_out = m.W_out
m.W_out = CleanSynapseMatrix(old_out.out_dim, old_out.in_dim, "W_out", seed=42)
m.W_out.W = old_out.W
m.W_out.A = old_out.A

m.seed_conversational_curriculum()
for ep in range(15):
    res = m.train_epoch()
    print(f"Epoch {ep+1:2d}: acc={res['accuracy']:.3f} correct={res['correct']}/{res['total']}")

for p in ["hello", "how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=25)
    print(f"\nPROMPT: '{p}' -> '{out}'")
    print("Tokens:", [t["pred_token"] for t in tel])
