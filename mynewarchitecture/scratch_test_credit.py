from lumenmax import LumenMAX, tproj
from typing import Optional
import numpy as np

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()

# Patch train_step to update all active rows in s instead of [:6], [:8], [:4]
def patched_train_step(self, input_tok: str, target_tok: str, domain: str = "general", *, keep_state: bool = False, prev_state: Optional[np.ndarray] = None):
    tgt_id, _ = self.symbols.encode(target_tok, domain=domain)
    self.token_freq[target_tok] = self.token_freq.get(target_tok, 0) + 1
    self.token_freq[input_tok] = self.token_freq.get(input_tok, 0) + 1

    rare = max(1, 8 - min(7, self.token_freq.get(target_tok, 1) // 4))

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
    eos_boost = 2 if target_tok == self.eos_token else 0

    active_s = np.where(s != 0)[0]
    active_prev = np.where(prev != 0)[0]

    if correct:
        self.W_out.reinforce(tgt_id, s, magnitude=2 + rare // 2 + eos_boost)
        self.router.update(info["expert"], success=True)
        # Update recurrent connections for active state dimensions
        if len(active_prev) > 0:
            for r in active_s:
                self.W_rec.reinforce(int(r), prev, magnitude=1)
    else:
        self.W_out.reinforce(tgt_id, s, magnitude=2 + rare // 2 + eos_boost)
        if pred_id >= 0 and pred_id != tgt_id:
            self.W_out.suppress(pred_id, s, magnitude=2)
        self.router.update(info["expert"], success=False)
        _, x = self.symbols.encode(input_tok, domain=domain)
        # Train input matrix so input token drives active state dimensions
        for r in active_s:
            self.W_in.reinforce(int(r), x, magnitude=1)
        if pred_id >= 0 and pred_id != tgt_id and len(active_prev) > 0:
            for r in active_s:
                self.W_rec.suppress(int(r), prev, magnitude=1)

    info["correct"] = correct
    info["target_id"] = tgt_id
    info["target_token"] = target_tok
    return info

LumenMAX.train_step = patched_train_step

for ep in range(15):
    res = m.train_epoch()
    print(f"Epoch {ep+1:2d}: acc={res['accuracy']:.3f} correct={res['correct']}/{res['total']}")

for p in ["hello", "how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=25)
    print(f"\nPROMPT: '{p}' -> '{out}'")
    print("Tokens:", [t["pred_token"] for t in tel])
