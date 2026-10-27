from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)

def custom_train_step(self, input_tok: str, target_tok: str, domain: str = "general", *, keep_state: bool = False, prev_state = None):
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
    margin = info.get("margin", 0)
    s = self._readout_state()
    prev = prev_state if prev_state is not None else np.zeros(self.N, dtype=np.int8)

    # Perceptron / Margin rule: Only update if wrong, or if margin is weak (< 2)
    if not correct:
        self.W_out.reinforce(tgt_id, s, magnitude=2)
        if pred_id >= 0 and pred_id != tgt_id:
            self.W_out.suppress(pred_id, s, magnitude=2)
        self.router.update(info["expert"], success=False)
        _, x = self.symbols.encode(input_tok, domain=domain)
        for r in np.where(np.abs(s) > 0)[0][:12]:
            self.W_in.reinforce(int(r), x, magnitude=1)
        if pred_id >= 0 and pred_id != tgt_id:
            for r in np.where(np.abs(s) > 0)[0][:8]:
                self.W_rec.suppress(int(r), prev, magnitude=1)
    elif margin < 2:
        # Boost margin gently
        self.W_out.reinforce(tgt_id, s, magnitude=1)
        self.router.update(info["expert"], success=True)
        for r in np.where(np.abs(s) > 0)[0][:8]:
            self.W_rec.reinforce(int(r), prev, magnitude=1)
    else:
        self.router.update(info["expert"], success=True)

    info["correct"] = correct
    info["target_id"] = tgt_id
    info["target_token"] = target_tok
    return info

LumenMAX.train_step = custom_train_step

seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
]

for ep in range(20):
    correct = 0
    total = 0
    for toks, dom in seqs:
        res = m.train_sequence(toks, domain=dom)
        correct += res["correct"]
        total += res["total"]
    if (ep + 1) % 2 == 0:
        print(f"Epoch {ep+1:2d}: {correct}/{total} = {correct/total*100:.1f}%")

for p in ["how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=10)
    print(f"'{p}' -> '{out}' (tokens={[t['pred_token'] for t in tel]})")
