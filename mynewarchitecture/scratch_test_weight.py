from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)

def custom_readout(self):
    # If the last transition was from <sep>, prompt_context is paramount (weight 3)
    # If inside answer transitions, recurrent state is paramount (weight 3)
    is_at_sep = (self.state[-1] == m.symbols.encode(m.sep_token)[1][-1]) # or check
    return tproj(
        self.state.astype(np.int32) * 3
        + self.prompt_context.astype(np.int32) * 2
        + self.context.astype(np.int32),
        theta=2
    )

LumenMAX._readout_state = custom_readout

seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
]

for ep in range(15):
    correct = 0
    total = 0
    for toks, dom in seqs:
        res = m.train_sequence(toks, domain=dom)
        correct += res["correct"]
        total += res["total"]
    if (ep + 1) % 3 == 0 or ep == 14:
        print(f"Epoch {ep+1:2d}: {correct}/{total} = {correct/total*100:.1f}%")

for p in ["how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=10)
    print(f"'{p}' -> '{out}' (tokens={[t['pred_token'] for t in tel]})")
