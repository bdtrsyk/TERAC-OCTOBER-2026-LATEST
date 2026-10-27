from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
]

for ep in range(10):
    correct = 0
    total = 0
    for toks, dom in seqs:
        res = m.train_sequence(toks, domain=dom)
        correct += res["correct"]
        total += res["total"]
    print(f"Epoch {ep+1:2d}: {correct}/{total} = {correct/total*100:.1f}%")

# Now test generation on both
for p in ["how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=10)
    print(f"'{p}' -> '{out}' (tokens={[t['pred_token'] for t in tel]})")
