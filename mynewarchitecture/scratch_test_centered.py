from lumenmax import LumenMAX
import numpy as np

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()
for ep in range(15):
    m.train_epoch()

for tok in ["how", "you", "today", "help", "i", "am", "lumenmax"]:
    tid = m.symbols.tok2id.get(tok)
    acc = m.W_out.A[tid]
    mean_a = int(np.mean(acc))
    centered = acc - mean_a
    w = np.zeros(len(acc), dtype=np.int8)
    w[centered >= 3] = 1
    w[centered <= -3] = -1
    print(f"Token '{tok:10s}': centered weights: +1={np.sum(w==1):2d}, -1={np.sum(w==-1):2d}, 0={np.sum(w==0):2d}")
