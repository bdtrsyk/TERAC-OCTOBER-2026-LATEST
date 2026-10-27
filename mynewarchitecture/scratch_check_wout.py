from lumenmax import LumenMAX
import numpy as np

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()
for ep in range(15):
    m.train_epoch()

for tok in ["how", "you", "today", "help", "i", "am", "lumenmax"]:
    tid = m.symbols.tok2id.get(tok)
    if tid is not None:
        row = m.W_out.W[tid]
        acc = m.W_out.A[tid]
        n_pos = np.sum(row == 1)
        n_neg = np.sum(row == -1)
        n_zero = np.sum(row == 0)
        print(f"Token '{tok:10s}' (id={tid:2d}): +1={n_pos:2d}, -1={n_neg:2d}, 0={n_zero:2d} | mean_acc={np.mean(acc):.2f} max_acc={np.max(acc)}")
