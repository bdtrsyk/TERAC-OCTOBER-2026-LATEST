from lumenmax import LumenMAX
import numpy as np

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()
for ep in range(15):
    m.train_epoch()

for theta in [1, 2, 3, 4, 5]:
    tid = m.symbols.tok2id.get("you")
    acc = m.W_out.A[tid]
    w = np.zeros(len(acc), dtype=np.int8)
    w[acc >= theta] = 1
    w[acc <= -theta] = -1
    print(f"theta={theta}: 'you' weights: +1={np.sum(w==1):2d}, -1={np.sum(w==-1):2d}, 0={np.sum(w==0):2d}")
