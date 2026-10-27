from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
tokens = ["hello", "how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"]
for theta in [1, 2, 3, 4]:
    norms = []
    for tok in tokens:
        _, x = m.symbols.encode(tok)
        accum = m.W_in.W.astype(np.int32) @ x.astype(np.int32)
        out = tproj(accum, theta=theta)
        norms.append(np.sum(np.abs(out)))
    print(f"theta={theta}: average active (non-zero) elements = {np.mean(norms):.1f} / 64 (zeros = {64 - np.mean(norms):.1f})")
