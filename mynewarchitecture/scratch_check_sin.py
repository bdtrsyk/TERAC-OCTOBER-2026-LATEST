from lumenmax import LumenMAX
import numpy as np

m = LumenMAX(seed=42)
tokens = ["hello", "how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"]
for tok in tokens:
    _, x = m.symbols.encode(tok)
    s_in = m.W_in.forward(x)
    print(f"Token '{tok:6s}': x norm={np.sum(np.abs(x))}, s_in norm={np.sum(np.abs(s_in))}")
    if tok in ("am", "doing"):
        print(f"  {tok} s_in[:10] = {s_in[:10]}")
