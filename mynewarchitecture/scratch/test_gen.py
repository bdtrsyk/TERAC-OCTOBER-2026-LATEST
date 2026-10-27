import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumenmax import LumenMAX

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()
for ep in range(15):
    res = m.train_epoch()
    print(f"Epoch {ep+1}: acc={res['accuracy']:.3f} correct={res['correct']}/{res['total']}")

for p in ["hello", "how are you", "who are you"]:
    out, tel = m.generate(p, max_tokens=25)
    print(f"\nPROMPT: '{p}' -> '{out}'")
    print("Tokens:", [t["pred_token"] for t in tel])
