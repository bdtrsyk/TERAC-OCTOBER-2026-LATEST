"""15-pair discrete credit regression — stock LumenMAX, no monkey-patches."""
from lumenmax import LumenMAX

m = LumenMAX(state_dim=64, input_dim=64, max_vocab=512, seed=42)
pairs = [
    ("hello", "world"), ("good", "morning"), ("cat", "dog"),
    ("sun", "moon"), ("hot", "cold"), ("up", "down"),
    ("left", "right"), ("yes", "no"), ("fast", "slow"),
    ("big", "small"), ("day", "night"), ("open", "close"),
    ("push", "pull"), ("start", "stop"), ("black", "white"),
]
for inp, tgt in pairs:
    m.add_training_pair(inp, tgt)

for ep in range(15):
    met = m.train_epoch()
    print(f"Epoch {ep+1:2d}: {met['correct']:2d}/{met['total']} = {met['accuracy']*100:.1f}%")

print("\nFinal Check:")
all_pass = True
for ex in m.dataset:
    m.reset_state()
    info = m.forward_step(ex["input"])
    ok = info["pred_token"] == ex["target"]
    if not ok:
        all_pass = False
        print(f"  FAIL: {ex['input']} -> exp '{ex['target']}', got '{info['pred_token']}'")
if all_pass:
    print("  ALL 15 PAIRS PASSED 100.0% PERFECT!")
else:
    raise SystemExit(1)
