"""Quick test of LumenMAX learning convergence."""
from lumenmax import LumenMAX

m = LumenMAX(state_dim=64, input_dim=64, max_vocab=512)

# User-supplied training pairs (NO pre-given data)
pairs = [
    ("hello", "world"), ("good", "morning"), ("cat", "dog"),
    ("sun", "moon"), ("hot", "cold"), ("up", "down"),
    ("left", "right"), ("yes", "no"), ("fast", "slow"),
    ("big", "small"), ("day", "night"), ("open", "close"),
    ("push", "pull"), ("start", "stop"), ("black", "white"),
]
for inp, tgt in pairs:
    m.add_training_pair(inp, tgt)

print(f"Dataset: {len(m.dataset)} pairs")
print(f"Parameters: {m.param_count()}")
print(f"Ternary audit: {m.get_ternary_audit()}")
print()

for ep in range(25):
    met = m.train_epoch()
    print(f"Epoch {ep+1:2d}: {met['correct']:2d}/{met['total']} = {met['accuracy']*100:5.1f}%")

print("\nEvaluation:")
for ex in m.dataset:
    m.reset_state()
    info = m.forward_step(ex['input'])
    status = "OK" if info['pred_token'] == ex['target'] else f"FAIL (got {info['pred_token']})"
    print(f"  {ex['input']} -> {ex['target']}: {status}")
