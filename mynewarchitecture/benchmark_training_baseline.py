import time
from lumenmax import LumenMAX

m = LumenMAX(seed=42)
m.seed_conversational_curriculum()

t0 = time.perf_counter()
for ep in range(5):
    res = m.train_epoch()
    acc = res["accuracy"] * 100
    c = res["correct"]
    tot = res["total"]
    print(f"Epoch {ep+1}: acc={acc:.2f}% | correct={c}/{tot}")
elapsed = time.perf_counter() - t0
print(f"Elapsed: {elapsed*1000:.2f} ms ({5/elapsed:.1f} epochs/sec)")

prompts = ["hello", "how are you", "who are you", "what can you do"]
print("\nGeneration test:")
for p in prompts:
    out, tel = m.generate(p, max_tokens=25)
    print(f"  '{p}' -> '{out}' (stop={tel[-1].get('stop_reason')})")
