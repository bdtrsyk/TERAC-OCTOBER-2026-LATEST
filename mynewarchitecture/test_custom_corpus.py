from lumenmax import LumenMAX

m = LumenMAX(state_dim=64, input_dim=64, max_vocab=512)

# User adds custom programming corpus
code_sample = "def add(x, y): return x + y"
n_added = m.add_text_corpus(code_sample)
print(f"Custom Corpus Ingested: {n_added} sequential transitions")

for ex in m.dataset:
    print(f"  {ex['input']} -> {ex['target']}")

print("\nRunning Automatic Discrete Reward/Punishment Training Loop:")
for ep in range(12):
    met = m.train_epoch()
    print(f"  Epoch {ep+1:2d}: Accuracy = {met['accuracy']*100:5.1f}% | Rewards: +{met['rewards']:2d} | Punishments: -{met['punishments']:2d}")

# Generate in corpus domain (Python substrate; Julia remains cognitive reasoner)
print("\nDynamic Generation Test:")
m.reset_state()
resp, telem = m.generate("def", max_tokens=10, domain="corpus")
print(f"  Prompt: 'def' -> Response: '{resp}' stop={telem[-1].get('stop_reason')}")
m.reset_state()
resp, telem = m.generate("return", max_tokens=6, domain="corpus")
print(f"  Prompt: 'return' -> Response: '{resp}' stop={telem[-1].get('stop_reason')}")
