from pathlib import Path
from lumenmax import LumenMAX, tproj, assert_ternary
import numpy as np

# Test the complete integrated system
m = LumenMAX(seed=42)

# Load dataset.txt
dataset_txt = Path("dataset.txt").read_text(encoding="utf-8")
added = m.ingest_dataset_text(dataset_txt)
print(f"Ingested {added} items from dataset.txt")
print(f"Total dataset items: {len(m.dataset)}, sequences: {len(m.sequences)}")

# Train 5 epochs
for ep in range(5):
    res = m.train_epoch()
    print(f"Epoch {ep+1}: acc={res['accuracy']:.3f} correct={res['correct']}/{res['total']}")

# Test conversational prompts
prompts = [
    "hello",
    "how are you",
    "who are you",
    "what is default parameters?",
    "what is searching?",
]
print("\n=== GENERATION SAMPLES ===")
for p in prompts:
    out, tel = m.generate(p, max_tokens=30)
    print(f"Q: '{p}'")
    print(f"A: '{out}' (stop={tel[-1].get('stop_reason')})")
    print()
