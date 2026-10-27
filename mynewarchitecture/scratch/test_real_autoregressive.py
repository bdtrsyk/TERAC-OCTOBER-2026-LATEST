import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumenmax import LumenMAX

m = LumenMAX()
print("Device:", m.accel.name)
print("Allocated VRAM:", m.accel.memory_info())
print("VRAM Pools count:", len(m.vram_pools))
m.seed_conversational_curriculum()
for ep in range(15):
    m.train_epoch()

for p in ["hello", "how are you", "who are you"]:
    out, tel = m.generate(p)
    print(f'PROMPT: "{p}" -> RESULT: "{out}" | stop={tel[-1].get("stop_reason")} | grounded={tel[-1].get("grounded")}')
