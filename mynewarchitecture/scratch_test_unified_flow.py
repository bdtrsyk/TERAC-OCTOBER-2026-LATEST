from pathlib import Path
from tcai_bridge import UnifiedEngine, find_julia

# Test Unified Hybrid Engine with both Julia and Python
u = UnifiedEngine(engine="unified", scale="tiny")
print(f"UnifiedEngine active: {u.active}, julia={u.julia is not None}")

test_queries = [
    "hello",
    "how are you",
    "who are you",
    "does sun cause heat?",
    "what is default parameters?",
    "what is searching?",
]

print("\n=== UNIFIED HYBRID ENGINE RESULTS ===")
for q in test_queries:
    resp = u.ask(q, max_tokens=36)
    print(f"QUERY: '{q}'")
    print(f"OUTPUT: '{resp.get('output')}'")
    print(f"QUALITY: {resp.get('quality')} | CONF: {resp.get('confidence'):.2f} | STOP: {resp.get('stop_reason')}")
    print(f"JULIA: concepts={resp.get('julia', {}).get('concepts')} | PYTHON: params={resp.get('python', {}).get('param_count')}")
    print("-" * 60)

u.close()
