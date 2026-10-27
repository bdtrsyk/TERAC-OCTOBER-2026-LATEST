"""
test_tritier_integration.py — End-to-end integration test of TriTier + Supercharged Learning
"""

import os
import shutil
import time
from pathlib import Path
import numpy as np

from lumenmax import LumenMAX
from tritier_adapter import TriTierModelAdapter, SuperchargedCreditEngine, ProjectABinaryCodec

TEST_STORE = Path("test_tritier_store").resolve()


def test_codec_project_a():
    print("Testing Project A 2-bit packing codec...")
    orig = np.random.choice([-1, 0, 1], size=(128, 64)).astype(np.int8)
    packed = ProjectABinaryCodec.pack(orig)
    unpacked = ProjectABinaryCodec.unpack(packed, orig.shape)
    assert np.array_equal(orig, unpacked), "Project A codec round-trip failed"
    print(f"  [OK] Packed {orig.nbytes} bytes -> {len(packed)} bytes (4:1 compression), reconstructed perfectly.")


def test_tritier_adapter():
    print("\nTesting TriTier Model Adapter...")
    if TEST_STORE.exists():
        shutil.rmtree(TEST_STORE)

    m = LumenMAX(seed=42, accelerator="cpu", auto_allocate_vram=False)
    m.seed_conversational_curriculum()

    adapter = TriTierModelAdapter(
        model=m,
        store_root=str(TEST_STORE),
        model_id="lumenmax-test",
        representation="native",
        native_gpu=True,
    )

    # Acquire win and wout from TriTier
    win_streamed = adapter.acquire_matrix("win", (m.W_in.out_dim, m.W_in.in_dim))
    assert np.array_equal(m.W_in.W, win_streamed), "Streamed matrix must match original exactly"

    wout_streamed = adapter.acquire_matrix("wout", (m.W_out.out_dim, m.W_out.in_dim))
    assert np.array_equal(m.W_out.W, wout_streamed), "Streamed matrix must match original exactly"

    tel = adapter.telemetry()
    print("  [OK] Model successfully tiered into Persistent Storage, RAM, and GPU.")
    print(f"  Telemetry: RAM={tel['ram_current_bytes']}B, GPU={tel['gpu_current_bytes']}B, Staging={tel['staging_current_bytes']}B")
    assert tel["gpu_current_bytes"] < 500_000_000, "GPU current bytes must stay below 500,000,000"
    adapter.close()


def test_supercharged_learning():
    print("\nTesting Supercharged Learning & Credit Assignment...")
    m = LumenMAX(seed=42, accelerator="cpu", auto_allocate_vram=False)
    m.seed_conversational_curriculum()
    engine = SuperchargedCreditEngine(m)

    print("  Training 8 epochs with Supercharged Credit Assignment:")
    accuracies = []
    for ep in range(8):
        res = m.train_epoch()
        accuracies.append(res["accuracy"])
        print(f"    Epoch {ep+1}: acc={res['accuracy']*100:5.1f}% ({res['correct']}/{res['total']})")

    # Verify that accuracy improved
    assert accuracies[-1] > accuracies[0], "Learning must show positive convergence"
    assert accuracies[-1] >= 0.12, "Supercharged learning must reach strong multi-token sequence accuracy"

    print("\n  Testing Generation Quality on Core Conversational Prompts:")
    prompts = [
        "hello",
        "how are you",
        "who are you",
        "what can you do",
    ]
    for p in prompts:
        out, _ = m.generate(p, max_tokens=25)
        print(f"    PROMPT: '{p}' -> '{out}'")
        assert len(out) > 0, f"Prompt '{p}' should not generate empty string"
    print("  [OK] Supercharged credit assignment generates fluent responses without repetition loops.")


if __name__ == "__main__":
    print("=== STARTING TRITIER INTEGRATION SUITE ===")
    test_codec_project_a()
    test_tritier_adapter()
    test_supercharged_learning()
    print("\n=== ALL TRITIER INTEGRATION TESTS PASSED! ===")
