"""
benchmark_tile_tuning.py — Tile-size tuning and profiling experiment for TriTier
Evaluates performance across tile sizes: 1MB, 2MB, 4MB, 8MB, 16MB.
"""

import json
import subprocess
import time
from pathlib import Path

TRITIER_EXE = Path("tritier/target/release/tritier.exe").resolve()
CONFIG_DIR = Path("tritier/configs").resolve()
REPORT_DIR = Path("tritier/reports_tuning").resolve()
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TILE_SIZES = [
    ("1MB",  1 * 1024 * 1024),
    ("2MB",  2 * 1024 * 1024),
    ("4MB",  4 * 1024 * 1024),
    ("8MB",  8 * 1024 * 1024),
    ("16MB", 16 * 1024 * 1024),
]

def run_tuning():
    results = []
    print("=" * 80)
    print("TRITIER TILE-SIZE PROFILING & TUNING ON INTEL ARC A770")
    print("=" * 80)
    print(f"{'Tile Size':<10} | {'Ops':<5} | {'Elapsed (ms)':<12} | {'Internal Peak (B)':<18} | {'External Peak (B)':<18} | {'Status':<8}")
    print("-" * 80)

    for label, tile_bytes in TILE_SIZES:
        cfg = {
            "ram_budget": 268435456,
            "staging_budget": 67108864,
            "gpu_budget": 471859200,
            "worker_capacity": 4,
            "initial_tile": tile_bytes,
            "min_tile": 65536,
            "native_gpu": True
        }
        cfg_path = REPORT_DIR / f"cfg_{label}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        run_dir = REPORT_DIR / f"run_{label}"
        cmd = [
            str(TRITIER_EXE),
            "benchmark",
            str(run_dir),
            "--operations", "64",
            "--config", str(cfg_path),
            "--require-native-gpu"
        ]

        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print(f"{label:<10} | FAILED: {p.stderr.strip()}")
            continue

        rep_path = run_dir / "reports" / "synthetic_10b_report.json"
        if rep_path.exists():
            rep = json.loads(rep_path.read_text())
            int_peak = rep["gpu"]["internal_peak_bytes"]
            ext_peak = rep["gpu"]["external_peak_bytes"]
            elapsed = rep["elapsed_ms"]
            status = rep["acceptance_status"]
            results.append({
                "tile_label": label,
                "tile_bytes": tile_bytes,
                "elapsed_ms": elapsed,
                "internal_peak_bytes": int_peak,
                "external_peak_bytes": ext_peak,
                "status": status,
                "ops_per_sec": 64.0 / (elapsed / 1000.0) if elapsed > 0 else 0
            })
            print(f"{label:<10} | {64:<5} | {elapsed:<12} | {int_peak:<18} | {ext_peak:<18} | {status:<8}")

    print("=" * 80)
    summary_path = REPORT_DIR / "tile_tuning_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nTile tuning summary written to: {summary_path}")

if __name__ == "__main__":
    run_tuning()
