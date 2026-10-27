# PROJECT B — TRITIER RUNTIME
## A Persistent, GPU-Bounded, RAM-Managed, Continuously Learning Model Runtime

TriTier is a high-performance three-tier execution subsystem and memory manager built for discrete ternary and hybrid deep learning architectures. It decouples active execution from permanent parameter residency, guaranteeing that massive models (such as the 10-billion-parameter benchmark) can be continuously evaluated and trained while keeping GPU memory footprint strictly below **500,000,000 bytes** (500 MB).

---

## 1. Core Architecture

TriTier coordinates three physical memory layers into a single cohesive runtime:

```
+-------------------------------------------------------------------------+
|                  TIER P: PERSISTENT CANONICAL STORE                     |
|  - Immutable baseline parameter blocks (read-only reference)            |
|  - Content-addressed parameter files with FNV-1a checksums              |
|  - Transactional candidate updates & atomic manifest promotion          |
|  - Complete audit journal (models/<id>/journal.log) with provenance     |
+-------------------------------------------------------------------------+
                                    │
                                    ▼ (Bounded staging queue)
+-------------------------------------------------------------------------+
|                  TIER 2: WARM SYSTEM RAM & WORKERS                      |
|  - Bounded RAM parameter cache with LRU eviction                        |
|  - Reader-pin protection: active execution handles cannot be evicted     |
|  - Independent staging budget (prevents I/O bursts from exhausting RAM) |
|  - Strict 50/50 Scheduler: 50% Background Learning / 50% Doing Work     |
|  - Multi-objective candidate validation (loss, stability, invariants)   |
+-------------------------------------------------------------------------+
                                    │
                                    ▼ (Asynchronous DMA / OpenCL staging)
+-------------------------------------------------------------------------+
|                  TIER 1: HOT GPU VRAM (HARD BOUND < 500 MB)             |
|  - Hardware execution tile cache (8MB -> 4MB -> ... -> 64KB)            |
|  - Hard ceiling: 500,000,000 bytes process-attributable GPU allocation  |
|  - Dynamic tile reduction under sustained memory pressure               |
|  - Dual telemetry: Internal allocator tracking + External DXGI VRAM     |
+-------------------------------------------------------------------------+
```

### Key Architectural Invariants
1. **Model Baseline Immutability**: The original model checkpoint is strictly read-only. Learning writes versioned candidate deltas to isolated transaction directories and only commits after explicit multi-objective validation.
2. **Deterministic Version Switching & Rollback**: Switching model versions is instantaneous via atomic pointer swapping (`HEAD` manifest pointer). Instant rollback to `baseline` restores initial state with zero parameter reconstruction.
3. **Strict 50/50 Scheduler Policy**: Tier 2 background capacity is partitioned into dedicated `Learning` and `Doing` lanes. Capacity borrowing only occurs when a lane is empty, and borrowing events are explicitly audited.
4. **Hardware VRAM Boundedness**: Peak GPU memory stays strictly below 500,000,000 bytes regardless of model scale (tested on a 10,000,000,000 parameter benchmark).

---

## 2. Directory Structure

```
tritier/
├── build.bat                   # Master build, test, and benchmark script
├── Cargo.toml                  # Rust package manifest (rlib + cdylib)
├── build.rs                    # Auto-links native C++ GPU backend
├── src/
│   ├── lib.rs                  # Core TriTier engine, caches, scheduler, C-ABI FFI
│   ├── cli.rs                  # TriTier CLI (`tritier init|benchmark|recover`)
│   └── native_sycl.rs          # Rust FFI wrapper for native GPU backend
├── cpp/
│   ├── CMakeLists.txt          # CMake definition for C++ native backend
│   ├── include/
│   │   └── tritier_gpu.h       # C-ABI header for native GPU allocator
│   └── src/
│       ├── tritier_gpu_native.cpp # OpenCL allocator & DXGI VRAM telemetry
│       └── tritier_gpu_sycl.cpp   # SYCL alternative backend
├── julia/
│   ├── TriTierAdapter.jl       # High-performance Julia candidate validator
│   ├── bridge.jl               # Line-delimited IPC protocol bridge
│   └── test_adapter.jl         # Julia validation test suite
├── configs/
│   ├── benchmark_10b.json      # 10B parameter benchmark runtime profile
│   ├── production.json         # High-concurrency production profile
│   ├── low_memory.json         # Tight memory edge profile (128MB RAM / 256MB GPU)
│   └── debug_cpu.json          # CPU fallback profile
└── tests/
    └── acceptance_tests.rs     # 22 comprehensive spec acceptance tests
```

---

## 3. Quick Start & Build Instructions

### Prerequisites
- **Windows 10/11 x64**
- **MSVC Build Tools** (Visual Studio 2022 or 2026 with C++ desktop workload)
- **Rust toolchain** (1.80+ recommended)
- **Python 3.10+** (with `numpy` and `torch`)
- *(Optional)* **Julia 1.10+** for Julia candidate evaluation

### Automated Master Build & Run
Run the top-level orchestration script from the repository root:
```cmd
build_tritier.bat
```
This executes:
1. Compilation of C++ native OpenCL/DXGI static library (`tritier_sycl.lib`)
2. Compilation of Rust core runtime (`tritier_runtime.dll` and `tritier.exe`) with native GPU support
3. Complete 29-test Rust unit and acceptance suite
4. 10-billion parameter benchmark with hardware DXGI telemetry verification
5. Python end-to-end integration and supercharged credit assignment tests
6. Comparative performance benchmark (Un-Tiered Baseline vs TriTier)

### Manual Step-by-Step Build
```cmd
REM 1. Compile C++ GPU allocator
cd tritier\cpp
mkdir build
cl /nologo /O2 /MD /EHsc /c /Iinclude src\tritier_gpu_native.cpp /Fobuild\tritier_gpu_native.obj
lib /nologo /OUT:build\tritier_sycl.lib build\tritier_gpu_native.obj dxgi.lib
cd ..

REM 2. Compile Rust runtime in release mode
cargo build --release --features sycl

REM 3. Run acceptance tests
cargo test --features sycl
```

---

## 4. CLI Usage

The `tritier` command-line utility provides administrative and benchmarking functions:

### 1. Initialize a Model Store
```cmd
tritier.exe init <store_directory>
```

### 2. Run the 10-Billion Parameter Benchmark
```cmd
tritier.exe benchmark <store_directory> --operations 64 --config configs/benchmark_10b.json --require-native-gpu
```
Outputs a verified JSON report to `<store_directory>/reports/synthetic_10b_report.json` containing:
- Process-attributable internal peak GPU bytes
- DXGI hardware adapter video memory peak bytes
- 50/50 scheduler busy times and completed units
- Pass/fail status against the 500,000,000-byte hard limit

### 3. Recover an Interrupted Transaction
```cmd
tritier.exe recover <store_directory> <model_id>
```
Scans for orphaned staging directories, rolls back unfinished transactions, and verifies the active `HEAD` pointer.

---

## 5. Python Integration & Supercharged Learning

TriTier connects seamlessly to Python architectures (such as `LumenMAX`) via ctypes FFI.

```python
from lumenmax import LumenMAX
from tritier_adapter import TriTierModelAdapter, SuperchargedCreditEngine

# 1. Initialize model and wrap with TriTier adapter
model = LumenMAX(accelerator="cpu")
adapter = TriTierModelAdapter(
    model=model,
    store_root="./persistent_store",
    model_id="lumenmax-v1",
    native_gpu=True,
)

# 2. Attach supercharged credit engine
credit_engine = SuperchargedCreditEngine(model)

# 3. Train with multi-token eligibility traces & full-dimensional attribution
for epoch in range(10):
    result = model.train_epoch()
    print(f"Epoch {epoch}: Accuracy = {result['accuracy']*100:.1f}%")

# 4. Telemetry audit
print(adapter.telemetry())
adapter.close()
```

### Supercharged Credit Innovations
- **Temporal Eligibility Traces**: Decaying geometric traces across multi-step recurrent trajectories credit earlier decisions for terminal sequence success.
- **Margin-Deficit Plasticity**: Synaptic updates scale proportionally to decision margins, prioritizing unresolved decisions.
- **Full-Dimensional Attribution**: Replaces legacy sub-dimensional state slices with full-rank associative updates across all active neurons.
- **Dynamic Meta-Plasticity**: Accumulator threshold scaling prevents saturation loops and stabilizes continuous background learning.

---

## 6. Runtime Configuration Profiles

Configuration files are located in `configs/`:

| Profile | RAM Budget | Staging Budget | GPU Budget | Workers | Initial Tile | Target Environment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `benchmark_10b.json` | 256 MB | 64 MB | 450 MB | 4 | 8 MB | 10B Parameter benchmark (<500MB VRAM) |
| `production.json` | 512 MB | 64 MB | 400 MB | 8 | 4 MB | High-throughput production service |
| `low_memory.json` | 128 MB | 32 MB | 256 MB | 2 | 2 MB | Edge/Embedded deployment |
| `debug_cpu.json` | 64 MB | 16 MB | 128 MB | 2 | 1 MB | Headless CI/CD testing without GPU |

---

## 7. Verification & Acceptance Criteria Matrix

All 17 requirements and stress scenarios are validated via automated tests in `tests/acceptance_tests.rs`:

| Test # | Requirement / Criterion | Status |
| :--- | :--- | :--- |
| **01** | Complete parameter set remains addressable across all tiers | PASS |
| **02** | Blocks retrieved from Storage, RAM, and VRAM with hit accounting | PASS |
| **03** | Requests consistently return correct requested parameter version | PASS |
| **04** | GPU allocator enforces strictly bounded memory budget | PASS |
| **05** | Internal + external DXGI telemetry captures actual process peak | PASS |
| **06** | Streaming remains bounded under repeated cache misses | PASS |
| **07** | Prefetch pipeline cannot exhaust staging budget | PASS |
| **08** | Eviction never removes active reader-protected (pinned) blocks | PASS |
| **09** | Concurrent readers never observe uncommitted/in-flight updates | PASS |
| **10** | Background learning cannot mutate immutable baseline checkpoint | PASS |
| **11** | 50/50 scheduler allocation enforced and auditable | PASS |
| **12** | Corrupted blocks detected via FNV-1a checksums | PASS |
| **13** | Interrupted updates recover safely to valid previous version | PASS |
| **14** | Rollback restores original baseline without data loss | PASS |
| **15** | Tiered execution matches reference weights with 0 tolerance | PASS |
| **16** | Learning updates traceable to provenance and journal entries | PASS |
| **17** | 10-Billion parameter benchmark GPU memory strictly < 500 MB | PASS |
| **18** | Concurrency stress: 4 reader threads + background learning | PASS |
| **19** | Sustained memory pressure dynamically reduces tile size | PASS |
| **20** | Repeated restarts maintain valid persistent HEAD state | PASS |
| **21** | Disk I/O simulation: graceful error isolation on missing blocks | PASS |
| **22** | Cross-representation support: Native int8 vs Project A 2-bit packed | PASS |
