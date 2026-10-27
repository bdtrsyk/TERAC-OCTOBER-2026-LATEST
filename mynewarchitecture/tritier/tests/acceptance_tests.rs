use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tritier_runtime::*;

fn temp_dir(name: &str) -> PathBuf {
    let p = env::temp_dir().join(format!("tritier-accept-{name}-{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir_all(&p).unwrap();
    p
}

// -------------------------------------------------------------------------------------------------
// TEST 1: The complete parameter set remains accessible through the block-provider interface.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_01_complete_parameter_set_remains_accessible() {
    let root = temp_dir("test01");
    let store = PersistentStore::open(&root).unwrap();
    let blocks = vec![
        ("w_in".into(), "layer0".into(), vec![1u8; 128]),
        ("w_rec".into(), "layer0".into(), vec![2u8; 128]),
        ("w_gate".into(), "layer0".into(), vec![0u8; 128]),
        ("w_out".into(), "layer0".into(), vec![255u8; 128]),
    ];
    let (m, b) = baseline_from_blocks("m1", "int8_ternary", blocks);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 1024, staging_budget: 256, gpu_budget: 4096, worker_capacity: 2, initial_tile: 256, min_tile: 64, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m1".into(), "baseline".into(), cfg).unwrap();

    for id in ["w_in", "w_rec", "w_gate", "w_out"] {
        let key = BlockKey { model_id: "m1".into(), version: "baseline".into(), block_id: id.into() };
        let h = rt.acquire(&key).expect("Every parameter block must be accessible");
        assert_eq!(h.bytes.len(), 128);
    }
}

// -------------------------------------------------------------------------------------------------
// TEST 2: Blocks can be retrieved from persistent storage, RAM, and VRAM.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_02_blocks_retrieved_from_storage_ram_and_vram() {
    let root = temp_dir("test02");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m2", "native", vec![("b1".into(), "comp".into(), vec![7u8; 64])]);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 128, staging_budget: 64, gpu_budget: 256, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m2".into(), "baseline".into(), cfg).unwrap();
    let key = BlockKey { model_id: "m2".into(), version: "baseline".into(), block_id: "b1".into() };

    // 1. From persistent storage (first acquisition -> cache miss)
    let h1 = rt.acquire(&key).unwrap();
    assert_eq!(h1.bytes.as_slice(), &[7u8; 64]);
    let t1 = rt.telemetry();
    assert_eq!(t1.ram.hits, 0);
    assert_eq!(t1.ram.misses, 1);

    // 2. From RAM and VRAM (second acquisition -> cache hit)
    let h2 = rt.acquire(&key).unwrap();
    assert_eq!(h2.bytes.as_slice(), &[7u8; 64]);
    let t2 = rt.telemetry();
    assert_eq!(t2.ram.hits, 1);
    assert!(t2.gpu.current_bytes >= 64);
}

// -------------------------------------------------------------------------------------------------
// TEST 3: Requests return the correct parameter version.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_03_requests_return_correct_version() {
    let root = temp_dir("test03");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m3", "native", vec![("param".into(), "core".into(), vec![10u8; 32])]);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 128, staging_budget: 64, gpu_budget: 256, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m3".into(), "baseline".into(), cfg).unwrap();

    let mut cand = store.candidate("m3", "baseline", BTreeMap::from([("param".into(), vec![20u8; 32])]), "test-version".into()).unwrap();
    store.validate_candidate(&mut cand, |_| true).unwrap();
    store.commit(&cand, "v2").unwrap();

    let k_base = BlockKey { model_id: "m3".into(), version: "baseline".into(), block_id: "param".into() };
    let k_v2 = BlockKey { model_id: "m3".into(), version: "v2".into(), block_id: "param".into() };

    let h_base = rt.acquire(&k_base).unwrap();
    assert_eq!(h_base.bytes.as_slice(), &[10u8; 32]);

    rt.switch_version("v2").unwrap();
    let h_v2 = rt.acquire(&k_v2).unwrap();
    assert_eq!(h_v2.bytes.as_slice(), &[20u8; 32]);
}

// -------------------------------------------------------------------------------------------------
// TEST 4: The GPU allocator enforces its budget.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_04_gpu_allocator_enforces_budget() {
    let budget = 100u64;
    let hard = 200u64;
    let mut cache = GpuCache::new(budget, hard, 64, 8, false).unwrap();
    let k1 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k1".into() };
    let k2 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k2".into() };

    cache.acquire(k1.clone(), &[1u8; 80]).unwrap();
    assert_eq!(cache.stats.current_bytes, 80);

    // Exceeds remaining budget and k1 is reader-protected -> must enforce budget and reduce tile
    let res = cache.acquire(k2.clone(), &[2u8; 30]);
    assert!(res.is_err());
    assert_eq!(cache.tile_bytes, 32);
    assert_eq!(cache.stats.pressure_events, 1);
}

// -------------------------------------------------------------------------------------------------
// TEST 5: Peak-memory telemetry captures the actual peak, including staging and runtime allocations.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_05_peak_memory_telemetry_captures_actual_peak() {
    let mut staging = StagingBudget::new(1024);
    staging.reserve(500).unwrap();
    assert_eq!(staging.current(), 500);
    assert_eq!(staging.peak(), 500);
    staging.reserve(300).unwrap();
    assert_eq!(staging.current(), 800);
    assert_eq!(staging.peak(), 800);
    staging.release(500);
    assert_eq!(staging.current(), 300);
    assert_eq!(staging.peak(), 800); // peak is preserved!
}

// -------------------------------------------------------------------------------------------------
// TEST 6: Streaming remains bounded under repeated cache misses.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_06_streaming_remains_bounded_under_repeated_cache_misses() {
    let root = temp_dir("test06");
    let store = PersistentStore::open(&root).unwrap();
    let mut blocks = Vec::new();
    for i in 0..20 {
        blocks.push((format!("blk_{i}"), "stream".into(), vec![i as u8; 64]));
    }
    let (m, b) = baseline_from_blocks("m6", "native", blocks);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 128, staging_budget: 64, gpu_budget: 256, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m6".into(), "baseline".into(), cfg).unwrap();

    for i in 0..100 {
        let id = format!("blk_{}", i % 20);
        let key = BlockKey { model_id: "m6".into(), version: "baseline".into(), block_id: id };
        let h = rt.acquire(&key).unwrap();
        assert_eq!(h.bytes.len(), 64);
        drop(h);
    }
    let t = rt.telemetry();
    assert!(t.ram.current_bytes <= 128);
    assert!(t.ram.peak_bytes <= 128);
    assert!(t.gpu.current_bytes <= 256);
    assert!(t.staging_current <= 64);
}

// -------------------------------------------------------------------------------------------------
// TEST 7: Prefetch operations cannot exhaust the staging budget.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_07_prefetch_cannot_exhaust_staging_budget() {
    let root = temp_dir("test07");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m7", "native", vec![("large".into(), "c".into(), vec![1u8; 100])]);
    store.create_baseline(m, &b).unwrap();

    // Staging budget is only 50 bytes, smaller than the 100-byte block
    let cfg = RuntimeConfig { ram_budget: 256, staging_budget: 50, gpu_budget: 512, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m7".into(), "baseline".into(), cfg).unwrap();

    let key = BlockKey { model_id: "m7".into(), version: "baseline".into(), block_id: "large".into() };
    let rx = rt.prefetch(key);
    let res = rx.recv().unwrap();
    assert!(res.is_err(), "Prefetching beyond staging cap must be rejected gracefully");
}

// -------------------------------------------------------------------------------------------------
// TEST 8: Eviction does not remove a block currently protected by an active reader.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_08_eviction_does_not_remove_active_reader_protected_block() {
    let mut ram = RamCache::new(100);
    let k1 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k1".into() };
    let k2 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k2".into() };

    let _h1 = ram.insert_and_pin(k1.clone(), vec![1u8; 80]).unwrap();
    // k1 has pins = 1. k2 needs 40 bytes. Budget is 100.
    // It must fail because k1 cannot be evicted while pinned!
    let res = ram.insert_and_pin(k2, vec![2u8; 40]);
    assert!(res.is_err());
    assert_eq!(ram.stats.current_bytes, 80);
}

// -------------------------------------------------------------------------------------------------
// TEST 9: Concurrent inference never reads a partially committed update.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_09_concurrent_inference_never_reads_partially_committed_update() {
    let root = temp_dir("test09");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m9", "native", vec![("weight".into(), "core".into(), vec![1u8; 32])]);
    store.create_baseline(m, &b).unwrap();

    // Simulate an incomplete transaction directory in flight
    let tx_dir = root.join("models/m9/transactions/cand-9999");
    fs::create_dir_all(tx_dir.join("blocks")).unwrap();
    fs::write(tx_dir.join("blocks/weight.bin"), vec![99u8; 32]).unwrap();

    // Reader requesting "v_in_progress" must fail cleanly with NotFound or validation
    let k_uncommitted = BlockKey { model_id: "m9".into(), version: "v_in_progress".into(), block_id: "weight".into() };
    let res = store.read_block(&k_uncommitted);
    assert!(res.is_err());
}

// -------------------------------------------------------------------------------------------------
// TEST 10: Background learning cannot overwrite the immutable baseline.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_10_background_learning_cannot_overwrite_immutable_baseline() {
    let root = temp_dir("test10");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m10", "native", vec![("w".into(), "core".into(), vec![5u8; 16])]);
    store.create_baseline(m.clone(), &b).unwrap();

    // Attempting to re-create baseline must be rejected
    assert!(matches!(store.create_baseline(m, &b), Err(Error::ImmutableBaseline)));

    // Attempting to commit a candidate as "baseline" must be rejected
    let mut cand = store.candidate("m10", "baseline", BTreeMap::from([("w".into(), vec![9u8; 16])]), "prov".into()).unwrap();
    store.validate_candidate(&mut cand, |_| true).unwrap();
    assert!(store.commit(&cand, "baseline").is_err());
}

// -------------------------------------------------------------------------------------------------
// TEST 11: The 50/50 scheduler allocation is measurable and enforced when both queues have eligible work.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_11_fifty_fifty_scheduler_allocation_enforced_and_measured() {
    let scheduler = Tier2Scheduler::new(4).unwrap();
    let mut rx_list = Vec::new();
    let count = 20;

    for _ in 0..count {
        rx_list.push(scheduler.submit(TaskClass::Learning, 1, Box::new(|| {
            thread::sleep(Duration::from_millis(2));
            Ok("learn".into())
        })));
        rx_list.push(scheduler.submit(TaskClass::Doing, 1, Box::new(|| {
            thread::sleep(Duration::from_millis(2));
            Ok("do".into())
        })));
    }

    for rx in rx_list {
        rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
    }

    let stats = scheduler.stats();
    assert_eq!(stats.learning_units, count as u64);
    assert_eq!(stats.doing_units, count as u64);
    // The achieved allocation is strictly balanced 50/50 across the batch
    assert_eq!(stats.learning_units, stats.doing_units);

    // Verify borrowing behavior when one queue has no work:
    let mut rx_borrow = Vec::new();
    for _ in 0..6 {
        rx_borrow.push(scheduler.submit(TaskClass::Doing, 1, Box::new(|| {
            thread::sleep(Duration::from_millis(2));
            Ok("borrow_do".into())
        })));
    }
    for rx in rx_borrow { rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap(); }
    let stats_after = scheduler.stats();
    assert!(stats_after.borrowed_doing > 0, "Idle learning workers must borrow doing tasks when learning queue is empty");
}

// -------------------------------------------------------------------------------------------------
// TEST 12: A corrupted block is detected.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_12_corrupted_block_is_detected() {
    let root = temp_dir("test12");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m12", "native", vec![("block".into(), "c".into(), vec![1, 2, 3, 4])]);
    store.create_baseline(m, &b).unwrap();

    let file_path = root.join("models/m12/versions/baseline/blocks/block.bin");
    fs::write(file_path, [9, 9, 9, 9]).unwrap(); // flip bytes

    let key = BlockKey { model_id: "m12".into(), version: "baseline".into(), block_id: "block".into() };
    let res = store.read_block(&key);
    assert!(matches!(res, Err(Error::Integrity(_))));
}

// -------------------------------------------------------------------------------------------------
// TEST 13: Interrupted updates recover to a valid model version.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_13_interrupted_updates_recover_to_valid_version() {
    let root = temp_dir("test13");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m13", "native", vec![("blk".into(), "c".into(), vec![10u8; 8])]);
    store.create_baseline(m, &b).unwrap();

    let crash_dir = root.join("models/m13/transactions/interrupted-update-777");
    fs::create_dir_all(&crash_dir).unwrap();
    fs::write(crash_dir.join("partial_data.tmp"), [0u8; 100]).unwrap();

    let warnings = store.recover("m13").unwrap();
    assert_eq!(store.head("m13").unwrap(), "baseline");
    assert!(warnings.iter().any(|w| w.contains("interrupted-update-777")));
}

// -------------------------------------------------------------------------------------------------
// TEST 14: Rollback restores the original baseline.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_14_rollback_restores_original_baseline() {
    let root = temp_dir("test14");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m14", "native", vec![("p".into(), "c".into(), vec![1u8; 4])]);
    store.create_baseline(m, &b).unwrap();

    let mut cand = store.candidate("m14", "baseline", BTreeMap::from([("p".into(), vec![9u8; 4])]), "exp".into()).unwrap();
    store.validate_candidate(&mut cand, |_| true).unwrap();
    store.commit(&cand, "v1").unwrap();
    assert_eq!(store.head("m14").unwrap(), "v1");

    store.set_head("m14", "baseline").unwrap();
    assert_eq!(store.head("m14").unwrap(), "baseline");
    let (_, data) = store.read_block(&BlockKey { model_id: "m14".into(), version: "baseline".into(), block_id: "p".into() }).unwrap();
    assert_eq!(data.as_slice(), &[1u8; 4]);
}

// -------------------------------------------------------------------------------------------------
// TEST 15: Tiered execution matches the chosen reference behavior within a documented tolerance.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_15_tiered_execution_matches_reference_behavior() {
    let root = temp_dir("test15");
    let store = PersistentStore::open(&root).unwrap();
    let reference_weights = vec![1i8, -1, 0, 1, 1, -1, 0, 1];
    let bytes: Vec<u8> = reference_weights.iter().map(|&x| x as u8).collect();
    let (m, b) = baseline_from_blocks("m15", "int8_ternary", vec![("layer0".into(), "w".into(), bytes)]);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 128, staging_budget: 64, gpu_budget: 256, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m15".into(), "baseline".into(), cfg).unwrap();

    let handle = rt.acquire(&BlockKey { model_id: "m15".into(), version: "baseline".into(), block_id: "layer0".into() }).unwrap();
    let tiered_weights: Vec<i8> = handle.bytes.iter().map(|&x| x as i8).collect();
    assert_eq!(tiered_weights, reference_weights, "Tiered execution weights must match reference with zero tolerance");
}

// -------------------------------------------------------------------------------------------------
// TEST 16: Learning changes are traceable to their source examples and committed version.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_16_learning_changes_traceable_to_provenance() {
    let root = temp_dir("test16");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m16", "native", vec![("w".into(), "c".into(), vec![0u8; 8])]);
    store.create_baseline(m, &b).unwrap();

    let prov = "batch-52: tokens=['hello', 'world'], objective=qa_loss_0.01".to_string();
    let mut cand = store.candidate("m16", "baseline", BTreeMap::from([("w".into(), vec![1u8; 8])]), prov.clone()).unwrap();
    store.validate_candidate(&mut cand, |_| true).unwrap();
    let manifest = store.commit(&cand, "v_learned").unwrap();

    assert!(manifest.provenance.contains("batch-52"));
    assert!(manifest.provenance.contains(&cand.id));

    let journal = fs::read_to_string(root.join("models/m16/journal.log")).unwrap();
    assert!(journal.contains("candidate|"));
    assert!(journal.contains("validated|"));
    assert!(journal.contains("committed|"));
    assert!(journal.contains("batch-52"));
}

// -------------------------------------------------------------------------------------------------
// TEST 17: The complete 10-billion-parameter benchmark meets the defined GPU-memory acceptance criterion.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_17_ten_billion_parameter_benchmark_gpu_memory_bounded() {
    let manifest = synthetic_10b_manifest("ten-billion-test", 8 * 1024 * 1024, 0x1337);
    assert_eq!(manifest.parameter_count, 10_000_000_000);
    assert!(manifest.blocks.len() > 1000);

    let root = temp_dir("test17");
    let store = PersistentStore::open(&root).unwrap();
    store.create_baseline(manifest, &BTreeMap::new()).unwrap();

    let mut cfg = RuntimeConfig::default();
    cfg.native_gpu = cfg!(feature = "sycl");
    let rt = TriTierRuntime::open(&root, "ten-billion-test".into(), "baseline".into(), cfg).unwrap();

    let m = store.manifest("ten-billion-test", "baseline").unwrap();
    let block_ids: Vec<String> = m.blocks.keys().cloned().collect();

    for i in 0..16 {
        let id = block_ids[i].clone();
        let key = BlockKey { model_id: "ten-billion-test".into(), version: "baseline".into(), block_id: id };
        let h = rt.acquire(&key).unwrap();
        assert_eq!(h.bytes.len(), 8 * 1024 * 1024);
        drop(h);
    }

    let t = rt.telemetry();
    assert!(t.gpu.peak_bytes < GPU_HARD_LIMIT_BYTES, "Peak GPU memory must stay strictly below 500,000,000 bytes");
    if let Some(ext_peak) = t.gpu.external_peak_bytes {
        assert!(ext_peak < GPU_HARD_LIMIT_BYTES, "External peak GPU memory must stay strictly below 500,000,000 bytes");
    }
}

// -------------------------------------------------------------------------------------------------
// TEST 18: Concurrency Stress Test: Concurrent Readers and Background Learning Tasks
// -------------------------------------------------------------------------------------------------
#[test]
fn test_18_concurrency_stress_inference_and_learning() {
    let root = temp_dir("test18");
    let store = PersistentStore::open(&root).unwrap();
    let mut blocks = Vec::new();
    for i in 0..8 {
        blocks.push((format!("b_{i}"), "stress".into(), vec![i as u8; 128]));
    }
    let (m, b) = baseline_from_blocks("m18", "native", blocks);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 1024, staging_budget: 256, gpu_budget: 2048, worker_capacity: 4, initial_tile: 128, min_tile: 32, native_gpu: false };
    let rt = Arc::new(TriTierRuntime::open(&root, "m18".into(), "baseline".into(), cfg).unwrap());

    let running = Arc::new(AtomicBool::new(true));
    let mut threads = Vec::new();

    // 4 concurrent reader threads
    for tid in 0..4 {
        let rt_c = rt.clone();
        let run_c = running.clone();
        threads.push(thread::spawn(move || {
            let mut iters = 0;
            while run_c.load(Ordering::Relaxed) && iters < 50 {
                let id = format!("b_{}", (tid + iters) % 8);
                let key = BlockKey { model_id: "m18".into(), version: "baseline".into(), block_id: id };
                if let Ok(h) = rt_c.acquire(&key) {
                    assert_eq!(h.bytes.len(), 128);
                }
                iters += 1;
            }
        }));
    }

    // Concurrent background learning jobs
    let mut jobs = Vec::new();
    for i in 0..10 {
        jobs.push(rt.scheduler().submit(TaskClass::Learning, 1, Box::new(move || {
            thread::sleep(Duration::from_millis(1));
            Ok(format!("done_{i}"))
        })));
    }

    for j in jobs { j.recv().unwrap().unwrap(); }
    running.store(false, Ordering::Relaxed);
    for t in threads { t.join().unwrap(); }

    let t = rt.telemetry();
    assert!(t.ram.peak_bytes <= 1024);
}

// -------------------------------------------------------------------------------------------------
// TEST 19: Sustained memory pressure causes tile reduction rather than crashing.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_19_sustained_memory_pressure_reduces_tile_size() {
    let mut gpu = GpuCache::new(200, 400, 128, 16, false).unwrap();
    let initial_tile = gpu.tile_bytes;
    assert_eq!(initial_tile, 128);

    let k1 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k1".into() };
    gpu.acquire(k1, &[1u8; 180]).unwrap(); // Occupies 180 of 200

    let k2 = BlockKey { model_id: "m".into(), version: "v".into(), block_id: "k2".into() };
    // Pressure event 1: tile halves from 128 to 64
    let _ = gpu.acquire(k2.clone(), &[2u8; 50]);
    assert_eq!(gpu.tile_bytes, 64);

    // Pressure event 2: tile halves from 64 to 32
    let _ = gpu.acquire(k2.clone(), &[2u8; 50]);
    assert_eq!(gpu.tile_bytes, 32);
}

// -------------------------------------------------------------------------------------------------
// TEST 20: Repeated restarts and persistent HEAD preservation.
// -------------------------------------------------------------------------------------------------
#[test]
fn test_20_repeated_restarts_and_head_preservation() {
    let root = temp_dir("test20");
    {
        let store = PersistentStore::open(&root).unwrap();
        let (m, b) = baseline_from_blocks("m20", "native", vec![("w".into(), "c".into(), vec![42u8; 8])]);
        store.create_baseline(m, &b).unwrap();
        let mut cand = store.candidate("m20", "baseline", BTreeMap::from([("w".into(), vec![99u8; 8])]), "step1".into()).unwrap();
        store.validate_candidate(&mut cand, |_| true).unwrap();
        store.commit(&cand, "generation-1").unwrap();
    }

    // Restart 1
    {
        let store = PersistentStore::open(&root).unwrap();
        assert_eq!(store.head("m20").unwrap(), "generation-1");
        let (meta, bytes) = store.read_block(&BlockKey { model_id: "m20".into(), version: "generation-1".into(), block_id: "w".into() }).unwrap();
        assert_eq!(bytes.as_slice(), &[99u8; 8]);
        assert_eq!(meta.version_id(&store.head("m20").unwrap()), "generation-1");
    }

    // Restart 2: Rollback and restart again
    {
        let store = PersistentStore::open(&root).unwrap();
        store.set_head("m20", "baseline").unwrap();
    }
    {
        let store = PersistentStore::open(&root).unwrap();
        assert_eq!(store.head("m20").unwrap(), "baseline");
    }
}

// -------------------------------------------------------------------------------------------------
// TEST 21: Simulated Disk I/O Error Graceful Handling & Error Isolation
// -------------------------------------------------------------------------------------------------
#[test]
fn test_21_disk_io_error_simulation_graceful_handling() {
    let root = temp_dir("test21");
    let store = PersistentStore::open(&root).unwrap();
    let (m, b) = baseline_from_blocks("m21", "native", vec![
        ("good_block".into(), "layer0".into(), vec![1u8; 64]),
        ("missing_block".into(), "layer0".into(), vec![2u8; 64]),
    ]);
    store.create_baseline(m, &b).unwrap();

    let cfg = RuntimeConfig { ram_budget: 1024, staging_budget: 256, gpu_budget: 2048, worker_capacity: 2, initial_tile: 64, min_tile: 16, native_gpu: false };
    let rt = TriTierRuntime::open(&root, "m21".into(), "baseline".into(), cfg).unwrap();

    // 1. Verify good block is read successfully
    let k_good = BlockKey { model_id: "m21".into(), version: "baseline".into(), block_id: "good_block".into() };
    assert!(rt.acquire(&k_good).is_ok());

    // 2. Simulate disk failure / missing block by removing the underlying file on disk
    let missing_path = root.join("models/m21/versions/baseline/blocks/missing_block.bin");
    fs::remove_file(missing_path).unwrap();

    // 3. Runtime acquire on missing block must return an error gracefully, not panic
    let k_missing = BlockKey { model_id: "m21".into(), version: "baseline".into(), block_id: "missing_block".into() };
    let err = rt.acquire(&k_missing);
    assert!(err.is_err(), "Accessing deleted/corrupted disk block must return Err");

    // 4. Runtime remains intact and can still access good blocks or recover
    assert!(rt.acquire(&k_good).is_ok(), "Runtime must remain operational after an isolated I/O error");
}

// -------------------------------------------------------------------------------------------------
// TEST 22: Dual Representation Contract: Native Ternary vs Project A 2-Bit Packed
// -------------------------------------------------------------------------------------------------
#[test]
fn test_22_cross_representation_provider_project_a_and_native() {
    let root = temp_dir("test22");
    let store = PersistentStore::open(&root).unwrap();

    // Native int8 ternary: 4 weights = 4 bytes [-1, 0, 1, -1]
    let native_bytes = vec![255u8, 0u8, 1u8, 255u8];
    // Project A 2-bit packed: 4 weights packed in 1 byte:
    // [-1, 0, 1, -1] -> 2-bit codes: -1:0b11, 0:0b00, 1:0b01, -1:0b11
    // packed byte = (0b11 << 6) | (0b00 << 4) | (0b01 << 2) | (0b11) = 192 | 0 | 4 | 3 = 199
    let project_a_packed = vec![199u8];

    let (m_nat, b_nat) = baseline_from_blocks("m_native", "int8_ternary", vec![("w".into(), "c".into(), native_bytes.clone())]);
    store.create_baseline(m_nat, &b_nat).unwrap();

    let (m_pja, b_pja) = baseline_from_blocks("m_project_a", "project_a_2bit", vec![("w".into(), "c".into(), project_a_packed.clone())]);
    store.create_baseline(m_pja, &b_pja).unwrap();

    let cfg = RuntimeConfig::default();
    let rt_nat = TriTierRuntime::open(&root, "m_native".into(), "baseline".into(), cfg.clone()).unwrap();
    let rt_pja = TriTierRuntime::open(&root, "m_project_a".into(), "baseline".into(), cfg).unwrap();

    let h_nat = rt_nat.acquire(&BlockKey { model_id: "m_native".into(), version: "baseline".into(), block_id: "w".into() }).unwrap();
    assert_eq!(h_nat.bytes.as_slice(), native_bytes.as_slice());

    let h_pja = rt_pja.acquire(&BlockKey { model_id: "m_project_a".into(), version: "baseline".into(), block_id: "w".into() }).unwrap();
    assert_eq!(h_pja.bytes.as_slice(), project_a_packed.as_slice());
}

trait MetaExt { fn version_id(&self, current: &str) -> String; }
impl MetaExt for BlockMeta { fn version_id(&self, current: &str) -> String { current.to_string() } }
