use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tritier_runtime::*;

fn usage() {
    eprintln!("tritier init <root> | benchmark <root> [--operations N] [--require-native-gpu] [--config <path>] | recover <root> <model>");
}

fn report(rt: &TriTierRuntime, ops: u64, elapsed_ms: u128, accepted: bool, reason: &str) -> String {
    let t = rt.telemetry();
    format!(
        "{{\n  \"benchmark\": \"synthetic_10b\",\n  \"logical_parameter_count\": 10000000000,\n  \"operations\": {ops},\n  \"elapsed_ms\": {elapsed_ms},\n  \"acceptance_status\": \"{}\",\n  \"reason\": \"{}\",\n  \"gpu\": {{\n    \"native_backend\": {},\n    \"internal_current_bytes\": {},\n    \"internal_peak_bytes\": {},\n    \"external_current_bytes\": {},\n    \"external_peak_bytes\": {},\n    \"hard_limit_bytes\": 500000000\n  }},\n  \"ram\": {{\"current_bytes\": {}, \"peak_bytes\": {}}},\n  \"staging\": {{\"current_bytes\": {}, \"peak_bytes\": {}}},\n  \"scheduler\": {{\"learning_units\": {}, \"doing_units\": {}, \"learning_busy_ms\": {}, \"doing_busy_ms\": {}}}\n}}\n",
        if accepted { "passed" } else { "not_passed" }, reason.replace('"', "'"), t.gpu_native,
        t.gpu.current_bytes, t.gpu.peak_bytes,
        t.gpu.external_current_bytes.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
        t.gpu.external_peak_bytes.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
        t.ram.current_bytes, t.ram.peak_bytes, t.staging_current, t.staging_peak,
        t.scheduler.learning_units, t.scheduler.doing_units,
        t.scheduler.learning_busy.as_millis(), t.scheduler.doing_busy.as_millis()
    )
}

fn benchmark(root: PathBuf, operations: u64, require_native_gpu: bool, config_path: Option<PathBuf>) -> Result<()> {
    let store = PersistentStore::open(&root)?;
    let manifest = synthetic_10b_manifest("synthetic-10b", 8 * 1024 * 1024, 0x5eed);
    let none: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    match store.create_baseline(manifest, &none) { Ok(()) | Err(Error::ImmutableBaseline) => {}, Err(e) => return Err(e) }
    let mut cfg = if let Some(ref p) = config_path {
        RuntimeConfig::from_file(p)?
    } else {
        RuntimeConfig::default()
    };
    if require_native_gpu || cfg!(feature = "sycl") {
        cfg.native_gpu = true;
    }
    let runtime = TriTierRuntime::open(&root, "synthetic-10b".into(), "baseline".into(), cfg)?;
    if require_native_gpu && !runtime.telemetry().gpu_native {
        return Err(Error::Unsupported("native SYCL GPU backend was not built; refusing to claim GPU acceptance".into()));
    }
    let manifest = store.manifest("synthetic-10b", "baseline")?;
    let ids: Vec<String> = manifest.blocks.keys().cloned().collect();
    let began = Instant::now();
    let mut learning = Vec::new();
    for i in 0..operations {
        let key = BlockKey { model_id: "synthetic-10b".into(), version: "baseline".into(), block_id: ids[(i as usize * 17) % ids.len()].clone() };
        runtime.prefetch(key.clone()).recv().map_err(|e| Error::Io(e.to_string()))??;
        let handle = runtime.acquire(&key)?;
        let _: u64 = handle.bytes.iter().take(128).map(|v| *v as u64).sum();
        learning.push(runtime.scheduler().submit(TaskClass::Learning, 1, Box::new(move || { let _ = i.wrapping_mul(7919); Ok("candidate evaluated".into()) })));
    }
    for r in learning { r.recv().map_err(|e| Error::Io(e.to_string()))??; }
    let elapsed = began.elapsed().as_millis();
    let t = runtime.telemetry();
    let passed = t.gpu_native && t.gpu.peak_bytes < GPU_HARD_LIMIT_BYTES && t.gpu.external_peak_bytes.map(|v| v < GPU_HARD_LIMIT_BYTES).unwrap_or(false);
    let reason = if passed { "native GPU internal/external measurements are below hard limit" }
        else if !t.gpu_native { "CPU synthetic dry run: storage, bounded paging and 50/50 queues exercised; native GPU telemetry is absent, so this is not an acceptance pass" }
        else { "native backend lacks external process telemetry; acceptance is intentionally withheld" };
    let body = report(&runtime, operations, elapsed, passed, reason);
    fs::create_dir_all(root.join("reports"))?;
    let path = root.join("reports/synthetic_10b_report.json");
    fs::write(&path, &body)?;
    print!("{body}");
    println!("report={}", path.display());
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 { usage(); return Ok(()); }
    match a[1].as_str() {
        "init" => {
            let store = PersistentStore::open(&a[2])?;
            let (manifest, blocks) = baseline_from_blocks("example", "int8_ternary", vec![("weights-0".into(), "example".into(), vec![0, 1, 255, 0])]);
            store.create_baseline(manifest, &blocks)?;
            println!("immutable baseline initialized");
        }
        "recover" if a.len() >= 4 => { for warning in PersistentStore::open(&a[2])?.recover(&a[3])? { println!("{warning}"); } }
        "benchmark" => {
            let mut operations = 64;
            if let Some(i) = a.iter().position(|x| x == "--operations") { operations = a.get(i + 1).ok_or_else(|| Error::Validation("missing operations".into()))?.parse().map_err(|_| Error::Validation("invalid operations".into()))?; }
            let config_path = a.iter().position(|x| x == "--config").and_then(|i| a.get(i + 1)).map(PathBuf::from);
            benchmark(PathBuf::from(&a[2]), operations, a.iter().any(|x| x == "--require-native-gpu"), config_path)?;
        }
        _ => usage(),
    }
    Ok(())
}
