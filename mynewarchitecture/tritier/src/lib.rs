//! TriTier's Rust core owns block identity, cache ownership, transactions,
//! queue admission, and all resource limits.  Model math stays outside this
//! crate behind [`ParameterProvider`], so parameter encoding is never coupled
//! to storage placement.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(feature = "sycl")]
mod native_sycl;

pub const GPU_HARD_LIMIT_BYTES: u64 = 500_000_000;

#[derive(Debug, Clone)]
pub enum Error {
    Io(String), Format(String), Integrity(String), NotFound(String), Budget(String),
    ImmutableBaseline, Validation(String), Unsupported(String), Cancelled,
}
impl Display for Error { fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") } }
impl std::error::Error for Error {}
impl From<std::io::Error> for Error { fn from(e: std::io::Error) -> Self { Error::Io(e.to_string()) } }
pub type Result<T> = std::result::Result<T, Error>;

fn now_ns() -> u128 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() }

/// FNV-1a 64 is used as an on-disk corruption checksum. It is intentionally
/// not an authenticity primitive; trusted storage may add a signed manifest.
pub fn checksum(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    format!("fnv1a64:{h:016x}")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BlockKey { pub model_id: String, pub version: String, pub block_id: String }
impl BlockKey { pub fn text(&self) -> String { format!("{}:{}:{}", self.model_id, self.version, self.block_id) } }

#[derive(Debug, Clone)]
pub enum ContentSource { File(String), Synthetic { seed: u64 } }
#[derive(Debug, Clone)]
pub struct BlockMeta {
    pub block_id: String, pub component: String, pub offset: u64, pub byte_size: u64,
    pub parameter_count: u64, pub encoding: String, pub checksum: String, pub source: ContentSource,
}
#[derive(Debug, Clone)]
pub struct Manifest {
    pub model_id: String, pub version: String, pub parent_version: Option<String>, pub baseline_version: String,
    pub immutable_baseline: bool, pub representation: String, pub parameter_count: u64,
    pub created_at_ns: u128, pub status: String, pub blocks: BTreeMap<String, BlockMeta>,
    pub provenance: String,
}

/// Line-oriented manifest avoids a JSON dependency in the critical runtime.
/// Values are percent encoded so the format is stable across Rust, C++, Julia.
fn esc(s: &str) -> String { s.bytes().flat_map(|b| match b { b'%'|b'|'|b'\n'|b'\r' => format!("%{b:02X}").bytes().collect::<Vec<_>>(), _ => vec![b] }).map(char::from).collect() }
fn unesc(s: &str) -> Result<String> {
    let mut o = Vec::new(); let b = s.as_bytes(); let mut i = 0;
    while i < b.len() { if b[i] == b'%' { if i + 2 >= b.len() { return Err(Error::Format("bad escape".into())); }
        let x = std::str::from_utf8(&b[i+1..i+3]).map_err(|_| Error::Format("utf8".into()))?;
        o.push(u8::from_str_radix(x, 16).map_err(|_| Error::Format("hex".into()))?); i += 3;
    } else { o.push(b[i]); i += 1; } }
    String::from_utf8(o).map_err(|_| Error::Format("utf8".into()))
}
impl Manifest {
    fn encode(&self) -> String {
        let parent = self.parent_version.clone().unwrap_or_default();
        let mut out = format!("TRITIER-MANIFEST|1\nmodel|{}\nversion|{}\nparent|{}\nbaseline|{}\nimmutable|{}\nrepresentation|{}\nparameters|{}\ncreated_ns|{}\nstatus|{}\nprovenance|{}\n",
            esc(&self.model_id), esc(&self.version), esc(&parent), esc(&self.baseline_version), self.immutable_baseline as u8,
            esc(&self.representation), self.parameter_count, self.created_at_ns, esc(&self.status), esc(&self.provenance));
        for m in self.blocks.values() {
            let (kind, source) = match &m.source { ContentSource::File(p) => ("file", p.clone()), ContentSource::Synthetic{seed} => ("synthetic", seed.to_string()) };
            out.push_str(&format!("block|{}|{}|{}|{}|{}|{}|{}|{}|{}\n", esc(&m.block_id), esc(&m.component), m.offset, m.byte_size, m.parameter_count, esc(&m.encoding), esc(&m.checksum), kind, esc(&source)));
        } out
    }
    fn decode(text: &str) -> Result<Self> {
        let mut kv = HashMap::new(); let mut blocks = BTreeMap::new();
        let mut lines = text.lines(); if lines.next() != Some("TRITIER-MANIFEST|1") { return Err(Error::Format("manifest magic/version".into())); }
        for l in lines { let p: Vec<&str> = l.split('|').collect(); if p.is_empty() { continue; }
            if p[0] == "block" { if p.len() != 10 { return Err(Error::Format("block columns".into())); }
                let source = match p[8] { "file" => ContentSource::File(unesc(p[9])?), "synthetic" => ContentSource::Synthetic { seed: unesc(p[9])?.parse().map_err(|_| Error::Format("seed".into()))? }, _ => return Err(Error::Format("source kind".into())) };
                let m = BlockMeta { block_id: unesc(p[1])?, component: unesc(p[2])?, offset: p[3].parse().map_err(|_| Error::Format("offset".into()))?, byte_size: p[4].parse().map_err(|_| Error::Format("size".into()))?, parameter_count: p[5].parse().map_err(|_| Error::Format("count".into()))?, encoding: unesc(p[6])?, checksum: unesc(p[7])?, source };
                blocks.insert(m.block_id.clone(), m);
            } else if p.len() == 2 { kv.insert(p[0].to_string(), unesc(p[1])?); } else { return Err(Error::Format("manifest line".into())); }
        }
        let get = |k: &str| kv.get(k).cloned().ok_or_else(|| Error::Format(format!("missing {k}")));
        Ok(Self { model_id:get("model")?, version:get("version")?, parent_version:match get("parent")? { x if x.is_empty()=>None,x=>Some(x)}, baseline_version:get("baseline")?, immutable_baseline:get("immutable")?=="1", representation:get("representation")?, parameter_count:get("parameters")?.parse().map_err(|_|Error::Format("parameters".into()))?, created_at_ns:get("created_ns")?.parse().map_err(|_|Error::Format("created".into()))?, status:get("status")?, provenance:get("provenance")?, blocks })
    }
}

#[derive(Debug, Clone)]
pub struct CandidateUpdate { pub id: String, pub model_id: String, pub parent_version: String, pub changes: BTreeMap<String, Vec<u8>>, pub provenance: String, pub created_at_ns: u128, pub validated: bool }

/// Persistent canonical store. Baselines are written exactly once. New values
/// exist only in transactions and committed version directories.
pub struct PersistentStore { root: PathBuf, lock: Mutex<()> }
impl PersistentStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> { let root = root.into(); fs::create_dir_all(root.join("models"))?; Ok(Self{root,lock:Mutex::new(())}) }
    fn model_root(&self, model: &str) -> PathBuf { self.root.join("models").join(model) }
    fn manifest_path(&self, model: &str, version: &str) -> PathBuf { self.model_root(model).join("versions").join(version).join("manifest.tier") }
    fn head_path(&self, model: &str) -> PathBuf { self.model_root(model).join("HEAD") }
    fn journal(&self, model: &str, event: &str) -> Result<()> {
        let path = self.model_root(model).join("journal.log");
        fs::create_dir_all(path.parent().unwrap())?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{}|{}", now_ns(), event.replace('\n', " "))?;
        file.sync_data()?;
        Ok(())
    }
    fn atomic_write(path: &Path, data: &[u8]) -> Result<()> {
        let tmp = path.with_extension(format!("tmp-{}", now_ns()));
        let mut f = File::create(&tmp).map_err(|e| Error::Io(format!("create {}: {e}", tmp.display())))?;
        f.write_all(data).map_err(|e| Error::Io(format!("write {}: {e}", tmp.display())))?;
        f.sync_all().map_err(|e| Error::Io(format!("sync {}: {e}", tmp.display())))?;
        drop(f);
        fs::rename(&tmp, path).map_err(|e| Error::Io(format!("rename {} -> {}: {e}", tmp.display(), path.display())))?;
        Ok(())
    }
    pub fn create_baseline(&self, manifest: Manifest, payloads: &BTreeMap<String, Vec<u8>>) -> Result<()> {
        let _g=self.lock.lock().unwrap(); if !manifest.immutable_baseline || manifest.version != manifest.baseline_version { return Err(Error::Validation("baseline manifest must be immutable and self-identifying".into())); }
        let p=self.manifest_path(&manifest.model_id,&manifest.version); if p.exists(){return Err(Error::ImmutableBaseline);}
        for m in manifest.blocks.values() { if let ContentSource::File(rel)=&m.source { let data=payloads.get(&m.block_id).ok_or_else(||Error::Validation(format!("missing baseline payload {}",m.block_id)))?; if data.len() as u64!=m.byte_size || checksum(data)!=m.checksum{return Err(Error::Integrity(m.block_id.clone()));} let target=self.model_root(&manifest.model_id).join(rel); fs::create_dir_all(target.parent().unwrap())?; Self::atomic_write(&target,data)?; } }
        fs::create_dir_all(p.parent().unwrap())?; Self::atomic_write(&p,manifest.encode().as_bytes())?; Self::atomic_write(&self.head_path(&manifest.model_id),manifest.version.as_bytes())?; Ok(())
    }
    pub fn manifest(&self, model:&str, version:&str)->Result<Manifest>{ let mut s=String::new(); File::open(self.manifest_path(model,version))?.read_to_string(&mut s)?; Manifest::decode(&s) }
    pub fn head(&self, model:&str)->Result<String>{ Ok(fs::read_to_string(self.head_path(model))?.trim().to_string()) }
    pub fn set_head(&self, model:&str, version:&str)->Result<()> { let _g=self.lock.lock().unwrap(); let _=self.manifest(model,version)?; Self::atomic_write(&self.head_path(model),version.as_bytes()) }
    fn synthetic(meta:&BlockMeta, seed:u64)->Vec<u8>{ let mut x=seed ^ meta.offset ^ meta.parameter_count.rotate_left(17); let mut out=vec![0u8;meta.byte_size as usize]; for b in &mut out { x^=x<<13;x^=x>>7;x^=x<<17; *b=(x%3) as u8; } out }
    pub fn read_block(&self,key:&BlockKey)->Result<(BlockMeta,Vec<u8>)>{ let m=self.manifest(&key.model_id,&key.version)?; let meta=m.blocks.get(&key.block_id).ok_or_else(||Error::NotFound(key.text()))?.clone(); let b=match &meta.source { ContentSource::File(rel)=>fs::read(self.model_root(&key.model_id).join(rel))?, ContentSource::Synthetic{seed}=>Self::synthetic(&meta,*seed)}; if b.len() as u64!=meta.byte_size{return Err(Error::Integrity(format!("{} wrong length",key.text())));} if matches!(meta.source,ContentSource::File(_)) && checksum(&b)!=meta.checksum{return Err(Error::Integrity(format!("{} checksum",key.text())));} Ok((meta,b)) }
    pub fn candidate(&self,model:&str,parent:&str,changes:BTreeMap<String,Vec<u8>>,provenance:String)->Result<CandidateUpdate>{ let _guard=self.lock.lock().unwrap(); let parent_manifest=self.manifest(model,parent)?; for (id,b) in &changes { let m=parent_manifest.blocks.get(id).ok_or_else(||Error::NotFound(id.clone()))?; if b.len() as u64!=m.byte_size{return Err(Error::Validation(format!("{id}: byte length changes are prohibited")));} } let candidate=CandidateUpdate{id:format!("cand-{}",now_ns()),model_id:model.into(),parent_version:parent.into(),changes,provenance,created_at_ns:now_ns(),validated:false}; self.journal(model,&format!("candidate|{}|parent={}|changes={}|provenance={}",candidate.id,candidate.parent_version,candidate.changes.len(),candidate.provenance))?; Ok(candidate) }
    pub fn validate_candidate(&self,c:&mut CandidateUpdate, objective:impl Fn(&CandidateUpdate)->bool)->Result<()> { if c.changes.is_empty(){return Err(Error::Validation("empty update".into()));} if !objective(c){self.journal(&c.model_id,&format!("rejected|{}|objective",c.id))?;return Err(Error::Validation("candidate objective rejected update".into()));} c.validated=true; self.journal(&c.model_id,&format!("validated|{}",c.id))?; Ok(()) }
    pub fn commit(&self,c:&CandidateUpdate,new_version:&str)->Result<Manifest>{ let _g=self.lock.lock().unwrap(); if !c.validated{return Err(Error::Validation("candidate not validated".into()));} let parent=self.manifest(&c.model_id,&c.parent_version)?; if parent.status!="committed"{return Err(Error::Validation("parent not committed".into()));} if self.manifest_path(&c.model_id,new_version).exists(){return Err(Error::Validation("version exists".into()));}
        let mr=self.model_root(&c.model_id); let tx=mr.join("transactions").join(format!("{}-{}",c.id,now_ns())); fs::create_dir_all(&tx)?; let mut blocks=parent.blocks.clone();
        for (id,data) in &c.changes { let old=blocks.get(id).unwrap().clone(); let rel=format!("versions/{new_version}/blocks/{id}.bin"); let staged=tx.join("blocks").join(format!("{id}.bin")); fs::create_dir_all(staged.parent().unwrap())?; Self::atomic_write(&staged,data)?; let mut n=old; n.checksum=checksum(data); n.source=ContentSource::File(rel); blocks.insert(id.clone(),n); }
        let manifest=Manifest{model_id:c.model_id.clone(),version:new_version.into(),parent_version:Some(parent.version.clone()),baseline_version:parent.baseline_version.clone(),immutable_baseline:false,representation:parent.representation.clone(),parameter_count:parent.parameter_count,created_at_ns:now_ns(),status:"committed".into(),blocks,provenance:format!("candidate={};{}",c.id,c.provenance)};
        Self::atomic_write(&tx.join("manifest.tier"),manifest.encode().as_bytes())?; let final_dir=mr.join("versions").join(new_version); fs::create_dir_all(final_dir.parent().unwrap())?; fs::rename(&tx,&final_dir)?; Self::atomic_write(&self.head_path(&c.model_id),new_version.as_bytes())?; self.journal(&c.model_id,&format!("committed|{}|version={new_version}",c.id))?; Ok(manifest) }
    /// Ignore incomplete transactions. A transaction only becomes visible when
    /// its directory atomically becomes `versions/<version>` with manifest.
    pub fn recover(&self,model:&str)->Result<Vec<String>>{ let mut warnings=Vec::new(); let tx=self.model_root(model).join("transactions"); if tx.exists(){for e in fs::read_dir(tx)?{warnings.push(format!("ignored incomplete transaction {}",e?.file_name().to_string_lossy()));}} let h=self.head(model)?; let _=self.manifest(model,&h)?; Ok(warnings) }
}

#[derive(Default, Clone)]
pub struct CacheStats { pub hits:u64,pub misses:u64,pub evictions:u64,pub current_bytes:u64,pub peak_bytes:u64,pub allocation_failures:u64 }
struct RamEntry { data:Arc<Vec<u8>>, pins:u32,last:u64 }
pub struct RamCache { budget:u64, clock:u64, entries:HashMap<BlockKey,RamEntry>, pub stats:CacheStats }
impl RamCache {
    pub fn new(budget:u64)->Self{Self{budget,clock:0,entries:HashMap::new(),stats:CacheStats::default()}}
    fn touch(&mut self){self.clock+=1;}
    pub fn get(&mut self,key:&BlockKey)->Option<Arc<Vec<u8>>>{self.touch();let clk=self.clock;if let Some(e)=self.entries.get_mut(key){e.pins+=1;e.last=clk;self.stats.hits+=1;return Some(e.data.clone())}self.stats.misses+=1;None}
    pub fn insert_and_pin(&mut self,key:BlockKey,data:Vec<u8>)->Result<Arc<Vec<u8>>>{ let sz=data.len()as u64;if sz>self.budget{self.stats.allocation_failures+=1;return Err(Error::Budget("block exceeds RAM cache budget".into()));} while self.stats.current_bytes+sz>self.budget {let victim=self.entries.iter().filter(|(_,e)|e.pins==0).min_by_key(|(_,e)|e.last).map(|(k,_)|k.clone());match victim{Some(k)=>{let e=self.entries.remove(&k).unwrap();self.stats.current_bytes-=e.data.len()as u64;self.stats.evictions+=1;}None=>{self.stats.allocation_failures+=1;return Err(Error::Budget("RAM cache has only reader-protected blocks".into()));}}} self.touch();let a=Arc::new(data);self.entries.insert(key,RamEntry{data:a.clone(),pins:1,last:self.clock});self.stats.current_bytes+=sz;self.stats.peak_bytes=self.stats.peak_bytes.max(self.stats.current_bytes);Ok(a)}
    pub fn release(&mut self,key:&BlockKey){if let Some(e)=self.entries.get_mut(key){e.pins=e.pins.saturating_sub(1)}}
}

/// Enforces bounded double/multi-buffer staging independently from cached RAM.
pub struct StagingBudget { cap:u64, current:u64, peak:u64 }
impl StagingBudget { pub fn new(cap:u64)->Self{Self{cap,current:0,peak:0}} pub fn reserve(&mut self,n:u64)->Result<()> {if self.current+n>self.cap{return Err(Error::Budget("staging budget exhausted".into()))}self.current+=n;self.peak=self.peak.max(self.current);Ok(())} pub fn release(&mut self,n:u64){self.current=self.current.saturating_sub(n)} pub fn current(&self)->u64{self.current} pub fn peak(&self)->u64{self.peak} }

#[derive(Default,Clone)]
pub struct GpuStats { pub current_bytes:u64,pub peak_bytes:u64,pub external_current_bytes:Option<u64>,pub external_peak_bytes:Option<u64>,pub hits:u64,pub misses:u64,pub evictions:u64,pub allocation_failures:u64,pub pressure_events:u64 }
struct GpuEntry { bytes:u64,pins:u32,last:u64,native_ptr:usize }
enum NativeGpu {
    Disabled,
    #[cfg(feature = "sycl")]
    Sycl(native_sycl::Device),
}
impl NativeGpu {
    fn initialize(requested: bool, hard: u64) -> Result<Self> {
        #[cfg(feature = "sycl")]
        if requested { return Ok(Self::Sycl(native_sycl::Device::create(hard)?)); }
        #[cfg(not(feature = "sycl"))]
        if requested { return Err(Error::Unsupported("native SYCL requested but this Rust binary was built without --features sycl".into())); }
        Ok(Self::Disabled)
    }
    fn enabled(&self) -> bool {
        #[cfg(feature = "sycl")]
        { return matches!(self, Self::Sycl(_)); }
        #[cfg(not(feature = "sycl"))]
        { false }
    }
    fn upload(&mut self, data: &[u8]) -> Result<usize> {
        match self {
            Self::Disabled => Ok(0),
            #[cfg(feature = "sycl")]
            Self::Sycl(device) => device.upload(data),
        }
    }
    fn free(&mut self, ptr: usize) {
        #[cfg(feature = "sycl")]
        if let Self::Sycl(device) = self { device.free(ptr); }
        let _ = ptr;
    }
    fn sample(&self) -> (Option<u64>, Option<u64>) {
        match self {
            Self::Disabled => (None, None),
            #[cfg(feature = "sycl")]
            Self::Sycl(device) => { let m = device.metrics(); if m.external_telemetry_available != 0 { (Some(m.external_current_bytes), Some(m.external_peak_bytes)) } else { (None, None) } }
        }
    }
}
/// A safe logical residency map. Actual SYCL allocations are delegated through
/// the C ABI in `cpp/`; without the `sycl` feature this reports no external
/// telemetry and cannot be used for an acceptance benchmark.
pub struct GpuCache { budget:u64,hard:u64,clock:u64,entries:HashMap<BlockKey,GpuEntry>,pub stats:GpuStats,pub tile_bytes:u64,min_tile_bytes:u64, native:NativeGpu }
impl GpuCache {
    pub fn new(budget:u64,hard:u64,tile:u64,min_tile:u64,native_enabled:bool)->Result<Self>{if budget>=hard||hard>GPU_HARD_LIMIT_BYTES{return Err(Error::Budget("GPU budget must stay below 500,000,000 bytes".into()))}let native=NativeGpu::initialize(native_enabled,hard)?;Ok(Self{budget,hard,clock:0,entries:HashMap::new(),stats:GpuStats::default(),tile_bytes:tile,min_tile_bytes:min_tile,native})}
    pub fn native_enabled(&self)->bool{self.native.enabled()}
    pub fn acquire(&mut self,key:BlockKey,data:&[u8])->Result<()>{ let bytes=data.len()as u64;self.clock+=1;if let Some(e)=self.entries.get_mut(&key){e.pins+=1;e.last=self.clock;self.stats.hits+=1;return Ok(())}self.stats.misses+=1;if bytes>self.budget{return self.pressure("block exceeds hot budget")};while self.stats.current_bytes+bytes>self.budget{let v=self.entries.iter().filter(|(_,e)|e.pins==0).min_by_key(|(_,e)|e.last).map(|(k,_)|k.clone());if let Some(k)=v{let e=self.entries.remove(&k).unwrap();self.native.free(e.native_ptr);self.stats.current_bytes-=e.bytes;self.stats.evictions+=1;}else{return self.pressure("all hot blocks reader-protected")}}if self.stats.current_bytes+bytes>=self.hard{return self.pressure("would reach hard GPU limit")}let native_ptr=match self.native.upload(data){Ok(ptr)=>ptr,Err(e)=>{self.stats.allocation_failures+=1;return Err(e)}};self.entries.insert(key,GpuEntry{bytes,pins:1,last:self.clock,native_ptr});self.stats.current_bytes+=bytes;self.stats.peak_bytes=self.stats.peak_bytes.max(self.stats.current_bytes);let(external_current,external_peak)=self.native.sample();self.stats.external_current_bytes=external_current;self.stats.external_peak_bytes=external_peak;Ok(()) }
    fn pressure(&mut self,msg:&str)->Result<()> {self.stats.pressure_events+=1;self.tile_bytes=(self.tile_bytes/2).max(self.min_tile_bytes);self.stats.allocation_failures+=1;Err(Error::Budget(format!("GPU pressure: {msg}; tile bytes now {}",self.tile_bytes)))}
    pub fn release(&mut self,key:&BlockKey){if let Some(e)=self.entries.get_mut(key){e.pins=e.pins.saturating_sub(1)}}
}

#[derive(Clone,Copy,Debug,PartialEq,Eq)] pub enum TaskClass { Learning, Doing }
type Job = Box<dyn FnOnce() -> Result<String> + Send + 'static>;
struct Queued { priority:u8, submitted:Instant, job:Job, result:mpsc::Sender<Result<String>> }
#[derive(Default,Clone)] pub struct SchedulerStats { pub learning_units:u64,pub doing_units:u64,pub learning_busy:Duration,pub doing_busy:Duration,pub learning_wait:Duration,pub doing_wait:Duration,pub borrowed_learning:u64,pub borrowed_doing:u64 }
struct SchedulerState { learning:VecDeque<Queued>,doing:VecDeque<Queued>,shutdown:bool,stats:SchedulerStats }
pub struct Tier2Scheduler { state:Arc<(Mutex<SchedulerState>,Condvar)>, workers:Vec<thread::JoinHandle<()>>, capacity:usize }
impl Tier2Scheduler {
    pub fn new(capacity:usize)->Result<Self>{if capacity<2||capacity%2!=0{return Err(Error::Validation("Tier 2 worker capacity must be even and >=2".into()))}let state=Arc::new((Mutex::new(SchedulerState{learning:VecDeque::new(),doing:VecDeque::new(),shutdown:false,stats:SchedulerStats::default()}),Condvar::new()));let mut workers=Vec::new();for i in 0..capacity{let s=state.clone();let own=if i<capacity/2{TaskClass::Learning}else{TaskClass::Doing};workers.push(thread::spawn(move||worker(s,own)));}Ok(Self{state,workers,capacity})}
    pub fn submit(&self,class:TaskClass,priority:u8,job:Job)->mpsc::Receiver<Result<String>>{let(tx,rx)=mpsc::channel();let q=Queued{priority,submitted:Instant::now(),job,result:tx};let(lock,cv)=&*self.state;let mut s=lock.lock().unwrap();let queue=match class{TaskClass::Learning=>&mut s.learning,TaskClass::Doing=>&mut s.doing};let at=queue.iter().position(|x|x.priority<priority).unwrap_or(queue.len());queue.insert(at,q);cv.notify_one();rx}
    pub fn stats(&self)->SchedulerStats{self.state.0.lock().unwrap().stats.clone()}
    pub fn capacity(&self)->usize{self.capacity}
}
fn worker(state:Arc<(Mutex<SchedulerState>,Condvar)>,own:TaskClass){loop{let task={let(lock,cv)=&*state;let mut s=lock.lock().unwrap();loop{if s.shutdown{return}let ownq=match own{TaskClass::Learning=>&mut s.learning,TaskClass::Doing=>&mut s.doing};if let Some(q)=ownq.pop_front(){break Some((q,own,false))}// borrow only when this reserved lane has no eligible work
let other=match own{TaskClass::Learning=>TaskClass::Doing,TaskClass::Doing=>TaskClass::Learning};let otherq=match other{TaskClass::Learning=>&mut s.learning,TaskClass::Doing=>&mut s.doing};if let Some(q)=otherq.pop_front(){break Some((q,other,true))}s=cv.wait(s).unwrap();}};if let Some((q,class,borrowed))=task{let wait=q.submitted.elapsed();let began=Instant::now();let result=(q.job)();let elapsed=began.elapsed();let _=q.result.send(result);let mut s=state.0.lock().unwrap();match class{TaskClass::Learning=>{s.stats.learning_units+=1;s.stats.learning_busy+=elapsed;s.stats.learning_wait+=wait;if borrowed{s.stats.borrowed_learning+=1}},TaskClass::Doing=>{s.stats.doing_units+=1;s.stats.doing_busy+=elapsed;s.stats.doing_wait+=wait;if borrowed{s.stats.borrowed_doing+=1}}}}}}
impl Drop for Tier2Scheduler { fn drop(&mut self){let(lock,cv)=&*self.state;{let mut s=lock.lock().unwrap();s.shutdown=true;cv.notify_all();}for h in self.workers.drain(..){let _=h.join();}} }

pub trait ParameterProvider: Send+Sync { fn acquire(&self,key:&BlockKey)->Result<BlockHandle>; fn prefetch(&self,key:BlockKey)->mpsc::Receiver<Result<String>>; fn active_version(&self)->String; }
pub struct BlockHandle { pub key:BlockKey,pub bytes:Arc<Vec<u8>>,ram:Arc<Mutex<RamCache>>,gpu:Arc<Mutex<GpuCache>> }
impl Drop for BlockHandle {fn drop(&mut self){self.ram.lock().unwrap().release(&self.key);self.gpu.lock().unwrap().release(&self.key)}}
#[derive(Clone, Debug, PartialEq, Eq)] pub struct RuntimeConfig {pub ram_budget:u64,pub staging_budget:u64,pub gpu_budget:u64,pub worker_capacity:usize,pub initial_tile:u64,pub min_tile:u64,pub native_gpu:bool}
impl Default for RuntimeConfig {fn default()->Self{Self{ram_budget:256*1024*1024,staging_budget:64*1024*1024,gpu_budget:450*1024*1024,worker_capacity:4,initial_tile:8*1024*1024,min_tile:64*1024,native_gpu:false}}}
impl RuntimeConfig {
    pub fn to_json(&self) -> String {
        format!(
            "{{\n  \"ram_budget\": {},\n  \"staging_budget\": {},\n  \"gpu_budget\": {},\n  \"worker_capacity\": {},\n  \"initial_tile\": {},\n  \"min_tile\": {},\n  \"native_gpu\": {}\n}}\n",
            self.ram_budget, self.staging_budget, self.gpu_budget, self.worker_capacity, self.initial_tile, self.min_tile, self.native_gpu
        )
    }
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::Io(e.to_string()))?;
        Self::from_str(&content)
    }
    pub fn from_str(text: &str) -> Result<Self> {
        let mut cfg = Self::default();
        for raw_line in text.lines() {
            let line = raw_line.trim().trim_end_matches(',');
            if line.is_empty() || line.starts_with('{') || line.starts_with('}') || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            let mut parts = line.splitn(2, ':');
            if parts.clone().count() < 2 {
                parts = line.splitn(2, '=');
            }
            if let (Some(k), Some(v)) = (parts.next(), parts.next()) {
                let k = k.trim().trim_matches('"').trim_matches('\'').trim();
                let v = v.trim().trim_matches('"').trim_matches('\'').trim();
                match k {
                    "ram_budget" => cfg.ram_budget = v.parse().map_err(|_| Error::Validation(format!("invalid ram_budget: {v}")))?,
                    "staging_budget" => cfg.staging_budget = v.parse().map_err(|_| Error::Validation(format!("invalid staging_budget: {v}")))?,
                    "gpu_budget" => cfg.gpu_budget = v.parse().map_err(|_| Error::Validation(format!("invalid gpu_budget: {v}")))?,
                    "worker_capacity" => cfg.worker_capacity = v.parse().map_err(|_| Error::Validation(format!("invalid worker_capacity: {v}")))?,
                    "initial_tile" => cfg.initial_tile = v.parse().map_err(|_| Error::Validation(format!("invalid initial_tile: {v}")))?,
                    "min_tile" => cfg.min_tile = v.parse().map_err(|_| Error::Validation(format!("invalid min_tile: {v}")))?,
                    "native_gpu" => cfg.native_gpu = v.parse().map_err(|_| Error::Validation(format!("invalid native_gpu: {v}")))?,
                    _ => {}
                }
            }
        }
        Ok(cfg)
    }
}
pub struct TriTierRuntime { pub store:Arc<PersistentStore>,ram:Arc<Mutex<RamCache>>,staging:Arc<Mutex<StagingBudget>>,gpu:Arc<Mutex<GpuCache>>,scheduler:Tier2Scheduler,active:Arc<Mutex<String>>,model:String }
impl TriTierRuntime {
    pub fn open(root:impl Into<PathBuf>,model:String,version:String,cfg:RuntimeConfig)->Result<Self>{let store=Arc::new(PersistentStore::open(root)?);let gpu=GpuCache::new(cfg.gpu_budget,GPU_HARD_LIMIT_BYTES,cfg.initial_tile,cfg.min_tile,cfg.native_gpu)?;Ok(Self{store,ram:Arc::new(Mutex::new(RamCache::new(cfg.ram_budget))),staging:Arc::new(Mutex::new(StagingBudget::new(cfg.staging_budget))),gpu:Arc::new(Mutex::new(gpu)),scheduler:Tier2Scheduler::new(cfg.worker_capacity)?,active:Arc::new(Mutex::new(version)),model})}
    pub fn switch_version(&self,version:&str)->Result<()> {let _=self.store.manifest(&self.model,version)?;*self.active.lock().unwrap()=version.into();Ok(())}
    pub fn rollback(&self,version:&str)->Result<()> {self.store.set_head(&self.model,version)?;self.switch_version(version)}
    pub fn candidate(&self,changes:BTreeMap<String,Vec<u8>>,provenance:String)->Result<CandidateUpdate>{let parent=self.active_version();self.store.candidate(&self.model,&parent,changes,provenance)}
    pub fn commit(&self,c:&CandidateUpdate,new_version:&str)->Result<Manifest>{let m=self.store.commit(c,new_version)?;self.switch_version(new_version)?;Ok(m)}
    pub fn manifest(&self,version:&str)->Result<Manifest>{self.store.manifest(&self.model,version)}
    pub fn model_id(&self)->&str{&self.model}
    pub fn telemetry(&self)->RuntimeTelemetry{
     // Keep each guard's lifetime explicit. In a struct literal a temporary
     // mutex guard may survive through later fields, which would deadlock when
     // the same cache is sampled twice for metrics.
     let ram = self.ram.lock().unwrap().stats.clone();
     let (gpu, gpu_native) = { let cache = self.gpu.lock().unwrap(); (cache.stats.clone(), cache.native_enabled()) };
     let (staging_current, staging_peak) = { let stage = self.staging.lock().unwrap(); (stage.current(), stage.peak()) };
     RuntimeTelemetry { ram, gpu, staging_current, staging_peak, scheduler:self.scheduler.stats(), version:self.active_version(), gpu_native }
 }
pub fn scheduler(&self)->&Tier2Scheduler{&self.scheduler}
}
impl ParameterProvider for TriTierRuntime {
    fn acquire(&self,key:&BlockKey)->Result<BlockHandle>{
        if key.model_id!=self.model{return Err(Error::NotFound("model mismatch".into()))}
        let cached = self.ram.lock().unwrap().get(key);
        let bytes=if let Some(x) = cached {
            x
        }else{
            let(meta,data)=self.store.read_block(key)?;
            {
                let mut s=self.staging.lock().unwrap();
                s.reserve(meta.byte_size)?;
            }
            let result=self.ram.lock().unwrap().insert_and_pin(key.clone(),data);
            self.staging.lock().unwrap().release(meta.byte_size);
            result?
        };
        self.gpu.lock().unwrap().acquire(key.clone(),bytes.as_slice())?;
        Ok(BlockHandle{key:key.clone(),bytes,ram:self.ram.clone(),gpu:self.gpu.clone()})
    }
fn prefetch(&self,key:BlockKey)->mpsc::Receiver<Result<String>>{let store=self.store.clone();let ram=self.ram.clone();let staging=self.staging.clone();self.scheduler.submit(TaskClass::Doing,1,Box::new(move||{if ram.lock().unwrap().get(&key).is_some(){ram.lock().unwrap().release(&key);return Ok(format!("cache hit {}",key.text()))}let(meta,data)=store.read_block(&key)?;staging.lock().unwrap().reserve(meta.byte_size)?;let r=ram.lock().unwrap().insert_and_pin(key.clone(),data);staging.lock().unwrap().release(meta.byte_size);r?;ram.lock().unwrap().release(&key);Ok(format!("prefetched {}",key.text()))}))}
fn active_version(&self)->String{self.active.lock().unwrap().clone()} }
#[derive(Clone)] pub struct RuntimeTelemetry {pub ram:CacheStats,pub gpu:GpuStats,pub staging_current:u64,pub staging_peak:u64,pub scheduler:SchedulerStats,pub version:String,pub gpu_native:bool}

/// Native model adapter: exports a representation-preserving baseline from
/// caller-provided byte blocks. It never aliases or mutates the source model.
pub fn baseline_from_blocks(model:&str,representation:&str,blocks:Vec<(String,String,Vec<u8>)>)->(Manifest,BTreeMap<String,Vec<u8>>){let mut metas=BTreeMap::new();let mut payloads=BTreeMap::new();let mut offset=0;for(id,component,data)in blocks{let n=data.len()as u64;metas.insert(id.clone(),BlockMeta{block_id:id.clone(),component,offset,byte_size:n,parameter_count:n,encoding:representation.into(),checksum:checksum(&data),source:ContentSource::File(format!("versions/baseline/blocks/{id}.bin"))});payloads.insert(id,data);offset+=n;}let m=Manifest{model_id:model.into(),version:"baseline".into(),parent_version:None,baseline_version:"baseline".into(),immutable_baseline:true,representation:representation.into(),parameter_count:offset,created_at_ns:now_ns(),status:"committed".into(),blocks:metas,provenance:"native-adapter representation-preserving snapshot".into()};(m,payloads)}

/// Defined synthetic 10-billion-param artifact.  It persists an authoritative
/// manifest plus deterministic generator identity, not 10 GB of duplicate RAM.
pub fn synthetic_10b_manifest(model:&str,block_bytes:u64,seed:u64)->Manifest{let total=10_000_000_000u64;let mut blocks=BTreeMap::new();let count=(total+block_bytes-1)/block_bytes;for i in 0..count{let offset=i*block_bytes;let n=(total-offset).min(block_bytes);let id=format!("p-{i:06}");blocks.insert(id.clone(),BlockMeta{block_id:id,component:"synthetic.ternary".into(),offset,byte_size:n,parameter_count:n,encoding:"int8_ternary".into(),checksum:format!("generator-fnv1a64:{seed:016x}:{i:06}:{n}"),source:ContentSource::Synthetic{seed:seed^i}});}Manifest{model_id:model.into(),version:"baseline".into(),parent_version:None,baseline_version:"baseline".into(),immutable_baseline:true,representation:"synthetic-int8-ternary".into(),parameter_count:total,created_at_ns:now_ns(),status:"committed".into(),blocks,provenance:"synthetic generator; each block is deterministic, addressable, and materialized only through bounded staging".into()}}

#[cfg(test)] mod tests { use super::*; use std::env;
fn temp(name:&str)->PathBuf{let p=env::temp_dir().join(format!("tritier-{name}-{}",now_ns()));fs::create_dir_all(&p).unwrap();p}
#[test]fn baseline_update_rollback_and_corruption(){let p=temp("store");let s=PersistentStore::open(&p).unwrap();let(m,b)=baseline_from_blocks("m","native",vec![("a".into(),"W".into(),vec![1,0,255])]);s.create_baseline(m,&b).unwrap();assert_eq!(s.read_block(&BlockKey{model_id:"m".into(),version:"baseline".into(),block_id:"a".into()}).unwrap().1,vec![1,0,255]);let mut c=s.candidate("m","baseline",BTreeMap::from([("a".into(),vec![1,1,1])]),"test-example".into()).unwrap();s.validate_candidate(&mut c, |_|true).unwrap();s.commit(&c,"v1").unwrap();assert_eq!(s.head("m").unwrap(),"v1");s.set_head("m","baseline").unwrap();assert_eq!(s.head("m").unwrap(),"baseline");let f=p.join("models/m/versions/baseline/blocks/a.bin");fs::write(f,[9u8,9,9]).unwrap();assert!(matches!(s.read_block(&BlockKey{model_id:"m".into(),version:"baseline".into(),block_id:"a".into()}),Err(Error::Integrity(_))));}
#[test]fn reader_protection_and_budgets(){let mut c=RamCache::new(8);let a=BlockKey{model_id:"m".into(),version:"v".into(),block_id:"a".into()};c.insert_and_pin(a.clone(),vec![1;8]).unwrap();assert!(matches!(c.insert_and_pin(BlockKey{model_id:"m".into(),version:"v".into(),block_id:"b".into()},vec![2;1]),Err(Error::Budget(_))));c.release(&a);c.insert_and_pin(BlockKey{model_id:"m".into(),version:"v".into(),block_id:"b".into()},vec![2;1]).unwrap();let mut g=GpuCache::new(100,500,64,8,false).unwrap();g.acquire(a.clone(),&[1;90]).unwrap();assert!(g.acquire(BlockKey{model_id:"m".into(),version:"v".into(),block_id:"b".into()},&[2;20]).is_err());assert_eq!(g.tile_bytes,32);}
#[test]fn strict_scheduler_lanes(){let sch=Tier2Scheduler::new(4).unwrap();let mut rs=Vec::new();for _ in 0..8{rs.push(sch.submit(TaskClass::Learning,1,Box::new(||{thread::sleep(Duration::from_millis(3));Ok("l".into())})));rs.push(sch.submit(TaskClass::Doing,1,Box::new(||{thread::sleep(Duration::from_millis(3));Ok("d".into())})));}for r in rs{r.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();}let st=sch.stats();assert_eq!(st.learning_units,8);assert_eq!(st.doing_units,8);}
#[test]fn ten_billion_is_addressable(){let m=synthetic_10b_manifest("ten",8*1024*1024,7);assert_eq!(m.parameter_count,10_000_000_000);assert!(m.blocks.len()>1000);}
#[test]fn staging_streaming_and_version_transition_remain_bounded(){
    let p=temp("stream"); let s=PersistentStore::open(&p).unwrap();
    let(m,b)=baseline_from_blocks("m","native",vec![
        ("a".into(),"W".into(),vec![1;4]), ("b".into(),"W".into(),vec![2;4]), ("c".into(),"W".into(),vec![0;4])
    ]); s.create_baseline(m,&b).unwrap();
    let cfg=RuntimeConfig{ram_budget:8,staging_budget:4,gpu_budget:32,worker_capacity:2,initial_tile:8,min_tile:1,native_gpu:false};
    let rt=TriTierRuntime::open(&p,"m".into(),"baseline".into(),cfg).unwrap();
    let first=BlockKey{model_id:"m".into(),version:"baseline".into(),block_id:"a".into()};
    let protected=rt.acquire(&first).unwrap(); assert_eq!(protected.bytes.as_slice(),&[1;4]);
    let mut c=s.candidate("m","baseline",BTreeMap::from([("a".into(),vec![9;4])]),"example-42".into()).unwrap();
    s.validate_candidate(&mut c, |_| true).unwrap(); s.commit(&c,"v1").unwrap();
    // An already acquired baseline handle remains immutable while a new version publishes.
    assert_eq!(protected.bytes.as_slice(),&[1;4]); rt.switch_version("v1").unwrap();
    assert_eq!(rt.acquire(&BlockKey{model_id:"m".into(),version:"v1".into(),block_id:"a".into()}).unwrap().bytes.as_slice(),&[9;4]);
    drop(protected);
    for id in ["b","c","b","c"] {
        let _h=rt.acquire(&BlockKey{model_id:"m".into(),version:"baseline".into(),block_id:id.into()}).unwrap();
    }
    let t=rt.telemetry(); assert!(t.ram.peak_bytes<=8); assert!(t.gpu.peak_bytes<GPU_HARD_LIMIT_BYTES);
    let journal=fs::read_to_string(p.join("models/m/journal.log")).unwrap(); assert!(journal.contains("candidate|")&&journal.contains("validated|")&&journal.contains("committed|"));
}
#[test]fn interrupted_transaction_is_ignored_and_baseline_recovers(){
    let p=temp("recovery");let s=PersistentStore::open(&p).unwrap();let(m,b)=baseline_from_blocks("m","native",vec![("a".into(),"W".into(),vec![1])]);s.create_baseline(m,&b).unwrap();
    let orphan=p.join("models/m/transactions/crashed-candidate");fs::create_dir_all(&orphan).unwrap();fs::write(orphan.join("partial"),[1u8]).unwrap();
    let warnings=s.recover("m").unwrap();assert_eq!(s.head("m").unwrap(),"baseline");assert!(warnings.iter().any(|x|x.contains("crashed-candidate")));
}
#[cfg(feature = "sycl")]
#[test]fn native_gpu_allocation_and_telemetry(){
    let mut g = GpuCache::new(400_000_000, GPU_HARD_LIMIT_BYTES, 8 * 1024 * 1024, 64 * 1024, true).unwrap();
    assert!(g.native_enabled());
    let key = BlockKey { model_id: "test".into(), version: "v1".into(), block_id: "b0".into() };
    let data = vec![42u8; 1024 * 1024];
    g.acquire(key.clone(), &data).unwrap();
    assert_eq!(g.stats.current_bytes, 1024 * 1024);
    assert!(g.stats.peak_bytes >= 1024 * 1024);
    assert!(g.stats.peak_bytes < GPU_HARD_LIMIT_BYTES);
    assert!(g.stats.external_current_bytes.is_some());
    assert!(g.stats.external_peak_bytes.is_some());
    g.release(&key);
}
}

pub mod ffi {
    use super::*;
    use std::ffi::{c_char, CStr};

    #[repr(C)]
    pub struct RawTelemetry {
        pub ram_current: u64,
        pub ram_peak: u64,
        pub ram_hits: u64,
        pub ram_misses: u64,
        pub ram_evictions: u64,
        pub gpu_current: u64,
        pub gpu_peak: u64,
        pub gpu_ext_current: u64,
        pub gpu_ext_peak: u64,
        pub gpu_hits: u64,
        pub gpu_misses: u64,
        pub gpu_evictions: u64,
        pub gpu_native: i32,
        pub staging_current: u64,
        pub staging_peak: u64,
        pub learning_units: u64,
        pub doing_units: u64,
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_open(
        root_path: *const c_char,
        model_id: *const c_char,
        version: *const c_char,
        ram_budget: u64,
        staging_budget: u64,
        gpu_budget: u64,
        worker_capacity: usize,
        native_gpu: bool,
    ) -> *mut TriTierRuntime {
        if root_path.is_null() || model_id.is_null() || version.is_null() {
            return std::ptr::null_mut();
        }
        let root = match CStr::from_ptr(root_path).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let model = match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return std::ptr::null_mut(),
        };
        let ver = match CStr::from_ptr(version).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return std::ptr::null_mut(),
        };
        let cfg = RuntimeConfig {
            ram_budget,
            staging_budget,
            gpu_budget,
            worker_capacity,
            initial_tile: 8 * 1024 * 1024,
            min_tile: 64 * 1024,
            native_gpu,
        };
        match TriTierRuntime::open(root, model, ver, cfg) {
            Ok(rt) => Box::into_raw(Box::new(rt)),
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_close(rt: *mut TriTierRuntime) {
        if !rt.is_null() {
            drop(Box::from_raw(rt));
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_acquire(
        rt: *mut TriTierRuntime,
        version: *const c_char,
        block_id: *const c_char,
        out_ptr: *mut *const u8,
        out_len: *mut usize,
    ) -> *mut BlockHandle {
        if rt.is_null() || version.is_null() || block_id.is_null() || out_ptr.is_null() || out_len.is_null() {
            return std::ptr::null_mut();
        }
        let rt_ref = &*rt;
        let ver = match CStr::from_ptr(version).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let bid = match CStr::from_ptr(block_id).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        let key = BlockKey {
            model_id: rt_ref.model_id().to_string(),
            version: ver.to_string(),
            block_id: bid.to_string(),
        };
        match rt_ref.acquire(&key) {
            Ok(handle) => {
                *out_ptr = handle.bytes.as_ptr();
                *out_len = handle.bytes.len();
                Box::into_raw(Box::new(handle))
            }
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_handle_release(handle: *mut BlockHandle) {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_prefetch(
        rt: *mut TriTierRuntime,
        version: *const c_char,
        block_id: *const c_char,
    ) -> i32 {
        if rt.is_null() || version.is_null() || block_id.is_null() {
            return -1;
        }
        let rt_ref = &*rt;
        let ver = match CStr::from_ptr(version).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let bid = match CStr::from_ptr(block_id).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        };
        let key = BlockKey {
            model_id: rt_ref.model_id().to_string(),
            version: ver.to_string(),
            block_id: bid.to_string(),
        };
        let rx = rt_ref.prefetch(key);
        match rx.recv() {
            Ok(Ok(_)) => 0,
            _ => -1,
        }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_switch_version(
        rt: *mut TriTierRuntime,
        version: *const c_char,
    ) -> i32 {
        if rt.is_null() || version.is_null() { return -1; }
        let ver = match CStr::from_ptr(version).to_str() { Ok(s) => s, Err(_) => return -1 };
        match (&*rt).switch_version(ver) { Ok(_) => 0, Err(_) => -1 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_rollback(
        rt: *mut TriTierRuntime,
        version: *const c_char,
    ) -> i32 {
        if rt.is_null() || version.is_null() { return -1; }
        let ver = match CStr::from_ptr(version).to_str() { Ok(s) => s, Err(_) => return -1 };
        match (&*rt).rollback(ver) { Ok(_) => 0, Err(_) => -1 }
    }

    #[no_mangle]
    pub unsafe extern "C" fn tt_runtime_telemetry(
        rt: *mut TriTierRuntime,
        out: *mut RawTelemetry,
    ) -> i32 {
        if rt.is_null() || out.is_null() { return -1; }
        let t = (&*rt).telemetry();
        *out = RawTelemetry {
            ram_current: t.ram.current_bytes,
            ram_peak: t.ram.peak_bytes,
            ram_hits: t.ram.hits,
            ram_misses: t.ram.misses,
            ram_evictions: t.ram.evictions,
            gpu_current: t.gpu.current_bytes,
            gpu_peak: t.gpu.peak_bytes,
            gpu_ext_current: t.gpu.external_current_bytes.unwrap_or(0),
            gpu_ext_peak: t.gpu.external_peak_bytes.unwrap_or(0),
            gpu_hits: t.gpu.hits,
            gpu_misses: t.gpu.misses,
            gpu_evictions: t.gpu.evictions,
            gpu_native: if t.gpu_native { 1 } else { 0 },
            staging_current: t.staging_current,
            staging_peak: t.staging_peak,
            learning_units: t.scheduler.learning_units,
            doing_units: t.scheduler.doing_units,
        };
        0
    }
}
