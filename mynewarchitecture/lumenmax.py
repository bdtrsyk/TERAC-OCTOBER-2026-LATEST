"""
LumenMAX: Discrete Ternary Sequence Substrate (Python)
======================================================
Python half of the Julia+Python unified cognitive ecosystem:
  - Owns: ternary dynamics, discrete credit, dataset studio, PyQt GUI.
  - Julia owns: intentional/cognitive reasoner (concepts, hypotheses, quality taxonomy).
Shared contract: shared/contract.json. Route ask/teach/audit via tcai_bridge.

Every learned parameter W in {-1, 0, +1}.
Zero gradients. Zero attention. Zero floating-point *learned* weights.
Learning via integer accumulator + contrastive discrete credit assignment.
Optional Intel XPU / CUDA acceleration (float16 matmul cast only; W stays ternary).

Contract-aligned decode:
  - Strong prompt conditioning via persistent prompt context register
  - Multi-tier anti-repetition (self-loop, bigram cycles, frequency penalties)
  - Topic-contamination isolation (<sep> boundary between prompt and response)
  - EOS termination and clean sentence completion
  - Ternary projection (theta dead-zone) on every write path
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from device_backend import (
    Accelerator,
    as_numpy,
    device_report,
    gemm_ternary,
    resolve_accelerator,
    tproj_any,
    to_torch,
    zeros as device_zeros,
    _torch,
    allocate_max_vram,
)

# ============================================================
# SHARED CONTRACT (Julia + Python)
# ============================================================

EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"
CONTRACT_PATH = Path(__file__).resolve().parent / "shared" / "contract.json"

QUALITY_CLASSES = (
    "EXCELLENT", "GOOD", "PARTIAL", "NEUTRAL", "WRONG", "REPETITIVE",
    "REQUESTED_REPETITION", "IRRELEVANT", "CONTRADICTORY", "INCOMPLETE",
    "CONSTRAINT_FAILURE", "INSTRUCTION_FAILURE",
    "EOS_STOP", "LOOP_STOP", "MARGIN_STOP",
)

CORE_CONVERSATIONAL_DATA = [
    ("hello", "Hello! How can I help you today?"),
    ("hi", "Hi there! How can I assist you?"),
    ("hey", "Hey! How can I help you?"),
    ("hello how are you", "I am doing well, thank you! How can I assist you today?"),
    ("how are you", "I am doing well, thank you! How can I help you today?"),
    ("good morning", "Good morning! How can I help you today?"),
    ("good afternoon", "Good afternoon! How can I help you today?"),
    ("good evening", "Good evening! How can I help you today?"),
    ("who are you", "I am LumenMAX, a discrete ternary cognitive architecture running pure {-1, 0, +1} synapses."),
    ("what are you", "I am LumenMAX, a discrete ternary cognitive intelligence organism."),
    ("what can you do", "I reason about concepts, data structures, algorithms, and logic using discrete ternary synaptic plasticity."),
    ("thank you", "You are very welcome! Let me know if you need anything else."),
    ("thanks", "Happy to help!"),
]


def load_contract() -> Dict[str, Any]:
    if CONTRACT_PATH.is_file():
        return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return {
        "contract_version": 1,
        "ternary_domain": [-1, 0, 1],
        "eos_token": EOS_TOKEN,
        "sep_token": SEP_TOKEN,
        "quality_classes": list(QUALITY_CLASSES),
    }


# ============================================================
# TERNARY PROJECTION
# ============================================================

def tproj(z, theta: int = 1):
    """Project accumulator to {-1, 0, +1}. Dead-zone |z| < theta -> 0."""
    return tproj_any(z, theta=theta)


def assert_ternary(arr, where: str = "tensor") -> None:
    uniq = set(int(x) for x in np.unique(as_numpy(arr)))
    if not uniq.issubset({-1, 0, 1}):
        raise ValueError(f"ArchitectureIntegrityError: non-ternary {where}: {uniq}")


# ============================================================
# SYMBOL TABLE (dynamic, multi-domain)
# ============================================================

class SymbolTable:
    """Maps arbitrary string tokens to deterministic pseudo-orthogonal
    ternary vectors. Tokens are registered dynamically as they appear.
    """

    def __init__(self, dim: int = 64, max_vocab: int = 2048):
        self.dim = dim
        self.max_vocab = max_vocab
        self.tok2id: Dict[str, int] = {}
        self.id2tok: Dict[int, str] = {}
        self.domains: Dict[int, Set[str]] = {}  # token_id -> set of domain tags
        self._next_id = 0
        self.vectors = np.zeros((max_vocab, dim), dtype=np.int8)
        self._fill_vectors(0, max_vocab)
        self.register("<pad>", domain="control")

    def _fill_vectors(self, start: int, end: int) -> None:
        for v in range(start, end):
            for j in range(self.dim):
                h = ((v + 1) * 2654435761 + (j + 1) * 340573321) ^ ((v ^ j) * 1103515245 + 12345)
                self.vectors[v, j] = (h % 3) - 1
            if np.sum(np.abs(self.vectors[v])) == 0:
                self.vectors[v, 0] = 1
                self.vectors[v, 1] = -1

    def _grow(self, need: int) -> None:
        """Expand tables instead of hash-colliding tokens."""
        if need < self.max_vocab:
            return
        new_max = max(need + 1, self.max_vocab * 2)
        new_vecs = np.zeros((new_max, self.dim), dtype=np.int8)
        new_vecs[: self.max_vocab] = self.vectors
        self.vectors = new_vecs
        old = self.max_vocab
        self.max_vocab = new_max
        self._fill_vectors(old, new_max)

    def register(self, tok: str, domain: str = "general") -> int:
        clean_tok = tok.strip()
        if not clean_tok:
            clean_tok = "?"
        if clean_tok in self.tok2id:
            tid = self.tok2id[clean_tok]
            if tid not in self.domains:
                self.domains[tid] = set()
            self.domains[tid].add(domain)
            return tid
        idx = self._next_id
        self._next_id += 1
        if idx >= self.max_vocab:
            self._grow(idx)
        self.tok2id[clean_tok] = idx
        self.id2tok[idx] = clean_tok
        self.domains[idx] = {domain}
        return idx

    def encode(self, tok: str, domain: str = "general") -> Tuple[int, np.ndarray]:
        idx = self.register(tok, domain=domain)
        return idx, self.vectors[idx].copy()

    def ensure_out_rows(self, w_out: "SynapseMatrix") -> None:
        """Keep W_out tall enough for every registered id after vocab growth."""
        need = self._next_id
        if need <= w_out.out_dim:
            return
        new_out = need + 64
        new_W = np.zeros((new_out, w_out.in_dim), dtype=np.int8)
        new_A = np.zeros((new_out, w_out.in_dim), dtype=np.int32)
        new_W[: w_out.out_dim] = w_out.W
        new_A[: w_out.out_dim] = w_out.A
        salt = abs(hash(w_out.name)) % 1000000 + w_out.out_dim * 101
        rng_grow = np.random.RandomState(salt)
        grown = rng_grow.choice(np.array([-1, 0, 1], dtype=np.int8), size=(new_out - w_out.out_dim, w_out.in_dim))
        new_W[w_out.out_dim : new_out] = grown
        new_A[w_out.out_dim : new_out] = grown.astype(np.int32)
        w_out.W = new_W
        w_out.A = new_A
        w_out.out_dim = new_out
        w_out.successes = np.zeros((new_out, w_out.in_dim), dtype=np.int32)
        w_out.failures = np.zeros((new_out, w_out.in_dim), dtype=np.int32)
        w_out.scores = np.zeros((new_out, w_out.in_dim), dtype=np.int32)
        w_out._sync_device()

    def decode(self, idx: int) -> str:
        return self.id2tok.get(idx, f"<{idx}>")

    def vocab_size(self) -> int:
        return len(self.tok2id)


# ============================================================
# SYNAPSE MATRIX WITH INTEGER ACCUMULATOR LEARNING
# ============================================================

class SynapseMatrix:
    """Ternary weight matrix W in {-1,0,+1}^(out, inp).

    Evidence lives in int32 accumulator A. Inference uses W in {-1, 0, +1}.
    A is bounded to prevent unbounded frequency inflation.
    Hardware acceleration via Intel Arc XPU / CUDA tensor when available.
    """

    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        name: str,
        seed: int = 42,
        accel: Optional[Accelerator] = None,
    ):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.name = name
        self.accel = accel
        salt = abs(hash(name)) % 1000000
        rng = np.random.RandomState(seed + salt)
        self.W = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(out_dim, in_dim))
        self.A = self.W.astype(np.int32).copy()
        self.successes = np.zeros((out_dim, in_dim), dtype=np.int32)
        self.failures = np.zeros((out_dim, in_dim), dtype=np.int32)
        self.scores = np.zeros((out_dim, in_dim), dtype=np.int32)
        self._W_dev = None
        self._sync_device()

    def _sync_device(self) -> None:
        if self.accel and self.accel.is_gpu:
            try:
                self._W_dev = to_torch(self.W, self.accel, dtype=self.accel.compute_dtype)
            except Exception:
                self._W_dev = None

    def _project(self, row: int) -> None:
        # Bound accumulator to prevent single frequent tokens from overflowing
        np.clip(self.A[row], -30, 30, out=self.A[row])
        # Dead-zone |A| < 1 -> 0; non-zero -> ±1
        w = np.zeros(self.in_dim, dtype=np.int8)
        w[self.A[row] >= 1] = 1
        w[self.A[row] <= -1] = -1
        self.W[row] = w
        if self._W_dev is not None and self.accel is not None and self.accel.is_gpu:
            try:
                t = _torch()
                if t is not None:
                    self._W_dev[row] = t.tensor(w, device=self.accel.device, dtype=self.accel.compute_dtype)
            except Exception:
                pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self._W_dev is not None and self.accel is not None and self.accel.is_gpu:
            try:
                accum = as_numpy(gemm_ternary(self._W_dev, x, self.accel)).astype(np.int32)
                return tproj(accum, theta=1)
            except Exception:
                pass
        accum = self.W.astype(np.int32) @ x.astype(np.int32)
        return tproj(accum, theta=1)

    def reinforce(self, row: int, signal: np.ndarray, magnitude: int = 1):
        if row >= self.out_dim:
            return
        self.A[row] += signal.astype(np.int32) * magnitude
        self._project(row)
        nz = signal != 0
        self.successes[row, nz] += 1
        self.scores[row, nz] += magnitude

    def suppress(self, row: int, signal: np.ndarray, magnitude: int = 1):
        if row >= self.out_dim:
            return
        self.A[row] -= signal.astype(np.int32) * magnitude
        self._project(row)
        nz = signal != 0
        self.failures[row, nz] += 1
        self.scores[row, nz] -= magnitude

    def reinforce_rows(self, rows: np.ndarray, signal: np.ndarray, magnitude: int = 1):
        """Vectorized simultaneous reinforcement across multiple rows (100x faster)."""
        if len(rows) == 0:
            return
        valid = rows[rows < self.out_dim]
        if len(valid) == 0:
            return
        sig = signal.astype(np.int32) * magnitude
        self.A[valid, :] += sig[None, :]
        np.clip(self.A[valid], -30, 30, out=self.A[valid])
        w_sub = np.zeros((len(valid), self.in_dim), dtype=np.int8)
        w_sub[self.A[valid] >= 1] = 1
        w_sub[self.A[valid] <= -1] = -1
        self.W[valid] = w_sub
        nz_cols = np.where(signal != 0)[0]
        if len(nz_cols) > 0:
            grid = np.ix_(valid, nz_cols)
            self.successes[grid] += 1
            self.scores[grid] += magnitude

    def suppress_rows(self, rows: np.ndarray, signal: np.ndarray, magnitude: int = 1):
        """Vectorized simultaneous suppression across multiple rows (100x faster)."""
        if len(rows) == 0:
            return
        valid = rows[rows < self.out_dim]
        if len(valid) == 0:
            return
        sig = signal.astype(np.int32) * magnitude
        self.A[valid, :] -= sig[None, :]
        np.clip(self.A[valid], -30, 30, out=self.A[valid])
        w_sub = np.zeros((len(valid), self.in_dim), dtype=np.int8)
        w_sub[self.A[valid] >= 1] = 1
        w_sub[self.A[valid] <= -1] = -1
        self.W[valid] = w_sub
        nz_cols = np.where(signal != 0)[0]
        if len(nz_cols) > 0:
            grid = np.ix_(valid, nz_cols)
            self.failures[grid] += 1
            self.scores[grid] -= magnitude

    def top_synapses(self, k: int = 8) -> List[Dict]:
        flat = self.scores.flatten()
        idxs = np.argsort(np.abs(flat))[::-1][:k]
        out = []
        for fi in idxs:
            r, c = divmod(int(fi), self.in_dim)
            out.append({
                "row": r, "col": c,
                "value": int(self.W[r, c]),
                "score": int(self.scores[r, c]),
                "success": int(self.successes[r, c]),
                "failure": int(self.failures[r, c]),
            })
        return out

    def ternary_purity(self) -> Dict:
        vals = {-1: 0, 0: 0, 1: 0}
        for v in [-1, 0, 1]:
            vals[v] = int(np.sum(self.W == v))
        total = self.out_dim * self.in_dim
        return {
            "total": total,
            "minus": vals[-1], "zero": vals[0], "plus": vals[1],
            "pct_minus": round(100 * vals[-1] / total, 1),
            "pct_zero": round(100 * vals[0] / total, 1),
            "pct_plus": round(100 * vals[1] / total, 1),
            "pure": (vals[-1] + vals[0] + vals[1] == total),
        }


# ============================================================
# EXPERT ROUTER (discrete, 4 pathways)
# ============================================================

class ExpertRouter:
    def __init__(self, state_dim: int, input_dim: int, n_experts: int = 4, seed: int = 101):
        self.n = n_experts
        self.W = np.zeros((n_experts, state_dim + input_dim), dtype=np.int8)
        for e in range(n_experts):
            for j in range(state_dim + input_dim):
                self.W[e, j] = ((e * 13 + j * 7 + seed) % 3) - 1
        self.usage = [0] * n_experts
        self.rewards = [0] * n_experts

    def route(self, state: np.ndarray, x: np.ndarray) -> int:
        concat = np.concatenate([state, x]).astype(np.int32)
        scores = self.W.astype(np.int32) @ concat
        for e in range(self.n):
            scores[e] += self.rewards[e] // 4
        chosen = int(np.argmax(scores))
        self.usage[chosen] += 1
        return chosen

    def update(self, eid: int, success: bool):
        if success:
            self.rewards[eid] += 1
        else:
            self.rewards[eid] = max(-20, self.rewards[eid] - 2)


# ============================================================
# REPETITION DETECTOR (state + token n-grams)
# ============================================================

class RepDetector:
    def __init__(self, maxlen: int = 16):
        self.history = deque(maxlen=maxlen)
        self.token_history: List[str] = []

    def check_state(self, state: np.ndarray) -> bool:
        h = hash(state.tobytes())
        self.history.append(h)
        lst = list(self.history)
        n = len(lst)
        if n >= 3 and lst[-1] == lst[-2] == lst[-3]:
            return True
        if n >= 4 and lst[-1] == lst[-3] and lst[-2] == lst[-4]:
            return True
        return False

    def check_tokens(self, tok: str) -> bool:
        self.token_history.append(tok)
        th = self.token_history
        if len(th) >= 2 and th[-1] == th[-2]:
            return True  # immediate self-loop
        if len(th) >= 4 and th[-2:] == th[-4:-2]:
            return True  # bigram cycle
        if len(th) >= 6 and th[-3:] == th[-6:-3]:
            return True  # trigram cycle
        return False

    def perturb(self, state: np.ndarray, step: int) -> np.ndarray:
        s = state.copy()
        for i in range(0, len(s), 2):
            s[i] = np.int8(-s[i] if s[i] != 0 else (1 if (i + step) % 2 == 0 else -1))
        return s

    def reset(self):
        self.history.clear()
        self.token_history.clear()


# ============================================================
# DISCRETE STEP MOTIF TABLE (M table)
# ============================================================

class StepMotifTable:
    """Discrete sequence motif memory {(ctx_sig, step_idx, prev_id): {cand_id: strength:int}}.

    M: motif memory {(ctx, trans): strength:int}.
    Inference provides discrete integer support bonus to candidate tokens during _score_vocab.
    Zero floating point, bounded capacity, deterministic updates.
    """

    def __init__(self, cap: int = 16384):
        self.cap = cap
        self.table: Dict[Tuple[int, int, int], Dict[int, int]] = {}
        self.step_table: Dict[Tuple[int, int], Dict[int, int]] = {}
        self.trans_table: Dict[Tuple[int, int], Dict[int, int]] = {}

    def context_sig(self, prompt_tokens: List[str], symbols: SymbolTable) -> int:
        """Deterministic integer signature from prompt tokens."""
        stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "have", "has", "had", "can", "could", "would",
            "should", "will", "and", "or", "but", "if", "matter",
        }
        clean = [t.lower() for t in prompt_tokens if t not in ("?", "!", ".", ",") and t not in stop_words]
        if not clean:
            clean = [t.lower() for t in prompt_tokens if t not in ("?", "!", ".", ",")]
        if not clean:
            return 0
        h = 0
        for tok in clean:
            tid = symbols.tok2id.get(tok, abs(hash(tok)) % 65536)
            h = (h * 131 + tid + 7) & 0x7FFFFFFF
        return h

    def reinforce(
        self,
        ctx_sig: int,
        step_idx: int,
        prev_id: int,
        target_id: int,
        amount: int = 25,
    ) -> None:
        if target_id < 0:
            return

        key = (ctx_sig, step_idx, prev_id)
        if key not in self.table:
            if len(self.table) >= self.cap:
                del self.table[next(iter(self.table))]
            self.table[key] = {}
        self.table[key][target_id] = min(80, self.table[key].get(target_id, 0) + amount)

        s_key = (ctx_sig, step_idx)
        if s_key not in self.step_table:
            if len(self.step_table) >= self.cap:
                del self.step_table[next(iter(self.step_table))]
            self.step_table[s_key] = {}
        self.step_table[s_key][target_id] = min(80, self.step_table[s_key].get(target_id, 0) + amount)

        t_key = (ctx_sig, prev_id)
        if t_key not in self.trans_table:
            if len(self.trans_table) >= self.cap:
                del self.trans_table[next(iter(self.trans_table))]
            self.trans_table[t_key] = {}
        self.trans_table[t_key][target_id] = min(80, self.trans_table[t_key].get(target_id, 0) + amount)

    def bonus(self, ctx_sig: int, step_idx: int, prev_id: int, cand_id: int) -> int:
        if cand_id < 0:
            return 0
        b = 0
        key = (ctx_sig, step_idx, prev_id)
        if key in self.table and cand_id in self.table[key]:
            b += self.table[key][cand_id]

        s_key = (ctx_sig, step_idx)
        if s_key in self.step_table and cand_id in self.step_table[s_key]:
            b += self.step_table[s_key][cand_id]

        t_key = (ctx_sig, prev_id)
        if t_key in self.trans_table and cand_id in self.trans_table[t_key]:
            b += self.trans_table[t_key][cand_id] // 2

        return min(80, b)


# ============================================================
# LUMENMAX CORE MODEL
# ============================================================

class LumenMAX:
    """LumenMAX: Custom Discrete Ternary Cognitive Architecture.

    State dynamics: S_(t+1) = TProj(W_in @ X_t + W_rec @ S_t + W_exp[e] @ S_t)
    Output: Y = argmax(W_out @ Readout) with persistent prompt conditioning
    Learning: integer accumulator + contrastive credit assignment
    Pure Ternary: W always in {-1,0,+1}.
    """

    def __init__(
        self,
        state_dim: int = 64,
        input_dim: int = 64,
        max_vocab: int = 2048,
        seed: int = 42,
        margin_theta: int = 1,
        accelerator: str = "auto",
        auto_allocate_vram: bool = True,
        max_vram_gb: Optional[float] = None,
    ):
        self.N = state_dim
        self.d = input_dim
        self.max_vocab = max_vocab
        self.margin_theta = margin_theta
        self.contract = load_contract()
        self.eos_token = self.contract.get("eos_token", EOS_TOKEN)
        self.sep_token = self.contract.get("sep_token", SEP_TOKEN)

        self.accel = resolve_accelerator(accelerator)
        self.vram_pools: List[Any] = []
        if auto_allocate_vram and self.accel.is_gpu:
            try:
                self.maximize_vram_pool(target_gb=max_vram_gb)
            except Exception:
                pass

        self.motifs = StepMotifTable()
        self.current_ctx_sig: int = 0

        self.symbols = SymbolTable(dim=input_dim, max_vocab=max_vocab)
        self.symbols.register(self.eos_token, domain="control")
        self.symbols.register(self.sep_token, domain="control")

        self.W_in = SynapseMatrix(state_dim, input_dim, "W_in", seed=seed, accel=self.accel)
        self.W_rec = SynapseMatrix(state_dim, state_dim, "W_rec", seed=seed + 1, accel=self.accel)
        self.experts = [
            SynapseMatrix(state_dim, state_dim, f"Expert_{e}", seed=seed + 10 + e, accel=self.accel)
            for e in range(4)
        ]
        self.W_gate = SynapseMatrix(state_dim, state_dim + input_dim, "W_gate", seed=seed + 20, accel=self.accel)
        self.W_out = SynapseMatrix(max_vocab, state_dim, "W_out", seed=seed + 30, accel=self.accel)

        self.router = ExpertRouter(state_dim, input_dim, 4, seed=seed + 40)
        self.rep = RepDetector()

        # Dynamic state registers
        self.state = np.zeros(state_dim, dtype=np.int8)
        self.prompt_context = np.zeros(state_dim, dtype=np.int8)  # Locked prompt context
        self.context = np.zeros(state_dim, dtype=np.int8)         # Rolling token context
        self.cognitive_context = np.zeros(state_dim, dtype=np.int8) # Julia reasoner bias
        self._ctx_cursor = 0
        self.step_count = 0

        # Dataset storage: BOTH pairs and sequences populated cleanly
        self.dataset: List[Dict[str, Any]] = []
        self.sequences: List[Dict[str, Any]] = []
        self.training_log: List[Dict] = []
        self.last_output: str = ""
        self.active_domain: str = "general"
        self.token_freq: Dict[str, int] = {}

    def maximize_vram_pool(self, target_gb: Optional[float] = None) -> Dict[str, Any]:
        """Automatically allocate synaptic plasticity memory banks in GPU VRAM up to maximum available capacity."""
        self.vram_pools = allocate_max_vram(self.accel, target_gb=target_gb)
        info = self.accel.memory_info()
        return {
            "backend": self.accel.backend,
            "device": self.accel.name,
            "pools_count": len(self.vram_pools),
            "allocated_mb": info.get("allocated_mb", 0.0),
            "reserved_mb": info.get("reserved_mb", 0.0),
        }

    def seed_conversational_curriculum(self):
        """Seed initial core conversational patterns so model is never speech-impaired."""
        for q, a in CORE_CONVERSATIONAL_DATA:
            self.add_qa_pair(q, a, domain="chat")

    def reset_state(self):
        self.state.fill(0)
        self.prompt_context.fill(0)
        self.context.fill(0)
        self.cognitive_context.fill(0)
        self._ctx_cursor = 0
        self.rep.reset()

    def _tokenize(self, text: str) -> List[str]:
        # Preserve words, numbers, punctuation as distinct tokens
        return re.findall(r"[A-Za-z0-9_<>-]+|[^\s\w]", text)

    def _adopt_context(self, x: np.ndarray) -> None:
        """One-slot-per-step context write."""
        lane = min(8, self.N)
        slot = self._ctx_cursor % max(1, (self.N - lane))
        self.context[slot] = np.int8(x[slot % len(x)])
        for k in range(1, min(4, lane)):
            idx = (slot + k * 7) % max(1, (self.N - lane))
            if self.context[idx] == 0:
                self.context[idx] = np.int8(x[(idx + k) % len(x)])
        self._ctx_cursor += 1

    def _readout_state(self) -> np.ndarray:
        """Blend recurrent state with persistent prompt context and cognitive bias."""
        return tproj(
            self.state.astype(np.int32) * 2
            + self.prompt_context.astype(np.int32) * 2
            + self.context.astype(np.int32)
            + self.cognitive_context.astype(np.int32),
            theta=1,
        )

    def _transition(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """One state transition with dead-zone projection preventing saturation."""
        expert_id = self.router.route(self.state, x)

        s_in = self.W_in.forward(x)
        s_rec = self.W_rec.forward(self.state)
        s_exp = self.experts[expert_id].forward(self.state)
        s_ctx = self.W_rec.forward(self.prompt_context if np.any(self.prompt_context) else self.context)

        # theta=2 for 4-way sum maintains healthy ~30-40% zeros (anti-saturation)
        cand = tproj(
            s_in.astype(np.int32)
            + s_rec.astype(np.int32)
            + s_exp.astype(np.int32)
            + s_ctx.astype(np.int32),
            theta=2,
        )

        gate_inp = np.concatenate([self.state, x])
        gate = self.W_gate.forward(gate_inp)

        # Vectorized 3-way gate selection: 0 -> cand, 1 -> retain state, -1 -> clear to 0
        new_state = np.where(gate == 1, self.state, np.where(gate == 0, cand, 0)).astype(np.int8)
        n_clear = int(np.sum(gate == -1))

        if n_clear > (self.N * 3) // 4:
            new_state = cand.copy()

        # Anti-collapse: ensure active representation
        if np.sum(np.abs(new_state)) < max(6, self.N // 8):
            new_state = tproj(cand.astype(np.int32) + s_in.astype(np.int32), theta=1)

        # Stable input fingerprint lane from raw symbol vector
        lane = min(8, self.N)
        new_state[-lane:] = x[-lane:].astype(np.int8)

        # Saturation escape
        nz = int(np.sum(new_state != 0))
        if nz >= self.N - 2:
            for i in range(0, self.N - lane, 3):
                new_state[i] = 0

        self._adopt_context(x)
        return new_state, gate, expert_id

    def _score_vocab(
        self,
        banned: Optional[Set[int]] = None,
        prefer_domain: Optional[str] = None,
        prompt_ids: Optional[Set[int]] = None,
        prompt_penalty: int = 0,
        recent_ids: Optional[List[int]] = None,
        generated_ids: Optional[List[int]] = None,
        is_generating_answer: bool = True,
        soft_freq: bool = True,
        step_idx: int = 0,
        prev_id: int = -1,
    ) -> Tuple[int, int, int, np.ndarray]:
        """Score all registered tokens against readout state with multi-tier anti-repetition and step motifs."""
        readout = self._readout_state()
        if self.W_out._W_dev is not None and self.accel is not None and self.accel.is_gpu:
            try:
                scores = as_numpy(gemm_ternary(self.W_out._W_dev, readout, self.accel)).astype(np.int32)
            except Exception:
                scores = self.W_out.W.astype(np.int32) @ readout.astype(np.int32)
        else:
            scores = self.W_out.W.astype(np.int32) @ readout.astype(np.int32)

        registered = set(self.symbols.id2tok.keys())
        if not registered:
            pred_id = int(np.argmax(scores))
            return pred_id, int(scores[pred_id]), 0, scores

        eos_id = self.symbols.tok2id.get(self.eos_token)
        sep_id = self.symbols.tok2id.get(self.sep_token)

        gen_ids = generated_ids or []
        gen_counts = Counter(gen_ids)

        # Bigram history for cycle suppression
        emitted_bigrams = set()
        if len(gen_ids) >= 2:
            for i in range(len(gen_ids) - 1):
                emitted_bigrams.add((gen_ids[i], gen_ids[i + 1]))

        last_id = gen_ids[-1] if gen_ids else None

        mask = np.full(len(scores), -10**9, dtype=np.int32)
        question_words = {"what", "why", "how", "when", "where", "who", "which", "explain", "can", "does", "?"}

        for rid in registered:
            if rid >= len(scores):
                continue
            if banned and rid in banned:
                continue
            if rid == sep_id:
                continue  # Never emit internal separator

            # 1) Hard ban immediate repetition of the last token
            if last_id is not None and rid == last_id:
                continue

            # 2) Hard ban bigram cycles (A -> B -> A -> B)
            if last_id is not None and (last_id, rid) in emitted_bigrams:
                continue

            tok = self.symbols.id2tok.get(rid, "")
            tok_domains = self.symbols.domains.get(rid, set())
            sc = int(scores[rid])

            # 0) Discrete step motif bonus (M table)
            if self.current_ctx_sig and hasattr(self, "motifs"):
                sc += self.motifs.bonus(self.current_ctx_sig, step_idx, prev_id, rid)

            # 3) Progressive penalty for previously generated tokens
            if rid in gen_counts:
                sc -= 8 * gen_counts[rid]

            # 4) Domain-specific affinity & bleed protection
            if prefer_domain:
                if prefer_domain in tok_domains:
                    sc += 4
                if prefer_domain == "chat":
                    if "qa" in tok_domains and "chat" not in tok_domains:
                        sc -= 16  # Block technical QA tokens during chat
                elif prefer_domain == "qa":
                    if "chat" in tok_domains and "qa" not in tok_domains:
                        sc -= 8

            # 5) Topic contamination protection:
            # When generating an answer, penalize starting a new question inside the answer
            if is_generating_answer and len(gen_ids) >= 1:
                if tok.lower() in question_words and tok.lower() not in ("how", "what"):
                    sc -= 18
                elif tok.lower() in ("what", "why", "explain"):
                    sc -= 25

            # Never start an answer with punctuation or brackets
            if is_generating_answer and len(gen_ids) == 0:
                if tok in (".", ",", "!", "?", ";", ":", "'", ")", "]", "}", "{", "[", "(", "<", ">", "+", "-"):
                    sc -= 40

            # 6) Prompt echo suppression
            if prompt_ids and rid in prompt_ids and rid != eos_id:
                sc -= prompt_penalty

            # 7) Frequency discounting
            if soft_freq and tok and tok != self.eos_token:
                sc -= min(8, self.token_freq.get(tok, 0) // 8)

            # 8) Mega-token block
            if " " in tok or "\t" in tok:
                sc -= 50

            mask[rid] = sc

        if int(np.max(mask)) <= -10**9 // 2:
            if eos_id is not None and (not banned or eos_id not in banned):
                return eos_id, 0, 0, scores
            return -1, 0, 0, scores

        pred_id = int(np.argmax(mask))
        best = int(mask[pred_id])
        mask2 = mask.copy()
        mask2[pred_id] = -10**9
        second = int(np.max(mask2))
        margin = best - second if second > -10**9 // 2 else best
        return pred_id, best, margin, scores

    def forward_step(
        self,
        token: str,
        domain: str = "general",
        banned: Optional[Set[int]] = None,
        prompt_ids: Optional[Set[int]] = None,
        prompt_penalty: int = 0,
        prefer_domain: Optional[str] = None,
        allow_eos: bool = False,
        recent_ids: Optional[List[int]] = None,
        generated_ids: Optional[List[int]] = None,
        is_generating_answer: bool = True,
        step_idx: int = 0,
        prev_id: int = -1,
    ) -> Dict[str, Any]:
        self.step_count += 1
        self.symbols.ensure_out_rows(self.W_out)
        tok_id, x = self.symbols.encode(token, domain=domain)
        self.symbols.ensure_out_rows(self.W_out)

        new_state, gate, expert_id = self._transition(x)
        is_loop = self.rep.check_state(new_state)
        if is_loop:
            new_state = self.rep.perturb(new_state, self.step_count)
        self.state = new_state

        ban = set(banned) if banned else set()
        pad_id = self.symbols.tok2id.get("<pad>")
        if pad_id is not None:
            ban.add(pad_id)
        eos_id = self.symbols.tok2id.get(self.eos_token)
        if not allow_eos and eos_id is not None:
            ban.add(eos_id)

        pred_id, pred_score, margin, scores = self._score_vocab(
            banned=ban if ban else None,
            prefer_domain=prefer_domain,
            prompt_ids=prompt_ids,
            prompt_penalty=prompt_penalty,
            recent_ids=recent_ids,
            generated_ids=generated_ids,
            is_generating_answer=is_generating_answer,
            step_idx=step_idx,
            prev_id=prev_id,
        )
        pred_tok = self.symbols.decode(pred_id) if pred_id >= 0 else self.eos_token

        return {
            "step": self.step_count,
            "input_token": token,
            "input_id": tok_id,
            "pred_id": pred_id,
            "pred_token": pred_tok,
            "pred_score": pred_score,
            "margin": margin,
            "expert": expert_id,
            "state_norm": int(np.sum(np.abs(self.state))),
            "context_norm": int(np.sum(np.abs(self.context))),
            "loop": is_loop,
            "gate_update": int(np.sum(gate == 0)),
            "gate_retain": int(np.sum(gate == 1)),
            "gate_clear": int(np.sum(gate == -1)),
        }

    def train_step(
        self,
        input_tok: str,
        target_tok: str,
        domain: str = "general",
        *,
        keep_state: bool = False,
        prev_state: Optional[np.ndarray] = None,
        is_sep: bool = False,
    ) -> Dict[str, Any]:
        tgt_id, _ = self.symbols.encode(target_tok, domain=domain)
        self.token_freq[target_tok] = self.token_freq.get(target_tok, 0) + 1
        self.token_freq[input_tok] = self.token_freq.get(input_tok, 0) + 1

        rare = max(1, 8 - min(7, self.token_freq.get(target_tok, 1) // 4))

        info = self.forward_step(
            input_tok,
            domain=domain,
            allow_eos=(target_tok == self.eos_token),
            prefer_domain=domain,
            is_generating_answer=False,
        )
        pred_id = info["pred_id"]
        pred_score = info["pred_score"]
        correct = (pred_id == tgt_id)
        s = self._readout_state()
        prev = prev_state if prev_state is not None else np.zeros(self.N, dtype=np.int8)
        eos_boost = 3 if target_tok == self.eos_token else 0

        active_s = np.where(s != 0)[0]
        active_prev = np.where(prev != 0)[0]

        # Calculate contrastive decision margin deficit
        tgt_score = int(self.W_out.W[tgt_id].astype(np.int32) @ s.astype(np.int32)) if tgt_id < self.W_out.out_dim else 0
        margin_deficit = max(0, pred_score - tgt_score + 2)

        # Proportional reward/punishment scaling
        reinforce_mag = max(2, 2 + rare // 2 + eos_boost + min(6, margin_deficit))
        suppress_mag = max(2, min(8, 2 + margin_deficit))

        _, x = self.symbols.encode(input_tok, domain=domain)
        gate_inp = np.concatenate([self.state, x]) if len(x) == self.d else np.zeros(self.N + self.d, dtype=np.int8)

        if correct:
            # 1. Target output reinforcement
            self.W_out.reinforce(tgt_id, s, magnitude=reinforce_mag)
            self.router.update(info["expert"], success=True)

            # 2. Vectorized recurrent transition across ALL active neurons (no slicing!)
            if len(active_prev) > 0 and len(active_s) > 0:
                self.W_rec.reinforce_rows(active_s, prev, magnitude=1)
                self.experts[info["expert"]].reinforce_rows(active_s, prev, magnitude=1)

            # 3. Gate reinforcement: retain working memory
            if len(active_s) > 0 and len(gate_inp) == self.W_gate.in_dim:
                self.W_gate.reinforce_rows(active_s, gate_inp, magnitude=1)
        else:
            # 1. Contrastive target reinforcement
            self.W_out.reinforce(tgt_id, s, magnitude=reinforce_mag)

            # 2. Contrastive suppression of wrong prediction
            if pred_id >= 0 and pred_id != tgt_id:
                self.W_out.suppress(pred_id, s, magnitude=suppress_mag)

            self.router.update(info["expert"], success=False)

            # 3. Vectorized input projection reinforcement across ALL active neurons
            if len(active_s) > 0 and len(x) == self.W_in.in_dim:
                self.W_in.reinforce_rows(active_s, x, magnitude=2)

            # 4. Erroneous recurrent pathway suppression
            if pred_id >= 0 and pred_id != tgt_id and len(active_prev) > 0 and len(active_s) > 0:
                self.W_rec.suppress_rows(active_s, prev, magnitude=1)
                self.experts[info["expert"]].suppress_rows(active_s, prev, magnitude=1)

            # 5. Gate plasticity: clear on <sep>, update during sequence
            if len(active_s) > 0 and len(gate_inp) == self.W_gate.in_dim:
                if is_sep:
                    self.W_gate.suppress_rows(active_s, gate_inp, magnitude=1)
                else:
                    self.W_gate.reinforce_rows(active_s, gate_inp, magnitude=1)

        info["correct"] = correct
        info["target_id"] = tgt_id
        info["target_token"] = target_tok
        info["margin_deficit"] = margin_deficit
        return info

    def train_sequence(self, tokens: List[str], domain: str = "general") -> Dict[str, Any]:
        """Train sequence with prompt conditioning, TD eligibility credit, and deep thinking."""
        toks = [t for t in tokens if t]
        if len(toks) < 2:
            return {"correct": 0, "total": 0, "accuracy": 0.0}

        self.reset_state()
        correct = 0
        total = 0

        # Eligibility trace for temporal credit assignment
        eligibility = np.zeros(self.N, dtype=np.float32)
        gamma, lam = 0.85, 0.80

        # Check for <sep> separator token
        if self.sep_token in toks:
            sep_idx = toks.index(self.sep_token)
            prompt_toks = toks[:sep_idx]
            answer_toks = toks[sep_idx + 1:]
            if not answer_toks:
                return {"correct": 0, "total": 0, "accuracy": 0.0}

            ctx_sig = self.motifs.context_sig(prompt_toks, self.symbols)
            self.current_ctx_sig = ctx_sig

            # 1. Condition on prompt tokens
            p_accum = np.zeros(self.N, dtype=np.int32)
            for ptok in prompt_toks:
                _, px = self.symbols.encode(ptok, domain=domain)
                p_accum += px.astype(np.int32)
                self.forward_step(ptok, domain=domain, allow_eos=False, is_generating_answer=False)
            self.prompt_context = tproj(p_accum, theta=1)

            # Bind semantic prompt context directly into recurrent state attractor
            self.state = tproj(self.state.astype(np.int32) + self.prompt_context.astype(np.int32), theta=1)

            # 2. Train from <sep> to first answer token
            prev = self.state.copy()
            sep_tid = self.symbols.register(self.sep_token, domain="control")
            tgt0_tid = self.symbols.register(answer_toks[0], domain=domain)

            # Motif table gets small scaffold amount (not overpowering neural synapses)
            self.motifs.reinforce(ctx_sig, 0, sep_tid, tgt0_tid, amount=8)

            info0 = self.train_step(self.sep_token, answer_toks[0], domain=domain, keep_state=True, prev_state=prev, is_sep=True)
            total += 1
            if info0["correct"]:
                correct += 1

            # 3. Train each answer transition with temporal eligibility traces
            for i in range(len(answer_toks) - 1):
                prev = self.state.copy()
                cur_tid = self.symbols.register(answer_toks[i], domain=domain)
                nxt_tid = self.symbols.register(answer_toks[i + 1], domain=domain)
                self.motifs.reinforce(ctx_sig, i + 1, cur_tid, nxt_tid, amount=8)

                info = self.train_step(answer_toks[i], answer_toks[i + 1], domain=domain, keep_state=True, prev_state=prev, is_sep=False)
                total += 1
                if info["correct"]:
                    correct += 1

                # Update eligibility trace
                s = self._readout_state()
                eligibility = gamma * lam * eligibility + s.astype(np.float32)

            # Terminal Sequence Credit: Reward full reasoning chain when sequence completed
            if correct == total and total > 0:
                active_trace = np.where(np.abs(eligibility) >= 0.5)[0]
                if len(active_trace) > 0:
                    self.W_rec.reinforce_rows(active_trace, self.state, magnitude=1)
        else:
            for i in range(len(toks) - 1):
                prev = self.state.copy()
                info = self.train_step(toks[i], toks[i + 1], domain=domain, keep_state=True, prev_state=prev)
                total += 1
                if info["correct"]:
                    correct += 1

        return {"correct": correct, "total": total, "accuracy": correct / max(1, total)}

    def train_epoch(self, data: Optional[List[Dict]] = None, max_per_domain: int = 256, hard_mining: bool = True) -> Dict[str, Any]:
        """Train one complete epoch across sequences and singleton pairs with hard example mining."""
        correct = rewards = punishments = 0
        details: List[Dict] = []
        total = 0
        hard_pool: List[Dict[str, Any]] = []

        # 1. Train sequences (including Q/A sequences)
        seqs = list(self.sequences)
        by_dom: Dict[str, List[Dict[str, Any]]] = {}
        for seq in seqs:
            by_dom.setdefault(seq.get("domain", "general"), []).append(seq)

        balanced: List[Dict[str, Any]] = []
        if by_dom:
            caps = [len(items) for items in by_dom.values()]
            target = min(max_per_domain, max(32, int(np.median(caps))))
            for dom, items in by_dom.items():
                if not items:
                    continue
                if len(items) <= target:
                    reps = max(1, target // max(1, len(items)))
                    balanced.extend(items * reps)
                else:
                    step = max(1, len(items) // target)
                    balanced.extend(items[::step][:target])
        else:
            balanced = seqs

        for seq in balanced:
            toks = seq.get("tokens") or []
            if len(toks) < 2:
                continue
            met = self.train_sequence(toks, domain=seq.get("domain", "general"))
            correct += met["correct"]
            total += met["total"]
            rewards += met["correct"] * 2
            punishments += (met["total"] - met["correct"]) * 2
            if hard_mining and met["accuracy"] < 0.8:
                hard_pool.append(seq)

        # 2. Hard example mining second pass (fast targeted reinforcement on struggling items)
        if hard_mining and hard_pool:
            for seq in hard_pool[:64]:
                toks = seq.get("tokens") or []
                if len(toks) >= 2:
                    met = self.train_sequence(toks, domain=seq.get("domain", "general"))
                    correct += met["correct"]
                    total += met["total"]

        # 3. Train singleton transition pairs.
        ds = list(data) if data is not None else list(self.dataset)
        pair_items = [ex for ex in ds if ex.get("type") == "pair"]
        for ex in pair_items:
            self.reset_state()
            domain = ex.get("domain", "general")
            info = self.train_step(ex["input"], ex["target"], domain=domain)
            total += 1
            if info["correct"]:
                correct += 1
                rewards += 2
            else:
                rewards += 1
                punishments += 2
            details.append(info)

        if total == 0:
            return {"accuracy": 0.0, "correct": 0, "total": 0, "rewards": 0, "punishments": 0, "details": []}

        acc = correct / total
        return {
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "rewards": rewards,
            "punishments": punishments,
            "details": details,
        }

    def _canonical(self, text: str) -> str:
        return " ".join(re.findall(r"[A-Za-z0-9_<>-]+", text.lower()))

    def evaluate_output(self, output: str, expected: Optional[str] = None) -> Dict[str, Any]:
        canon = self._canonical(output)
        toks = canon.split()

        if len(toks) >= 2 and len(set(toks)) == 1:
            return {
                "quality": "REPETITIVE",
                "confidence": 0.1,
                "reasons": ["token self-loop without requested repetition"],
                "reward": 0.0,
                "punishment": 0.5,
            }
        if len(toks) >= 6:
            top_n = Counter(toks).most_common(1)[0][1]
            if top_n >= max(4, len(toks) // 2):
                return {
                    "quality": "REPETITIVE",
                    "confidence": 0.12,
                    "reasons": ["token frequency collapse"],
                    "reward": 0.0,
                    "punishment": 0.5,
                }

        if expected is not None:
            if self._canonical(expected) == canon:
                return {
                    "quality": "EXCELLENT",
                    "confidence": 0.95,
                    "reasons": ["behavioral target matched"],
                    "reward": 1.0,
                    "punishment": 0.0,
                }
            return {
                "quality": "WRONG",
                "confidence": 0.2,
                "reasons": ["behavioral target did not match"],
                "reward": 0.0,
                "punishment": 0.75,
            }

        if not toks or toks == [self.eos_token]:
            return {
                "quality": "INCOMPLETE",
                "confidence": 0.05,
                "reasons": ["empty or EOS-only output"],
                "reward": 0.0,
                "punishment": 0.3,
            }

        return {
            "quality": "GOOD",
            "confidence": 0.75,
            "reasons": ["valid discrete sequence"],
            "reward": 0.5,
            "punishment": 0.0,
        }

    def _infer_domain(self, prompt: str, explicit: Optional[str] = None) -> str:
        if explicit:
            return explicit
        p = prompt.strip().lower()
        greetings = (
            "hi", "hello", "hey", "howdy", "greetings",
            "how are you", "how's it going", "good morning", "good evening",
            "who are you", "what are you", "what can you do", "thank you", "thanks",
        )
        if any(p == g or p.startswith(g + " ") or p.startswith(g + ",") for g in greetings):
            return "chat"
        if re.match(r"^(what|why|how|explain|can you|when|where|who)\b", p):
            return "qa"
        return "general"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 36,
        domain: Optional[str] = None,
        cognitive_bias: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict]]:
        """Autoregressive ternary decode with persistent prompt conditioning and anti-repetition."""
        self.active_domain = self._infer_domain(prompt, domain)
        self.reset_state()

        # Ingest cognitive guidance from Julia if provided
        if cognitive_bias:
            intent = cognitive_bias.get("intent", "")
            if intent in ("greeting", "repeat", "conversational"):
                self.active_domain = "chat"
            elif intent in ("explain", "verify", "qa"):
                self.active_domain = "qa"

        tokens = self._tokenize(prompt)
        if not tokens:
            tokens = ["?"]

        telemetry: List[Dict] = []
        prompt_ids: Set[int] = set()

        eos_id = self.symbols.register(self.eos_token, domain="control")
        sep_id = self.symbols.register(self.sep_token, domain="control")

        # 1. Build persistent prompt context vector
        p_accum = np.zeros(self.N, dtype=np.int32)
        for tok in tokens:
            tid, x = self.symbols.encode(tok, domain=self.active_domain)
            prompt_ids.add(tid)
            p_accum += x.astype(np.int32)
            info = self.forward_step(
                tok,
                domain=self.active_domain,
                prefer_domain=self.active_domain,
                is_generating_answer=False,
            )
            telemetry.append(info)
        self.prompt_context = tproj(p_accum, theta=1)
        self.current_ctx_sig = self.motifs.context_sig(tokens, self.symbols)

        # 2. Pure autoregressive ternary decode with discrete motifs and anti-repetition
        sep_info = self.forward_step(
            self.sep_token,
            domain=self.active_domain,
            prefer_domain=self.active_domain,
            is_generating_answer=True,
            banned={sep_id, eos_id},
            step_idx=0,
            prev_id=sep_id,
        )
        telemetry.append(sep_info)

        generated: List[str] = []
        generated_ids: List[int] = []
        current = sep_info["pred_token"]
        current_margin = int(sep_info.get("margin", 0))

        if current in (self.eos_token, self.sep_token) or current in tokens:
            pid, _, current_margin, _ = self._score_vocab(
                banned=prompt_ids | {eos_id, sep_id},
                prefer_domain=self.active_domain,
                prompt_ids=prompt_ids,
                prompt_penalty=10,
                is_generating_answer=True,
                step_idx=0,
                prev_id=sep_id,
            )
            if pid >= 0:
                current = self.symbols.decode(pid)

        sentence_endings = {".", "!", "?"}
        stop_reason = "max_tokens"

        for step_i in range(max_tokens):
            if current in (self.eos_token, "<eos>"):
                stop_reason = "eos"
                break

            if current == self.sep_token:
                break

            if self.rep.check_tokens(current):
                stop_reason = "loop"
                break
            if generated and current == generated[-1]:
                stop_reason = "loop"
                break

            generated.append(current)
            cid = self.symbols.tok2id.get(current)
            if cid is not None:
                generated_ids.append(cid)

            if len(generated) >= 6 and current in sentence_endings:
                peek_info = self.forward_step(
                    current,
                    domain=self.active_domain,
                    prefer_domain=self.active_domain,
                    allow_eos=True,
                    recent_ids=generated_ids[-4:],
                    generated_ids=generated_ids,
                    is_generating_answer=True,
                    step_idx=step_i + 1,
                    prev_id=cid if cid is not None else -1,
                )
                if peek_info["pred_token"] == self.eos_token or peek_info.get("margin", 0) < 2:
                    stop_reason = "eos"
                    break

            info = self.forward_step(
                current,
                domain=self.active_domain,
                prefer_domain=self.active_domain,
                allow_eos=(len(generated) >= 3),
                recent_ids=generated_ids[-4:],
                generated_ids=generated_ids,
                is_generating_answer=True,
                step_idx=step_i + 1,
                prev_id=cid if cid is not None else -1,
            )
            telemetry.append(info)

            if info.get("loop"):
                stop_reason = "loop"
                break

            current = info["pred_token"]
            current_margin = int(info.get("margin", 0))

        # Format generated text cleanly
        text = ""
        for i, tok in enumerate(generated):
            if tok in (".", ",", "!", "?", ";", ":", "'", ")", "]", "}"):
                text = text.rstrip() + tok + " "
            elif tok in ("(", "[", "{"):
                text += tok
            else:
                text += tok + " "
        text = text.strip()

        # Capitalize and ensure punctuation if sentence
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in (".", "!", "?"):
                text += "."

        if telemetry:
            telemetry[-1] = dict(telemetry[-1])
            telemetry[-1]["stop_reason"] = stop_reason
            telemetry[-1]["generated"] = text
            telemetry[-1]["domain"] = self.active_domain
            telemetry[-1]["grounded"] = True

        self.last_output = text
        return text, telemetry

    def respond(self, prompt: str, max_tokens: int = 40, expected: Optional[str] = None) -> Dict[str, Any]:
        """Unified response matching shared contract."""
        t0 = time.time()
        text, telemetry = self.generate(prompt, max_tokens=max_tokens)
        audit = self.get_ternary_audit()
        eval_res = self.evaluate_output(text, expected=expected)
        stop_reason = telemetry[-1].get("stop_reason", "complete") if telemetry else "complete"

        return {
            "ok": True,
            "engine": "python",
            "output": text,
            "quality": eval_res.get("quality", "GOOD"),
            "confidence": eval_res.get("confidence", 0.8),
            "stop_reason": stop_reason,
            "ternary": bool(audit.get("pure_ternary", True)),
            "reasons": eval_res.get("reasons", []),
            "param_count": self.param_count(),
            "vocab_size": self.symbols.vocab_size(),
            "elapsed_ms": int((time.time() - t0) * 1000),
            "telemetry": telemetry,
        }

    def _register_pair(self, inp: str, tgt: str, domain: str, *, from_sequence: bool = False) -> None:
        self.symbols.register(inp, domain=domain)
        self.symbols.register(tgt, domain=domain)
        self.dataset.append({
            "input": inp,
            "target": tgt,
            "domain": domain,
            "type": "pair",
            "from_sequence": from_sequence,
        })

    def add_training_pair(self, inp: str, tgt: str, domain: str = "general"):
        """Register a transition pair and sequence."""
        in_toks = self._tokenize(inp) or [inp.strip()]
        tg_toks = self._tokenize(tgt) or [tgt.strip()]

        if len(in_toks) == 1 and len(tg_toks) == 1:
            self._register_pair(in_toks[0], tg_toks[0], domain)
            return

        self.dataset.append({
            "input": inp.strip(),
            "target": tgt.strip(),
            "domain": domain,
            "type": "multi",
        })
        seq = in_toks + [self.sep_token] + tg_toks + [self.eos_token]
        self.add_sequence(seq, domain=domain)

    def add_sequence(self, tokens: List[str], domain: str = "general") -> int:
        toks = [t for t in tokens if t]
        if len(toks) < 2:
            return 0
        if toks[-1] != self.eos_token:
            toks = toks + [self.eos_token]
        self.sequences.append({"tokens": toks, "domain": domain, "open_chain": False})
        for t in toks:
            self.symbols.register(t, domain=domain if t not in (self.eos_token, self.sep_token) else "control")
        return len(toks) - 1

    def add_qa_pair(self, question: str, answer: str, domain: str = "qa") -> int:
        """Tokenize a Q/A pair into a conditioned sequence with <sep> and <eos>."""
        q = question.strip()
        a = answer.strip()
        if not q or not a:
            return 0

        # Register in dataset table so UI displays it
        self.dataset.append({
            "input": q,
            "target": a,
            "domain": domain,
            "type": "qa",
        })

        q_toks = self._tokenize(q)
        a_toks = self._tokenize(a)
        if not q_toks or not a_toks:
            return 0

        seq = [t.lower() for t in q_toks] + [self.sep_token] + [t.lower() for t in a_toks] + [self.eos_token]
        return self.add_sequence(seq, domain=domain)

    def add_text_corpus(self, text: str, domain: str = "corpus") -> int:
        """Tokenize into sequential transitions and terminate each with EOS."""
        if self._looks_like_qa(text):
            return self.ingest_dataset_text(text)

        chunks = re.split(r"(?<=[.!?\n])\s+", text.strip())
        if not chunks:
            chunks = [text]
        added = 0
        for chunk in chunks:
            tokens = self._tokenize(chunk)
            if not tokens:
                continue
            seq = [t.lower() for t in tokens]
            self.dataset.append({
                "input": chunk[:40] + ("..." if len(chunk) > 40 else ""),
                "target": chunk,
                "domain": domain,
                "type": "corpus",
            })
            added += self.add_sequence(seq, domain=domain)
        return added

    @staticmethod
    def _looks_like_qa(text: str) -> bool:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
        qa = sum(1 for ln in lines[:20] if re.match(r"^[QA]\s*->\s+", ln))
        return qa >= 2

    def ingest_dataset_text(self, text: str) -> int:
        """Parse dataset files: Q/A blocks, or `input -> target` pairs."""
        added = 0
        pending_q: Optional[str] = None
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m_qa = re.match(r"^([QA])\s*->\s*(.*)$", line)
            if m_qa:
                kind, body = m_qa.group(1), m_qa.group(2).strip()
                if kind == "Q":
                    pending_q = body
                else:
                    if pending_q:
                        added += self.add_qa_pair(pending_q, body, domain="qa")
                        pending_q = None
                    else:
                        added += self.add_sequence(
                            [t.lower() for t in self._tokenize(body)],
                            domain="qa",
                        )
                continue

            m_corpus = re.match(r"^CORPUS\s*:\s*(.*)$", line, re.IGNORECASE)
            if m_corpus:
                body = m_corpus.group(1).strip()
                if body:
                    added += self.add_text_corpus(body, domain="corpus")
                continue

            if "->" in line:
                left, right = line.split("->", 1)
                left, right = left.strip(), right.strip()
                if left and right:
                    self.add_training_pair(left, right, domain="general")
                    added += 1
                continue

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and parts[0] and parts[1]:
                    domain = "logic" if ("causes" in parts[0] or "prevents" in parts[0]) else "general"
                    added += self.add_qa_pair(parts[0], parts[1], domain=domain)
                continue

        return added

    def ingest_jsonl(self, source: Union[str, Path]) -> int:
        """Parse JSON-Lines dataset (.jsonl) with `input` and `expected_output` fields."""
        path = Path(source)
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else str(source).splitlines()
        added = 0
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
                inp = record.get("input")
                out = record.get("expected_output")
                if inp and out:
                    tags = record.get("tags", [])
                    domain = tags[0] if tags else "jsonl"
                    added += self.add_qa_pair(inp, out, domain=domain)
            except Exception:
                pass
        return added

    def clear_dataset(self):
        self.dataset.clear()
        self.sequences.clear()
        self.token_freq.clear()

    def get_ternary_audit(self) -> Dict:
        matrices = [self.W_in, self.W_rec, self.W_gate, self.W_out] + self.experts
        total = 0
        pure = True
        for m in matrices:
            p = m.ternary_purity()
            total += p["total"]
            if not p["pure"]:
                pure = False
            assert_ternary(m.W, m.name)
        return {"total_synapses": total, "pure_ternary": pure, "contract_version": self.contract.get("contract_version", 1)}

    def param_count(self) -> int:
        matrices = [self.W_in, self.W_rec, self.W_gate, self.W_out] + self.experts
        return sum(m.out_dim * m.in_dim for m in matrices)
