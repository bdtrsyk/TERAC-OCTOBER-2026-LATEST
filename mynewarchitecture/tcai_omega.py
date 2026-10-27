"""
TCAI-Omega (standalone research prototype -- NOT the gui.py backend).
=====================================================================
NOTE: this file defines class TCAIOmega, a parallel prototype to
lumenmax.py's LumenMAX. The PyQt6 GUI (gui.py) uses lumenmax.LumenMAX;
nothing imports this module (it is a self-contained experiment with its
own __main__ demo). The two similarly-named classes used to share the
name TCAIOmega; this one was renamed to prevent silent wrong-model use.
"""

import numpy as np
import time
import sys
import hashlib
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Set, Any


def _stable_hash(data: bytes) -> int:
    """Deterministic 64-bit hash (see lumenmax.py: builtin hash() is
    salted per process for bytes/str and would break determinism)."""
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# CONSTANTS
# ============================================================
TERNARY = np.array([-1, 0, 1], dtype=np.int8)
U_MAX       = 1000
H_MAX       = 500
I_MAX       = 200
M_MAX       = 100
T_CHAIN     = 16
G_MAX       = 4
K_PROV      = 16
K_CF        = 12         # Higher budget for deep counterfactual
N_CAND      = 3          # Competing candidates per state cell
NEIGH       = 2          # Neighborhood radius
REP_WINDOW  = 10
NOVELTY_BUF = 500

# ============================================================
# 1. TERNARY PACKING  (2-bit: 00->-1, 01->0, 10->+1, 11->reserved)
# ============================================================
class TernaryPack:
    @staticmethod
    def pack(a: np.ndarray) -> np.ndarray:
        m = (a.astype(np.int8) + 1).astype(np.uint8)
        pl = (len(a) + 3) // 4
        p = np.zeros(pl, dtype=np.uint8)
        for i in range(len(a)):
            p[i // 4] |= (m[i] & 0b11) << ((i % 4) * 2)
        return p

    @staticmethod
    def unpack(p: np.ndarray, n: int) -> np.ndarray:
        o = np.zeros(n, dtype=np.int8)
        for i in range(n):
            o[i] = ((p[i // 4] >> ((i % 4) * 2)) & 0b11) - 1
        return o

def _mix32(x: int) -> int:
    """Deterministic 32-bit integer mixer (MurmurHash3 finalizer).
    Replaces naive (i*seed)%3 which collapses to a regular cycle
    because 2654435761 % 3 == 1 -> weights become [0,1,-1,0,1,-1,...]."""
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x

# ============================================================
# 2. TERNARY PROJECTION  (adaptive multi-threshold)
# ============================================================
def tproj(z: np.ndarray, fan_in: int = 1) -> np.ndarray:
    """Project integer accumulator -> {-1,0,+1} with adaptive threshold.
    theta ~ 0.6*sqrt(fan_in) keeps roughly half the accumulator signal
    (ternary dot products have std ~ (2/3)*sqrt(fan_in))."""
    theta = max(1, int(round(0.6 * np.sqrt(max(1, fan_in)))))
    r = np.zeros_like(z, dtype=np.int8)
    r[z >= theta] = 1
    r[z <= -theta] = -1
    return r

def tproj1(z: int, theta: int = 1) -> int:
    if z >= theta: return 1
    if z <= -theta: return -1
    return 0

# ============================================================
# 3. TOKEN ENCODING  (deterministic pseudo-orthogonal ternary)
# ============================================================
def _primes(n: int) -> List[int]:
    ps, c = [], 2
    while len(ps) < n:
        if all(c % p for p in ps if p * p <= c):
            ps.append(c)
        c += 1
    return ps

class TokenEncoder:
    def __init__(self, V: int, d: int):
        self.V, self.d = V, d
        self.emb = np.zeros((V, d), dtype=np.int8)
        for v in range(V):
            for j in range(d):
                # (v*prime)%3 maps every token divisible by 3 to the same
                # all -1 vector; use a proper mixer for distinct codes.
                self.emb[v, j] = (_mix32(v * 0x9E3779B1 + j * 0x85EBCA6B) % 3) - 1
            if np.sum(np.abs(self.emb[v])) == 0:
                self.emb[v, 0] = 1
                self.emb[v, 1] = -1

    def encode(self, tok: int) -> np.ndarray:
        return self.emb[tok % self.V].copy()

# ============================================================
# 4. PROVENANCE
# ============================================================
CPRI = {'D': 5, 'I': 4, 'R': 3, 'O': 2, 'C': 1, 'X': 0}

class PAtom:
    __slots__ = ['idx', 'ct', 'sgn', 'n']
    def __init__(self, idx: int, ct: str, sgn: int, n: int = 1):
        self.idx = idx; self.ct = ct; self.sgn = sgn; self.n = n

class Prov:
    """Bounded sparse provenance set."""
    def __init__(self, K: int = K_PROV):
        self.K = K
        self.a: Dict[int, PAtom] = {}

    def add(self, atom: PAtom):
        e = self.a.get(atom.idx)
        if e:
            e.n += atom.n
            if CPRI.get(atom.ct, 0) > CPRI.get(e.ct, 0):
                e.ct = atom.ct; e.sgn = atom.sgn
        else:
            self.a[atom.idx] = atom
        if len(self.a) > self.K:
            self._prune()

    def _prune(self):
        if len(self.a) <= self.K: return
        items = sorted(self.a.values(), key=lambda x: (CPRI.get(x.ct, 0), x.n), reverse=True)
        self.a = {x.idx: x for x in items[:self.K]}

    def causal(self, k: int = K_CF) -> List[int]:
        cs = [x for x in self.a.values() if x.ct in ('D', 'I')]
        cs.sort(key=lambda x: (CPRI.get(x.ct, 0), x.n), reverse=True)
        return [x.idx for x in cs[:k]]

    def all_idx(self) -> Set[int]:
        return set(self.a.keys())

    @staticmethod
    def merge(p1: 'Prov', p2: 'Prov') -> 'Prov':
        r = Prov(max(p1.K, p2.K))
        for idx in set(p1.a) | set(p2.a):
            a1, a2 = p1.a.get(idx), p2.a.get(idx)
            if a1 and not a2:
                r.a[idx] = PAtom(idx, 'D', a1.sgn, a1.n)
            elif a2 and not a1:
                r.a[idx] = PAtom(idx, 'D', a2.sgn, a2.n)
            else:
                ct = 'R' if a1.sgn == a2.sgn else 'O'
                r.a[idx] = PAtom(idx, ct, a1.sgn if ct == 'R' else 0, a1.n + a2.n)
        r._prune()
        return r

# ============================================================
# 5. CONTEXT SIGNATURE
# ============================================================
class CtxSig:
    def __init__(self, d: int = 16):
        self.d = d; self.phi = np.zeros(d, dtype=np.int8)
    def update(self, x: np.ndarray):
        n = min(self.d, len(x))
        s = np.roll(self.phi, 1).astype(np.int32)
        s[:n] += x[:n].astype(np.int32)
        self.phi = np.clip(s, -1, 1).astype(np.int8)
    def match(self, o: np.ndarray) -> int:
        n = min(len(self.phi), len(o))
        return int(np.sum(self.phi[:n] == o[:n]))
    def get(self) -> np.ndarray: return self.phi.copy()
    def reset(self): self.phi.fill(0)
    def hash(self) -> int: return _stable_hash(self.phi.tobytes())

# ============================================================
# 6. REPETITION DETECTOR
# ============================================================
class RepDet:
    def __init__(self, w: int = REP_WINDOW):
        self.w = w; self.outs: List[int] = []; self.hashes: List[int] = []
    def update(self, tok: int, sh: int) -> str:
        self.outs.append(tok); self.hashes.append(sh)
        if len(self.outs) > self.w: self.outs.pop(0); self.hashes.pop(0)
        oc = self.outs.count(tok)
        hc = self.hashes.count(sh)
        if oc > self.w // 2 and hc > self.w // 3: return "harmful"
        if oc > 2: return "productive"
        if oc <= 1: return "neutral"
        return "required"
    def reset(self): self.outs.clear(); self.hashes.clear()

# ============================================================
# 7. NOVELTY DETECTOR
# ============================================================
class NovDet:
    def __init__(self, budget: int = NOVELTY_BUF, dim: int = 32):
        self.budget = budget; self.th = max(1, dim // 4)
        self.pats: Dict[bytes, int] = {}
    def check(self, s: np.ndarray) -> bool:
        sb = s.tobytes()
        if sb in self.pats: self.pats[sb] += 1; return False
        for kb in self.pats:
            k = np.frombuffer(kb, dtype=np.int8)
            if len(k) == len(s) and int(np.sum(s != k)) <= self.th: return False
        if len(self.pats) >= self.budget:
            mk = max(self.pats, key=self.pats.get); del self.pats[mk]
        self.pats[sb] = 1; return True
    def reset(self): self.pats.clear()

# ============================================================
# 8. STRUCTURED ERROR
# ============================================================
class SError:
    NAMES = ['content','structural','temporal','repetition',
             'context','contradiction','prov_conflict','strategy']
    @staticmethod
    def evaluate(tgt: int, pred: int, rep: str, prov_conflict: bool,
                 strat_fail: bool, ctx_miss: bool = False,
                 temp_err: bool = False, struct_err: bool = False) -> np.ndarray:
        e = np.zeros(8, dtype=np.int32)
        e[0] = int(tgt != pred)
        e[1] = int(struct_err)
        e[2] = int(temp_err)
        e[3] = int(rep == "harmful")
        e[4] = int(ctx_miss)
        e[6] = int(prov_conflict)
        e[7] = int(strat_fail)
        return e

# ============================================================
# 9. MULTI-SCALE UTILITY
# ============================================================
class Utility:
    """7 integer utility dimensions per parameter, all bounded [-U_MAX, U_MAX].
    Multi-timescale: short (per-step decay), long (per-epoch accumulate),
    causal, interaction, novelty, repetition, context.
    """
    def __init__(self, P: int):
        self.P = P
        self.short   = np.zeros(P, dtype=np.int32)
        self.long    = np.zeros(P, dtype=np.int32)
        self.causal  = np.zeros(P, dtype=np.int32)
        self.inter   = np.zeros(P, dtype=np.int32)
        self.novel   = np.zeros(P, dtype=np.int32)
        self.rep     = np.zeros(P, dtype=np.int32)
        self.ctx     = np.zeros(P, dtype=np.int32)
        self.consec  = np.zeros(P, dtype=np.int32)
        self.best_ctx = np.zeros((P, 16), dtype=np.int8)

    def update(self, i: int, delta: int, syn: int = 0,
               is_nov: bool = False, rep_s: int = 0, ctx_s: int = 0):
        s = 1 if delta > 0 else (-1 if delta < 0 else 0)
        cl = np.clip
        self.short[i] = cl((self.short[i] * 3) // 4 + delta * 4, -U_MAX, U_MAX)
        self.long[i]  = cl(self.long[i] + s, -U_MAX, U_MAX)
        self.causal[i]= cl(self.causal[i] + delta, -U_MAX, U_MAX)
        self.inter[i] = cl(self.inter[i] + syn, -U_MAX, U_MAX)
        nv = (1 if is_nov else 0) * s * 2
        self.novel[i] = cl((self.novel[i] * 7) // 8 + nv, -U_MAX, U_MAX)
        self.rep[i]   = cl(self.rep[i] + rep_s, -U_MAX, U_MAX)
        self.ctx[i]   = cl((self.ctx[i] * 7) // 8 + ctx_s, -U_MAX, U_MAX)

    def composite(self, i: int) -> int:
        return int(2*self.long[i] + self.short[i] + self.causal[i] +
                   self.inter[i] + self.novel[i] + self.rep[i] + self.ctx[i])

    def composite_all(self) -> np.ndarray:
        return 2*self.long + self.short + self.causal + self.inter + self.novel + self.rep + self.ctx

    def is_protected(self, i: int, sgn: int) -> bool:
        if self.P < 10: return False
        th = np.partition(self.long, -max(1, self.P//10))[-max(1, self.P//10)]
        if self.long[i] >= th:
            if self.consec[i] * sgn > 0: self.consec[i] += sgn
            else: self.consec[i] = sgn
            return abs(self.consec[i]) < 2
        return False

    def decay_short(self):
        self.short = (self.short * 3) // 4

# ============================================================
# 10. INTERACTION HYPERGRAPH
# ============================================================
class IGraph:
    def __init__(self):
        self.edges: Dict[tuple, List[int]] = {}
    def observe(self, g: tuple, ok: bool):
        if len(g) < 2 or len(g) > G_MAX: return
        k = tuple(sorted(g))
        if k not in self.edges: self.edges[k] = [0, 0]
        self.edges[k][1] += 1
        self.edges[k][0] += 1 if ok else -1
        if len(self.edges) > I_MAX:
            ranked = sorted(self.edges.items(), key=lambda kv: abs(kv[1][0])/(kv[1][1]+1), reverse=True)
            self.edges = dict(ranked[:I_MAX])
    def synergy(self, g: tuple) -> int:
        k = tuple(sorted(g))
        return self.edges[k][0] if k in self.edges else 0
    def groups_of(self, i: int) -> List[tuple]:
        return [g for g in self.edges if i in g]

# ============================================================
# 11. MOTIF MEMORY
# ============================================================
class Motifs:
    def __init__(self):
        self.m: Dict[int, dict] = {}; self.nxt = 0
    def find(self, pk: tuple, sh: int) -> int:
        for mid, mo in self.m.items():
            if mo['pk'] == pk and mo['sh'] == sh: return mid
        return -1
    def record(self, pk: tuple, sh: int, sh2: int, ch: int, ok: bool) -> int:
        mid = self.find(pk, sh)
        if mid >= 0:
            self.m[mid]['n'] += 1; self.m[mid]['s'] += 1 if ok else -1; return mid
        mid = self.nxt; self.nxt += 1
        self.m[mid] = {'pk': pk, 'sh': sh, 'sh2': sh2, 'ch': ch, 's': 1 if ok else -1, 'n': 1}
        if len(self.m) > M_MAX:
            ranked = sorted(self.m.items(), key=lambda kv: abs(kv[1]['s'])/(kv[1]['n']+1), reverse=True)
            self.m = dict(ranked[:M_MAX])
        return mid
    def failing(self, mid: int) -> bool:
        if mid < 0 or mid not in self.m: return False
        mo = self.m[mid]; return mo['s'] < -(int(np.sqrt(mo['n'])) + 2)

# ============================================================
# 12. CAUSAL CHAIN BUFFER  (temporal credit)
# ============================================================
class CChain:
    def __init__(self, L: int = T_CHAIN):
        self.L = L; self.ch: List[List[tuple]] = []
    def add(self, changes: List[tuple]):
        self.ch.append(changes)
        if len(self.ch) > self.L: self.ch.pop(0)
    def trace(self, cells: Set[int]) -> Dict[int, int]:
        cr: Dict[int, int] = defaultdict(int)
        active = set(cells)
        for dist, step in enumerate(reversed(self.ch)):
            if not active: break
            na = set()
            for ci, pis, pv, nv in step:
                if ci in active:
                    for pi in pis: cr[pi] += max(1, 4 // (1 + dist))
                    na.update(pis)
            active = na
        return dict(cr)
    def reset(self): self.ch.clear()

# ============================================================
# 13. REPLAY BUFFER
# ============================================================
class Replay:
    def __init__(self, cap: int = H_MAX):
        self.cap = cap; self.buf: List[dict] = []
    def add(self, rec: dict):
        self.buf.append(rec)
        if len(self.buf) > self.cap * 2: self._consolidate()
    def _consolidate(self):
        if len(self.buf) <= self.cap: return
        self.buf.sort(key=lambda r: r.get('pri', 0), reverse=True)
        top = self.buf[:self.cap // 2]
        errs = [r for r in self.buf[self.cap // 2:] if r.get('err')]
        self.buf = (top + errs)[:self.cap]
    def sample(self, n: int) -> List[dict]:
        if not self.buf: return []
        n = min(n, len(self.buf))
        pri = np.array([max(r.get('pri', 1), 1) for r in self.buf], dtype=np.int32)
        return [self.buf[i] for i in np.argsort(pri)[-n:]]

# ============================================================
# 14. PARAMETER BLOCK
# ============================================================
class PBlock:
    """Named block of ternary parameters with flat-index mapping.
    Initialized with deterministic balanced ternary pattern.
    """
    def __init__(self, name: str, shape: tuple, offset: int, seed: int = 31):
        self.name = name; self.shape = shape; self.offset = offset
        self.size = int(np.prod(shape))
        # Deterministic balanced init: ~33% each of {-1, 0, +1}
        d = np.zeros(self.size, dtype=np.int8)
        for i in range(self.size):
            v = _mix32((i * 0x9E3779B1) ^ (offset * 0x85EBCA6B) ^ (seed * 0xC2B2AE35)) % 3
            d[i] = v - 1  # maps 0->-1, 1->0, 2->+1
        self.data = d.reshape(shape)

    def get_flat(self, fi: int) -> int:
        loc = fi - self.offset
        if 0 <= loc < self.size: return int(self.data.flat[loc])
        return 0

    def set_flat(self, fi: int, v: int):
        loc = fi - self.offset
        if 0 <= loc < self.size: self.data.flat[loc] = np.int8(v)

# ============================================================
# 15. TCAIOmega  --  THE COMPLETE MODEL
# ============================================================
class TCAIOmega:
    """TCAIOmega: A Ternary Competitive Arithmetic Intelligence.

    State dynamics: three coupled components (Memory, Evidence, Decision)
    with 3-candidate competition and 3-way gated updates.
    Learning: causal counterfactual evaluation with deep re-computation
    through state dynamics, not just the output layer.
    """

    def __init__(self, N: int = 32, V: int = 16, d: int = 32, L: int = 2):
        self.N, self.V, self.d, self.L = N, V, d, L

        # --- Parameter blocks ---
        self.blocks: List[PBlock] = []
        off = 0

        # Memory candidate generators: 3 x (N, N+d)
        self.PM = []
        for c in range(N_CAND):
            b = PBlock(f'PM{c}', (N, N + d), off, seed=31 + c * 7)
            self.blocks.append(b); self.PM.append(b); off += b.size

        # Evidence candidate generators: 3 x (N, 2N+d)
        self.PE = []
        for c in range(N_CAND):
            b = PBlock(f'PE{c}', (N, 2 * N + d), off, seed=53 + c * 11)
            self.blocks.append(b); self.PE.append(b); off += b.size

        # Decision candidate generators: 3 x (N, 3N)
        self.PD = []
        for c in range(N_CAND):
            b = PBlock(f'PD{c}', (N, 3 * N), off, seed=79 + c * 13)
            self.blocks.append(b); self.PD.append(b); off += b.size

        # Gate blocks: 3 x (N, N)  -- one per state component
        self.PG = []
        for c in range(3):
            b = PBlock(f'PG{c}', (N, N), off, seed=97 + c * 17)
            self.blocks.append(b); self.PG.append(b); off += b.size

        # Output projection: (V, N)
        self.Pout = PBlock('Pout', (V, N), off, seed=127)
        self.blocks.append(self.Pout); off += self.Pout.size

        self.P = off  # total parameter count

        # --- State ---
        # Initialize with small deterministic noise to escape all-zero basin
        self.M = np.zeros(N, dtype=np.int8)
        self.E = np.zeros(N, dtype=np.int8)
        self.D = np.zeros(N, dtype=np.int8)
        self._init_state_noise()

        # --- Subsystems ---
        self.enc   = TokenEncoder(V, d)
        self.ctx   = CtxSig(min(16, d))
        self.util  = Utility(self.P)
        self.igraph= IGraph()
        self.motifs= Motifs()
        self.chain = CChain()
        self.replay= Replay()
        self.repdet= RepDet()
        self.novdet= NovDet(dim=N)
        self.step_n = 0
        self.ablate: set = set()

        # --- Block index for fast lookup ---
        self._block_ranges = [(b.offset, b.offset + b.size, b) for b in self.blocks]

    def _init_state_noise(self):
        """Initialize state with small deterministic ternary noise to escape all-zero basin."""
        for i in range(self.N):
            self.M[i] = (_mix32(i * 0x9E3779B1 + 1) % 3) - 1
            self.E[i] = (_mix32(i * 0x9E3779B1 + 2) % 3) - 1
            self.D[i] = (_mix32(i * 0x9E3779B1 + 3) % 3) - 1

    # ---------- parameter access ----------
    def _blk(self, fi: int) -> Optional[PBlock]:
        for lo, hi, b in self._block_ranges:
            if lo <= fi < hi: return b
        return None

    def _get(self, fi: int) -> int:
        b = self._blk(fi)
        return b.get_flat(fi) if b else 0

    def _set(self, fi: int, v: int):
        b = self._blk(fi)
        if b: b.set_flat(fi, v)

    # ---------- state dynamics ----------
    def _candidates(self, gens: List[PBlock], state: np.ndarray,
                    extra: np.ndarray, fan: int) -> List[np.ndarray]:
        inp = np.concatenate([state, extra]).astype(np.int32)
        return [tproj(g.data.astype(np.int32) @ inp, fan) for g in gens]

    def _compete(self, cands: List[np.ndarray], old: np.ndarray) -> Tuple[np.ndarray, int]:
        N = len(old)
        best_i, best_score = 0, -999999
        for ci, c in enumerate(cands):
            # Score = activity (non-zero count) + coherence (neighbor agreement) - change_penalty
            activity = int(np.sum(np.abs(c)))
            coherence = 0
            for j in range(N):
                for k in range(1, NEIGH + 1):
                    if j + k < N: coherence += int(c[j]) * int(c[j + k])
                    if j - k >= 0: coherence += int(c[j]) * int(c[j - k])
            # Small penalty for changing from old, but not when old is zero
            change = int(np.sum(np.abs(c.astype(np.int32) - old.astype(np.int32))))
            old_activity = int(np.sum(np.abs(old)))
            change_penalty = change if old_activity > 0 else 0
            score = activity * 2 + coherence - change_penalty
            if score > best_score:
                best_score, best_i = score, ci
        return cands[best_i], best_i

    def _gate(self, winner: np.ndarray, old: np.ndarray, gb: PBlock) -> np.ndarray:
        diff = old.astype(np.int32) - winner.astype(np.int32)
        g = tproj(gb.data.astype(np.int32) @ diff, self.N)
        ns = np.zeros(self.N, dtype=np.int8)
        ns[g == 0] = winner[g == 0]
        ns[g == 1] = old[g == 1]
        # g == -1 => ns stays 0 (clear)
        return ns

    def _collect_prov(self, gen: PBlock, out: np.ndarray) -> Prov:
        """Collect provenance from a generator computation.
        Tracks parameters that contributed most to non-zero outputs.
        """
        prov = Prov()
        count = 0
        for j in range(min(len(out), gen.shape[0])):
            if out[j] != 0 and count < K_PROV:
                row = gen.data[j]
                nz = np.nonzero(row)[0]
                # Take up to 3 non-zero params per row for richer provenance
                for k_idx in range(min(3, len(nz))):
                    pidx = int(nz[k_idx])
                    flat = gen.offset + j * gen.shape[1] + pidx
                    prov.add(PAtom(flat, 'D', int(out[j] * row[pidx])))
                    count += 1
                    if count >= K_PROV: break
        return prov

    def _shash(self) -> int:
        return _stable_hash(np.concatenate([self.M, self.E, self.D]).tobytes())

    # ---------- forward ----------
    def _forward_core(self, x: np.ndarray) -> Tuple[int, np.ndarray, Prov, dict]:
        """Core forward pass: state dynamics + output projection.
        Returns (prediction, scores, provenance, internals_dict).
        internals_dict stores everything needed for deep counterfactual.
        """
        oM, oE, oD = self.M.copy(), self.E.copy(), self.D.copy()

        # Memory
        mc = self._candidates(self.PM, self.M, x, self.N + self.d)
        mw, mi = self._compete(mc, self.M)
        self.M = self._gate(mw, oM, self.PG[0])
        mp = self._collect_prov(self.PM[mi], self.M)

        # Evidence
        ee = np.concatenate([self.M, x])
        ec = self._candidates(self.PE, self.E, ee, 2 * self.N + self.d)
        ew, ei = self._compete(ec, self.E)
        self.E = self._gate(ew, oE, self.PG[1])
        ep = self._collect_prov(self.PE[ei], self.E)

        # Decision
        de = np.concatenate([self.E, self.M])
        dc = self._candidates(self.PD, self.D, de, 3 * self.N)
        dw, di = self._compete(dc, self.D)
        self.D = self._gate(dw, oD, self.PG[2])
        dp = self._collect_prov(self.PD[di], self.D)

        # Output
        scores = self.Pout.data.astype(np.int32) @ self.D.astype(np.int32)
        pred = int(np.argmax(scores))

        # Merge provenance
        prov = Prov.merge(Prov.merge(mp, ep), dp)
        # Add output-layer provenance
        for j in range(self.N):
            if self.D[j] != 0 and self.Pout.data[pred, j] != 0:
                flat = self.Pout.offset + pred * self.N + j
                prov.add(PAtom(flat, 'D', int(self.D[j] * self.Pout.data[pred, j])))

        # Record state changes for temporal credit
        changes = []
        for j in range(self.N):
            if self.M[j] != oM[j]:
                changes.append((j, list(mp.all_idx())[:4], int(oM[j]), int(self.M[j])))
            if self.E[j] != oE[j]:
                changes.append((self.N+j, list(ep.all_idx())[:4], int(oE[j]), int(self.E[j])))
            if self.D[j] != oD[j]:
                changes.append((2*self.N+j, list(dp.all_idx())[:4], int(oD[j]), int(self.D[j])))
        if 'no_temporal' not in self.ablate:
            self.chain.add(changes)

        internals = {
            'x': x.copy(), 'oM': oM, 'oE': oE, 'oD': oD,
            'mi': mi, 'ei': ei, 'di': di,
        }
        return pred, scores, prov, internals

    def forward_step(self, x: np.ndarray) -> Tuple[int, np.ndarray, Prov]:
        pred, scores, prov, _ = self._forward_core(x)
        return pred, scores, prov

    # ---------- DEEP COUNTERFACTUAL ----------
    def _deep_cf(self, target: int, cur_score: int, cur_pred: int,
                 prov: Prov, internals: dict) -> Dict[int, Tuple[int, int]]:
        """Deep counterfactual evaluation.

        For each candidate parameter:
        - If in Pout: re-compute output projection only (cheap)
        - If in state dynamics: re-run affected state transition + downstream + output (deep)

        Returns {flat_idx: (best_value, delta)}.
        """
        candidates = list(prov.causal(K_CF))

        # Always include relevant Pout parameters for target and pred
        for j in range(self.N):
            if self.D[j] != 0:
                for tok in [target, cur_pred]:
                    fi = self.Pout.offset + tok * self.N + j
                    if fi not in candidates: candidates.append(fi)

        results: Dict[int, Tuple[int, int]] = {}
        x = internals['x']

        for fi in candidates:
            cv = self._get(fi)
            best_v, best_d = cv, 0

            blk = self._blk(fi)
            if blk is None: continue

            for av in TERNARY:
                av = int(av)
                if av == cv: continue

                self._set(fi, av)

                # Determine re-computation scope based on block
                bn = blk.name
                if bn == 'Pout':
                    # Cheap: just output projection
                    alt_sc = blk.data.astype(np.int32) @ self.D.astype(np.int32)
                elif bn.startswith('PM') or bn.startswith('PG0'):
                    # Deep: re-run M -> E -> D -> output
                    mc = self._candidates(self.PM, internals['oM'], x, self.N+self.d)
                    mw, _ = self._compete(mc, internals['oM'])
                    tM = self._gate(mw, internals['oM'], self.PG[0])
                    ee = np.concatenate([tM, x])
                    ec = self._candidates(self.PE, internals['oE'], ee, 2*self.N+self.d)
                    ew, _ = self._compete(ec, internals['oE'])
                    tE = self._gate(ew, internals['oE'], self.PG[1])
                    de = np.concatenate([tE, tM])
                    dc = self._candidates(self.PD, internals['oD'], de, 3*self.N)
                    dw, _ = self._compete(dc, internals['oD'])
                    tD = self._gate(dw, internals['oD'], self.PG[2])
                    alt_sc = self.Pout.data.astype(np.int32) @ tD.astype(np.int32)
                elif bn.startswith('PE') or bn.startswith('PG1'):
                    # Medium: re-run E -> D -> output
                    ee = np.concatenate([self.M, x])
                    ec = self._candidates(self.PE, internals['oE'], ee, 2*self.N+self.d)
                    ew, _ = self._compete(ec, internals['oE'])
                    tE = self._gate(ew, internals['oE'], self.PG[1])
                    de = np.concatenate([tE, self.M])
                    dc = self._candidates(self.PD, internals['oD'], de, 3*self.N)
                    dw, _ = self._compete(dc, internals['oD'])
                    tD = self._gate(dw, internals['oD'], self.PG[2])
                    alt_sc = self.Pout.data.astype(np.int32) @ tD.astype(np.int32)
                elif bn.startswith('PD') or bn.startswith('PG2'):
                    # Shallow: re-run D -> output
                    de = np.concatenate([self.E, self.M])
                    dc = self._candidates(self.PD, internals['oD'], de, 3*self.N)
                    dw, _ = self._compete(dc, internals['oD'])
                    tD = self._gate(dw, internals['oD'], self.PG[2])
                    alt_sc = self.Pout.data.astype(np.int32) @ tD.astype(np.int32)
                else:
                    alt_sc = self.Pout.data.astype(np.int32) @ self.D.astype(np.int32)

                self._set(fi, cv)  # restore

                delta = int(alt_sc[target]) - cur_score
                if cur_pred >= 0 and cur_pred != target:
                    # Also consider suppressing the wrong prediction
                    delta += max(0, cur_score - int(alt_sc[cur_pred]))
                if delta > best_d:
                    best_d = delta; best_v = av

            if best_d > 0 and best_v != cv:
                results[fi] = (best_v, best_d)

        return results

    # ---------- TRAINING ----------
    def train_step(self, xtok: int, ytgt: int) -> Tuple[bool, dict]:
        self.step_n += 1
        xenc = self.enc.encode(xtok)
        self.ctx.update(xenc)

        pred, scores, prov, internals = self._forward_core(xenc)
        ok = (pred == ytgt)
        sh = self._shash()

        rep_cls = self.repdet.update(pred, sh)
        is_nov = False if 'no_novelty' in self.ablate else self.novdet.check(self.D)
        prov_conf = any(a.ct == 'O' for a in prov.a.values())
        mk = tuple(sorted(list(prov.all_idx())[:4]))
        mid = self.motifs.find(mk, sh)
        sf = self.motifs.failing(mid) if mid >= 0 else False
        ev = SError.evaluate(ytgt, pred, rep_cls, prov_conf, sf)
        is_err = not ok

        updates: Dict[int, Tuple[int, int]] = {}
        tcredits: Dict[int, int] = {}

        if is_err:
            cur_sc = int(scores[ytgt])
            updates = self._deep_cf(ytgt, cur_sc, pred, prov, internals)
            ecells = {2 * self.N + j for j in range(self.N) if self.D[j] != 0}
            tcredits = {} if 'no_temporal' in self.ablate else self.chain.trace(ecells)

        self.motifs.record(mk, sh, self._shash(), self.ctx.hash(), ok)

        pidx = list(prov.all_idx())
        if len(pidx) >= 2 and 'no_interaction' not in self.ablate:
            self.igraph.observe(tuple(sorted(pidx[:G_MAX])), ok)

        rep_s = -2 if rep_cls == "harmful" else (1 if rep_cls == "required" else 0)
        for fi in prov.all_idx():
            delta = 0
            if fi in updates: delta = updates[fi][1]
            elif ok: delta = 1
            elif is_err: delta = -1
            if fi in tcredits: delta += tcredits[fi]
            syn = 0 if 'no_interaction' in self.ablate else sum(self.igraph.synergy(g) for g in self.igraph.groups_of(fi))
            ctx_s = 1 if ok else -1
            self.util.update(fi, delta, syn, is_nov, rep_s, ctx_s)

        nu = 0
        for fi, (bv, bd) in updates.items():
            sgn = 1 if bd > 0 else -1
            if not self.util.is_protected(fi, sgn):
                self._set(fi, bv); nu += 1

        # Temporal credit driven updates
        for fi, cr in tcredits.items():
            if fi not in updates and cr > 2:
                cv = self._get(fi)
                for av in TERNARY:
                    av = int(av)
                    if av == cv:
                        continue
                    self._set(fi, av)
                    asc = self.Pout.data.astype(np.int32) @ self.D.astype(np.int32)
                    if int(asc[ytgt]) > int(scores[ytgt]):
                        if not self.util.is_protected(fi, 1):
                            nu += 1; break
                        else:
                            self._set(fi, cv)
                    else:
                        self._set(fi, cv)

        nze = int(np.sum(np.abs(ev)))
        pri = 3 * nze + 2 * int(is_err) + int(is_nov)
        self.replay.add({
            'inp': xtok, 'tgt': ytgt, 'pred': pred,
            'ev': ev.copy(), 'pidx': pidx[:8],
            't': self.step_n, 'mid': mid,
            'err': is_err, 'pri': pri, 'ih': hash(xtok)
        })

        return ok, {'ok': ok, 'pred': pred, 'tgt': ytgt, 'nu': nu,
                     'rep': rep_cls, 'nov': is_nov, 'ev': ev, 'ntc': len(tcredits)}

    def train(self, data: List[Tuple[List[int], List[int]]],
              epochs: int, verbose: bool = True) -> Dict[str, Any]:
        met = {'accuracy': [], 'times': [], 'updates': []}
        for ep in range(epochs):
            t0 = time.time()
            nc, tot, nu = 0, 0, 0
            for sx, sy in data:
                self._init_state_noise()
                self.ctx.reset(); self.repdet.reset(); self.chain.reset()
                for xt, yt in zip(sx, sy):
                    ok, diag = self.train_step(xt, yt)
                    nc += int(ok); nu += diag['nu']; tot += 1

            # Replay
            if 'no_replay' not in self.ablate:
                for rec in self.replay.sample(min(8, len(self.replay.buf))):
                    if rec['err']:
                        xenc = self.enc.encode(rec['inp'])
                        self._init_state_noise()
                        pred, scores, prov, ints = self._forward_core(xenc)
                        if pred != rec['tgt']:
                            ups = self._deep_cf(rec['tgt'], int(scores[rec['tgt']]), pred, prov, ints)
                            for fi, (bv, bd) in ups.items():
                                hd = bd // 2
                                if hd > 0:
                                    self.util.update(fi, hd)
                                    if not self.util.is_protected(fi, 1):
                                        self._set(fi, bv)

            self.util.decay_short()
            self._diversity()
            self._anticollapse()
            self._verify()

            el = time.time() - t0
            acc = nc / max(1, tot)
            met['accuracy'].append(acc); met['times'].append(el); met['updates'].append(nu)
            if verbose:
                print(f"  Epoch {ep+1}/{epochs}: acc={acc:.4f} upd={nu} t={el:.2f}s")
        return met

    # ---------- ANTI-COLLAPSE ----------
    def _diversity(self):
        comp = self.util.composite_all()
        dom = np.abs(comp) > int(0.9 * U_MAX)
        if np.any(dom):
            self.util.long[dom] = np.clip(self.util.long[dom], -int(0.7*U_MAX), int(0.7*U_MAX))
        dead = comp < -int(0.9 * U_MAX)
        if np.any(dead):
            self.util.long[dead] = 0; self.util.short[dead] = 0

    def _anticollapse(self):
        for b in self.blocks:
            nz = int(np.sum(b.data == 0))
            if nz > b.size * 2 // 3:
                zi = np.where(b.data.flat == 0)[0]
                nf = min(len(zi), b.size // 3)
                for i in range(nf):
                    idx = zi[i]
                    b.data.flat[idx] = np.int8(1 if (idx + self.step_n) % 2 == 0 else -1)
            # Prevent all-same
            u = np.unique(b.data)
            if len(u) == 1 and b.size > 1:
                for i in range(min(b.size // 3, 100)):
                    b.data.flat[i] = np.int8(((i * 7 + 3) % 3) - 1)

    def _verify(self):
        """Verify ternary invariants after every epoch."""
        for b in self.blocks:
            mask = (b.data != -1) & (b.data != 0) & (b.data != 1)
            if np.any(mask):
                b.data[mask] = np.int8(0)
        for s in [self.M, self.E, self.D]:
            mask = (s != -1) & (s != 0) & (s != 1)
            if np.any(mask):
                s[mask] = np.int8(0)

    # ---------- INFERENCE ----------
    def predict(self, seq: List[int]) -> List[int]:
        self.M.fill(0); self.E.fill(0); self.D.fill(0); self.ctx.reset()
        ps = []
        for t in seq:
            p, _, _ = self.forward_step(self.enc.encode(t))
            ps.append(p)
        return ps

    # ---------- EXPLAIN ----------
    def explain_prediction(self, seq: List[int]) -> Dict[str, Any]:
        self.M.fill(0); self.E.fill(0); self.D.fill(0)
        self.ctx.reset(); self.repdet.reset()
        preds, last_prov, last_sc = [], Prov(), np.zeros(self.V, dtype=np.int32)
        for t in seq:
            xenc = self.enc.encode(t); self.ctx.update(xenc)
            p, sc, prov = self.forward_step(xenc)
            self.repdet.update(p, self._shash())
            preds.append(p); last_prov = prov; last_sc = sc
        fp = preds[-1]
        ranked = np.argsort(last_sc)[::-1]
        competing = [(int(ranked[i]), int(last_sc[ranked[i]])) for i in range(min(5, len(ranked)))]
        dom, opp = [], []
        for j in range(self.N):
            c = int(self.Pout.data[fp, j]) * int(self.D[j])
            fi = self.Pout.offset + fp * self.N + j
            if c > 0: dom.append(fi)
            elif c < 0: opp.append(fi)
        return {
            'predicted_tokens': preds, 'final': fp,
            'competing': competing, 'dominant': dom[:10], 'opposing': opp[:10],
            'M': self.M.tolist(), 'E': self.E.tolist(), 'D': self.D.tolist(),
            'interactions': len(self.igraph.edges),
            'motifs': len(self.motifs.m),
            'novelty_patterns': len(self.novdet.pats),
            'replay_size': len(self.replay.buf),
            'evidence_score': int(last_sc[fp]),
        }


# ============================================================
# 16. SELF-AUDIT
# ============================================================
def audit(model: TCAIOmega) -> bool:
    print("  [AUDIT] Verifying TCAIOmega constraints...")
    assert not hasattr(model, 'grad'), "gradient object"
    assert not hasattr(model, 'backward'), "backward method"
    assert not hasattr(model, 'optimizer'), "optimizer"
    for b in model.blocks:
        vals = np.unique(b.data)
        for v in vals: assert v in (-1, 0, 1), f"param {v} in {b.name}"
        assert b.data.dtype == np.int8, f"{b.name} dtype {b.data.dtype}"
    for nm, s in [('M', model.M), ('E', model.E), ('D', model.D)]:
        for v in np.unique(s): assert v in (-1, 0, 1), f"state {v} in {nm}"
        assert s.dtype == np.int8, f"{nm} dtype {s.dtype}"
    assert 'torch' not in sys.modules
    assert 'tensorflow' not in sys.modules
    for a in dir(model): assert 'attention' not in a.lower(), f"attention: {a}"
    model.M.fill(0); model.E.fill(0); model.D.fill(0)
    x = model.enc.encode(0)
    _, s1, _ = model.forward_step(x)
    model.M.fill(0); model.E.fill(0); model.D.fill(0)
    _, s2, _ = model.forward_step(x)
    assert np.array_equal(s1, s2), "non-deterministic"
    print("  [AUDIT] All 9 checks PASSED")
    return True

# ============================================================
# 17. DIAGNOSTICS
# ============================================================
def param_util(model: TCAIOmega) -> Dict[str, Any]:
    nz = sum(int(np.sum(b.data != 0)) for b in model.blocks)
    c = model.util.composite_all()
    return {
        'total': model.P, 'nonzero': nz,
        'nz_pct': 100.0 * nz / max(1, model.P),
        'mean_util': float(np.mean(c)), 'std_util': float(np.std(c)),
        'max_util': int(np.max(c)), 'min_util': int(np.min(c)),
    }

def complexity(m: TCAIOmega):
    N, V, d = m.N, m.V, m.d
    print(f"\n  === COMPLEXITY ===")
    print(f"  Params: {m.P}  State: {N}  Vocab: {V}  Input: {d}")
    print(f"  Inference/step: O({N*(N+d) + N*(2*N+d) + N*3*N + V*N})")
    print(f"  Train/step:     O(Kc*[full_forward]) = O({K_CF}*{N*(N+d)+N*(2*N+d)+N*3*N+V*N})")
    print(f"  Memory: params={m.P}B packed={m.P//4}B util={m.P*7*4}B state={3*N}B")

# ============================================================
# 18. BENCHMARKS
# ============================================================
def gen_copy(n, ln, V, seed=42):
    rng = np.random.RandomState(seed)
    return [([int(x) for x in rng.randint(1, V, ln)],) * 2 for _ in range(n)]

def gen_delayed(n, ln, delay, V, seed=42):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        s = [int(x) for x in rng.randint(1, V, ln)]
        d.append((s + [0]*delay, [0]*delay + s))
    return d

def gen_pattern(n, pl=3, seed=42):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        p = [int(x) for x in rng.randint(1, 5, pl)]
        s = p * 3; d.append((s[:-1], s[1:]))
    return d

def gen_arith(n, seed=42):
    rng = np.random.RandomState(seed)
    return [([int(a), int(b)], [int(b), min(int(a)+int(b), 15)])
            for a, b in zip(rng.randint(1, 5, n), rng.randint(1, 5, n))]

def gen_longrange(n, ln=8, seed=42):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        f = int(rng.randint(1, 5))
        mid = [int(x) for x in rng.randint(1, 5, ln - 2)]
        d.append(([f] + mid + [0], [0]*(ln-1) + [f]))
    return d

def gen_ctx(n, seed=42):
    rng = np.random.RandomState(seed)
    return [([int(c), 5, 5], [0, int(c), int(c)]) for c in rng.randint(1, 4, n)]

def gen_repsens(n, seed=42):
    return [([1, 1, 1, 2, 3], [1, 1, 0, 2, 3]) for _ in range(n)]

def gen_contradiction(n, seed=42):
    return [([1, 2, 1, 3], [2, 1, 3, 0]) for _ in range(n)]

# ============================================================
# 19. CONTROLS
# ============================================================
class RandomCtrl(TCAIOmega):
    def train_step(self, xt, yt):
        self.step_n += 1; xenc = self.enc.encode(xt)
        p, sc, prov = self.forward_step(xenc); ok = p == yt
        if not ok:
            for i in range(K_CF):
                fi = (self.step_n * 7 + i) % self.P
                self._set(fi, (self.step_n + i) % 3 - 1)
        return ok, {'ok': ok, 'pred': p, 'tgt': yt, 'nu': K_CF if not ok else 0,
                     'rep': 'n', 'nov': False, 'ev': np.zeros(8, np.int32), 'ntc': 0}

class SignCtrl(TCAIOmega):
    def train_step(self, xt, yt):
        self.step_n += 1; xenc = self.enc.encode(xt)
        p, sc, prov = self.forward_step(xenc); ok = p == yt; nu = 0
        if not ok:
            for fi in list(prov.all_idx())[:K_CF]:
                cv = self._get(fi)
                nv = np.clip(cv + (1 if sc[yt] < sc[p] else -1), -1, 1)
                self._set(fi, int(nv)); nu += 1
        return ok, {'ok': ok, 'pred': p, 'tgt': yt, 'nu': nu,
                     'rep': 'n', 'nov': False, 'ev': np.zeros(8, np.int32), 'ntc': 0}

class StaticCtrl(TCAIOmega):
    def train_step(self, xt, yt):
        self.step_n += 1; xenc = self.enc.encode(xt)
        p, sc, prov = self.forward_step(xenc); ok = p == yt
        return ok, {'ok': ok, 'pred': p, 'tgt': yt, 'nu': 0,
                     'rep': 'n', 'nov': False, 'ev': np.zeros(8, np.int32), 'ntc': 0}

# ============================================================
# 20. ABLATION FRAMEWORK
# ============================================================
class Ablation:
    @staticmethod
    def run(task_fn, epochs=10, verbose=False):
        configs = ['full', 'no_replay', 'no_temporal', 'no_interaction', 'no_novelty']
        res = {}
        for cfg in configs:
            m = TCAIOmega(N=16, V=10, d=16, L=1)
            m.ablate = set() if cfg == 'full' else {cfg}
            data = task_fn()
            met = m.train(data, epochs, verbose=verbose)
            res[cfg] = met['accuracy'][-1] if met['accuracy'] else 0.0
        return res

# ============================================================
# 21. MAIN
# ============================================================
def main():
    print("=" * 65)
    print("  TCAIOmega -- Ternary Competitive Arithmetic Intelligence")
    print("  TCAI-Omega: Zero-Gradient Discrete Self-Optimizing Organism")
    print("=" * 65)

    # --- Create ---
    print("\n[1] Building TCAIOmega (N=32, V=16, d=32)...")
    m = TCAIOmega(N=32, V=16, d=32, L=2)
    print(f"    Parameters: {m.P}")
    print(f"    Blocks: {len(m.blocks)}")
    for b in m.blocks:
        nz = int(np.sum(b.data != 0))
        print(f"      {b.name:6s}: {str(b.shape):12s} size={b.size:5d}  nonzero={100*nz/b.size:.0f}%")

    # --- Audit ---
    print("\n[2] Self-Audit...")
    audit(m)

    # --- Copy Task ---
    print("\n[3] Copy Task (100 seq, len 4, vocab 16)...")
    cd = gen_copy(100, 4, 16)
    cm = m.train(cd, epochs=25, verbose=True)

    # --- Explain ---
    print("\n[4] Explanation...")
    sample = cd[0][0]
    ex = m.explain_prediction(sample)
    print(f"    Input:       {sample}")
    print(f"    Predicted:   {ex['predicted_tokens']}")
    print(f"    Final:       {ex['final']}")
    print(f"    Competing:   {ex['competing'][:5]}")
    print(f"    Dominant:    {ex['dominant'][:5]}")
    print(f"    Opposing:    {ex['opposing'][:5]}")
    print(f"    Interactions:{ex['interactions']}")
    print(f"    Motifs:      {ex['motifs']}")
    print(f"    Evidence:    {ex['evidence_score']}")

    # --- Utilization ---
    print("\n[5] Parameter Utilization...")
    u = param_util(m)
    for k, v in u.items(): print(f"    {k}: {v}")

    # --- Complexity ---
    print("\n[6] Complexity...")
    complexity(m)

    # --- Determinism ---
    print("\n[7] Determinism Check...")
    sd = gen_copy(10, 3, 8, seed=123)
    m1 = TCAIOmega(N=16, V=8, d=16, L=1)
    r1 = m1.train(sd, 5, verbose=False)
    m2 = TCAIOmega(N=16, V=8, d=16, L=1)
    r2 = m2.train(sd, 5, verbose=False)
    if r1['accuracy'] == r2['accuracy']:
        print("    [PASS] Deterministic: identical results from identical conditions")
    else:
        print(f"    [FAIL] {r1['accuracy']} != {r2['accuracy']}")

    # --- Controls ---
    print("\n[8] Control Experiments...")
    ctd = gen_copy(50, 3, 8, seed=99)
    models = {
        'TCAIOmega': TCAIOmega(N=16, V=8, d=16, L=1),
        'Random':   RandomCtrl(N=16, V=8, d=16, L=1),
        'SignError': SignCtrl(N=16, V=8, d=16, L=1),
        'Static':   StaticCtrl(N=16, V=8, d=16, L=1),
    }
    for nm, mdl in models.items():
        mr = mdl.train(ctd, 15, verbose=False)
        print(f"    {nm:12s}: final={mr['accuracy'][-1]:.4f}  best={max(mr['accuracy']):.4f}")

    # --- Multi-task ---
    print("\n[9] Multi-Task Benchmark...")
    tasks = {
        'copy':          lambda: gen_copy(50, 4, 10),
        'pattern':       lambda: gen_pattern(50, 3),
        'arithmetic':    lambda: gen_arith(50),
        'context':       lambda: gen_ctx(50),
        'long_range':    lambda: gen_longrange(50, 6),
        'rep_sensitive': lambda: gen_repsens(50),
        'contradiction': lambda: gen_contradiction(50),
    }
    for tn, tf in tasks.items():
        mdl = TCAIOmega(N=16, V=16, d=16, L=1)
        data = tf()
        mr = mdl.train(data, 20, verbose=False)
        print(f"    {tn:16s}: final={mr['accuracy'][-1]:.4f}  "
              f"best={max(mr['accuracy']):.4f}  "
              f"upd={sum(mr['updates'])}")

    print("\n" + "=" * 65)
    print("  TCAIOmega -- COMPLETE")
    print("=" * 65)
    print("\n  HONEST ASSESSMENT:")
    print("  - Research prototype. Not a production system.")
    print("  - Performance on toy tasks != general capability.")
    print("  - No claim of superiority over existing architectures.")
    print("  - Primary contribution: concrete specification of")
    print("    gradient-free discrete causal learning with deep")
    print("    counterfactual evaluation through state dynamics.")

if __name__ == '__main__':
    main()
