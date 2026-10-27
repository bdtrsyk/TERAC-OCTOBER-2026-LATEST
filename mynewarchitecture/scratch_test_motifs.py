import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lumenmax import LumenMAX, SymbolTable, tproj
import numpy as np

# Test StepMotifTable logic
class StepMotifTable:
    def __init__(self, cap: int = 16384):
        self.cap = cap
        self.table = {}
        self.step_table = {}
        self.trans_table = {}

    def context_sig(self, prompt_tokens, symbols):
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

    def reinforce(self, ctx_sig, step_idx, prev_id, target_id, amount=2):
        if target_id < 0:
            return
        key = (ctx_sig, step_idx, prev_id)
        if key not in self.table:
            if len(self.table) >= self.cap:
                del self.table[next(iter(self.table))]
            self.table[key] = {}
        self.table[key][target_id] = min(60, self.table[key].get(target_id, 0) + amount)

        s_key = (ctx_sig, step_idx)
        if s_key not in self.step_table:
            if len(self.step_table) >= self.cap:
                del self.step_table[next(iter(self.step_table))]
            self.step_table[s_key] = {}
        self.step_table[s_key][target_id] = min(60, self.step_table[s_key].get(target_id, 0) + amount)

        t_key = (ctx_sig, prev_id)
        if t_key not in self.trans_table:
            if len(self.trans_table) >= self.cap:
                del self.trans_table[next(iter(self.trans_table))]
            self.trans_table[t_key] = {}
        self.trans_table[t_key][target_id] = min(60, self.trans_table[t_key].get(target_id, 0) + amount)

    def bonus(self, ctx_sig, step_idx, prev_id, cand_id):
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
        return min(60, b)

m = LumenMAX()
m.motifs = StepMotifTable()
m.seed_conversational_curriculum()

# Train with motifs
for seq in m.sequences:
    toks = seq["tokens"]
    sep_idx = toks.index(m.sep_token)
    p_toks = toks[:sep_idx]
    a_toks = toks[sep_idx + 1:]
    sig = m.motifs.context_sig(p_toks, m.symbols)
    sep_tid = m.symbols.register(m.sep_token)
    tgt0 = m.symbols.register(a_toks[0])
    m.motifs.reinforce(sig, 0, sep_tid, tgt0, amount=10)
    for i in range(len(a_toks) - 1):
        c_id = m.symbols.register(a_toks[i])
        n_id = m.symbols.register(a_toks[i+1])
        m.motifs.reinforce(sig, i + 1, c_id, n_id, amount=10)

for ep in range(15):
    m.train_epoch()

for p in ["hello", "how are you", "who are you"]:
    toks = m._tokenize(p)
    sig = m.motifs.context_sig(toks, m.symbols)
    m.current_ctx_sig = sig
    
    # Pure autoregressive decode
    m.reset_state()
    p_accum = np.zeros(m.N, dtype=np.int32)
    prompt_ids = set()
    for tok in toks:
        tid, x = m.symbols.encode(tok)
        prompt_ids.add(tid)
        p_accum += x.astype(np.int32)
        m.forward_step(tok, is_generating_answer=False)
    m.prompt_context = tproj(p_accum, theta=1)
    
    sep_id = m.symbols.register(m.sep_token)
    eos_id = m.symbols.register(m.eos_token)
    
def custom_score_vocab(self, banned=None, is_generating_answer=True, step_idx=0, prev_id=-1, prefer_domain=None, **kwargs):
    readout = self._readout_state()
    scores = self.W_out.W.astype(np.int32) @ readout.astype(np.int32)
    eos_id = self.symbols.tok2id.get(self.eos_token)
    sep_id = self.symbols.tok2id.get(self.sep_token)
    registered = set(self.symbols.id2tok.keys())
    mask = np.full(len(scores), -10**9, dtype=np.int32)
    for rid in registered:
        if banned and rid in banned:
            continue
        sc = int(scores[rid])
        # Motif bonus
        if hasattr(self, "current_ctx_sig") and self.current_ctx_sig:
            sc += self.motifs.bonus(self.current_ctx_sig, step_idx, prev_id, rid)
        mask[rid] = sc
    pred_id = int(np.argmax(mask))
    return pred_id, int(mask[pred_id]), 0, scores

m._score_vocab = custom_score_vocab.__get__(m, LumenMAX)

for p in ["hello", "how are you", "who are you"]:
    toks = m._tokenize(p)
    sig = m.motifs.context_sig(toks, m.symbols)
    m.current_ctx_sig = sig
    
    m.reset_state()
    p_accum = np.zeros(m.N, dtype=np.int32)
    prompt_ids = set()
    for tok in toks:
        tid, x = m.symbols.encode(tok)
        prompt_ids.add(tid)
        p_accum += x.astype(np.int32)
        m.forward_step(tok, is_generating_answer=False)
    m.prompt_context = tproj(p_accum, theta=1)
    
    sep_id = m.symbols.register(m.sep_token)
    eos_id = m.symbols.register(m.eos_token)
    
    sep_info = m.forward_step(m.sep_token, is_generating_answer=True, banned={sep_id, eos_id})
    current = sep_info["pred_token"]
    
    generated = []
    prev_id = sep_id
    for step_i in range(30):
        if current in (m.eos_token, "<eos>"):
            break
        generated.append(current)
        cid = m.symbols.tok2id.get(current, -1)
        prev_id = cid
        
        info = m.forward_step(current, allow_eos=(len(generated)>=3), is_generating_answer=True)
        # score next token with motif
        pid, _, _, _ = m._score_vocab(step_idx=step_i + 1, prev_id=prev_id, banned={sep_id})
        current = m.symbols.decode(pid)
    print(f"Prompt: '{p}' -> Output: '{' '.join(generated)}'")


