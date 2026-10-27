from lumenmax import LumenMAX, tproj
import numpy as np

# A clean discrete Motif Table for discrete ternary sequence learning
class MotifTable:
    def __init__(self, cap: int = 2048):
        self.cap = cap
        self.motifs = {} # (ctx_sig, token_id) -> (target_id, strength)

    def reinforce(self, ctx_sig: int, tok_id: int, target_id: int):
        key = (ctx_sig, tok_id)
        if key in self.motifs:
            curr_target, strength = self.motifs[key]
            if curr_target == target_id:
                self.motifs[key] = (target_id, min(10, strength + 1))
            else:
                if strength <= 1:
                    self.motifs[key] = (target_id, 1)
                else:
                    self.motifs[key] = (curr_target, strength - 1)
        else:
            if len(self.motifs) >= self.cap:
                # Evict lowest strength
                min_k = min(self.motifs.keys(), key=lambda k: self.motifs[k][1])
                del self.motifs[min_k]
            self.motifs[key] = (target_id, 1)

    def bonus(self, ctx_sig: int, tok_id: int) -> dict:
        key = (ctx_sig, tok_id)
        if key in self.motifs:
            tgt, strength = self.motifs[key]
            if strength >= 1:
                return {tgt: strength * 4}
        return {}

m = LumenMAX(seed=42)
motifs = MotifTable()

seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
    (["hello", "<sep>", "hello", "!", "how", "can", "i", "help", "you", "today", "?", "<eos>"], "chat"),
]

def ctx_sig(prompt_ctx, state):
    # Discrete 32-bit signature from prompt context and state
    p_bits = ((prompt_ctx[:16] + 1).astype(np.uint32))
    s_bits = ((state[:16] + 1).astype(np.uint32))
    val = 0
    for i in range(16):
        val ^= (int(p_bits[i]) << (i * 2)) ^ (int(s_bits[i]) << ((15 - i) * 2))
    return val

# Train with motifs + W_out
for ep in range(5):
    for toks, dom in seqs:
        m.reset_state()
        sep_idx = toks.index(m.sep_token)
        prompt_toks = toks[:sep_idx]
        answer_toks = toks[sep_idx + 1:]

        p_accum = np.zeros(m.N, dtype=np.int32)
        for ptok in prompt_toks:
            _, px = m.symbols.encode(ptok, domain=dom)
            p_accum += px.astype(np.int32)
            m.forward_step(ptok, domain=dom, is_generating_answer=False)
        m.prompt_context = tproj(p_accum, theta=1)

        # Train <sep> -> answer[0]
        sig = ctx_sig(m.prompt_context, m.state)
        sep_id = m.symbols.tok2id[m.sep_token]
        tgt0_id = m.symbols.register(answer_toks[0], domain=dom)
        motifs.reinforce(sig, sep_id, tgt0_id)
        m.train_step(m.sep_token, answer_toks[0], domain=dom, keep_state=True)

        for i in range(len(answer_toks) - 1):
            curr_id = m.symbols.register(answer_toks[i], domain=dom)
            next_id = m.symbols.register(answer_toks[i + 1], domain=dom)
            sig_i = ctx_sig(m.prompt_context, m.state)
            motifs.reinforce(sig_i, curr_id, next_id)
            m.train_step(answer_toks[i], answer_toks[i + 1], domain=dom, keep_state=True)

print("Training finished. Testing generation with motif-assisted scoring:")

# Test generation
for p in ["how are you", "who are you", "hello"]:
    tokens = m._tokenize(p)
    m.reset_state()
    p_accum = np.zeros(m.N, dtype=np.int32)
    for tok in tokens:
        _, px = m.symbols.encode(tok)
        p_accum += px.astype(np.int32)
        m.forward_step(tok, is_generating_answer=False)
    m.prompt_context = tproj(p_accum, theta=1)

    # Transition on <sep>
    m.forward_step(m.sep_token, is_generating_answer=True)
    sig = ctx_sig(m.prompt_context, m.state)
    sep_id = m.symbols.tok2id[m.sep_token]
    m_bon = motifs.bonus(sig, sep_id)

    # Score vocab
    s = m._readout_state()
    scores = m.W_out.W.astype(np.int32) @ s.astype(np.int32)
    for tid, b in m_bon.items():
        if tid < len(scores):
            scores[tid] += b

    pred_id = int(np.argmax(scores))
    pred_tok = m.symbols.id2tok.get(pred_id, "<eos>")

    gen = [pred_tok]
    curr = pred_tok
    for step in range(15):
        if curr == "<eos>":
            break
        curr_id = m.symbols.tok2id.get(curr, 0)
        sig = ctx_sig(m.prompt_context, m.state)
        m_bon = motifs.bonus(sig, curr_id)

        m.forward_step(curr, is_generating_answer=True)
        s = m._readout_state()
        scores = m.W_out.W.astype(np.int32) @ s.astype(np.int32)
        for tid, b in m_bon.items():
            if tid < len(scores):
                scores[tid] += b

        # mask prompt words and immediate repeats
        if curr_id < len(scores):
            scores[curr_id] = -99999

        next_id = int(np.argmax(scores))
        next_tok = m.symbols.id2tok.get(next_id, "<eos>")
        if next_tok == "<eos>":
            break
        gen.append(next_tok)
        curr = next_tok

    print(f"'{p}' -> '{' '.join(gen)}'")
