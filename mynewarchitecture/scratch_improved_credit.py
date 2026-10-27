from lumenmax import LumenMAX, tproj
from typing import Optional, Dict, Any, List
import numpy as np

def create_supercharged_lumenmax():
    m = LumenMAX(seed=42, accelerator="cpu", auto_allocate_vram=False)
    m.seed_conversational_curriculum()

    # Track synaptic stability across all matrices
    m.W_out_stability = np.zeros_like(m.W_out.W, dtype=np.float32)
    m.W_rec_stability = np.zeros_like(m.W_rec.W, dtype=np.float32)
    m.W_in_stability = np.zeros_like(m.W_in.W, dtype=np.float32)

    # Sequence eligibility trace
    m.eligibility = np.zeros(m.N, dtype=np.float32)
    gamma = 0.85
    lam = 0.80

    def enhanced_train_step(
        self,
        input_tok: str,
        target_tok: str,
        domain: str = "general",
        *,
        keep_state: bool = False,
        prev_state: Optional[np.ndarray] = None,
        seq_step: int = 0,
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

        # Update eligibility trace
        self.eligibility = gamma * lam * self.eligibility + s.astype(np.float32)
        trace_nz = np.where(np.abs(self.eligibility) > 0.05)[0]

        # Calculate target score under current state
        tgt_score = int(self.W_out.W[tgt_id].astype(np.int32) @ s.astype(np.int32)) if tgt_id < self.W_out.out_dim else 0
        margin_deficit = max(0, pred_score - tgt_score + 2)

        active_s = np.where(s != 0)[0]
        active_prev = np.where(prev != 0)[0]

        # Adaptive magnitude based on margin deficit and rarity
        reinforce_mag = max(2, 2 + rare // 2 + eos_boost + min(6, margin_deficit))
        suppress_mag = max(2, min(8, 2 + margin_deficit))

        if correct:
            self.W_out.reinforce(tgt_id, s, magnitude=reinforce_mag)
            self.W_out_stability[tgt_id, active_s] += 0.5
            self.router.update(info["expert"], success=True)

            # Reinforce recurrent transitions across all active state dimensions
            if len(active_prev) > 0:
                for r in active_s:
                    self.W_rec.reinforce(int(r), prev, magnitude=1)
                    self.W_rec_stability[int(r), active_prev] += 0.2
        else:
            self.W_out.reinforce(tgt_id, s, magnitude=reinforce_mag)
            if pred_id >= 0 and pred_id != tgt_id:
                # Modulate suppression by stability (anti-forgetting)
                stab = self.W_out_stability[pred_id, active_s]
                effective_suppress = np.maximum(1, (suppress_mag - stab * 0.5)).astype(int)
                for i, r in enumerate(active_s):
                    mag = int(effective_suppress[i])
                    if mag > 0:
                        self.W_out.A[pred_id, r] -= s[r] * mag
                self.W_out._project(pred_id)

            self.router.update(info["expert"], success=False)
            _, x = self.symbols.encode(input_tok, domain=domain)
            active_x = np.where(x != 0)[0]

            # Train input projection for all active state dimensions
            for r in active_s:
                self.W_in.reinforce(int(r), x, magnitude=2)

            # Suppress erroneous recurrent pathway if pred was wrong
            if pred_id >= 0 and pred_id != tgt_id and len(active_prev) > 0:
                for r in active_s:
                    self.W_rec.suppress(int(r), prev, magnitude=1)

            # Memory gate credit: if error on transition, modulate gate
            gate_inp = np.concatenate([self.state, x])
            if is_sep:
                # Clear on sep
                for r in active_s:
                    self.W_gate.suppress(int(r), gate_inp, magnitude=1)
            else:
                # Retain on valid generation
                for r in active_s:
                    self.W_gate.reinforce(int(r), gate_inp, magnitude=1)

        info["correct"] = correct
        info["target_id"] = tgt_id
        info["target_token"] = target_tok
        return info

    def enhanced_train_sequence(self, tokens: List[str], domain: str = "general") -> Dict[str, Any]:
        toks = [t for t in tokens if t]
        if len(toks) < 2:
            return {"correct": 0, "total": 0, "accuracy": 0.0}

        self.reset_state()
        self.eligibility.fill(0)
        correct = 0
        total = 0

        if self.sep_token in toks:
            sep_idx = toks.index(self.sep_token)
            prompt_toks = toks[:sep_idx]
            answer_toks = toks[sep_idx + 1:]
            if not answer_toks:
                return {"correct": 0, "total": 0, "accuracy": 0.0}

            ctx_sig = self.motifs.context_sig(prompt_toks, self.symbols)
            self.current_ctx_sig = ctx_sig

            p_accum = np.zeros(self.N, dtype=np.int32)
            for ptok in prompt_toks:
                _, px = self.symbols.encode(ptok, domain=domain)
                p_accum += px.astype(np.int32)
                self.forward_step(ptok, domain=domain, allow_eos=False, is_generating_answer=False)
            self.prompt_context = tproj(p_accum, theta=1)

            # First transition: <sep> -> answer_toks[0]
            prev = self.state.copy()
            sep_tid = self.symbols.register(self.sep_token, domain="control")
            tgt0_tid = self.symbols.register(answer_toks[0], domain=domain)
            self.motifs.reinforce(ctx_sig, 0, sep_tid, tgt0_tid, amount=16)

            info0 = self.train_step(self.sep_token, answer_toks[0], domain=domain, keep_state=True, prev_state=prev, seq_step=0, is_sep=True)
            total += 1
            if info0["correct"]:
                correct += 1

            for i in range(len(answer_toks) - 1):
                prev = self.state.copy()
                cur_tid = self.symbols.register(answer_toks[i], domain=domain)
                nxt_tid = self.symbols.register(answer_toks[i + 1], domain=domain)
                self.motifs.reinforce(ctx_sig, i + 1, cur_tid, nxt_tid, amount=16)

                info = self.train_step(answer_toks[i], answer_toks[i + 1], domain=domain, keep_state=True, prev_state=prev, seq_step=i + 1, is_sep=False)
                total += 1
                if info["correct"]:
                    correct += 1
        else:
            for i in range(len(toks) - 1):
                prev = self.state.copy()
                info = self.train_step(toks[i], toks[i + 1], domain=domain, keep_state=True, prev_state=prev, seq_step=i)
                total += 1
                if info["correct"]:
                    correct += 1

        return {"correct": correct, "total": total, "accuracy": correct / max(1, total)}

    m.train_step = enhanced_train_step.__get__(m, LumenMAX)
    m.train_sequence = enhanced_train_sequence.__get__(m, LumenMAX)
    return m

if __name__ == "__main__":
    m = create_supercharged_lumenmax()
    for ep in range(10):
        res = m.train_epoch()
        print(f"Epoch {ep+1:2d}: acc={res['accuracy']*100:5.1f}% ({res['correct']}/{res['total']})")

    print("\n--- INFERENCE GENERATION RESULTS ---")
    for prompt in ["hello", "how are you", "who are you", "what can you do"]:
        resp, _ = m.generate(prompt, max_tokens=25)
        print(f"PROMPT: '{prompt}' -> '{resp}'")
