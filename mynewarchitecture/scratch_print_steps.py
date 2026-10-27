from lumenmax import LumenMAX, tproj
import numpy as np

m = LumenMAX(seed=42)
seqs = [
    (["how", "are", "you", "<sep>", "i", "am", "doing", "well", "<eos>"], "qa"),
    (["who", "are", "you", "<sep>", "i", "am", "lumenmax", "<eos>"], "qa"),
]

for ep in range(10):
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

        info0 = m.train_step(m.sep_token, answer_toks[0], domain=dom, keep_state=True)
        if ep == 9:
            print(f"[{' '.join(prompt_toks)}] <sep> -> exp '{answer_toks[0]}', got '{info0['pred_token']}' (correct={info0['correct']})")

        for i in range(len(answer_toks) - 1):
            info = m.train_step(answer_toks[i], answer_toks[i + 1], domain=dom, keep_state=True)
            if ep == 9:
                print(f"[{' '.join(prompt_toks)}] '{answer_toks[i]}' -> exp '{answer_toks[i+1]}', got '{info['pred_token']}' (correct={info['correct']})")
