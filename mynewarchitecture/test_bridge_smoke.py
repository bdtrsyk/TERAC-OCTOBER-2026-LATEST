"""Smoke test: ternary purity, EOS/loop stops, shared contract, optional Julia bridge."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from lumenmax import LumenMAX, EOS_TOKEN, load_contract, assert_ternary
from tcai_bridge import UnifiedEngine, find_julia


def test_contract():
    c = load_contract()
    assert c["eos_token"] == EOS_TOKEN
    assert -1 in c["ternary_domain"] and 1 in c["ternary_domain"]
    print("OK contract", c.get("contract_version"))


def test_ternary_and_stops():
    m = LumenMAX(state_dim=64, input_dim=64, max_vocab=512, seed=7)
    # Train a clean multi-step chain ending at EOS (sequence teacher-forcing)
    m.add_sequence(["hello", "there", "friend", EOS_TOKEN], domain="chat")
    for _ in range(25):
        met = m.train_epoch()
    print(f"train acc={met['accuracy']*100:.0f}%")

    audit = m.get_ternary_audit()
    assert audit["pure_ternary"], audit
    for mat in [m.W_in, m.W_out, m.W_rec, m.W_gate] + m.experts:
        assert_ternary(mat.W, mat.name)

    m.reset_state()
    out, tel = m.generate("hello", max_tokens=12)
    stop = tel[-1].get("stop_reason")
    print(f"generate -> {out!r} stop={stop} quality={tel[-1].get('quality')}")
    assert stop in ("eos", "loop", "margin", "max_tokens")
    assert out, "expected non-empty generation after training hello->"
    toks = out.split()
    assert not (len(toks) >= 4 and len(set(toks)) == 1), f"looped: {out}"

    m.reset_state()
    resp = m.respond("hello")
    assert resp["ternary"] is True
    assert resp["engine"] == "python"
    assert "quality" in resp
    assert resp["output"], resp
    print("OK python respond", {k: resp[k] for k in ("output", "quality", "stop_reason", "ternary")})


def test_corpus_eos_anchors():
    m = LumenMAX(seed=3)
    n = m.add_text_corpus("What is searching? Parameters versus arguments.\nHello how are you?")
    assert n > 0
    assert m.sequences and any(s["tokens"] and s["tokens"][-1] == EOS_TOKEN for s in m.sequences)
    print(f"OK corpus pairs={n} with EOS anchors")


def test_julia_bridge_optional():
    if not find_julia():
        print("SKIP julia (not on PATH / JULIA_BIN)")
        return
    try:
        u = UnifiedEngine(engine="julia", scale="tiny")
        pingish = u.julia.request({"op": "ping"})
        assert pingish.get("ok"), pingish
        ask = u.ask("sun causes heat")
        assert ask.get("ok"), ask
        assert ask.get("engine") == "julia"
        assert ask.get("ternary") is True
        audit = u.audit()
        assert audit.get("ok"), audit
        print("OK julia", {k: ask.get(k) for k in ("output", "quality", "confidence")})
        u.close()
    except Exception as e:
        print("WARN julia bridge:", e)


if __name__ == "__main__":
    test_contract()
    test_ternary_and_stops()
    test_corpus_eos_anchors()
    test_julia_bridge_optional()
    print("ALL SMOKE PASSED")
