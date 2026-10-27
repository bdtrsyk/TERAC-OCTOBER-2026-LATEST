"""
tcai_bridge.py — Unified Julia + Python ecosystem boundary.

One ecosystem, two owners (do not duplicate full stacks):
  - Python keeps: LumenMAX sequence substrate, discrete credit, PyQt GUI,
    dataset studio, file IO / desktop packaging.
  - Julia keeps: TernaryCognitiveAI intentional/cognitive layer (concepts,
    hypotheses, richer quality taxonomy / reasoning), performance-sensitive
    formal cognitive ops.

Both speak the same JSON response shape (shared/contract.json):

  { ok, engine, output, quality, confidence, stop_reason, ternary, reasons }

Backends:
  - python  : in-process LumenMAX (sequence substrate + UI path)
  - julia   : subprocess JSON-lines into TernaryCognitiveAI (cognitive reasoner)
  - auto    : prefer Julia if available, else Python
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lumenmax import LumenMAX, load_contract, EOS_TOKEN

ROOT = Path(__file__).resolve().parent
CONTRACT = load_contract()


def find_julia() -> Optional[str]:
    env = os.environ.get("JULIA_BIN") or os.environ.get("JULIA")
    if env and Path(env).is_file():
        return env
    which = shutil.which("julia")
    if which:
        return which
    # Common Windows install locations
    local = Path.home() / "AppData" / "Local" / "Programs"
    if local.is_dir():
        candidates = []
        for p in local.glob("Julia-*/bin/julia.exe"):
            if not p.is_file():
                continue
            m = re.match(r"Julia-(\d+(?:\.\d+)*)", p.parent.parent.name)
            key = tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
            candidates.append((key, str(p)))
        if candidates:
            return max(candidates)[1]
    return None


class JuliaBridge:
    """Persistent JSON-lines session to Julia TernaryCognitiveAI."""

    def __init__(self, scale: str = "small", julia_bin: Optional[str] = None):
        self.scale = scale
        self.julia_bin = julia_bin or find_julia()
        if not self.julia_bin:
            raise FileNotFoundError(
                "Julia not found. Set JULIA_BIN or install Julia, or use engine='python'."
            )
        self.proc: Optional[subprocess.Popen] = None
        self._lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._stderr_buf: List[str] = []
        self._desynced = False
        self._restarting = False
        self._start()

    def _start(self) -> None:
        cmd = [
            self.julia_bin,
            f"--project={ROOT}",
            str(ROOT / "main.jl"),
            "--bridge",
            "--scale",
            self.scale,
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        # Dedicated reader threads: a blocking readline() would defeat any
        # timeout and can hang the caller forever if Julia stalls or dies.
        self._reader = threading.Thread(target=self._pump_stdout, daemon=True)
        self._reader.start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        ready = self._readline(timeout=120.0)
        if not ready or not ready.get("ok"):
            err = "".join(self._stderr_buf[-20:])
            self.close()
            raise RuntimeError(f"Julia bridge failed to start: {ready} {err}")

    def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        for line in self.proc.stderr:
            self._stderr_buf.append(line)
            if len(self._stderr_buf) > 200:
                del self._stderr_buf[:100]

    def _readline(self, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                return None
            if line is None:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Non-protocol stdout noise must not kill the session.
                continue

    def request(self, payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        # A timed-out request leaves an unknown response in flight; the only
        # safe recovery for a strict request/response protocol is a restart.
        if self._desynced and not self._restarting:
            try:
                self._restart()
            except Exception as e:
                return {"ok": False, "engine": "julia", "error": f"bridge restart failed: {e}", "ternary": True}
        if not self.proc or not self.proc.stdin or self.proc.poll() is not None:
            if self._restarting:
                return {"ok": False, "engine": "julia", "error": "bridge not running", "ternary": True}
            try:
                self._restart()
            except Exception as e:
                return {"ok": False, "engine": "julia", "error": f"bridge restart failed: {e}", "ternary": True}
        try:
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._desynced = True
            return {"ok": False, "engine": "julia", "error": f"bridge write failed: {e}", "ternary": True}
        resp = self._readline(timeout=timeout)
        if resp is None:
            self._desynced = True
            return {"ok": False, "engine": "julia", "error": "no response / timeout", "ternary": True}
        return resp

    def _restart(self) -> None:
        if self._restarting:
            return
        self._restarting = True
        try:
            self.close()
            self._lines = queue.Queue()
            self._stderr_buf = []
            self._desynced = False
            self._start()
        finally:
            self._restarting = False

    def ask(self, text: str, expected: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"op": "ask", "text": text}
        if expected is not None:
            payload["expected"] = expected
        return self.request(payload)

    def teach(self, text: str, expected: str) -> Dict[str, Any]:
        return self.request({"op": "teach", "text": text, "expected": expected})

    def audit(self) -> Dict[str, Any]:
        return self.request({"op": "audit"})

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            # Don't let close() route through request()'s desync restart path:
            # that would recurse back into close() forever.
            self._desynced = False
            try:
                self.request({"op": "shutdown"}, timeout=5.0)
            except Exception:
                pass
            try:
                self.proc.stdin and self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=5.0)
            except Exception:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=5.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class UnifiedEngine:
    """Unified hybrid cognitive architecture: Julia reasoner + Python sequence substrate together."""

    def __init__(self, engine: str = "unified", scale: str = "small", **lumen_kwargs):
        if engine not in ("unified", "auto", "julia", "python"):
            raise ValueError("engine must be 'unified', 'auto', 'python', or 'julia'")
        self.requested = engine
        self.scale = scale
        self.tritier_adapter: Optional[Any] = None
        self.credit_engine: Optional[Any] = None
        use_tritier = lumen_kwargs.pop("tritier", False)
        self.python = LumenMAX(**lumen_kwargs)
        self.julia: Optional[JuliaBridge] = None
        self.active = "python"
        self._fallback_reason: Optional[str] = None

        # Automatically ground Python substrate with conversational curriculum & dataset if present
        self.python.seed_conversational_curriculum()
        # Resolve relative to this file, not the caller's CWD: run.bat and the
        # GUI can be launched from anywhere and previously missed dataset.txt.
        ds_file = ROOT / "dataset.txt"
        if ds_file.is_file():
            try:
                self.python.ingest_dataset_text(ds_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if engine in ("unified", "auto", "julia"):
            try:
                self.julia = JuliaBridge(scale=scale)
                if engine == "julia":
                    self.active = "julia"
                else:
                    self.active = "unified"  # Julia and Python TOGETHER
            except Exception as e:
                if engine == "julia":
                    raise
                self.julia = None
                self.active = "python"
                self._fallback_reason = str(e)
        elif engine == "python":
            self.active = "python"
        if use_tritier:
            self.enable_tritier()

    def contract(self) -> Dict[str, Any]:
        return CONTRACT

    def set_engine(self, engine: str) -> str:
        if engine in ("unified", "auto"):
            if self.julia is None:
                self.julia = JuliaBridge(scale=self.scale)
            self.active = "unified" if self.julia is not None else "python"
        elif engine == "julia":
            if self.julia is None:
                self.julia = JuliaBridge(scale=self.scale)
            self.active = "julia"
        elif engine == "python":
            self.active = "python"
        else:
            raise ValueError("engine must be 'unified', 'auto', 'python', or 'julia'")
        return self.active
    def ask(self, text: str, expected: Optional[str] = None, max_tokens: int = 36) -> Dict[str, Any]:
        if self.active == "julia" and self.julia is not None:
            return self.julia.ask(text, expected=expected)

        if self.active == "unified" and self.julia is not None:
            # 1. Julia cognitive reasoner analyzes intent, relations, hypotheses
            j_resp = self.julia.ask(text, expected=expected)
            if not j_resp.get("ok", True):
                # Julia side failed: degrade to the Python substrate instead of
                # crashing on missing fields.
                resp = self.python.respond(text, max_tokens=max_tokens, expected=expected)
                resp["julia_error"] = j_resp.get("error", "julia request failed")
                return resp
            j_intent = ""
            j_reasons = j_resp.get("reasons")
            for r in j_reasons if isinstance(j_reasons, list) else []:
                if isinstance(r, str) and r.startswith("intent="):
                    j_intent = r.split("=", 1)[1]
                    break
            j_out = j_resp.get("output", "") or ""
            j_qual = j_resp.get("quality", "NEUTRAL")
            try:
                j_conf = float(j_resp.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                j_conf = 0.0

            # 2. Python ternary sequence substrate conditions on prompt & cognitive bias
            cog_bias = {
                "intent": j_intent,
                "concepts": j_resp.get("concepts", 0),
                "quality": j_qual,
                "confidence": j_conf,
                "output": j_out,
            }
            p_text, p_tel = self.python.generate(text, max_tokens=max_tokens, cognitive_bias=cog_bias)
            p_eval = self.python.evaluate_output(p_text, expected=expected)
            stop_reason = p_tel[-1].get("stop_reason", "complete") if p_tel else "complete"

            # 3. Choose cohesive output:
            # If Julia deduced a strong formal conclusion, use it
            if j_conf >= 0.45 and j_qual in ("GOOD", "EXCELLENT", "PARTIAL") and not j_out.startswith("Insufficient supported evidence"):
                final_out = j_out
            elif p_text and p_tel and p_tel[-1].get("stop_reason") == "eos" and p_tel[-1].get("grounded", False):
                # Python produced a grounded sequence
                final_out = p_text
            elif j_out:
                final_out = j_out
            else:
                final_out = p_text

            p_reasons = p_eval.get("reasons") if isinstance(p_eval.get("reasons"), list) else []
            combined_reasons = list(dict.fromkeys(
                ([f"julia_intent={j_intent}"] if j_intent else [])
                + p_reasons
                + (j_reasons if isinstance(j_reasons, list) else [])
            ))

            try:
                p_conf = float(p_eval.get("confidence", 0.75))
            except (TypeError, ValueError):
                p_conf = 0.0
            return {
                "ok": True,
                "engine": "unified",
                "output": final_out,
                "quality": p_eval.get("quality", j_qual),
                "confidence": max(p_conf, j_conf),
                "stop_reason": stop_reason,
                "ternary": True,
                "reasons": combined_reasons,
                "telemetry": p_tel,
                "julia": j_resp,
                "python": {
                    "output": p_text,
                    "param_count": self.python.param_count(),
                    "vocab_size": self.python.symbols.vocab_size(),
                },
            }

        return self.python.respond(text, max_tokens=max_tokens, expected=expected)

    def _train_recent_sequences(self, count: int) -> None:
        """Train just-added sequences directly.

        train_epoch() subsamples large domains, so a freshly taught example
        could otherwise never be presented to the model.
        """
        for seq in self.python.sequences[-count:] if count > 0 else []:
            self.python.train_sequence(seq["tokens"], domain=seq.get("domain", "qa"))

    def teach(self, text: str, expected: str) -> Dict[str, Any]:
        if self.active == "julia" and self.julia is not None:
            return self.julia.teach(text, expected)

        if self.active == "unified" and self.julia is not None:
            # Teach both Julia and Python simultaneously
            self.julia.teach(text, expected)
            domain = "chat" if self.python._infer_domain(text) == "chat" else "qa"
            if self.python.add_qa_pair(text, expected, domain=domain) > 0:
                self._train_recent_sequences(1)
            return self.ask(text, expected=expected)

        # Python-only fallback
        in_toks = self.python._tokenize(text)
        ex_toks = self.python._tokenize(expected)
        if in_toks and ex_toks:
            domain = "chat" if self.python._infer_domain(text) == "chat" else "qa"
            if self.python.add_qa_pair(text, expected, domain=domain) > 0:
                self._train_recent_sequences(1)
        return self.python.respond(text, expected=expected)

    def audit(self) -> Dict[str, Any]:
        p_audit = self.python.get_ternary_audit()
        j_audit = self.julia.audit() if self.julia is not None else None
        pure = p_audit.get("pure_ternary", True) and (j_audit.get("pure_ternary", True) if j_audit else True)
        return {
            "ok": True,
            "engine": self.active,
            "ternary": True,
            "pure_ternary": pure,
            "total_synapses": p_audit.get("total_synapses", 0),
            "contract_version": p_audit.get("contract_version", 1),
            "eos_token": EOS_TOKEN,
            "julia_audit": j_audit,
            "python_audit": p_audit,
        }

    def enable_tritier(self, store_root: str = "tritier_store", native_gpu: bool = True):
        try:
            from tritier_adapter import TriTierModelAdapter, SuperchargedCreditEngine
            self.tritier_adapter = TriTierModelAdapter(
                model=self.python,
                store_root=store_root,
                model_id="lumenmax-unified",
                native_gpu=native_gpu,
            )
            self.credit_engine = SuperchargedCreditEngine(self.python)
            return self.tritier_adapter
        except Exception as e:
            print(f"[WARN] Failed to enable TriTier: {e}")
            return None

    def close(self) -> None:
        if self.tritier_adapter is not None:
            try:
                self.tritier_adapter.close()
            except Exception:
                pass
            self.tritier_adapter = None
        if self.julia is not None:
            self.julia.close()
            self.julia = None


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    engine = "auto"
    if "--python" in argv:
        engine = "python"
    if "--julia" in argv:
        engine = "julia"
    # Drop engine flags before dispatch; otherwise "--python ask hello" was
    # treated as flag-only and fell into the interactive REPL.
    args = [a for a in argv if a not in ("--python", "--julia")]

    u = UnifiedEngine(engine=engine)
    print(json.dumps({"ok": True, "active": u.active, "contract_version": CONTRACT.get("contract_version")}, indent=2))
    if not args:
        # interactive mini REPL
        print(f"UnifiedEngine active={u.active}  (commands: ask <text> | teach <in> | <out> | audit | engine python|julia | quit)")
        while True:
            try:
                line = input("tcai> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line in ("quit", "exit", ":q"):
                break
            if line == "audit":
                print(json.dumps(u.audit(), indent=2))
                continue
            if line.startswith("engine "):
                print(u.set_engine(line.split(None, 1)[1].strip()))
                continue
            if line.startswith("teach "):
                body = line[6:]
                if "|" not in body:
                    print("usage: teach <input> | <expected>")
                    continue
                left, right = body.split("|", 1)
                print(json.dumps(u.teach(left.strip(), right.strip()), indent=2))
                continue
            if line.startswith("ask "):
                print(json.dumps(u.ask(line[4:].strip()), indent=2))
                continue
            print(json.dumps(u.ask(line), indent=2))
        u.close()
        return 0

    # one-shot: tcai_bridge.py ask "hello"
    if args[0] == "ask" and len(args) >= 2:
        print(json.dumps(u.ask(args[1]), indent=2))
        u.close()
        return 0
    if args[0] == "teach" and len(args) >= 3:
        print(json.dumps(u.teach(args[1], args[2]), indent=2))
        u.close()
        return 0
    if args[0] == "audit":
        print(json.dumps(u.audit(), indent=2))
        u.close()
        return 0
    print("usage: python tcai_bridge.py [--python|--julia] [ask TEXT|teach IN OUT|audit]")
    u.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
