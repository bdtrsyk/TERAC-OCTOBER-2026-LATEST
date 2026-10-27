"""
LumenMAX: PyQt6 Cognitive Intelligence Desktop Platform
=========================================================
100% Native Desktop GUI (PyQt6). Zero Web Server. Zero Pregiven Tasks.
Custom Discrete Ternary Cognitive Organism: W in {-1, 0, +1}.

Features:
- Tab 1: 💬 Dynamic Cognitive Chatting (pure state dynamics, zero hardcoded Q&A)
- Tab 2: 🧠 Custom Dataset Studio & Discrete Training Loop
  * Add custom transition pairs (e.g. "input -> target")
  * Ingest raw text/code corpus (auto-tokenized into sequential pairs)
  * Load/Save dataset files (.txt, .csv, .json)
  * Live evaluation table of all dataset pairs
  * High-performance discrete reward/punishment credit assignment loop
  * Synapse parameter accountability and competitive group telemetry
"""

import sys
import time
import json
import re
import html
from typing import List, Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QLineEdit, QPushButton, QProgressBar, QTableWidget,
    QTableWidgetItem, QSpinBox, QComboBox, QGroupBox, QSplitter, QHeaderView,
    QMessageBox, QFrame, QFileDialog, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from lumenmax import LumenMAX
from tcai_bridge import UnifiedEngine, find_julia

# ============================================================
# BACKGROUND WORKER THREADS
# ============================================================

class ChatWorker(QThread):
    step_ready = pyqtSignal(dict)
    finished = pyqtSignal(dict)

    def __init__(self, engine: UnifiedEngine, prompt: str, max_tokens: int = 48):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self):
        try:
            self.engine.python.reset_state()
            result = self.engine.ask(self.prompt, max_tokens=self.max_tokens)
            for t in result.get("telemetry") or []:
                self.step_ready.emit(t)
                time.sleep(0.015)
        except Exception as e:
            # Never leave the send button disabled on an exception.
            result = {
                "ok": False,
                "engine": "error",
                "output": f"[error] {e}",
                "quality": "WRONG",
                "stop_reason": "error",
                "confidence": 0.0,
                "reasons": [str(e)],
            }
        self.finished.emit(result)


class EvaluationWorker(QThread):
    """Evaluate dataset pairs off the UI thread.

    Evaluating the full dataset synchronously froze the window for minutes
    once dataset.txt was auto-ingested (8k+ items)."""
    progress = pyqtSignal(int, int)
    complete = pyqtSignal(list, int, int)  # results, correct, total

    def __init__(self, model: LumenMAX, max_items: int = 300):
        super().__init__()
        self.model = model
        self.max_items = max_items

    def run(self):
        items = list(self.model.dataset)
        # Always include everything when the dataset is small; otherwise cap
        # so the UI stays responsive (the table only shows what was evaluated).
        if len(items) > self.max_items:
            step = max(1, len(items) // self.max_items)
            items = items[::step][: self.max_items]
        results = []
        correct = 0
        try:
            for idx, ex in enumerate(items):
                self.model.reset_state()
                pred = ""
                try:
                    if ex.get("type") == "pair":
                        info = self.model.forward_step(ex['input'])
                        pred = info['pred_token']
                        ok = (pred.lower() == ex['target'].lower())
                    else:
                        pred, _ = self.model.generate(
                            ex['input'], max_tokens=len(ex['target'].split()) + 4
                        )
                        c_tgt = self.model._canonical(ex['target'])
                        c_pred = self.model._canonical(pred)
                        tgt_words = set(c_tgt.split())
                        pred_words = set(c_pred.split())
                        overlap = len(tgt_words & pred_words)
                        ok = (c_pred == c_tgt or c_tgt in c_pred
                              or overlap >= max(1, len(tgt_words) // 2))
                except Exception:
                    ok = False
                if ok:
                    correct += 1
                results.append((ex, pred, ok))
                self.progress.emit(idx + 1, len(items))
        finally:
            self.complete.emit(results, correct, len(items))


class TrainingWorker(QThread):
    epoch_progress = pyqtSignal(int, int, float, int, int) # epoch, total, acc, rewards, punishments
    training_complete = pyqtSignal(dict)

    def __init__(self, engine: UnifiedEngine, epochs: int):
        super().__init__()
        self.engine = engine
        self.model = engine.python
        self.epochs = epochs
        self.is_running = True

    def run(self):
        t0 = time.time()
        final_acc = 0.0
        tot_rewards = 0
        tot_punish = 0

        for ep in range(self.epochs):
            if not self.is_running:
                break
            met = self.model.train_epoch()
            final_acc = met['accuracy']
            tot_rewards += met.get('rewards', 0)
            tot_punish += met.get('punishments', 0)
            self.epoch_progress.emit(ep + 1, self.epochs, final_acc, tot_rewards, tot_punish)
            time.sleep(0.01)

        # In unified mode, synchronize key Q/A concepts into Julia cognitive agent
        if self.engine.julia is not None:
            try:
                for ex in self.model.dataset[:30]:
                    if ex.get("type") == "qa":
                        self.engine.julia.teach(ex["input"], ex["target"])
            except Exception:
                pass

        elapsed = time.time() - t0
        self.training_complete.emit({
            'epochs': self.epochs,
            'final_acc': final_acc,
            'elapsed': elapsed,
            'rewards': tot_rewards,
            'punishments': tot_punish
        })

    def stop(self):
        self.is_running = False


# ============================================================
# MAIN GUI WINDOW
# ============================================================

class LumenMAXGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LumenMAX - Unified Julia + Python Ternary Cognitive Platform")
        self.resize(1200, 820)

        # Unified hybrid architecture: Julia cognitive reasoner + Python sequence substrate TOGETHER
        self.engine = UnifiedEngine(engine="unified", state_dim=64, input_dim=64, max_vocab=1024, seed=42)
        self.model = self.engine.python
        self.model.seed_conversational_curriculum()

        self._init_ui()
        self._apply_dark_theme()

    def _xpu_status_text(self) -> str:
        accel = self.model.accel
        mem = accel.memory_info()
        alloc_gb = mem.get('allocated_mb', 0.0) / 1024.0
        reserved_gb = mem.get('reserved_mb', 0.0) / 1024.0
        if accel.is_gpu:
            vram_text = f"VRAM Alloc:{alloc_gb:.2f}GB"
            if reserved_gb > 0:
                vram_text += f" Res:{reserved_gb:.2f}GB"
            return (f"⚡ {accel.backend.upper()}: {accel.name} | {vram_text} | "
                    f"Pure Autoregressive W ∈ {{-1,0,+1}}")
        return f"⚡ CPU (numpy) | Pure Autoregressive W ∈ {{-1,0,+1}}"

    def _xpu_telemetry_text(self) -> str:
        accel = self.model.accel
        mem = accel.memory_info()
        alloc_gb = mem.get('allocated_mb', 0.0) / 1024.0
        reserved_gb = mem.get('reserved_mb', 0.0) / 1024.0
        if accel.is_gpu:
            return (f"Hardware: {accel.backend.upper()} | {accel.name}\n"
                    f"VRAM Allocated: {alloc_gb:.3f} GB\n"
                    f"VRAM Reserved:  {reserved_gb:.3f} GB\n"
                    f"Backend: {accel.backend}")
        return "Hardware: CPU (numpy)\nBackend: cpu"

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Top Header Banner: Unified Hybrid Cognitive Organism
        header = QFrame()
        header.setObjectName("HeaderFrame")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(15, 10, 15, 10)

        title_label = QLabel("⚡ LumenMAX + TernaryCognitiveAI")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))

        julia_ok = self.engine.julia is not None
        j_badge = "Connected" if julia_ok else "In-Process Fallback"
        self.status_label = QLabel(
            f"{self._xpu_status_text()} | "
            f"{self.model.param_count():,} Synapses | Julia: {j_badge}"
        )
        self.status_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        main_layout.addWidget(header)

        # Two Main Menus (Tabs)
        self.tabs = QTabWidget()
        self.tabs.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))

        # Menu 1: Chatting
        self.chat_tab = QWidget()
        self._setup_chat_tab()
        self.tabs.addTab(self.chat_tab, "💬 1. Dynamic Cognitive Chatting")

        # Menu 2: Training & Custom Datasets
        self.train_tab = QWidget()
        self._setup_train_tab()
        self.tabs.addTab(self.train_tab, "🧠 2. Custom Dataset Studio & Training")

        main_layout.addWidget(self.tabs)

    # ------------------------------------------------------------
    # MENU 1: CHATTING & REASONING (ZERO HARDCODED Q&A)
    # ------------------------------------------------------------
    def _setup_chat_tab(self):
        layout = QHBoxLayout(self.chat_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Live Chat Window
        chat_panel = QWidget()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(0, 0, 0, 0)

        banner = QLabel(
            "Fully ternary W∈{-1,0,+1}. Decode stops on <eos>, loop ban, or weak evidence margin. "
            "Ecosystem: Python owns LumenMAX sequence + GUI/dataset; Julia owns TernaryCognitiveAI "
            "cognitive reasoner — one contract (shared/contract.json), no duplicated stacks."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet("color: #89b4fa; padding: 6px; background: #181825; border-radius: 4px;")
        chat_layout.addWidget(banner)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFont(QFont("Consolas", 10))
        chat_layout.addWidget(self.chat_history)

        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter prompt (e.g. 'hello', 'def x', 'cat', or custom tokens)...")
        self.input_field.setFont(QFont("Consolas", 11))
        self.input_field.returnPressed.connect(self._send_chat_message)

        self.send_btn = QPushButton("Reason & Send")
        self.send_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.send_btn.clicked.connect(self._send_chat_message)

        self.clear_chat_btn = QPushButton("Reset State")
        self.clear_chat_btn.clicked.connect(self._reset_chat_state)

        input_row.addWidget(self.input_field, stretch=4)
        input_row.addWidget(self.send_btn)
        input_row.addWidget(self.clear_chat_btn)
        chat_layout.addLayout(input_row)

        splitter.addWidget(chat_panel)

        # Right: Telemetry Panel
        telemetry_panel = QGroupBox("🔍 Real-Time Cognitive Telemetry")
        telemetry_layout = QVBoxLayout(telemetry_panel)

        self.expert_label = QLabel("Active Expert: None")
        self.expert_label.setStyleSheet("color: #f9e2af; font-size: 13px; font-weight: bold;")

        self.state_norm_label = QLabel("Ternary State Norm: 0 / 64")
        self.loop_label = QLabel("Repetition Loop Status: Clear")
        self.stop_label = QLabel("Stop Reason: —")
        self.quality_label = QLabel("Quality: —")
        self.gate_label = QLabel("3-Way Gate Activity: Standby")
        self.vocab_label = QLabel("Active Registered Vocabulary: 0 tokens")

        self.xpu_label = QLabel(self._xpu_telemetry_text())
        self.xpu_label.setStyleSheet("color: #89dceb; font-size: 11px;")

        telemetry_layout.addWidget(self.expert_label)
        telemetry_layout.addWidget(self.state_norm_label)
        telemetry_layout.addWidget(self.loop_label)
        telemetry_layout.addWidget(self.stop_label)
        telemetry_layout.addWidget(self.quality_label)
        telemetry_layout.addWidget(self.gate_label)
        telemetry_layout.addWidget(self.vocab_label)
        telemetry_layout.addWidget(self.xpu_label)
        telemetry_layout.addStretch()

        telemetry_layout.addWidget(QLabel("Persistent State S_t in {-1, 0, +1}:"))
        self.state_view = QTextEdit()
        self.state_view.setReadOnly(True)
        self.state_view.setMaximumHeight(160)
        self.state_view.setFont(QFont("Consolas", 9))
        telemetry_layout.addWidget(self.state_view)

        splitter.addWidget(telemetry_panel)
        splitter.setSizes([720, 380])
        layout.addWidget(splitter)

    def _send_chat_message(self):
        worker = getattr(self, "chat_worker", None)
        if worker is not None and worker.isRunning():
            return
        text = self.input_field.text().strip()
        if not text:
            return
        self.input_field.clear()

        self.chat_history.append(
            f"<b style='color:#89b4fa;'>User:</b> {html.escape(text)}"
        )
        self.send_btn.setEnabled(False)
        self.chat_history.append(
            "<i style='color:#6c7086;'>[Julia reasoner + Python substrate] joint inference…</i>"
        )

        self.chat_worker = ChatWorker(self.engine, text, max_tokens=48)
        self.chat_worker.step_ready.connect(self._update_telemetry)
        self.chat_worker.finished.connect(self._on_chat_finished)
        self.chat_worker.start()

    def _update_telemetry(self, step_info: dict):
        exp_names = ["0: Code Syntax", "1: Mathematics", "2: Logic", "3: Structured Text"]
        exp_id = step_info.get('expert', 0)
        if isinstance(exp_id, int) and 0 <= exp_id < len(exp_names):
            self.expert_label.setText(f"Active Expert: {exp_names[exp_id]}")

        norm = step_info.get('state_norm', 0)
        self.state_norm_label.setText(f"Ternary State Norm: {norm} / {self.model.N}")

        if step_info.get('loop', False):
            self.loop_label.setText("Repetition Loop: LOOP STOPPED")
            self.loop_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        else:
            self.loop_label.setText("Repetition Loop: Clean trajectory")
            self.loop_label.setStyleSheet("color: #a6e3a1;")

        if 'stop_reason' in step_info:
            self.stop_label.setText(f"Stop Reason: {step_info.get('stop_reason')}")
        if 'quality' in step_info:
            try:
                conf = float(step_info.get('confidence', 0))
            except (TypeError, ValueError):
                conf = 0.0
            self.quality_label.setText(
                f"Quality: {step_info.get('quality')} | conf={conf:.2f}"
            )

        u = step_info.get('gate_update', 0)
        r = step_info.get('gate_retain', 0)
        c = step_info.get('gate_clear', 0)
        self.gate_label.setText(f"3-Way Gate: {u} Update | {r} Retain | {c} Clear")
        self.vocab_label.setText(f"Registered Vocabulary: {self.model.symbols.vocab_size()} tokens")

        s_str = " ".join([f"{v:+d}" for v in self.model.state])
        self.state_view.setText(f"S_t (dim={self.model.N}):\n[{s_str}]")

    def _on_chat_finished(self, result: dict):
        response = result.get("output", "")
        quality = result.get("quality", "?")
        stop = result.get("stop_reason", "?")
        eng = result.get("engine", "unified")
        self.stop_label.setText(f"Stop Reason: {stop}")
        try:
            conf = float(result.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        self.quality_label.setText(f"Quality: {quality} | conf={conf:.2f}")

        j_info = result.get("julia") or {}
        j_intent = ""
        reasons = j_info.get("reasons") if isinstance(j_info, dict) else None
        for r in reasons if isinstance(reasons, list) else []:
            if isinstance(r, str) and r.startswith("intent="):
                j_intent = r
                break
        meta_tag = f"[{eng} | {quality}" + (f" | {j_intent}" if j_intent else "") + f" | stop={stop}]"

        self.chat_history.append(
            f"<b style='color:#a6e3a1;'>Unified Engine:</b> "
            f"<span style='color:#cdd6f4; font-family:Consolas;'>{html.escape(response)}</span> "
            f"<span style='color:#6c7086;'>{html.escape(meta_tag)}</span>\n"
        )
        self.send_btn.setEnabled(True)
        self.input_field.setFocus()

    def _reset_chat_state(self):
        self.model.reset_state()
        self.model.last_output = ""
        self.chat_history.append("<i style='color:#f9e2af;'>[System] State S_t reset to zeros.</i>\n")
        self.state_view.clear()
        self.stop_label.setText("Stop Reason: —")
        self.quality_label.setText("Quality: —")

    # ------------------------------------------------------------
    # MENU 2: CUSTOM DATASET STUDIO & DISCRETE TRAINING
    # ------------------------------------------------------------
    def _setup_train_tab(self):
        layout = QVBoxLayout(self.train_tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top Section: Dataset Ingestion Studio
        ingest_box = QGroupBox("📥 Custom Dataset Ingestion (Zero Pregiven Tasks - Add Your Own Data)")
        ingest_layout = QHBoxLayout(ingest_box)

        # Ingest Sub-panel 1: Transition Pairs
        pairs_box = QFrame()
        pairs_layout = QVBoxLayout(pairs_box)
        pairs_layout.addWidget(QLabel("Option A: Paste Transition Pairs (one per line):"))
        self.pairs_text = QPlainTextEdit()
        self.pairs_text.setPlaceholderText("hello -> world\ncat -> dog\nsun -> moon\nhot -> cold\ndef -> solve\nx -> =")
        self.pairs_text.setMaximumHeight(120)
        pairs_layout.addWidget(self.pairs_text)

        ingest_pairs_btn = QPushButton("➕ Ingest Transition Pairs")
        ingest_pairs_btn.clicked.connect(self._ingest_pairs_text)
        pairs_layout.addWidget(ingest_pairs_btn)
        ingest_layout.addWidget(pairs_box, stretch=1)

        # Ingest Sub-panel 2: Raw Text / Code Corpus
        corpus_box = QFrame()
        corpus_layout = QVBoxLayout(corpus_box)
        corpus_layout.addWidget(QLabel("Option B: Paste Raw Text / Code (Auto-tokenized sequentially):"))
        self.corpus_text = QPlainTextEdit()
        self.corpus_text.setPlaceholderText("Paste any code or text here:\n\ndef calculate(x, y):\n    return x + y\n\nIt will automatically become sequential training transitions (def -> calculate -> ( -> x ...)")
        self.corpus_text.setMaximumHeight(120)
        corpus_layout.addWidget(self.corpus_text)

        ingest_corpus_btn = QPushButton("⚡ Auto-Tokenize into Dataset")
        ingest_corpus_btn.clicked.connect(self._ingest_corpus_text)
        corpus_layout.addWidget(ingest_corpus_btn)
        ingest_layout.addWidget(corpus_box, stretch=1)

        # Ingest Sub-panel 3: File Management
        file_box = QFrame()
        file_layout = QVBoxLayout(file_box)
        file_layout.addWidget(QLabel("Option C: File Actions:"))
        load_file_btn = QPushButton("📂 Load Dataset File (.txt/.json/.csv)")
        load_file_btn.clicked.connect(self._load_dataset_file)
        save_file_btn = QPushButton("💾 Save Dataset to File")
        save_file_btn.clicked.connect(self._save_dataset_file)
        clear_ds_btn = QPushButton("🗑️ Clear Entire Dataset")
        clear_ds_btn.setStyleSheet("color: #f38ba8;")
        clear_ds_btn.clicked.connect(self._clear_dataset)

        file_layout.addWidget(load_file_btn)
        file_layout.addWidget(save_file_btn)
        file_layout.addWidget(clear_ds_btn)
        file_layout.addStretch()
        ingest_layout.addWidget(file_box)

        main_splitter.addWidget(ingest_box)

        # Middle Section: Training Execution & Live Progress
        train_ctrl_box = QGroupBox("🚀 Automatic Reward / Punishment Training Loop")
        train_ctrl_layout = QVBoxLayout(train_ctrl_box)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Epochs:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(15)
        ctrl_row.addWidget(self.epoch_spin)

        self.start_train_btn = QPushButton("⚡ Start Discrete Training")
        self.start_train_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.start_train_btn.setStyleSheet("background-color: #a6e3a1; color: #11111b; padding: 6px 18px;")
        self.start_train_btn.clicked.connect(self._start_training)
        ctrl_row.addWidget(self.start_train_btn)

        self.eval_btn = QPushButton("🔍 Evaluate Accuracy on Dataset")
        self.eval_btn.clicked.connect(self._evaluate_dataset)
        ctrl_row.addWidget(self.eval_btn)

        self.reset_synapses_btn = QPushButton("Re-initialize Synapses")
        self.reset_synapses_btn.clicked.connect(self._reinit_synapses)
        ctrl_row.addWidget(self.reset_synapses_btn)

        self.vram_btn = QPushButton("🎮 Maximize VRAM Pool (Arc XPU)")
        self.vram_btn.setStyleSheet("background-color: #89b4fa; color: #11111b; padding: 4px 12px;")
        self.vram_btn.clicked.connect(self._maximize_vram)
        ctrl_row.addWidget(self.vram_btn)
        ctrl_row.addStretch()

        self.dataset_count_label = QLabel("Dataset Size: 0 pairs")
        self.dataset_count_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.dataset_count_label.setStyleSheet("color: #89b4fa;")
        ctrl_row.addWidget(self.dataset_count_label)

        train_ctrl_layout.addLayout(ctrl_row)

        self.train_progress = QProgressBar()
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(True)
        train_ctrl_layout.addWidget(self.train_progress)

        # Metrics Row
        self.metrics_label = QLabel("Metrics: Standing by. Add training data and click Start.")
        self.metrics_label.setFont(QFont("Consolas", 10))
        self.metrics_label.setStyleSheet("color: #cdd6f4;")
        train_ctrl_layout.addWidget(self.metrics_label)

        main_splitter.addWidget(train_ctrl_box)

        # Bottom Section: Active Dataset Table & Synapse Inspector
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Dataset Table
        table_box = QGroupBox("📋 Active Dataset Pairs & Predictions")
        table_layout = QVBoxLayout(table_box)
        self.dataset_table = QTableWidget(0, 4)
        self.dataset_table.setHorizontalHeaderLabels(["#", "Input Token", "Target Token", "Current Prediction"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.dataset_table)
        bottom_splitter.addWidget(table_box)

        # Right: Synapse Accountability
        syn_box = QGroupBox("🔬 Synaptic Parameter Accountability (Top W_out Synapses)")
        syn_layout = QVBoxLayout(syn_box)
        self.synapse_table = QTableWidget(6, 5)
        self.synapse_table.setHorizontalHeaderLabels(["Synapse [r,c]", "Weight W", "Score", "Success", "Failure"])
        self.synapse_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        syn_layout.addWidget(self.synapse_table)
        bottom_splitter.addWidget(syn_box)

        bottom_splitter.setSizes([600, 500])
        main_splitter.addWidget(bottom_splitter)

        main_splitter.setSizes([200, 140, 360])
        layout.addWidget(main_splitter)

        self._refresh_dataset_ui()
        self._refresh_synapse_table()

    # ------------------------------------------------------------
    # DATASET INGESTION HANDLERS
    # ------------------------------------------------------------
    def _ingest_pairs_text(self):
        text = self.pairs_text.toPlainText().strip()
        if not text:
            return
        # Prefer full-file parser (Q/A blocks + first-arrow pairs; never split on commas)
        added = self.model.ingest_dataset_text(text)
        if added == 0:
            for line in text.splitlines():
                line = line.strip()
                if not line or "->" not in line:
                    continue
                left, right = line.split("->", 1)
                inp, tgt = left.strip(), right.strip()
                if inp and tgt:
                    before = len(self.model.dataset)
                    self.model.add_training_pair(inp, tgt)
                    added += max(0, len(self.model.dataset) - before)
        self.pairs_text.clear()
        self._refresh_dataset_ui()
        QMessageBox.information(
            self,
            "Ingestion Complete",
            f"Successfully added {added} training transition(s). "
            f"Sequences={len(self.model.sequences)}",
        )

    def _ingest_corpus_text(self):
        text = self.corpus_text.toPlainText().strip()
        if not text:
            return
        added = self.model.add_text_corpus(text)
        self.corpus_text.clear()
        self._refresh_dataset_ui()
        QMessageBox.information(
            self,
            "Corpus Tokenized",
            f"Successfully auto-tokenized text into {added} sequential training pair(s)! "
            f"(Q/A files are word-tokenized; multi-word mega-tokens are refused.)",
        )

    def _load_dataset_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Dataset File", "", "Data Files (*.txt *.csv *.json);;All Files (*)")
        if not path:
            return
        try:
            added = 0
            lower = path.lower()
            if lower.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        kind = item.get("type", "pair")
                        if "tokens" in item:
                            added += self.model.add_sequence(
                                list(item["tokens"]),
                                domain=item.get("domain", "general"),
                            )
                        elif "input" not in item or "target" not in item:
                            continue
                        elif kind == "qa":
                            added += self.model.add_qa_pair(
                                item["input"], item["target"],
                                domain=item.get("domain", "qa"),
                            )
                        elif kind == "corpus":
                            added += self.model.add_text_corpus(
                                item.get("target", ""),
                                domain=item.get("domain", "corpus"),
                            )
                        else:
                            before = len(self.model.dataset)
                            self.model.add_training_pair(
                                item["input"], item["target"],
                                domain=item.get("domain", "general"),
                            )
                            added += max(0, len(self.model.dataset) - before)
            elif lower.endswith(".csv"):
                import csv
                with open(path, "r", encoding="utf-8", newline="") as f:
                    for row in csv.reader(f):
                        if len(row) < 2:
                            continue
                        inp, tgt = row[0].strip(), row[1].strip()
                        if inp.lower() in ("input", "prompt", "question") and tgt.lower() in ("target", "output", "answer"):
                            continue
                        if inp and tgt:
                            self.model.add_training_pair(inp, tgt)
                            added += 1
            else:
                with open(path, "r", encoding="utf-8") as f:
                    added = self.model.ingest_dataset_text(f.read())
            self._refresh_dataset_ui()
            QMessageBox.information(
                self,
                "Loaded",
                f"Loaded {added} transition(s) from {path}\n"
                f"Sequences={len(self.model.sequences)} Vocab={self.model.symbols.vocab_size()}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", str(e))

    def _save_dataset_file(self):
        if not self.model.dataset and not self.model.sequences:
            QMessageBox.warning(self, "Empty Dataset", "Dataset is empty, nothing to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Dataset File", "dataset.txt", "Text Files (*.txt);;JSON Files (*.json)")
        if not path:
            return
        try:
            lower = path.lower()
            if lower.endswith(".json"):
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.model.dataset, f, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for ex in self.model.dataset:
                        kind = ex.get("type")
                        if kind == "qa":
                            f.write(f"Q -> {ex['input']}\nA -> {ex['target']}\n")
                        elif kind == "corpus":
                            f.write(f"CORPUS: {ex['target']}\n")
                        else:
                            f.write(f"{ex['input']} -> {ex['target']}\n")
            QMessageBox.information(self, "Saved", f"Saved {len(self.model.dataset)} items to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error Saving File", str(e))

    def _clear_dataset(self):
        self.model.clear_dataset()
        self._refresh_dataset_ui()

    def _refresh_dataset_ui(self):
        n = len(self.model.dataset)
        n_seq = len(self.model.sequences)
        self.dataset_count_label.setText(f"Dataset Size: {n} items ({n_seq} sequences)")
        self.vocab_label.setText(f"Registered Vocabulary: {self.model.symbols.vocab_size()} tokens")

        # Cap table population: 8k+ rows rebuilt on every ingest froze the UI.
        max_rows = 500
        shown = min(n, max_rows)
        self.dataset_table.setRowCount(shown)
        for i in range(shown):
            ex = self.model.dataset[i]
            self.dataset_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.dataset_table.setItem(i, 1, QTableWidgetItem(ex['input']))
            self.dataset_table.setItem(i, 2, QTableWidgetItem(ex['target']))
            self.dataset_table.setItem(i, 3, QTableWidgetItem("-"))
        if n > max_rows:
            self.dataset_count_label.setText(
                f"Dataset Size: {n} items ({n_seq} sequences, showing first {max_rows})"
            )

    def _evaluate_dataset(self):
        worker = getattr(self, "eval_worker", None)
        if worker is not None and worker.isRunning():
            return
        if not self.model.dataset and not self.model.sequences:
            QMessageBox.warning(self, "No Dataset", "Please add some training pairs first.")
            return
        self.eval_btn.setEnabled(False)
        self.metrics_label.setText("Evaluating dataset…")
        self.eval_worker = EvaluationWorker(self.model)
        self.eval_worker.progress.connect(self._on_eval_progress)
        self.eval_worker.complete.connect(self._on_eval_complete)
        self.eval_worker.start()

    def _on_eval_progress(self, done: int, total: int):
        self.metrics_label.setText(f"Evaluating dataset… {done}/{total}")

    def _on_eval_complete(self, results: list, correct: int, total: int):
        for i, (ex, pred, ok) in enumerate(results):
            if i >= self.dataset_table.rowCount():
                break
            disp_pred = (pred[:36] + "...") if len(pred) > 36 else pred
            pred_item = QTableWidgetItem(f"{disp_pred} {'✓' if ok else '✗'}")
            pred_item.setForeground(QColor("#a6e3a1" if ok else "#f38ba8"))
            self.dataset_table.setItem(i, 3, pred_item)
        acc = correct / max(1, total)
        self.metrics_label.setText(
            f"Evaluation Accuracy: {acc*100:.1f}% ({correct}/{total} correct)"
        )
        self.eval_btn.setEnabled(True)
        self._refresh_synapse_table()

    # ------------------------------------------------------------
    # TRAINING HANDLERS
    # ------------------------------------------------------------
    def _start_training(self):
        worker = getattr(self, "train_worker", None)
        if worker is not None and worker.isRunning():
            return
        if not self.model.dataset and not self.model.sequences:
            QMessageBox.warning(self, "Empty Dataset", "Dataset is empty! Paste pairs or text in Option A/B above first.")
            return

        epochs = self.epoch_spin.value()
        self.start_train_btn.setEnabled(False)
        self.eval_btn.setEnabled(False)
        self.train_progress.setValue(0)
        self.metrics_label.setText(f"Training on {len(self.model.dataset)} items ({len(self.model.sequences)} seqs) for {epochs} epochs...")

        self.train_worker = TrainingWorker(self.engine, epochs)
        self.train_worker.epoch_progress.connect(self._on_epoch_progress)
        self.train_worker.training_complete.connect(self._on_training_complete)
        self.train_worker.start()

    def _on_epoch_progress(self, ep: int, total_ep: int, acc: float, rewards: int, punish: int):
        pct = int(100.0 * ep / total_ep)
        self.train_progress.setValue(pct)
        self.metrics_label.setText(
            f"Epoch {ep}/{total_ep} | Accuracy: {acc*100:.1f}% | Rewards: +{rewards} pts | Punishments: -{punish} pts"
        )
        self._refresh_synapse_table()

    def _on_training_complete(self, res: dict):
        self.start_train_btn.setEnabled(True)
        self.eval_btn.setEnabled(True)
        self.train_progress.setValue(100)
        self.metrics_label.setText(
            f"✅ Complete: {res['epochs']} epochs in {res['elapsed']:.2f}s | Final Accuracy: {res['final_acc']*100:.1f}% | Rewards: +{res['rewards']} | Punishments: -{res['punishments']}"
        )
        self._evaluate_dataset()
        self._refresh_synapse_table()

    def _reinit_synapses(self):
        tw = getattr(self, "train_worker", None)
        if tw is not None and tw.isRunning():
            QMessageBox.information(
                self, "Training Running",
                "Finish or stop training before re-initializing synapses.",
            )
            return
        ew = getattr(self, "eval_worker", None)
        if ew is not None and ew.isRunning():
            QMessageBox.information(
                self, "Evaluation Running",
                "Wait for the dataset evaluation to finish first.",
            )
            return
        ds = list(self.model.dataset)
        self.model = LumenMAX(state_dim=64, input_dim=64, max_vocab=1024, seed=42)
        self.engine.python = self.model
        for ex in ds:
            if ex.get("type") == "qa":
                self.model.add_qa_pair(ex["input"], ex["target"], domain=ex.get("domain", "qa"))
            else:
                self.model.add_training_pair(
                    ex["input"], ex["target"], domain=ex.get("domain", "general")
                )
        self.train_progress.setValue(0)
        self.metrics_label.setText("Model synapses re-initialized to balanced ternary baseline.")
        self._refresh_dataset_ui()
        self._refresh_synapse_table()

    def _maximize_vram(self):
        """Allocate GPU VRAM synaptic pools up to hardware limit."""
        try:
            info = self.model.maximize_vram_pool()
            alloc = info.get("allocated_mb", 0.0)
            pools = info.get("pools_count", 0)
            self.status_label.setText(
                f"{self._xpu_status_text()} | "
                f"{self.model.param_count():,} Synapses"
            )
            QMessageBox.information(
                self,
                "VRAM Pool Allocated",
                f"Allocated {pools} memory bank(s)\n"
                f"VRAM Usage: {alloc:.1f} MB\n"
                f"Device: {self.model.accel.name}",
            )
        except Exception as e:
            QMessageBox.warning(self, "VRAM Allocation", f"Could not allocate VRAM:\n{e}")

    def _refresh_synapse_table(self):
        top = self.model.W_out.top_synapses(k=6)
        for i, s in enumerate(top):
            self.synapse_table.setItem(i, 0, QTableWidgetItem(f"[{s['row']}, {s['col']}]"))
            self.synapse_table.setItem(i, 1, QTableWidgetItem(f"{s['value']:+d}"))
            self.synapse_table.setItem(i, 2, QTableWidgetItem(f"{s['score']:+d}"))
            self.synapse_table.setItem(i, 3, QTableWidgetItem(str(s['success'])))
            self.synapse_table.setItem(i, 4, QTableWidgetItem(str(s['failure'])))

    # ------------------------------------------------------------
    # WINDOW LIFECYCLE
    # ------------------------------------------------------------
    def closeEvent(self, event):
        """Stop background workers and shut the Julia bridge down cleanly."""
        for attr in ("chat_worker", "eval_worker", "train_worker"):
            worker = getattr(self, attr, None)
            if worker is not None and worker.isRunning():
                stop = getattr(worker, "stop", None)
                if callable(stop):
                    stop()
                worker.wait(3000)
        try:
            self.engine.close()
        except Exception:
            pass
        super().closeEvent(event)

    def _apply_dark_theme(self):
        style = """
        QMainWindow { background-color: #1e1e2e; }
        QWidget { color: #cdd6f4; background-color: #1e1e2e; font-family: 'Segoe UI', Arial, sans-serif; }
        #HeaderFrame { background-color: #181825; border: 1px solid #313244; border-radius: 6px; }
        QTabWidget::pane { border: 1px solid #313244; background-color: #1e1e2e; border-radius: 6px; }
        QTabBar::tab { background-color: #181825; color: #a6adc8; padding: 8px 18px; margin-right: 4px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
        QTabBar::tab:selected { background-color: #313244; color: #89b4fa; }
        QGroupBox { border: 1px solid #313244; border-radius: 6px; margin-top: 12px; padding-top: 10px; font-weight: bold; color: #89b4fa; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox { background-color: #11111b; border: 1px solid #313244; border-radius: 4px; padding: 6px; color: #cdd6f4; }
        QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; padding: 6px 12px; }
        QPushButton:hover { background-color: #45475a; color: #89b4fa; }
        QPushButton:pressed { background-color: #585b70; }
        QProgressBar { background-color: #11111b; border: 1px solid #313244; border-radius: 4px; text-align: center; color: #cdd6f4; }
        QProgressBar::chunk { background-color: #a6e3a1; border-radius: 3px; }
        QTableWidget { background-color: #11111b; border: 1px solid #313244; gridline-color: #313244; color: #cdd6f4; }
        QHeaderView::section { background-color: #181825; color: #89b4fa; border: 1px solid #313244; padding: 4px; }
        """
        self.setStyleSheet(style)


def main():
    app = QApplication(sys.argv)
    window = LumenMAXGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
