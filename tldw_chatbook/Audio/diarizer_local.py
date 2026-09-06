"""App-side diarizer backend: spawns and talks to the worker over pipes.

No torch is imported here -- SpeechBrain/torch live only in
`diarizer_worker.py`, a separate process. This module speaks a small wire
protocol to it (spec §3.4):

    stdin  (app -> worker):  one JSON control line per command; an "assign"
                             line is immediately followed by exactly ``n``
                             bytes of raw PCM16 (the length-prefix is the
                             ``n`` field on the control line).
    stdout (worker -> app):  one ``{"id": "S1"}`` / ``{"id": null}`` line per
                             assign; ``{"segments": [...]}`` for a diarize.
    stderr (worker -> app):  ``READY`` once, when the ECAPA model is warm.

Crash rule (spec §3.6): on any worker death / broken pipe / read timeout the
backend attempts exactly ONE restart; a second failure marks it permanently
degraded and every later ``assign`` returns ``None`` -- coarse labels for the
rest of the meeting. Best-effort throughout: a worker problem never raises
into the session.

Privacy: only PCM and cluster ids cross the pipe. Transcript text and speaker
names never reach the worker, and nothing here logs PCM, text, names, or
paths -- types and lengths only.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Audio.meeting_session import SpeakerSegment

WORKER_MODULE = "tldw_chatbook.Audio.diarizer_worker"
#: A single assign must reply within this; a slower reply is treated as a
#: read timeout (the window falls back to a coarse label).
ASSIGN_BUDGET_S = 2.0
#: The Stop-pass batch clusters the whole recording -- allow longer than an
#: assign, but bound it: this runs on the stop worker thread, so a wedged
#: worker delays meeting finalize/ingest by exactly this long. 60 s is ample
#: for a meeting-sized recording (final whole-branch review M3); a genuinely
#: hung worker is caught here instead of stalling the stop for 5 minutes.
DIARIZE_BUDGET_S = 60.0
#: First run downloads the ECAPA model; warm-up can take a while.
READY_TIMEOUT_S = 120.0

_SENTINEL = object()  # placed on the reply queue when the worker's stdout EOFs


class SpeechBrainDiarizer:
    """Talks to the SpeechBrain worker subprocess; degrades to coarse on failure."""

    def __init__(
        self,
        max_speakers: int = 8,
        *,
        spawn: Callable[..., Any] = subprocess.Popen,
        assign_budget_s: float = ASSIGN_BUDGET_S,
    ) -> None:
        self._max = max_speakers
        self._spawn = spawn
        self._budget = assign_budget_s
        self._proc: Any | None = None
        self._q: "queue.Queue[Any]" = queue.Queue()
        self._degraded = False
        self._restarted = False
        self._lock = threading.Lock()
        # Spawn eagerly: build_diarizer (Task 6) only constructs this once a
        # meeting with live diarization begins, so warm-up cost is expected.
        if not self._start():
            self._degraded = True

    # ---- process lifecycle ------------------------------------------------
    def _command(self) -> list[str]:
        """The worker argv. Isolated here so the frozen-app case has one seam.

        ponytail: frozen apps (PyInstaller) set ``sys.frozen`` and give a
        ``sys.executable`` that is the app, not a python that understands
        ``-m``. TODO(frozen): ship the worker as a bundled entry point (or a
        multiprocessing spawn via ``freeze_support``) and return that argv
        here. Until then a frozen build simply fails to spawn and degrades to
        coarse labels -- correct, just not diarized.
        """
        if getattr(sys, "frozen", False):
            logger.warning("diarizer: live diarization unsupported in frozen build; coarse labels only")
        return [sys.executable, "-m", WORKER_MODULE]

    def _start(self) -> bool:
        """Spawn the worker and wait for ``READY``. False -> caller degrades."""
        try:
            env = {**os.environ, "TLDW_DIARIZER_MAX_SPEAKERS": str(self._max)}
            self._proc = self._spawn(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("diarizer: worker spawn failed ({})", type(exc).__name__)
            self._proc = None
            return False
        proc = self._proc
        if proc.poll() is not None:
            logger.warning("diarizer: worker exited before READY")
            return False
        if not self._await_ready(proc):
            logger.warning("diarizer: worker never reported READY")
            self._kill(proc)
            return False
        # Fresh reply queue per worker session: a previous (dead) worker's
        # EOF sentinel must never be read as this worker's crash.
        self._q = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(proc, self._q), daemon=True, name="diarizer-stdout").start()
        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True, name="diarizer-stderr").start()
        return True

    def _await_ready(self, proc: Any) -> bool:
        """Block (bounded) until the worker prints READY on stderr."""
        result: dict[str, bool] = {}

        def _read() -> None:
            try:
                for raw in iter(proc.stderr.readline, b""):
                    if b"READY" in raw:
                        result["ok"] = True
                        return
                result["ok"] = False  # EOF without READY
            except Exception:  # noqa: BLE001
                result["ok"] = False

        t = threading.Thread(target=_read, daemon=True, name="diarizer-ready")
        t.start()
        t.join(READY_TIMEOUT_S)
        return result.get("ok", False)

    def _read_stdout(self, proc: Any, q: "queue.Queue[Any]") -> None:
        try:
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw)
        except Exception:  # noqa: BLE001
            pass
        finally:
            q.put(_SENTINEL)

    def _drain_stderr(self, proc: Any) -> None:
        # Keep the pipe empty so a chatty worker never blocks; contents are
        # worker diagnostics (types only) and are not logged here.
        try:
            for _ in iter(proc.stderr.readline, b""):
                pass
        except Exception:  # noqa: BLE001
            pass

    def _kill(self, proc: Any) -> None:
        try:
            if getattr(proc, "stdin", None):
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            pass

    def _fail(self) -> None:
        """One restart, else permanently degraded. Caller returns None."""
        proc, self._proc = self._proc, None
        if proc is not None:
            self._kill(proc)
        if self._restarted:
            self._degraded = True
            return
        self._restarted = True
        logger.warning("diarizer: worker lost; restarting once")
        if not self._start():
            self._degraded = True

    # ---- Diarizer protocol ------------------------------------------------
    def assign(self, pcm: bytes, sample_rate: int, seq: int) -> str | None:
        """Return a live cluster id for this PCM window, or None (coarse)."""
        with self._lock:
            if self._degraded:
                return None
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._fail()
                return None
            try:
                self._send(proc, {"cmd": "assign", "sr": sample_rate, "seq": seq, "n": len(pcm)}, pcm)
                raw = self._q.get(timeout=self._budget)
            except (queue.Empty, OSError, ValueError):
                self._fail()
                return None
            if raw is _SENTINEL:
                self._fail()
                return None
            try:
                return json.loads(raw).get("id")
            except Exception:  # noqa: BLE001
                return None

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]:
        """Batch Stop pass: reconciled live ids for the whole recording.

        Best-effort: any trouble returns ``[]`` and the session keeps the
        near-live labels ``assign`` already placed.
        """
        with self._lock:
            if self._degraded:
                return []
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return []
            try:
                self._send(proc, {"cmd": "diarize", "wav": str(wav_path), "start": start_s, "end": end_s})
                raw = self._q.get(timeout=DIARIZE_BUDGET_S)
            except (queue.Empty, OSError, ValueError):
                self._fail()
                return []
            if raw is _SENTINEL:
                self._fail()
                return []
        try:
            segs = json.loads(raw).get("segments", [])
            return [SpeakerSegment(start_s=s["start_s"], end_s=s["end_s"], speaker=s["speaker"]) for s in segs]
        except Exception:  # noqa: BLE001
            return []

    def centroids(self) -> dict[str, Any]:
        # The live centroids live in the worker (voice embeddings never cross
        # the pipe by design, spec §3.4); reconciliation runs there, so the
        # app never needs them. Kept for the Diarizer protocol.
        return {}

    def close(self) -> None:
        """Best-effort: ask the worker to exit, then tear it down. Idempotent."""
        with self._lock:
            self._degraded = True
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if getattr(proc, "stdin", None):
                try:
                    proc.stdin.write(b'{"cmd": "close"}\n')
                    proc.stdin.flush()
                except Exception:  # noqa: BLE001
                    pass
            proc.wait(timeout=2.0)
        except Exception:  # noqa: BLE001
            self._kill(proc)

    # ---- helpers ----------------------------------------------------------
    def _send(self, proc: Any, control: dict, pcm: bytes | None = None) -> None:
        proc.stdin.write((json.dumps(control) + "\n").encode())
        if pcm:
            proc.stdin.write(pcm)
        proc.stdin.flush()
