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

Crash rule (spec §7): a DEAD worker (exited process / broken pipe / stdout
EOF) sends the rest of the meeting to coarse labels -- cluster ids cannot
survive a restart, so a fresh worker's ``S1`` would inherit the first
meeting's ``S1`` name. Exactly ONE restart is still attempted so the
authoritative Stop pass survives a transient failure; a second death marks
the backend permanently degraded. A *slow* reply is NOT a crash: it is
backpressure (spec §6.3) -- the window keeps its coarse label and the worker
keeps its restart budget. Best-effort throughout: a worker problem never
raises into the session.

Warm-up (spec §7): construction NEVER blocks. The first run downloads the
ECAPA model, so ``READY`` can be minutes away; the recording must start
anyway. ``assign`` checks readiness without waiting (coarse until warm) and
only ``diarize`` -- the Stop pass, already off the UI thread -- waits, bounded
by its own budget.

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
import time
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Audio.meeting_session import SpeakerSegment

WORKER_MODULE = "tldw_chatbook.Audio.diarizer_worker"
#: A single assign must reply within this; a slower reply is treated as a
#: read timeout (the window falls back to a coarse label).
ASSIGN_BUDGET_S = 2.0
#: The Stop-pass batch clusters the whole recording, so its budget scales with
#: the recording (Qodo Q13: a fixed 60 s silently lost the final diarization of
#: anything long). Floor: a short meeting still gets a usable budget. Ceiling:
#: this runs on the stop worker thread, so a wedged worker delays meeting
#: finalize/ingest by exactly this long -- 10 minutes, not forever.
DIARIZE_BUDGET_FLOOR_S = 60.0
DIARIZE_BUDGET_CEILING_S = 600.0
#: First run downloads the ECAPA model; warm-up can take a while. Nothing
#: blocks on it -- see the module docstring.
READY_TIMEOUT_S = 120.0

#: Static, user-safe reasons for the "speaker labels unavailable" footer copy
#: (spec §7). Never a path, a name, or transcript text.
COARSE_UNAVAILABLE = "backend unavailable"
COARSE_CRASHED = "backend crashed"

_SENTINEL = object()  # placed on the reply queue when the worker's stdout EOFs


def diarize_budget_s(duration_s: float) -> float:
    """Seconds to allow the Stop pass for a recording of `duration_s`.

    Args:
        duration_s: The recording's length in seconds; junk values (negative,
            NaN-ish, None-shaped) collapse to the floor.

    Returns:
        Roughly one second of budget per second of audio, clamped to
        ``[DIARIZE_BUDGET_FLOOR_S, DIARIZE_BUDGET_CEILING_S]``.
    """
    try:
        wanted = float(duration_s)
    except (TypeError, ValueError):
        wanted = 0.0
    return min(DIARIZE_BUDGET_CEILING_S, max(DIARIZE_BUDGET_FLOOR_S, wanted))


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
        self._ready = threading.Event()
        self._ready_ok = False
        self._degraded = False
        #: Live labelling is over for this meeting (a crash), even though the
        #: restarted worker still serves the Stop pass (Qodo Q10).
        self._coarse_only = False
        self._restarted = False
        #: Highest cluster number any assign has returned ("S7" -> 7). The
        #: restarted worker starts past it, so its ids cannot collide with a
        #: pre-crash id the user may have NAMED (31749).
        self.max_id_seen = 0
        #: The `seq` of the assign in flight when the worker died; None until
        #: (and unless) that happens. The Stop pass re-labels only segments
        #: from this seq on -- everything before it keeps its near-live id,
        #: and so keeps the name attached to that id.
        self.crashed_at_seq: int | None = None
        #: Static reason the meeting is on coarse labels, for the footer.
        self.coarse_reason: str | None = None
        self._lock = threading.Lock()
        # Spawn and return: the READY handshake runs on its own thread so
        # Start is never held behind a cold model download (fix C1).
        if not self._start():
            self._degraded = True
            self._mark_coarse(COARSE_UNAVAILABLE)

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
        cmd = [sys.executable, "-m", WORKER_MODULE]
        if self.max_id_seen:
            # A restart (31749): the replacement's clusterer numbers from here,
            # so it can never re-mint an id the dead worker already gave out
            # (and the user may have named). Absent on the first spawn.
            cmd += ["--start-id", str(self.max_id_seen)]
        return cmd

    def _mark_coarse(self, reason: str) -> None:
        """Live labelling is over; keep the FIRST reason (the root cause)."""
        self._coarse_only = True
        if self.coarse_reason is None:
            self.coarse_reason = reason

    def _start(self) -> bool:
        """Spawn the worker; READY is awaited on a thread. False -> degrade."""
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
        # Fresh reply queue and READY gate per worker session: a previous
        # (dead) worker's EOF sentinel must never be read as this worker's
        # crash, nor its handshake as this worker's readiness.
        self._q = queue.Queue()
        self._ready = threading.Event()
        self._ready_ok = False
        threading.Thread(target=self._read_stdout, args=(proc, self._q), daemon=True, name="diarizer-stdout").start()
        threading.Thread(
            target=self._watch_stderr, args=(proc, self._ready), daemon=True, name="diarizer-stderr"
        ).start()
        return True

    def _watch_stderr(self, proc: Any, ready: threading.Event) -> None:
        """Open the READY gate, then keep the pipe drained for this worker.

        One thread does both jobs: nobody joins it, so the constructor never
        waits (C1), and a chatty worker can never block on a full stderr.
        Contents are worker diagnostics (types only) and are not logged here.
        """
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not ready.is_set() and b"READY" in raw:
                    self._ready_ok = True
                    ready.set()
        except Exception:  # noqa: BLE001
            pass
        finally:
            if not ready.is_set():
                logger.warning("diarizer: worker never reported READY")
                self._mark_coarse(COARSE_UNAVAILABLE)
            ready.set()  # unblock `wait_ready` -- `_ready_ok` says whether it worked

    def wait_ready(self, timeout: float) -> bool:
        """Block up to `timeout` seconds for the worker's warm-up handshake.

        Only the Stop pass (and tests) may call this; `assign` checks
        `_ready` without waiting so a cold model never stalls the transcript
        thread.

        Args:
            timeout: Seconds to wait at most.

        Returns:
            True when the worker reported READY within `timeout`.
        """
        return self._ready.wait(timeout) and self._ready_ok

    def _read_stdout(self, proc: Any, q: "queue.Queue[Any]") -> None:
        try:
            for raw in iter(proc.stdout.readline, b""):
                q.put(raw)
        except Exception:  # noqa: BLE001
            pass
        finally:
            q.put(_SENTINEL)

    def _kill(self, proc: Any) -> None:
        """Terminate, then force-kill (Qodo Q14): a worker that ignores
        SIGTERM must not survive with its model and accelerator memory while
        `_fail` spawns its replacement."""
        try:
            if getattr(proc, "stdin", None):
                proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
            return
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.kill()
            proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            logger.warning("diarizer: worker did not exit after kill")

    def _fail(self, seq: int | None = None) -> None:
        """A DEAD worker: coarse for the rest of the meeting, one restart.

        Cluster ids cannot survive a restart (the centroids live in the
        worker), so a fresh worker's "S1" would inherit the first worker's
        S1 name -- spec §7 sends the REST of the meeting to coarse labels and
        keeps the restarted worker only for the authoritative Stop pass.
        A second death degrades the backend permanently.

        Args:
            seq: The assign whose window was in flight when the death was
                detected, if any. The FIRST such seq is remembered as
                `crashed_at_seq`: it is the boundary the Stop pass must not
                re-label across (31749). A death detected outside an assign
                (during the batch pass) has no boundary and passes None.
        """
        if seq is not None and self.crashed_at_seq is None:
            self.crashed_at_seq = seq
        self._mark_coarse(COARSE_CRASHED)
        proc, self._proc = self._proc, None
        if proc is not None:
            self._kill(proc)
        if self._restarted:
            self._degraded = True
            return
        self._restarted = True
        logger.warning("diarizer: worker lost; restarting once, live labels stay coarse")
        if not self._start():
            self._degraded = True

    # ---- Diarizer protocol ------------------------------------------------
    def assign(self, pcm: bytes, sample_rate: int, seq: int) -> str | None:
        """Return a live cluster id for this PCM window, or None (coarse).

        Never waits for warm-up and never raises: a not-yet-READY worker, a
        crashed one, or one that is simply too slow all return None and the
        window keeps its coarse label.
        """
        if self._degraded or self._coarse_only:
            return None
        # Non-blocking readiness check (C1): the model may still be
        # downloading, and the transcript thread cannot wait for it.
        if not (self._ready.is_set() and self._ready_ok):
            return None
        with self._lock:
            if self._degraded or self._coarse_only:
                return None
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._fail(seq)
                return None
            try:
                self._send(proc, {"cmd": "assign", "sr": sample_rate, "seq": seq, "n": len(pcm)}, pcm)
            except (OSError, ValueError):
                self._fail(seq)
                return None
            sid = self._await_reply(seq)
            if sid and sid[:1] == "S" and sid[1:].isdigit():
                self.max_id_seen = max(self.max_id_seen, int(sid[1:]))
            return sid

    def _await_reply(self, seq: int) -> str | None:
        """Read this assign's reply within the budget; None means coarse.

        A budget overrun is BACKPRESSURE, not a crash (spec §6.3): it returns
        None, leaves the restart budget alone, and lets the worker keep
        going. The reply it eventually writes is discarded here by `seq`, so
        one slow window can never shift every later window's answer by one.
        """
        deadline = time.monotonic() + self._budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = self._q.get(timeout=remaining)
            except queue.Empty:
                return None
            if raw is _SENTINEL:
                self._fail(seq)
                return None
            try:
                reply = json.loads(raw)
            except Exception:  # noqa: BLE001 - a garbled line is not an answer
                continue
            if not isinstance(reply, dict) or "id" not in reply:
                continue
            reply_seq = reply.get("seq")
            if reply_seq is not None and reply_seq != seq:
                continue  # a late reply from a window that already gave up
            return reply.get("id")

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]:
        """Batch Stop pass: reconciled live ids for the whole recording.

        The only call that WAITS on warm-up, so a meeting whose model finished
        downloading mid-recording still gets an authoritative pass. The two
        waits are bounded SEPARATELY: warm-up by `READY_TIMEOUT_S` (never the
        clamped batch budget, or the Stop pass could stall for 2x it), the
        reply by the budget. Best-effort: any trouble returns ``[]`` and the
        session keeps the near-live labels ``assign`` already placed.
        """
        budget = diarize_budget_s(end_s - start_s)
        if self._degraded:
            return []
        if not self.wait_ready(min(budget, READY_TIMEOUT_S)):
            # Spawned but never warm (offline first-run download, wedged
            # worker): this IS the spec §7 "failed to become ready" case, so
            # record it -- otherwise the footer stays silent about a meeting
            # that ran entirely on coarse labels (re-review, item 1).
            self._mark_coarse(COARSE_UNAVAILABLE)
            return []
        with self._lock:
            if self._degraded:
                return []
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return []
            try:
                self._send(proc, {"cmd": "diarize", "wav": str(wav_path), "start": start_s, "end": end_s})
            except (OSError, ValueError):
                self._fail()
                return []
            segs = self._await_segments(budget)
        try:
            return [SpeakerSegment(start_s=s["start_s"], end_s=s["end_s"], speaker=s["speaker"]) for s in segs]
        except Exception:  # noqa: BLE001
            return []

    def _await_segments(self, budget: float) -> list[dict]:
        """Read the batch reply within `budget`; an overrun is a skip (Q13)."""
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning("diarizer: stop pass exceeded its budget; keeping near-live labels")
                return []
            try:
                raw = self._q.get(timeout=remaining)
            except queue.Empty:
                return []
            if raw is _SENTINEL:
                self._fail()
                return []
            try:
                reply = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(reply, dict) and "segments" in reply:
                return list(reply.get("segments") or [])

    def pin(self, cluster_id: str) -> None:
        """Best-effort: tell the worker's live clusterer to pin `cluster_id`.

        Fire-and-forget -- no reply is sent or awaited, so this never blocks
        the caller (the screen's rename handler, on the app thread) behind a
        subprocess round trip. Silently does nothing when there is no live
        worker to tell (not ready, coarse-only, or degraded).
        """
        if self._degraded or self._coarse_only:
            return
        if not (self._ready.is_set() and self._ready_ok):
            return
        with self._lock:
            if self._degraded or self._coarse_only:
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            try:
                self._send(proc, {"cmd": "pin", "id": cluster_id})
            except Exception as exc:  # noqa: BLE001 - best-effort, never raises
                logger.warning("diarizer: pin failed ({})", type(exc).__name__)

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
