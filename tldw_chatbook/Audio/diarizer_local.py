"""App-side diarizer backend: spawns and talks to the worker over pipes.

No torch is imported here -- SpeechBrain/torch live only in
`diarizer_worker.py`, a separate process. This module speaks a small wire
protocol to it (spec §3.4):

    stdin  (app -> worker):  one JSON control line per command; an "assign"
                             or "enroll_from_pcm" line is immediately
                             followed by exactly ``n`` bytes of raw PCM16
                             (the length-prefix is the ``n`` field on the
                             control line). "enroll" and "pin" get no reply.
                             "export_centroid"/"enroll_from_pcm" also carry
                             an ``op_id`` the worker must echo back (C1: lets
                             the caller drop a late reply for an op it
                             already gave up on).
    stdout (worker -> app):  one ``{"id": "S1", "seq": ..., "self": ...}`` /
                             ``{"id": null, ...}`` line per assign;
                             ``{"segments": [...], "self": ...}`` for a
                             diarize; ``{"centroid": [...] | null, "seconds":
                             ..., "op_id": ...}`` for
                             "export_centroid"/"enroll_from_pcm".
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

Privacy: only PCM, cluster ids, and (task 3) a self-voiceprint / centroid
vectors cross the pipe -- the app process holds those in memory only, via
`Audio/voiceprint.py`. Transcript text and speaker names never reach the
worker, and nothing here logs PCM, text, names, paths, or vector values --
types and lengths only.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
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
#: `pin` runs on the APP thread (the rename handler) and the backend lock is
#: held by `assign` for up to `ASSIGN_BUDGET_S`; a pin that cannot get the
#: lock within this is dropped rather than freezing the TUI (final review I1).
_PIN_LOCK_WAIT_S = 0.05
#: First run downloads the ECAPA model; warm-up can take a while. Nothing
#: blocks on it -- see the module docstring.
READY_TIMEOUT_S = 120.0
#: `export_centroid`/`enroll_from_pcm` (spec §3.4's learning offer and
#: explicit enrollment) share this one bounded wait -- lock acquire plus the
#: reply -- so a busy or wedged worker cannot hang either flow indefinitely.
CENTROID_BUDGET_S = 10.0

#: Static, user-safe reasons for the "speaker labels unavailable" footer copy
#: (spec §7). Never a path, a name, or transcript text.
COARSE_UNAVAILABLE = "backend unavailable"
COARSE_CRASHED = "backend crashed"
#: The ONNX engine's model fetch failed (spec §3/§8) -- distinct from
#: `COARSE_UNAVAILABLE` (a worker that spawned but never warmed up) so the
#: footer can tell "no models" from "worker trouble" apart.
COARSE_MODELS_UNAVAILABLE = "models unavailable"
#: Wall-clock ceiling for the ONNX engine's model fetch (spec §3), separate
#: from `READY_TIMEOUT_S` -- the fetch has its own budget and the READY
#: timeout only starts once the worker is actually spawned (task 4: 31827).
MODELS_DOWNLOAD_BUDGET_S = 600.0

#: Import-free alias of the SpeechBrain model id (spec §6), mirroring
#: `diarizer_worker.MODEL_ID` -- `model_id_for` must not import
#: `diarizer_engine_speechbrain` (or pull the ONNX module for a speechbrain
#: lookup) just to read one static string.
_SPEECHBRAIN_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb@unpinned"

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


class LocalDiarizer:
    """Talks to the diarizer worker subprocess; degrades to coarse on failure.

    `engine` selects which worker module does the embedding (spec §2): the
    default `"speechbrain"` starts the worker synchronously, exactly as
    before; `"onnx"` first fetches its ONNX models on a daemon warm-up
    thread (task 4: 31827) so construction still never blocks (spec §7)."""

    def __init__(
        self,
        engine: str = "speechbrain",
        max_speakers: int = 8,
        *,
        spawn: Callable[..., Any] = subprocess.Popen,
        assign_budget_s: float = ASSIGN_BUDGET_S,
        voiceprint: Sequence[float] | None = None,
        match_threshold: float = 0.2,
        match_min_seconds: float = 4.0,
        embedder: str | None = None,
        models_dir_override: Path | None = None,
        ensure_models: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine
        self._embedder = embedder
        self._models_dir_override = models_dir_override
        self._ensure_models = ensure_models
        self._max = max_speakers
        self._spawn = spawn
        self._budget = assign_budget_s
        # Held privately and sent to the worker (never logged, spec §3.3) as
        # soon as it reports READY; re-sent after a restart for free since
        # that spawns a fresh watcher (`_watch_stderr`). The constructor must
        # never raise (M2/M3): a vector that fails float() conversion (a
        # corrupt on-disk record) or is empty is treated as absent rather
        # than breaking meeting Start.
        try:
            parsed_voiceprint = None if voiceprint is None else [float(x) for x in voiceprint]
        except Exception as exc:  # noqa: BLE001 - never raise on a bad vector
            logger.warning("diarizer: voiceprint rejected ({})", type(exc).__name__)
            parsed_voiceprint = None
        self._voiceprint: list[float] | None = parsed_voiceprint or None
        self._match_threshold = float(match_threshold)
        self._match_min_seconds = float(match_min_seconds)
        #: The first cluster id the worker ever flagged `self: true` (§3.3):
        #: fixed here so a later, noisier match can never displace it. Backed
        #: by a private field; read via the `self_cluster_id` property.
        self._self_cluster_id: str | None = None
        #: Every `assign` reply flagged `self: true` counts here, matched
        #: cluster or not -- diagnostic only; the session acts on
        #: `self_cluster_id` alone. Read via the `self_candidates_seen` property.
        self._self_candidates_seen: int = 0
        #: The most recent Stop-pass (`diarize`) reply's `self` id, or None.
        #: Task 4's fallback: applied only when no live match was recorded.
        self.stop_self: str | None = None
        #: Monotonic id stamped on every `export_centroid`/`enroll_from_pcm`
        #: control line so `_await_centroid` can tell its own reply apart
        #: from a late one for an op this call already gave up on (C1).
        self._op_seq = 0
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
        #: What the rail shows for this backend's warm-up (task 4: 31827) --
        #: one of "downloading <a> / <b> MB", "warming up", "ready",
        #: "unavailable". Read via the `warmup_status` property.
        self._status = "warming up"
        if self._engine == "onnx":
            # Deviation from the brief's step 3 (see task-4-report.md): the
            # brief has the daemon thread itself set the initial "downloading
            # 0 / N MB" status as its first action. Measured: a freshly
            # started thread is NOT guaranteed to run even that far before
            # `Thread.start()` returns to this constructor, so a caller that
            # checks `warmup_status` right after construction can race an
            # empty thread and still see "warming up". Computed here instead,
            # synchronously, before the thread (which only runs the fetch
            # itself) is started -- `diarizer_engine_onnx` is still imported
            # lazily (a local import, not at module scope), just one call
            # frame earlier than the brief described.
            from tldw_chatbook.Audio.diarizer_engine_onnx import DEFAULT_EMBEDDER, EMBEDDERS, SEGMENTATION

            self._embedder = self._embedder or DEFAULT_EMBEDDER
            total_mb = ((SEGMENTATION.download_size or SEGMENTATION.size) + EMBEDDERS[self._embedder].size) // (
                1 << 20
            )
            self._status = f"downloading 0 / {total_mb} MB"
            # The model fetch (spec §3) is the first step of warm-up, ahead
            # of the spawn -- run it on its own daemon thread so construction
            # still returns at once (fix C1 extended to the ONNX engine).
            threading.Thread(target=self._warmup, daemon=True, name="diarizer-warmup").start()
        # Spawn and return: the READY handshake runs on its own thread so
        # Start is never held behind a cold model download (fix C1).
        elif not self._start():
            self._degraded = True
            self._mark_coarse(COARSE_UNAVAILABLE)
            self._status = "unavailable"

    @property
    def voiceprint(self) -> list[float] | None:
        """The enrolled voiceprint, or None. A copy -- the caller's mutation
        of the returned list must never reach the vector this backend
        (re-)sends to the worker on every restart."""
        return None if self._voiceprint is None else list(self._voiceprint)

    @property
    def self_cluster_id(self) -> str | None:
        """The first cluster id the worker ever flagged `self: true` (spec
        §3.3: a read-only attribute for the session, task 4, to read)."""
        return self._self_cluster_id

    @property
    def self_candidates_seen(self) -> int:
        """Every consumed `assign` reply flagged `self: true`, matched
        cluster or not -- diagnostic only."""
        return self._self_candidates_seen

    @property
    def warmup_status(self) -> str:
        """What the rail shows for this backend's warm-up (spec §3/§7):
        one of ``"downloading <a> / <b> MB"``, ``"warming up"``, ``"ready"``,
        ``"unavailable"``."""
        return self._status

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
        cmd = [sys.executable, "-m", WORKER_MODULE, "--engine", self._engine]
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
            # The ONNX worker has no other way to learn which embedder (and,
            # for an override/air-gapped install, which directory) the app
            # already fetched the models into (task 4: 31827, ruling 2) --
            # harmless to set for the speechbrain engine, which ignores them.
            if self._embedder:
                env["TLDW_DIARIZER_EMBEDDER"] = self._embedder
            if self._models_dir_override is not None:
                env["TLDW_DIARIZER_MODELS_DIR"] = str(self._models_dir_override)
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
        # crash, nor its handshake as this worker's readiness. The ONNX
        # engine's first `_start()` call runs on the warm-up thread (task 4:
        # 31827), asynchronously with respect to `__init__`'s caller -- a
        # `wait_ready` already blocked on the Event `__init__` created must
        # not be orphaned on a freshly-minted, never-to-be-set replacement,
        # so an event that is not yet set (never handshaked) is reused as-is;
        # only a set one (a previous worker's completed handshake) is
        # discarded.
        if self._ready.is_set():
            self._ready = threading.Event()
        self._ready_ok = False
        self._q = queue.Queue()
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

        The enroll send happens strictly BEFORE `ready.set()` (controller
        ruling 1): `assign`/`pin` only proceed once they observe `ready` set,
        so the enroll is guaranteed to be the first command this worker ever
        sees and can never interleave with one of them. A restart calls
        `_start()` again, which spawns a fresh watcher -- re-enrollment falls
        out of this same ordering for free.
        """
        try:
            for raw in iter(proc.stderr.readline, b""):
                if not ready.is_set() and b"READY" in raw:
                    self._ready_ok = True
                    self._send_enroll(proc)
                    ready.set()
                    self._status = "ready"
        except Exception:  # noqa: BLE001
            pass
        finally:
            if not ready.is_set():
                logger.warning("diarizer: worker never reported READY")
                self._mark_coarse(COARSE_UNAVAILABLE)
                self._status = "unavailable"
            ready.set()  # unblock `wait_ready` -- `_ready_ok` says whether it worked

    def _send_enroll(self, proc: Any) -> None:
        """Best-effort: hand the worker this session's voiceprint. Guarded by
        the same lock as every other command (pipe writes are not atomic
        across threads); a no-op when no voiceprint was configured. Never
        logs the vector -- its length only, and only on a send failure.

        A failed send (a broken pipe -- the worker died during warm-up) is a
        worker failure, not a silently-ignored one (ruling 6): it goes
        through the same `_fail` path as any other dead-worker detection, so
        the meeting ends up coarse-and-flagged rather than "ready" with
        self-matching silently off. `_fail` never takes `self._lock`, so
        calling it from inside this `with` block cannot deadlock.
        """
        if self._voiceprint is None:
            return
        with self._lock:
            try:
                self._send(
                    proc,
                    {
                        "cmd": "enroll",
                        "vector": self._voiceprint,
                        "threshold": self._match_threshold,
                        "min_seconds": self._match_min_seconds,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - best-effort, never raises
                logger.warning(
                    "diarizer: enroll send failed ({}, vector_len={})", type(exc).__name__, len(self._voiceprint)
                )
                self._fail()

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

    def _warmup(self) -> None:
        """`engine == "onnx"` only: fetch the models, then spawn (spec §3).

        Runs entirely on its own daemon thread, started by `__init__` (which
        returns immediately, same as the speechbrain path -- fix C1 extended
        to the model fetch). Touches only `_status`, `_degraded`,
        `coarse_reason` and `_ready` directly -- `_lock` is never acquired
        here (ruling 8), so a slow or wedged download can never block
        `assign`/`pin`, which check readiness without waiting. The READY
        timeout itself only starts once `_start()` actually spawns -- the
        fetch has its own separate `MODELS_DOWNLOAD_BUDGET_S` budget.

        `self._embedder` is already resolved (never None) by the time this
        runs -- `__init__` settles it, synchronously, before starting this
        thread (see the constructor's comment).
        """
        try:
            ensure = self._ensure_models
            if ensure is None:
                from tldw_chatbook.Audio.diarizer_engine_onnx import ensure_models as ensure
            ensure(
                self._embedder,
                models_dir_override=self._models_dir_override,
                progress=self._set_status,
                budget_s=MODELS_DOWNLOAD_BUDGET_S,
            )
        except Exception as exc:  # noqa: BLE001 - a daemon thread must never raise
            logger.warning("diarizer: models unavailable ({})", type(exc).__name__)
            self._mark_coarse(COARSE_MODELS_UNAVAILABLE)
            self._degraded = True
            self._status = "unavailable"
            self._ready.set()  # unblock `wait_ready` -- no worker will ever spawn
            return
        self._status = "warming up"
        if not self._start():
            self._degraded = True
            self._mark_coarse(COARSE_UNAVAILABLE)
            self._status = "unavailable"

    def _set_status(self, status: str) -> None:
        """`ensure_models`'s `progress` callback -- integers only (spec §8)."""
        self._status = status

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
            self._status = "unavailable"
            return
        self._restarted = True
        logger.warning("diarizer: worker lost; restarting once, live labels stay coarse")
        if not self._start():
            self._degraded = True
            self._status = "unavailable"

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
        # I2: a bounded acquire, not a blocking one -- `_centroid_op` can
        # hold this same lock for up to CENTROID_BUDGET_S (5x this budget).
        # A backend too busy to take the lock right now is exactly the same
        # backpressure `_await_reply`'s own timeout already models: skip to
        # coarse, do not touch the restart budget, never block the caller.
        if not self._lock.acquire(timeout=self._budget):
            return None
        try:
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
        finally:
            self._lock.release()

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
            sid = reply.get("id")
            if reply.get("self") is True:
                # Fixed here, once: the FIRST flagged cluster is `self`, for
                # good, no matter how many later windows also flag true
                # (spec §3.3 -- stable first match lives in the backend).
                self._self_candidates_seen += 1
                if self._self_cluster_id is None:
                    self._self_cluster_id = sid
            return sid

    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]:
        """Batch Stop pass: reconciled live ids for the whole recording.

        The only call that WAITS on warm-up, so a meeting whose model finished
        downloading mid-recording still gets an authoritative pass. The two
        waits are bounded SEPARATELY: warm-up by `READY_TIMEOUT_S` (never the
        clamped batch budget, or the Stop pass could stall for 2x it), the
        reply by the budget. Best-effort: any trouble returns ``[]`` and the
        session keeps the near-live labels ``assign`` already placed.
        """
        # M6: a timed-out or failed pass must not leave a PREVIOUS successful
        # pass's verdict standing -- reset up front, not just on success.
        self.stop_self = None
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
        # Bounded, like `assign`'s (final review Minor 8): `_centroid_op` can
        # hold this same lock for CENTROID_BUDGET_S, and a blocking acquire
        # let the Stop pass wait that out ON TOP OF its own budget. Giving up
        # keeps the near-live labels, which is what every other failure here
        # does too.
        if not self._lock.acquire(timeout=budget):
            return []
        try:
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
        finally:
            self._lock.release()
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
                # Every successful Stop pass overwrites this, including with
                # None -- Task 4's fallback reads the LATEST batch's verdict.
                self.stop_self = reply.get("self")
                return list(reply.get("segments") or [])

    def pin(self, cluster_id: str) -> None:
        """Best-effort: tell the worker's live clusterer to pin `cluster_id`.

        Fire-and-forget -- no reply is sent or awaited, and the lock is taken
        with a short timeout, so this never blocks the caller (the screen's
        rename handler, on the app thread) behind a subprocess round trip.
        `assign` holds the same lock across `_await_reply` for up to
        `ASSIGN_BUDGET_S`, so a blocking acquire here froze the TUI for up to
        two seconds whenever a rename landed while a window was in flight
        (final review I1). A pin dropped because the backend was busy is the
        same best-effort miss as one dropped because there is no live worker
        to tell (not ready, coarse-only, or degraded).
        """
        if self._degraded or self._coarse_only:
            return
        if not (self._ready.is_set() and self._ready_ok):
            return
        if not self._lock.acquire(timeout=_PIN_LOCK_WAIT_S):
            return
        try:
            if self._degraded or self._coarse_only:
                return
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return
            try:
                self._send(proc, {"cmd": "pin", "id": cluster_id})
            except Exception as exc:  # noqa: BLE001 - best-effort, never raises
                logger.warning("diarizer: pin failed ({})", type(exc).__name__)
        finally:
            self._lock.release()

    def centroids(self) -> dict[str, Any]:
        # The live centroids live in the worker (voice embeddings never cross
        # the pipe by design, spec §3.4); reconciliation runs there, so the
        # app never needs them. Kept for the Diarizer protocol.
        return {}

    def export_centroid(self, cluster_id: str) -> tuple[list[float], float] | None:
        """Best-effort: `cluster_id`'s batch centroid (spec §3.4's learning
        offer). None on any trouble -- never raises."""
        return self._centroid_op({"cmd": "export_centroid", "id": cluster_id})

    def enroll_from_pcm(self, pcm: bytes, sample_rate: int) -> tuple[list[float], float] | None:
        """Best-effort: embed `pcm` (explicit enrollment's ~30s mic sample)
        and return its unit centroid. None on any trouble -- never raises."""
        return self._centroid_op({"cmd": "enroll_from_pcm", "sr": sample_rate, "n": len(pcm)}, pcm)

    def _centroid_op(self, control: dict, pcm: bytes | None = None) -> tuple[list[float], float] | None:
        """Shared plumbing for `export_centroid`/`enroll_from_pcm`.

        Ready-gated like `assign`/`pin` (ruling 2 / I1): a pre-READY caller
        returns None at once, without ever taking `self._lock` -- so this can
        never block the stderr watcher's `_send_enroll`, which also takes
        that lock. Task 4 is expected to wait for readiness itself before
        enrolling; this is the structural backstop.

        Acquires the same lock `assign`/`diarize` use (bounded, like
        `assign`'s own acquire), so no other reply can be in flight while
        this waits. Every control line is stamped with a fresh `op_id`
        (C1): a queued reply for an op this call already gave up on is
        indistinguishable from a fresh one by shape alone (both are
        `{"centroid": ...}`), so `_await_centroid` also checks it matches.
        Bounded by `CENTROID_BUDGET_S` total, lock wait included. Never
        raises.
        """
        if self._degraded or self._coarse_only or not self.wait_ready(0):
            return None
        deadline = time.monotonic() + CENTROID_BUDGET_S
        if not self._lock.acquire(timeout=CENTROID_BUDGET_S):
            return None
        try:
            if self._degraded or self._coarse_only:
                return None
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return None
            self._op_seq += 1
            op_id = self._op_seq
            try:
                self._send(proc, {**control, "op_id": op_id}, pcm)
            except (OSError, ValueError):
                self._fail()
                return None
            remaining = max(0.0, deadline - time.monotonic())
            return self._await_centroid(remaining, op_id)
        except Exception as exc:  # noqa: BLE001 - best-effort, never raises (M7)
            logger.warning("diarizer: centroid op failed ({})", type(exc).__name__)
            return None
        finally:
            self._lock.release()

    def _await_centroid(self, budget: float, op_id: int) -> tuple[list[float], float] | None:
        """Read the `{"centroid": ..., "op_id": ...}` reply matching `op_id`
        within `budget`.

        A `centroid`-keyed reply whose `op_id` does not match is a late
        answer to an op THIS call (or an earlier one) already gave up on
        (C1) -- dropped and logged (ids only), never returned as this call's
        answer. An assign/diarize reply cannot be in flight while the caller
        holds `self._lock`, so those are simply skipped by the `"centroid"
        not in reply` check, same as before.
        """
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                raw = self._q.get(timeout=remaining)
            except queue.Empty:
                return None
            if raw is _SENTINEL:
                self._fail()
                return None
            try:
                reply = json.loads(raw)
            except Exception:  # noqa: BLE001 - a garbled line is not an answer
                continue
            if not isinstance(reply, dict) or "centroid" not in reply:
                continue
            if reply.get("op_id") != op_id:
                logger.warning(
                    "diarizer: dropped a stale centroid reply (op_id={}, want={})", reply.get("op_id"), op_id
                )
                continue
            centroid = reply.get("centroid")
            if centroid is None:
                return None
            return [float(x) for x in centroid], float(reply.get("seconds") or 0.0)

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


class SpeechBrainDiarizer(LocalDiarizer):
    """`LocalDiarizer` pinned to the SpeechBrain engine (spec §2).

    Kept as a real subclass -- not a `functools.partial` -- so
    `isinstance(x, SpeechBrainDiarizer)` and every existing import keep
    working unchanged; its `__init__` simply forces `engine="speechbrain"`
    and forwards everything else."""

    def __init__(self, max_speakers: int = 8, **kwargs: Any) -> None:
        kwargs.pop("engine", None)
        super().__init__("speechbrain", max_speakers, **kwargs)


def model_id_for(engine: str, embedder: str | None = None) -> str:
    """The voiceprint model id for `engine` (spec §6): stable across a run,
    changing only if the underlying model file or its manifest hash does.
    The owner reads this BEFORE any worker exists (a Start-time
    re-enrollment check), so it must work without spawning anything.

    Args:
        engine: `"speechbrain"` or `"onnx"`.
        embedder: For `"onnx"`, one of the manifest's embedder keys;
            `DEFAULT_EMBEDDER` when omitted. Ignored for `"speechbrain"`.

    Returns:
        `"speechbrain/spkrec-ecapa-voxceleb@unpinned"` for `"speechbrain"`;
        `"sherpa-onnx/<embedder file name>@<first 12 hex of its sha256>"`
        for `"onnx"`.

    Raises:
        ValueError: `engine` is neither known engine name.
    """
    if engine == "speechbrain":
        return _SPEECHBRAIN_MODEL_ID
    if engine == "onnx":
        from tldw_chatbook.Audio.diarizer_engine_onnx import DEFAULT_EMBEDDER, model_id_for_embedder

        return model_id_for_embedder(embedder or DEFAULT_EMBEDDER)
    raise ValueError(f"unknown engine: {engine}")
