"""One retained playback attempt on the qualified Windows WAV profile."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from enum import Enum
import os
from pathlib import Path
import subprocess
import threading
import time

from .audio_output_ownership import AudioOutputLease
from ._controlled_wav_windows import (
    SpawnRefused,
    UncertainBirth,
    spawn_player as _spawn,
)
from .controlled_wav_profile import (
    ControlledWavError,
    RATE,
    RetainedWav,
    WavBookmark,
    check_route,
    prepare_files,
)
from .controlled_wav_progress import (
    FRESH_S,
    STARTUP_S,
    ProgressClock,
    ProgressObservation,
    StatsReader,
)


_now = time.perf_counter
_SETTLE_S = 0.250
_STOP_WAIT_S = 1.5
_POLL_S = 0.020


class CompletionKind(str, Enum):
    ALREADY_AT_END = "already_at_end"
    NATURAL = "natural"
    STOPPED = "stopped"
    FAILED = "failed"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class PlaybackEvidence:
    launched_at_s: float | None
    stop_at_s: float | None
    reaped_at_s: float | None
    retired_at_s: float | None
    last_physical_frame: int | None
    advanced_span_s: float
    clock_skew_s: float | None
    last_observed_at_s: float | None


@dataclass(frozen=True)
class PlaybackCompletion:
    kind: CompletionKind
    reason: str
    bookmark: WavBookmark
    generation: int | None
    evidence: PlaybackEvidence


class ControlledWavPlayback:
    """Prepare once, then start under a lease or retire the prepared pins."""

    def __init__(self, artifact: RetainedWav, bookmark: WavBookmark):
        self._artifact = artifact
        self._bookmark = bookmark
        self._loop = asyncio.get_running_loop()
        self._ready = self._loop.create_future()
        self._started = self._loop.create_future()
        self._retired = self._loop.create_future()
        self._completion = self._loop.create_future()
        self._lease: AudioOutputLease | None = None
        self._command = threading.Event()
        self._stop = threading.Event()
        self._progress_lock = threading.Lock()
        self._clock: ProgressClock | None = None
        self._stop_at: float | None = None
        self._reaped_at: float | None = None
        self._retired_at: float | None = None
        self._reader_ok = False

    @classmethod
    async def prepare(
        cls,
        artifact: RetainedWav,
        bookmark: WavBookmark,
        *,
        executable: Path | None = None,
    ) -> ControlledWavPlayback:
        """Validate and pin off-loop before asking an existing owner to stop."""
        run = cls(artifact, bookmark)
        # Publish all futures and the dedicated custodian before it can enter.
        run._worker = threading.Thread(
            target=run._lifetime, args=(executable,), daemon=True
        )
        run._worker.start()
        try:
            refusal = await asyncio.shield(run._ready)
        except asyncio.CancelledError:
            run._interrupt()
            raise
        if refusal:
            raise ControlledWavError(refusal)
        return run

    @property
    def already_at_end(self) -> bool:
        return self._bookmark.frame == self._artifact.content_frames

    @property
    def completion(self) -> asyncio.Future[PlaybackCompletion]:
        """Return an individually cancellable observer of the retained result."""
        return asyncio.shield(self._completion)

    def position(self) -> ProgressObservation | None:
        """Return only a fresh current-generation observation while playing."""
        if self._lease is None or not self._lease.is_current() or self._retired.done():
            return None
        with self._progress_lock:
            return self._clock.position(_now()) if self._clock else None

    async def start(self, lease: AudioOutputLease) -> None:
        """Commit an already prepared attempt to the exact current output lease."""
        if self._lease is not None or self.already_at_end or self._stop.is_set():
            raise ControlledWavError("attempt_not_startable")
        if not isinstance(lease, AudioOutputLease) or not lease.authorizes(self.retire):
            self._interrupt()
            raise ControlledWavError("lease_revoked")
        self._lease = lease
        self._command.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._started), STARTUP_S)
        except TimeoutError:
            interrupted = self._interrupted()
            self._interrupt()
            lease.block()
            self._publish(
                CompletionKind.STOPPED if interrupted else CompletionKind.UNCERTAIN,
                "interrupted" if interrupted else "startup_deadline_exceeded",
            )
        except asyncio.CancelledError:
            self._interrupt()
            raise

    def _interrupt(self) -> None:
        if self._retired.done():
            return
        if self._lease:
            self._lease.revoke()
        with self._progress_lock:
            if not self._stop.is_set():
                self._stop_at = _now()
                self._stop.set()
        self._command.set()

    def retire(self) -> Awaitable[bool]:
        """Latch interruption synchronously; bound only the caller's wait."""
        self._interrupt()
        return self._observe_retirement()

    async def _observe_retirement(self) -> bool:
        try:
            return await asyncio.wait_for(asyncio.shield(self._retired), _STOP_WAIT_S)
        except TimeoutError:
            if self._lease is not None:
                self._lease.block()
            return False

    def _interrupted(self) -> bool:
        return (
            self._stop.is_set()
            or self._loop.is_closed()
            or (self._lease is not None and self._lease.revoked)
        )

    def _post(self, callback, *args) -> None:
        try:
            self._loop.call_soon_threadsafe(callback, *args)
        except RuntimeError:
            if not self._loop.is_closed():
                raise

    def _publish(self, kind: CompletionKind, reason: str) -> None:
        if not self._completion.done():
            clock = self._clock
            last = clock.last if clock else None
            evidence = PlaybackEvidence(
                clock.launched_at if clock else None,
                self._stop_at,
                self._reaped_at,
                self._retired_at,
                clock.last_physical_frame if clock else None,
                clock.advanced_span_s if clock else 0.0,
                clock.clock_skew_s if clock else None,
                last.observed_at_s if last else None,
            )
            self._completion.set_result(
                PlaybackCompletion(
                    kind,
                    reason,
                    replace(
                        self._bookmark,
                        frame=last.frame if last else self._bookmark.frame,
                    ),
                    self._lease.generation if self._lease else None,
                    evidence,
                )
            )

    def _finish(self, kind: CompletionKind, reason: str, retired: bool) -> None:
        if kind is not CompletionKind.ALREADY_AT_END and self._interrupted():
            kind, reason = CompletionKind.STOPPED, "interrupted"
        self._publish(kind, reason)
        if self._lease:
            if retired:
                self._lease.release()
            else:
                self._lease.block()
        if not self._retired.done():
            self._retired.set_result(retired)

    def _read(self, pipe, clock: ProgressClock) -> None:
        reader = StatsReader()
        try:
            while data := pipe.read(4096):
                stamp = _now()
                values = reader.feed(data)
                with self._progress_lock:
                    if not self._interrupted():
                        for value in values:
                            clock.observe(value, stamp, clock.generation, now_s=_now())
            self._reader_ok = True
        except (OSError, ValueError):
            pass

    def _watch(self, child, reader) -> tuple[CompletionKind, str]:
        reader_ended_at = None
        while child.poll() is None:
            if self._interrupted():
                return CompletionKind.STOPPED, "interrupted"
            now = _now()
            with self._progress_lock:
                expired = self._clock.expired(now)
            if not reader.is_alive() and reader_ended_at is None:
                reader_ended_at = now
            if expired or (
                reader_ended_at is not None and now - reader_ended_at > FRESH_S
            ):
                return CompletionKind.UNCERTAIN, "progress_unavailable"
            self._stop.wait(_POLL_S)
        return (
            (CompletionKind.NATURAL, "normal_exit")
            if child.returncode == 0
            else (CompletionKind.FAILED, "player_failed")
        )

    def _reap(self, child) -> None:
        # A bounded UI observer may already have left. This worker still owns
        # this exact handle and can confirm late retirement without the UI.
        while True:
            try:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=0.30)
                break
            except (OSError, subprocess.TimeoutExpired):
                try:
                    child.kill()
                    child.wait(timeout=0.30)
                    break
                except (OSError, subprocess.TimeoutExpired):
                    time.sleep(0.10)
        self._reaped_at = _now()

    def _lifetime(self, executable: Path | None) -> None:
        """Single native custodian through validation, delayed Popen and reap."""
        pins = ()
        child = reader = None
        ready = False
        clean = True
        birth_unknown = False
        kind, reason = CompletionKind.FAILED, "controlled_prepare_failed"
        try:
            exe, *pins = prepare_files(self._artifact, self._bookmark, executable)
            if self.already_at_end:
                kind, reason = CompletionKind.ALREADY_AT_END, "logical_eof"
                return
            self._post(self._ready.set_result, None)
            ready = True
            while not self._command.wait(0.25):
                if self._loop.is_closed():
                    return
            env = dict(os.environ)
            env.pop(
                "FFREPORT", None
            )  # The controlled adapter discards raw player output.
            check_route(env)
            if self._interrupted():
                kind, reason = CompletionKind.STOPPED, "interrupted"
                return
            lease = self._lease
            clock = self._clock = ProgressClock(
                self._bookmark.frame,
                self._artifact.content_frames,
                lease.generation,
                _now(),
            )
            argv = [
                str(exe),
                "-nodisp",
                "-autoexit",
                "-stats",
                "-sync",
                "audio",
                "-ss",
                f"{self._bookmark.frame/RATE:.6f}",
                str(pins[-1].physical_path),
            ]
            # The launcher also checks authority at the native creation boundary.
            if self._interrupted():
                kind, reason = CompletionKind.STOPPED, "interrupted"
                return
            reason = "player_start_failed"
            # Opening the launch boundary transfers custody to this attempt.
            # Only an owned child or checked refusal can clear the uncertainty.
            birth_unknown = True
            try:
                child = _spawn(
                    argv, env=env, start_allowed=lambda: not self._interrupted()
                )
            except SpawnRefused:
                birth_unknown = False
                kind, reason = CompletionKind.FAILED, "player_start_refused"
                return
            except BaseException as error:
                if isinstance(error, UncertainBirth):
                    self._partial_birth = error.child
                    if error.child.has_process:
                        child = error.child
                return
            if child is None:
                return
            birth_unknown = False
            reader = threading.Thread(
                target=self._read, args=(child.stderr, clock), daemon=True
            )
            reader.start()
            self._post(self._started.set_result, None)
            kind, reason = self._watch(child, reader)
        except ControlledWavError as error:
            reason = str(error)
        except Exception:
            # Native exception objects can contain private paths or player argv.
            pass
        finally:
            if child is not None:
                self._reap(child)
                if reader is not None and reader.ident is not None:
                    reader.join()
                if child.stderr is not None:
                    try:
                        child.stderr.close()
                    except BaseException:
                        clean = False
                        self._unclosed_pipe = child.stderr
                try:
                    child.close_process_handle()
                except BaseException:
                    clean = False
                    self._unclosed_process = child
                if kind is CompletionKind.NATURAL and not (
                    self._reader_ok and self._clock.natural_candidate(self._reaped_at)
                ):
                    kind, reason = (
                        CompletionKind.UNCERTAIN,
                        "terminal_progress_unconfirmed",
                    )
                time.sleep(max(0.0, _SETTLE_S - (_now() - self._reaped_at)))
            if not birth_unknown:
                for pin in reversed(pins):
                    if not pin.close():
                        clean = False
            else:
                self._unconfirmed_pins = pins
                clean = False
            if clean:
                self._retired_at = _now()
            else:
                kind, reason = (
                    CompletionKind.UNCERTAIN,
                    "player_birth_unconfirmed"
                    if birth_unknown
                    else "resource_retirement_unconfirmed",
                )
            self._post(self._finish, kind, reason, clean)
            if not ready:
                self._post(
                    self._ready.set_result,
                    None if kind is CompletionKind.ALREADY_AT_END else reason,
                )
            if child is None or reader is None or reader.ident is None:
                self._post(self._started.set_result, None)
