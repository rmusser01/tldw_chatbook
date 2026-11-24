"""File playback attempts retained by a single UI surface on the app loop."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..TTS.audio_output_ownership import AudioOutputIntent, AudioOutputLease
    from ..TTS.audio_output_runs import FileOutputRun


@dataclass(eq=False)
class FileOutputAttempt:
    """One user action, including its pre-admission work and exact output lease."""

    app: Any
    ticket: AudioOutputIntent
    run: FileOutputRun | None = None
    lease: AudioOutputLease | None = None

    def stop(self) -> asyncio.Future[bool]:
        """Fence immediately, including when preparation has yet to return."""
        self.ticket.cancel()
        if self.run is not None and not self.run.retirement_confirmed:
            if self.lease is not None:
                return self.app.audio_output.stop(self.lease)
            return asyncio.ensure_future(self.run.retire())
        stopped = asyncio.get_running_loop().create_future()
        stopped.set_result(True)
        return stopped


class FileOutputSurface:
    """Keep pending starts and the last granted output addressable on exit."""

    def __init__(self) -> None:
        self.latest: FileOutputAttempt | None = None
        self.active: FileOutputAttempt | None = None

    def begin(self, app: Any, ticket: AudioOutputIntent) -> FileOutputAttempt:
        self.latest = FileOutputAttempt(app, ticket)
        return self.latest

    async def play(
        self,
        attempt: FileOutputAttempt,
        path: Path,
        owner: str,
        current: Callable[[], bool],
        *,
        retain_handoff: Callable[[Callable[[], None]], None] | None = None,
        release: Callable[[], None] | None = None,
    ) -> bool:
        """Transfer artifact custody from preparation to positive retirement."""
        from ..TTS.audio_output_admission import acquire_audio_output
        from ..TTS.audio_output_runs import FileOutputRun

        def eligible() -> bool:
            return self.latest is attempt and attempt.ticket.is_current() and current()

        async def prepare_and_start() -> bool:
            run = None
            try:
                if not eligible():
                    return False
                run = attempt.run = await FileOutputRun.prepare(path)
                if run is None:
                    return False
                if release is not None:

                    def release_file() -> None:
                        if run.handed_off and retain_handoff is not None:
                            retain_handoff(release)
                        elif run.handed_off:
                            from ..TTS.audio_player import HAND_OFF_RETENTION_SECONDS

                            asyncio.get_running_loop().call_later(
                                HAND_OFF_RETENTION_SECONDS, release
                            )
                        else:
                            release()

                    run.on_retired(release_file)
                attempt.lease = await acquire_audio_output(
                    attempt.app,
                    attempt.ticket,
                    owner,
                    run,
                    current=eligible,
                    external=run.external,
                )
                if attempt.lease is None:
                    return False
                if not eligible() or not attempt.lease.authorizes(run.retire):
                    await attempt.app.audio_output.stop(attempt.lease)
                    if not run.retirement_confirmed:
                        await run.retire()
                    return False
                self.active = attempt
                started = await run.start(attempt.lease)
                if started and run.handed_off:
                    # The OS now owns playback. Start the file retention window
                    # after releasing our launcher, independent of host ACK.
                    await attempt.app.audio_output.stop(attempt.lease)
                return started
            finally:
                if run is None:
                    if release is not None:
                        release()
                elif attempt.lease is None:
                    await run.retire()

        # File probe/native launch can outlive a Textual worker. The retained
        # task installs the exact cleanup hook even if its observer is cancelled.
        work = asyncio.create_task(prepare_and_start())
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            attempt.stop()
            raise

    def confirming(self) -> bool:
        if self.latest is None:
            return False
        from ..TTS.audio_output_admission import is_audio_confirmation_for

        return is_audio_confirmation_for(self.latest.app, self.latest.ticket)

    def stop(self) -> asyncio.Future:
        attempts = {a for a in (self.active, self.latest) if a is not None}
        return asyncio.gather(*(a.stop() for a in attempts))
