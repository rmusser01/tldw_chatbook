"""Per-attempt worker custody for the app's existing file, PCM and media outputs."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from tldw_chatbook.Audio.streaming_sink import (
    SinkDrained,
    SinkFailed,
    SinkStopped,
    StreamingPcmSink,
)
from tldw_chatbook.Media_Playback.player_pipeline import (
    PlayerPipeline,
    PlayerProbe,
    PlayerRun,
    audio_pipe_supported,
)
from tldw_chatbook.TTS.audio_output_ownership import AudioOutputLease, OutputSuperseded
from tldw_chatbook.TTS.audio_player import (
    FilePlaybackPlan,
    PlaybackState,
    SimpleAudioPlayer,
)

if TYPE_CHECKING:
    from tldw_chatbook.Audio.speech_presentation import SpeechPresentation


_RETIRE_TIMEOUT = 1.5
_PCM_BLOCK_MS = 20
_T = TypeVar("_T")


class _OutputRun:
    """Keep submitted native work and its exact cleanup through observer cancellation."""

    def __init__(self, *, audible: bool = True) -> None:
        self.audible = audible
        self.lease: AudioOutputLease | None = None
        self.retirement_confirmed = False
        self._fence = threading.Event()
        self._bound = False
        self._pending: asyncio.Task | None = None
        self._retirement: asyncio.Task[bool] | None = None

    def _authorized(self) -> bool:
        """Fast worker guard. The host only makes admission decisions on its loop."""
        return not self._fence.is_set() and (
            not self.audible or (self.lease is not None and not self.lease.revoked)
        )

    def _bind(self, lease: AudioOutputLease | None) -> None:
        if self._bound:
            raise RuntimeError("Audio output run was already started")
        if self.audible != (lease is not None):
            raise ValueError("Only an audible output run takes an audio lease")
        if lease is not None and (
            not isinstance(lease, AudioOutputLease) or not lease.authorizes(self.retire)
        ):
            raise OutputSuperseded("Audio lease does not authorize this attempt")
        self.lease = lease
        self._bound = True
        if self.retirement_confirmed and lease is not None:
            lease.release()  # an unstarted local attempt was stopped before grant

    async def _submit(
        self, operation: Callable[[], Awaitable[_T]], *, external_handoff: bool = False
    ) -> _T:
        if not self._authorized():
            raise OutputSuperseded("Audio output authority revoked")
        previous = self._pending

        async def invoke():
            # Suspend even with an eager task factory. The run must hold this
            # task before a native call or a reentrant event can see the run.
            await asyncio.sleep(0)
            if previous is not None:
                await previous
            if not self._authorized():
                raise OutputSuperseded("Audio output authority revoked")
            return await operation()

        job = self._pending = asyncio.create_task(invoke())
        try:
            result = await asyncio.shield(job)
            if not external_handoff and not self._authorized():
                raise OutputSuperseded("Audio output authority revoked")
            return result
        except asyncio.CancelledError:
            self._begin_retirement()
            raise
        except Exception:
            self._begin_retirement()
            raise

    def _begin_retirement(self) -> asyncio.Task[bool]:
        self._fence.set()
        if self.lease is not None:
            self.lease.revoke()
        if self._retirement is None or (
            self._retirement.done() and not self.retirement_confirmed
        ):
            self._retirement = asyncio.create_task(self._finish_retirement())
        return self._retirement

    def retire(self) -> Awaitable[bool]:
        """Fence this attempt now. Bound each observer while the actual cleanup continues."""
        task = self._begin_retirement()

        async def observe() -> bool:
            try:
                return await asyncio.wait_for(asyncio.shield(task), _RETIRE_TIMEOUT)
            except TimeoutError:
                return False

        return observe()

    stop = retire

    async def _finish_retirement(self) -> bool:
        await asyncio.sleep(0)  # publish the retirement task before dispatch
        if self._pending is not None:
            try:
                await asyncio.shield(self._pending)
            except (Exception, asyncio.CancelledError):
                pass  # even a failed or interrupted launch may have allocated a child
        try:
            confirmed = await self._close_native()
        except (Exception, asyncio.CancelledError):
            return False
        if confirmed is True:
            self.retirement_confirmed = True
            if self.lease is not None:
                self.lease.release()
        return self.retirement_confirmed

    async def _close_native(self) -> bool:
        raise NotImplementedError


class FileOutputRun(_OutputRun):
    """One prepared artifact, private player and exact native attempt token."""

    def __init__(self, player: SimpleAudioPlayer, plan: FilePlaybackPlan) -> None:
        super().__init__()
        self._player = player
        self._plan = plan
        self.path = plan.file_path
        self.external = plan.external
        self.handed_off = False
        self._token = object()
        self._attempted = False
        self._terminal: PlaybackState | None = None
        self._monitor: asyncio.Task | None = None
        self._wake_monitor = asyncio.Event()

    @classmethod
    async def prepare(
        cls, path: Path, *, player: SimpleAudioPlayer | None = None
    ) -> FileOutputRun | None:
        """Resolve the launch plan on a worker before requesting output admission."""

        def prepare():
            private = player if player is not None else SimpleAudioPlayer()
            return private, private.prepare(path)

        private, plan = await asyncio.to_thread(prepare)
        return cls(private, plan) if plan is not None else None

    async def start(self, lease: AudioOutputLease) -> bool:
        try:
            self._bind(lease)
        except OutputSuperseded:
            return False

        async def launch():
            self._attempted = True

            def commit():
                if not self._authorized():
                    return False, False
                return True, self._player.play_prepared(
                    self._plan,
                    attempt_token=self._token,
                    start_authority=self._authorized,
                )

            self._attempted, accepted = await asyncio.to_thread(commit)
            if accepted is True and self.external:
                self.handed_off = True
                lease.record_handoff()  # before this retained job can settle
                return True
            if accepted is True and self._authorized():
                self._monitor = asyncio.create_task(self._watch_completion())
                return True
            return False

        try:
            accepted = await self._submit(launch, external_handoff=self.external)
        except Exception:
            accepted = False
        if not accepted:
            await self.retire()
        return accepted

    async def _watch_completion(self) -> None:
        await asyncio.sleep(0)  # publish the observer before its first getter
        while self._authorized():
            try:
                state = await self.get_state()
            except Exception:
                self._begin_retirement()
                return
            if state in (PlaybackState.FINISHED, PlaybackState.ERROR):
                if self._authorized():
                    self._terminal = state
                    self._begin_retirement()
                return
            try:
                await asyncio.wait_for(self._wake_monitor.wait(), timeout=0.1)
            except TimeoutError:
                pass

    async def get_state(self) -> PlaybackState:
        if self.retirement_confirmed and self._terminal is not None:
            return self._terminal
        state = await asyncio.to_thread(
            self._player.get_state, expected_token=self._token
        )
        return (
            self._terminal if state is PlaybackState.IDLE and self._terminal else state
        )

    async def get_position(self) -> float:
        return await asyncio.to_thread(self._player.get_position)

    async def _transport(self, action: Callable[..., bool]) -> bool:
        if not self._authorized():
            return False
        try:
            return await self._submit(
                lambda: asyncio.to_thread(action, expected_token=self._token)
            )
        except OutputSuperseded:
            return False

    async def pause(self) -> bool:
        return await self._transport(self._player.pause)

    async def resume(self) -> bool:
        return await self._transport(self._player.resume)

    async def _close_native(self) -> bool:
        self._wake_monitor.set()
        if self._monitor is not None:
            await self._monitor
        if not self._attempted:
            return True

        def close():
            if self._player.stop(expected_token=self._token) is True:
                return True
            if self.handed_off:
                return True  # accepted delegation is recorded separately on the host
            # A refusal or compensated restart can clear the live token before
            # our final stop. Ask for the native attempt's positive receipt.
            return self._player.retirement_confirmed(expected_token=self._token)

        return await asyncio.to_thread(close)


class PcmOutputRun(_OutputRun):
    """A managed PCM sink whose feed and pump interfaces remain native."""

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        *,
        on_event: Callable[[object], None],
        presentation: SpeechPresentation | None = None,
        stream_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        if type(sample_rate) is not int or sample_rate * _PCM_BLOCK_MS // 1000 <= 0:
            raise ValueError("Invalid PCM sample rate")
        if type(channels) is not int or channels <= 0:
            raise ValueError("Invalid PCM channel count")
        self._rate, self._channels = sample_rate, channels
        self._loop: asyncio.AbstractEventLoop | None = None

        def notify(event: object) -> None:
            try:
                on_event(event)
            finally:
                if (
                    isinstance(event, (SinkDrained, SinkStopped, SinkFailed))
                    and self._loop is not None
                ):
                    self._loop.call_soon_threadsafe(self._terminal)

        self.sink = StreamingPcmSink(
            on_event=notify,
            presentation=presentation,
            stream_factory=stream_factory,
            start_authority=self._authorized,
            blocksize_ms=_PCM_BLOCK_MS,
        )

    def _terminal(self) -> None:
        if not self._fence.is_set():
            self._begin_retirement()

    async def start(self, lease: AudioOutputLease) -> bool:
        try:
            self._bind(lease)
        except OutputSuperseded:
            return False
        self._loop = asyncio.get_running_loop()

        def open_sink():
            self.sink.open(self._rate, self._channels)
            return self.sink.state == "open" and self._authorized()

        try:
            opened = await self._submit(lambda: asyncio.to_thread(open_sink))
        except Exception:
            opened = False
        if not opened:
            await self.retire()
        return opened

    async def _close_native(self) -> bool:
        return await self.sink.retire()


class MediaOutputRun(_OutputRun):
    """One private demux pipeline, including its unpublished partial starts."""

    def __init__(
        self,
        source: str | Path,
        probe: PlayerProbe,
        *,
        pipeline_factory: Callable[
            [str | Path, PlayerProbe], PlayerPipeline
        ] = PlayerPipeline,
    ) -> None:
        if probe.has_audio and not audio_pipe_supported():
            raise RuntimeError("In-app media audio requires POSIX inherited pipes")
        super().__init__(audible=probe.has_audio)
        self.pipeline = pipeline_factory(source, probe)
        self.current_run: PlayerRun | None = None
        self._owned: PlayerRun | None = None

    async def _restart(self, offset: float, *, seek: bool) -> PlayerRun:
        async def launch():
            old = self.current_run
            if seek and old is None:
                raise RuntimeError("No current media run to seek")
            # The pipeline can own an unpublished partial pair if restart raises.
            self._owned = None
            if seek:
                next_run = await asyncio.to_thread(
                    self.pipeline.seek,
                    offset,
                    expected_run=old,
                    start_authority=self._authorized,
                )
            else:
                next_run = await asyncio.to_thread(
                    self.pipeline.start,
                    offset_seconds=offset,
                    start_authority=self._authorized,
                )
            self._owned = next_run
            if not self._authorized():
                raise OutputSuperseded("Media output authority revoked")
            self.current_run = next_run
            return next_run

        return await self._submit(launch)

    async def start(
        self, lease: AudioOutputLease | None, *, offset_seconds: float = 0.0
    ) -> PlayerRun:
        self._bind(lease)
        return await self._restart(offset_seconds, seek=False)

    async def seek(self, offset: float) -> PlayerRun:
        if not self._bound:
            raise RuntimeError("Media output has not started")
        return await self._restart(offset, seek=True)

    async def _transport(self, action: Callable[[], None]) -> bool:
        if self.current_run is None or not self._authorized():
            return False

        def act():
            if not self._authorized():
                return False
            action()
            return self._authorized()

        try:
            return await self._submit(lambda: asyncio.to_thread(act))
        except OutputSuperseded:
            return False

    async def pause(self) -> bool:
        return await self._transport(self.pipeline.pause)

    async def resume(self) -> bool:
        return await self._transport(self.pipeline.resume)

    async def _close_native(self) -> bool:
        confirmed = await asyncio.to_thread(
            self.pipeline.stop, expected_run=self._owned
        )
        if confirmed is True:
            self._owned = self.current_run = None
        return confirmed
