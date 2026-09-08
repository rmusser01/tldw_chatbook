"""One bounded Buddy speech queue, independent of execution and acknowledgement."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BuddySpeechItem:
    """A named immutable utterance with its source's live authority probe."""

    key: str
    title: str
    text: str = field(repr=False)
    question: bool = False
    is_current: Callable[[], bool] = field(
        default=lambda: True, repr=False, compare=False
    )
    context: object = field(default=None, repr=False, compare=False)

    @property
    def named_text(self) -> str:
        """Prefix the spoken content with a bounded, plain conversation name."""
        title = " ".join(self.title.split())[:180] or "Conversation"
        return f"{title}. {self.text}"[:5000]


@dataclass(frozen=True, slots=True)
class BuddySpeechState:
    paused: bool
    muted: bool
    queued: int
    current_title: str
    error: str
    input_active: bool = False


class BuddySpeechQueue:
    """Serialize playback and cleanup; controls touch no agent or receipt service.

    Pause stops current playback and restarts that utterance from its beginning
    on Resume. Question priority applies between utterances, without interrupting
    a response already being heard. All methods run on the app event loop.
    """

    def __init__(
        self,
        play: Callable[[BuddySpeechItem, Callable[[], bool]], Awaitable[bool]],
    ) -> None:
        self._play = play
        self._pending: list[BuddySpeechItem] = []
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._runner: asyncio.Task | None = None
        self._playback: asyncio.Task | None = None
        self._active: BuddySpeechItem | None = None
        self._replay = False
        self._epoch = 0
        self._paused = False
        self._muted = False
        self._closed = False
        self._error = ""
        self._input_owners: set[object] = set()

    @property
    def state(self) -> BuddySpeechState:
        return BuddySpeechState(
            self._paused,
            self._muted,
            len(self._pending),
            self._active.title if self._active else "",
            self._error,
            bool(self._input_owners),
        )

    @staticmethod
    def _valid(item: BuddySpeechItem) -> bool:
        try:
            return item.is_current() is True
        except Exception:  # noqa: BLE001 - stale authority fails closed
            return False

    def enqueue(self, item: BuddySpeechItem) -> bool:
        """Queue one new terminal item; duplicate/progress events never accumulate."""
        if (
            self._closed
            or self._muted
            or item.key in self._seen
            or not self._valid(item)
        ):
            return False
        if len(self._pending) >= 64:
            return False
        self._seen[item.key] = None
        while len(self._seen) > 512:
            self._seen.popitem(last=False)
        self._pending.append(item)
        self._start()
        return True

    def _start(self) -> None:
        if (
            not self._closed
            and not self._paused
            and not self._muted
            and self._pending
            and self._runner is None
            and not self._input_owners
        ):
            self._runner = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        try:
            while self._pending and not (
                self._closed or self._paused or self._muted or self._input_owners
            ):
                index = next(
                    (i for i, item in enumerate(self._pending) if item.question), 0
                )
                item = self._pending.pop(index)
                if not self._valid(item):
                    continue
                epoch = self._epoch
                self._active, self._replay = item, False

                def current(epoch=epoch, item=item) -> bool:
                    return (
                        epoch == self._epoch
                        and not self._closed
                        and not self._muted
                        and not self._paused
                        and not self._input_owners
                        and self._active is item
                        and self._valid(item)
                    )

                self._playback = asyncio.create_task(self._play(item, current))
                try:
                    if not await self._playback:
                        self._error = "Speech could not finish."
                except asyncio.CancelledError:
                    pass  # Exact playback cancellation; the app's work is untouched.
                except Exception:  # noqa: BLE001 - never retain provider/private error text
                    self._error = "Speech could not finish."
                finally:
                    if self._replay and epoch == self._epoch and self._valid(item):
                        self._pending.insert(0, item)
                    self._active, self._playback, self._replay = None, None, False
        finally:
            self._runner = None
            self._start()

    def pause(self) -> None:
        self._paused = True
        self._replay = self._active is not None
        if self._playback is not None and not self._playback.cancelling():
            self._playback.cancel()

    def resume(self) -> None:
        self._paused = False
        self._start()

    def skip(self) -> None:
        self._replay = False
        if self._playback is not None:
            if not self._playback.cancelling():
                self._playback.cancel()
        elif self._pending:
            self._pending.pop(0)

    def mute(self) -> None:
        self._muted = True
        self._pending.clear()
        self.skip()

    def unmute(self) -> None:
        self._muted = False
        self._start()

    def reset(self) -> None:
        """Invalidate all old binding/profile work before new items can play."""
        self._epoch += 1
        self._pending.clear()
        self._seen.clear()
        self._error = ""
        self.skip()

    def revalidate(self) -> None:
        """Discard revoked sources and interrupt only their owned playback."""
        self._pending[:] = [item for item in self._pending if self._valid(item)]
        if self._active is not None and not self._valid(self._active):
            self.skip()

    def set_input_active(self, owner: object, active: bool) -> None:
        """Hold output while an explicit microphone owner is active."""
        if active:
            self._input_owners.add(owner)
            self._replay = self._active is not None
            if self._playback is not None and not self._playback.cancelling():
                self._playback.cancel()
        else:
            self._input_owners.discard(owner)
            self._start()

    async def wait_idle(self) -> None:
        """Wait for the current drain, including cancelled playback cleanup."""
        while self._runner is not None:
            await asyncio.shield(self._runner)

    async def aclose(self) -> None:
        self._closed = True
        self.reset()
        await self.wait_idle()
