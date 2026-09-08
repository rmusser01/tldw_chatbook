"""Ephemeral, store-free projection of one speculative Console voice turn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


_STATUS_LABELS = {
    "preparing": "Preparing voice",
    "listening": "Listening",
    "transcribing": "Transcribing",
    "responding": "Responding",
    "speaking": "Speaking",
    "updating response": "Updating response",
    "aec warming": "AEC warming",
    "half duplex": "Half duplex · echo cancellation unavailable",
    "cleanup quarantine": "Voice cleanup quarantine",
}


def voice_status_label(status: str) -> str:
    """Return bounded user-facing copy for one content-free voice state."""

    if type(status) is not str or not status.strip() or len(status) > 80:
        raise ValueError("status must be a non-empty bounded string")
    normalized = status.strip().casefold()
    return _STATUS_LABELS.get(normalized, status.strip())


class VoiceStatusAnnouncementThrottle:
    """Announce meaningful voice-state transitions at a bounded cadence."""

    def __init__(
        self,
        announce: Callable[[str], None],
        *,
        clock: Callable[[], float] = time.monotonic,
        minimum_interval_seconds: float = 0.75,
    ) -> None:
        if not callable(announce) or not callable(clock):
            raise TypeError("announcement and clock callbacks must be callable")
        if (
            isinstance(minimum_interval_seconds, bool)
            or not isinstance(minimum_interval_seconds, (int, float))
            or minimum_interval_seconds <= 0
        ):
            raise ValueError("announcement interval must be positive")
        self._announce = announce
        self._clock = clock
        self._minimum_interval_seconds = float(minimum_interval_seconds)
        self._last_status: str | None = None
        self._last_announced_at: float | None = None

    def observe(self, status: str) -> bool:
        """Announce a changed status when the cadence permits it."""

        label = voice_status_label(status)
        normalized = status.strip().casefold()
        if normalized == self._last_status:
            return False
        now = float(self._clock())
        last = self._last_announced_at
        if last is not None and now - last < self._minimum_interval_seconds:
            return False
        self._announce(label)
        self._last_status = normalized
        self._last_announced_at = now
        return True


@dataclass(frozen=True, slots=True)
class VoicePreviewProjection:
    """Pure provisional text/status projection; never a transcript message."""

    turn_id: str
    attempt_epoch: int
    user_text: str = field(repr=False)
    assistant_text: str = field(repr=False)
    status: str

    def __post_init__(self) -> None:
        if type(self.turn_id) is not str or not self.turn_id or len(self.turn_id) > 256:
            raise ValueError("turn_id must be a non-empty bounded string")
        if type(self.attempt_epoch) is not int or self.attempt_epoch < 0:
            raise ValueError("attempt_epoch must be a non-negative integer")
        if type(self.user_text) is not str or type(self.assistant_text) is not str:
            raise TypeError("preview text must be strings")
        if type(self.status) is not str or not self.status or len(self.status) > 80:
            raise ValueError("status must be a non-empty bounded string")


class ConsoleVoicePreview(Vertical):
    """Two visually provisional rows kept outside durable transcript grouping."""

    can_focus = False
    BUNDLED_CSS = """
    ConsoleVoicePreview {height:auto;margin:0 1 1 1;padding:0 1;border-left:thick $accent 45%;background:$surface-lighten-1 55%;color:$text-muted;}
    ConsoleVoicePreview .console-voice-preview-status {color:$accent-lighten-1;text-style:italic;}
    ConsoleVoicePreview .console-voice-preview-user,
    ConsoleVoicePreview .console-voice-preview-assistant {height:auto;}
    """

    def __init__(
        self,
        projection: VoicePreviewProjection | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._projection = projection

    @property
    def projection(self) -> VoicePreviewProjection | None:
        """Return the current immutable view projection."""

        return self._projection

    def compose(self) -> ComposeResult:
        yield Static("", classes="console-voice-preview-status", markup=False)
        yield Static("", classes="console-voice-preview-user", markup=False)
        yield Static("", classes="console-voice-preview-assistant", markup=False)

    def on_mount(self) -> None:
        self._apply_projection()

    def set_projection(self, projection: VoicePreviewProjection) -> None:
        """Replace the current provisional projection in place."""

        if type(projection) is not VoicePreviewProjection:
            raise TypeError("projection must be a VoicePreviewProjection")
        self._projection = projection
        if self.is_mounted:
            self._apply_projection()

    def clear(self) -> None:
        """Hide and forget all provisional text."""

        self._projection = None
        if self.is_mounted:
            self._apply_projection()

    def _apply_projection(self) -> None:
        projection = self._projection
        self.display = projection is not None
        status = self.query_one(".console-voice-preview-status", Static)
        user = self.query_one(".console-voice-preview-user", Static)
        assistant = self.query_one(".console-voice-preview-assistant", Static)
        if projection is None:
            status.update("")
            user.update("")
            assistant.update("")
            assistant.display = False
            return
        status.update(voice_status_label(projection.status))
        user.update(f"You · {projection.user_text}" if projection.user_text else "")
        user.display = bool(projection.user_text)
        assistant.update(
            f"Assistant · {projection.assistant_text}"
            if projection.assistant_text
            else ""
        )
        assistant.display = bool(projection.assistant_text)


__all__ = [
    "ConsoleVoicePreview",
    "VoicePreviewProjection",
    "VoiceStatusAnnouncementThrottle",
    "voice_status_label",
]
