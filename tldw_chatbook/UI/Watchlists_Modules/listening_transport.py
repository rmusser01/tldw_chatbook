"""Compact display and commands for the app-owned article transport."""

from __future__ import annotations

from collections.abc import Callable
from sqlite3 import Error as SQLiteError
from typing import TYPE_CHECKING, Literal

from textual import on
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from ...Subscriptions.html_text import strip_control_characters

if TYPE_CHECKING:
    from ...Subscriptions.article_listening_service import (
        ArticleListeningService,
        ListeningRevision,
        ListeningSnapshot,
    )

TransportAction = Literal[
    "pause", "resume", "stop", "skip", "clear", "play_again", "retry", "regenerate"
]
AUDIO_HOST_NOTICE = "Audio plays on the computer running Chatbook."
CHECKING_OUTPUT = "Checking article audio output…"
LOCAL_ONLY = "Switch to Local to listen to saved articles."
START_ACTIONS = frozenset({"resume", "play_again", "retry", "regenerate"})

_PRIMARY = {
    "pause": "Pause",
    "resume": "Resume saved listening",
    "play_again": "Play again",
    "retry": "Retry",
    "regenerate": "Regenerate from start",
}
_PHASE_LABELS = {
    "preparing": "Preparing audio",
    "playing": "Playing",
    "paused": "Paused",
    "completed": "Completed",
    "retiring": "Waiting for audio output to stop",
    "unavailable": "Listening unavailable",
}
_REASONS = {
    "missing_item": "The cited item is no longer available.",
    "empty_body": "No saved body to read.",
    "unsupported_content": "Only saved article bodies can be read aloud.",
    "unsupported_platform": "Article listening requires Windows.",
    "unsupported_audio_route": "Use the default Windows audio output for article listening.",
    "qualified_ffplay_unavailable": "Install the supported ffplay build on the Chatbook computer.",
    "player_containment_unavailable": "The player could not start safely. Check the ffplay setup and retry.",
    "article_conversion_unavailable": "Audio conversion is unavailable. Check the audio dependencies.",
    "article_audio_missing": "Saved audio is missing. Regenerate to start at the beginning.",
    "article_audio_corrupt": "Saved audio is damaged. Regenerate to start at the beginning.",
    "explicit_regeneration_required": "Regenerate the saved audio to start at the beginning.",
    "article_audio_disk_full": "Free space in the Chatbook profile and retry.",
    "article_audio_database_unavailable": "Article listening storage is unavailable. Check the profile and retry.",
    "article_audio_storage_unavailable": "Article listening storage is unavailable. Check the profile and retry.",
    "article_audio_busy": "Article audio is busy. Try again when the current work finishes.",
    "listening_busy": "Article listening is busy. Try again when the current work finishes.",
    "listening_voice_changed": "The saved voice changed. Restore that voice before retrying.",
    "listening_voice_unavailable": "Check the default speech voice in Settings, then retry.",
    "listening_voice_unbound": "Choose a supported, explicit speech voice in Settings.",
    "article_native_budget_unavailable": "Choose another speech provider in Settings for article listening.",
    "listening_authority_changed": "The listening context changed. Try again from the current selection.",
    "local_listening_authority_required": LOCAL_ONLY,
    "listening_brief_unavailable": "This saved briefing is unavailable.",
    "no_playable_articles": "No playable saved article body was found.",
    "article_at_end": "The saved position is at the end of the audio.",
    "player_retirement_unconfirmed": "Audio has not confirmed it stopped. Try Stop again.",
    "previous_player_unconfirmed": "The previous player may still be running. Close it and retry.",
    "listening_replacement_required": "Review and confirm the replacement queue.",
}


def listening_reason(code: object) -> str:
    """Convert a bounded diagnostic to recovery copy without echoing inputs."""
    if type(code) is str:
        return _REASONS.get(
            code,
            "Article listening failed. Check speech and audio settings before retrying.",
        )
    return "Article listening is unavailable."


def listening_service(app) -> ArticleListeningService | None:
    try:
        return app.services.watchlists.listening
    except AttributeError:
        return None


def audio_host_notice(app) -> str:
    try:
        return AUDIO_HOST_NOTICE if app._served_canvas_mode is True else ""
    except AttributeError:
        return ""


async def output_issue() -> str:
    # No custody is requested by this display probe. Full output admission
    # stays with the listening service, using its exact playback owner.
    from ...TTS.controlled_wav_profile import controlled_wav_readiness

    code = await controlled_wav_readiness()
    return listening_reason(code) if code else ""


async def run_transport(
    app,
    service: ArticleListeningService,
    action: TransportAction,
    revision: ListeningRevision,
    *,
    view_is_current: Callable[[], bool] | None = None,
) -> None:
    """Dispatch the displayed action to the same live app owner."""
    from ...Subscriptions.article_listening_lease import ListeningBusy
    from ...Subscriptions.article_listening_models import ListeningConflict

    def current() -> bool:
        return (
            listening_service(app) is service
            and action in service.snapshot().actions
            and (view_is_current is None or view_is_current())
        )

    if not current():
        return
    if action in START_ACTIONS:
        issue = await output_issue()
        if not current():
            return
        if issue:
            app.notify(issue, severity="warning", markup=False)
            return
    try:
        if action == "regenerate":
            await service.retry(regenerate=True, expected_revision=revision)
        else:
            await getattr(service, action)(expected_revision=revision)
    except SQLiteError:
        app.notify(
            listening_reason("article_audio_database_unavailable"),
            severity="warning",
            markup=False,
        )
    except (ListeningConflict, ListeningBusy, OSError) as error:
        app.notify(
            listening_reason(error.args[0] if error.args else None),
            severity="warning",
            markup=False,
        )


def update_listening_note(note: Static, copy: str) -> None:
    if note.content != copy:
        note.update(copy)
    if note.display != bool(copy):
        note.display = bool(copy)


class ListeningTransportRequested(Message):
    def __init__(self, action: TransportAction, revision: ListeningRevision) -> None:
        self.action = action
        self.revision = revision
        super().__init__()


def _clock(frames: int, rate: int) -> str:
    minutes, seconds = divmod(frames // rate, 60)
    return f"{minutes}:{seconds:02}"


def _title(value: str) -> str:
    return " ".join(strip_control_characters(value).split())


def _status(snapshot: ListeningSnapshot) -> str:
    status = [_PHASE_LABELS.get(snapshot.phase, "Article listening")]
    if snapshot.phase == "preparing" and (
        snapshot.completed_chunks is not None and snapshot.total_chunks is not None
    ):
        status.append(f"{snapshot.completed_chunks}/{snapshot.total_chunks} chunks")
    if snapshot.sample_rate and snapshot.elapsed_frames is not None:
        timing = _clock(snapshot.elapsed_frames, snapshot.sample_rate)
        if snapshot.duration_frames is not None:
            timing += " / " + _clock(snapshot.duration_frames, snapshot.sample_rate)
        status.append(timing)
    if snapshot.current_title:
        status.append(_title(snapshot.current_title))
    return " · ".join(status)


class ListeningTransport(Vertical):
    """Project immutable listening state and publish only typed user intents."""

    def __init__(self) -> None:
        super().__init__(id="wl-listening", classes="wl-listening-hidden")
        self._revision: ListeningRevision | None = None
        self._primary: TransportAction | None = None

    def compose(self):
        yield Static("", id="wl-listening-status", markup=False)
        yield Static("", id="wl-listening-detail", markup=False)
        with Horizontal(id="wl-listening-actions", classes="destination-filter-strip"):
            yield Button("Pause", id="wl-listening-primary", compact=True)
            yield Button("Stop", id="wl-listening-stop", compact=True)
            yield Button("Skip", id="wl-listening-skip", compact=True)
            yield Button("Clear queue", id="wl-listening-clear", compact=True)

    def project(
        self, snapshot: ListeningSnapshot | None, issue: str, host: str
    ) -> None:
        visible = snapshot is not None and (snapshot.phase != "empty" or snapshot.error)
        self.set_class(not visible, "wl-listening-hidden")
        self._revision = snapshot.revision if visible else None
        if not visible:
            return
        try:
            status = self.query_one("#wl-listening-status", Static)
        except NoMatches:
            return

        title = _status(snapshot)
        if status.content != title:
            status.update(title, layout=False)

        details = []
        if snapshot.error:
            details.append(listening_reason(snapshot.error))
        if issue and snapshot.actions & START_ACTIONS:
            issue_copy = issue if issue not in details else ""
            if issue_copy:
                details.append(issue_copy)
        if snapshot.stored_feed_text:
            details.append("Saved feed text")
        if snapshot.queue_position is not None:
            details.append(f"Queue {snapshot.queue_position}/{snapshot.queue_count}")
        if snapshot.suspended_title is not None:
            details.append("Suspended: " + _title(snapshot.suspended_title))
        if host and snapshot.actions & START_ACTIONS:
            details.append(host)
        update_listening_note(
            self.query_one("#wl-listening-detail", Static), " · ".join(details)
        )

        allowed = snapshot.actions - START_ACTIONS if issue else snapshot.actions
        self._primary = next((action for action in _PRIMARY if action in allowed), None)
        primary = self.query_one("#wl-listening-primary", Button)
        if self._primary and primary.label.plain != _PRIMARY[self._primary]:
            primary.label = _PRIMARY[self._primary]
        clear = self.query_one("#wl-listening-clear", Button)
        clear_label = "Clear queue" if snapshot.queue_count else "Clear listening"
        if clear.label.plain != clear_label:
            clear.label = clear_label
        buttons = {
            primary: self._primary is not None,
            self.query_one("#wl-listening-stop", Button): "stop" in allowed,
            self.query_one("#wl-listening-skip", Button): "skip" in allowed,
            clear: "clear" in allowed,
        }
        for button, available in buttons.items():
            if available and not button.display:
                button.display = True
        for button, available in buttons.items():
            if not available and button.display:
                if button.has_focus:
                    successor = next(
                        (item for item, ready in buttons.items() if ready), None
                    )
                    if successor is not None:
                        successor.focus()
                button.display = False
        actions = self.query_one("#wl-listening-actions", Horizontal)
        if actions.display != any(buttons.values()):
            actions.display = any(buttons.values())

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._revision is None or not self.display or not event.button.display:
            return
        action = {
            "wl-listening-primary": self._primary,
            "wl-listening-stop": "stop",
            "wl-listening-skip": "skip",
            "wl-listening-clear": "clear",
        }.get(event.button.id)
        if action is not None:
            self.post_message(ListeningTransportRequested(action, self._revision))
