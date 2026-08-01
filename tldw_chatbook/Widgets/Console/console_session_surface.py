"""Console-native chat session surface."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from rich.markup import escape as _escape_markup
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, HorizontalScroll, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_RUN_MARKER_GLYPHS,
    CONSOLE_RUN_MARKER_MEANINGS,
    ConsoleRunMarker,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_glyphs import GLYPH_CLOSE
from tldw_chatbook.Chat.console_onboarding_state import ConsoleSetupCardState
from tldw_chatbook.Utils.console_background_effects import (
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards
from tldw_chatbook.Widgets.Console.console_background_effect import (
    ConsoleTranscriptSurface,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript


CONSOLE_CLOSE_TAB_BUTTON_WIDTH = 3
CONSOLE_CLOSE_TAB_BUTTON_HEIGHT = 1
CONSOLE_NEW_TAB_BUTTON_WIDTH = 12
CONSOLE_NEW_TAB_BUTTON_HEIGHT = 1
CONSOLE_SESSION_TAB_DISPLAY_CHARS = 19
CONSOLE_SESSION_TAB_WIDTH = 21
CONSOLE_TRANSCRIPT_TITLE = "Transcript / Event Stream"
#: Fleet-UX expert review F2 (task-1232): one-time coach-mark row mounted
#: under the tab strip, hidden until `show_fleet_coachmark` reveals it.
CONSOLE_FLEET_COACHMARK_DISMISS_WIDTH = 3


def _session_tab_tooltip(
    session: ConsoleChatSession,
    *,
    active: bool,
    marker: ConsoleRunMarker = ConsoleRunMarker.NONE,
) -> str:
    """Return action copy for a Console session tab.

    Fleet-UX expert review F4 (task-1233): decodes the tab's fleet
    run-marker glyph in context ("Blue Chat — waiting for approval.")
    rather than leaving the reader to infer ● / ◆ / ✓ / ✗ from shape alone.
    ``ConsoleRunMarker.NONE`` (the steady state) adds no suffix at all, so
    an unmarked tab's tooltip is byte-for-byte the pre-task-1233 copy.
    Every tooltip ends in a period, marked or not -- the sidebar's mirrored
    ``_marker_aware_tooltip`` (``console_workspace_context.py``) matches
    this convention (task-1233 review round 1).

    The whole assembled sentence is escaped exactly once, at the end, not
    per-fragment: the tooltip widget renders Rich markup (Textual's
    ``Tooltip`` is a ``Static`` with markup parsing on), so an unescaped
    ``"["`` anywhere in the sentence -- not just in ``session.title`` --
    would be read as a style-tag start. Escaping only the title fragment
    left a sibling bug in the sidebar row tooltip (a literal
    ``"[saved]"`` status badge concatenated in unescaped): an unrecognized
    tag name is silently DROPPED from the rendered text, not shown
    literally. Escaping the fully-assembled sentence once, here, avoids
    that class of bug even though today's fixed vocabulary (the marker
    meaning, "Click again to rename.") happens to contain no brackets.
    """
    meaning = CONSOLE_RUN_MARKER_MEANINGS.get(marker, "")
    tail = f" — {meaning}." if meaning else "."
    if active:
        text = f"Active Console tab: {session.title}{tail} Click again to rename."
    else:
        text = f"Switch to Console tab: {session.title}{tail}"
    return _escape_markup(text)


class ConsoleSessionTabButton(Button):
    """Console session tab that closes on middle-click.

    The ✕ close button stays as the visible, keyboard-reachable affordance;
    middle-click is the accelerator so heavy session users avoid precision
    targeting on a 3-cell glyph. ``Button._on_click`` activates on any mouse
    button and stops the event, so the middle-click path must live here.
    """

    # TASK-375: keep the (middle-truncated) label on one line so its ellipsis
    # renders instead of being word-wrapped onto a hidden second row.
    DEFAULT_CSS = """
    ConsoleSessionTabButton {
        text-wrap: nowrap;
        text-overflow: clip;
    }
    """

    def __init__(self, *args: Any, session_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._session_id = session_id

    async def _on_click(self, event) -> None:
        if getattr(event, "button", 1) == 2:
            event.stop()
            try:
                close_button = self.screen.query_one(
                    f"#console-close-session-tab-{self._session_id}", Button
                )
            except Exception:
                return
            close_button.press()
            return
        await super()._on_click(event)


class ConsoleSessionSurface(Vertical):
    """Host Console transcript/event stream sessions without legacy chat chrome."""

    def __init__(
        self,
        app_instance: Any,
        *,
        background_effect_settings: ConsoleBackgroundEffectSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.background_effect_settings = (
            background_effect_settings or ConsoleBackgroundEffectSettings()
        )
        self._session_sync_lock = asyncio.Lock()
        #: Title of the active conversation/session shown in the transcript
        #: header; ``None`` renders the static section label.
        self._session_title: str | None = None

    def compose(self) -> ComposeResult:
        title = Static(
            self._render_transcript_title(),
            id="console-transcript-title",
            classes="destination-section console-transcript-title",
        )
        title.styles.height = 1
        title.styles.min_height = 1
        yield title

        tab_strip = HorizontalScroll(
            id="console-native-tab-strip",
            classes="console-session-tab-strip",
        )
        tab_strip.styles.height = 1
        tab_strip.styles.min_height = 1
        tab_strip.styles.max_height = 1
        with tab_strip:
            yield self._build_new_tab_button()
        yield self._build_fleet_coachmark()
        yield ChatTaskCards(id="console-task-surface")
        yield ConsoleTranscriptSurface(
            self._transcript_background_effect_settings(
                self.background_effect_settings
            ),
            id="console-transcript-surface",
            classes="console-transcript-surface",
        )

    def _build_fleet_coachmark(self) -> Horizontal:
        """Build the (initially hidden) one-time parallel-agents coach-mark.

        Fleet-UX expert review F2 / Upgrade proposal 1 (task-1232): composed
        every time (mirrors the composer's `#console-clear-attachment`
        pattern of always mounting a `display: none` control and toggling
        it later) so visibility is driven by `show_fleet_coachmark`/
        `hide_fleet_coachmark` calls off real state, never by a one-shot
        value baked in at mount time.
        """
        row = Horizontal(
            id="console-fleet-coachmark",
            classes="console-fleet-coachmark",
        )
        row.styles.height = 1
        row.styles.min_height = 1
        row.styles.max_height = 1
        row.styles.display = "none"
        text = Static("", id="console-fleet-coachmark-text")
        text.styles.width = "1fr"
        dismiss = Button(
            GLYPH_CLOSE,
            id="console-fleet-coachmark-dismiss",
            compact=True,
        )
        dismiss.tooltip = "Dismiss"
        dismiss.styles.width = CONSOLE_FLEET_COACHMARK_DISMISS_WIDTH
        dismiss.styles.min_width = CONSOLE_FLEET_COACHMARK_DISMISS_WIDTH
        dismiss.styles.max_width = CONSOLE_FLEET_COACHMARK_DISMISS_WIDTH
        # Children are composed via `compose_add_child` rather than a
        # generator `with row: yield ...` block because THIS helper itself
        # returns a plain `Horizontal` (not a generator) so its caller can
        # `yield self._build_fleet_coachmark()` alongside the tab strip's own
        # already-established `with tab_strip: yield ...` pattern.
        row.compose_add_child(text)
        row.compose_add_child(dismiss)
        return row

    def show_fleet_coachmark(self, text: str) -> None:
        """Reveal the one-time parallel-agents coach-mark with the given copy.

        Args:
            text: Plain-text copy to show; rendered via ``rich.text.Text``
                (not markup-parsed) so the copy is safe even if it ever
                contains literal square brackets.
        """
        try:
            banner = self.query_one("#console-fleet-coachmark")
            content = self.query_one("#console-fleet-coachmark-text", Static)
        except Exception:
            return
        content.update(Text(text))
        banner.styles.display = "block"

    def hide_fleet_coachmark(self) -> None:
        """Hide the parallel-agents coach-mark (dismissed, or nothing to show)."""
        try:
            banner = self.query_one("#console-fleet-coachmark")
        except Exception:
            return
        banner.styles.display = "none"

    def _build_new_tab_button(self) -> Button:
        """Return the compact symbolic Console new-session control."""
        button = Button("New tab", id="console-new-chat-tab", compact=True)
        button.tooltip = "New Console tab"
        button.styles.width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.min_width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.max_width = CONSOLE_NEW_TAB_BUTTON_WIDTH
        button.styles.height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        button.styles.min_height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        button.styles.max_height = CONSOLE_NEW_TAB_BUTTON_HEIGHT
        return button

    def _build_new_temporary_tab_button(self) -> Button:
        """Return the tab-strip control for a chat that is never saved."""
        button = Button("Temporary", id="console-new-temporary-tab", compact=True)
        button.tooltip = "New temporary Console tab — not saved locally"
        for style, value in (
            ("width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("min_width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("max_width", CONSOLE_NEW_TAB_BUTTON_WIDTH),
            ("height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
            ("min_height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
            ("max_height", CONSOLE_NEW_TAB_BUTTON_HEIGHT),
        ):
            setattr(button.styles, style, value)
        return button

    @classmethod
    def _display_title(cls, title: str) -> str:
        """Return a tab label that preserves space for close/rename controls.

        TASK-375 originally middle-truncated here ("Long conv…local RAG")
        so a shared first word wouldn't collapse two conversations to the
        same fragment. Fleet-UX expert review F7 (task-1234): the mark
        lands mid-word often enough to read as GARBLED rather than
        truncated ("What is t…ate an."), which live UAT judged the worse
        defect. END-truncation now matches ``derive_console_session_title``
        (``console_chat_models.py``, the auto-title helper this label
        usually renders) -- one truncation convention, not two. TASK-375's
        own word-wrap fix (``ConsoleSessionTabButton``'s ``text-wrap:
        nowrap`` DEFAULT_CSS above) is untouched and still what keeps a
        single-line label from hiding the ellipsis off-screen; only the
        cut POSITION changes here. Trade-off accepted: two conversations
        sharing a long common prefix can once again render identical tab
        labels (the disambiguation TASK-375 added AC#2 for) -- the full
        title always remains one hover away in the tab's tooltip
        (``_session_tab_tooltip``).
        """
        normalized_title = title.strip() or "Untitled"
        if len(normalized_title) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS:
            return normalized_title
        keep = CONSOLE_SESSION_TAB_DISPLAY_CHARS - 1  # room for the ellipsis cell
        return f"{normalized_title[:keep].rstrip()}…"

    def _build_session_tab_button(
        self,
        session: ConsoleChatSession,
        *,
        active: bool,
        marker: ConsoleRunMarker = ConsoleRunMarker.NONE,
    ) -> Button:
        """Build a stable-width Console session tab title button."""
        classes = "console-session-tab"
        if active:
            classes = f"{classes} console-session-tab-active"
        button = ConsoleSessionTabButton(
            self._tab_label(session.title, marker=marker),
            id=f"console-session-tab-{session.id}",
            classes=classes,
            compact=True,
            session_id=session.id,
        )
        button.tooltip = _session_tab_tooltip(
            session, active=active, marker=marker
        )
        button.styles.width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.min_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.max_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.height = 1
        button.styles.min_height = 1
        button.styles.max_height = 1
        return button

    @classmethod
    def _tab_label(
        cls, title: str, *, marker: ConsoleRunMarker = ConsoleRunMarker.NONE
    ) -> str:
        """Return the tab label, prefixed with its fleet run-marker glyph.

        Parallel-agents spec PA-T8: sourced from ``CONSOLE_RUN_MARKER_GLYPHS``
        so RUNNING/NEEDS_APPROVAL/FINISHED_OK/FINISHED_FAILED all render here,
        not just the legacy streaming-only glyph. ``ConsoleRunMarker.NONE``'s
        glyph is the empty string, so an unmarked tab gets no stray leading
        space.
        """
        label = cls._display_title(title)
        glyph = CONSOLE_RUN_MARKER_GLYPHS.get(marker, "")
        if glyph:
            return f"{glyph} {label}"
        return label

    @staticmethod
    def _resolve_tab_marker(
        session_id: str,
        *,
        streaming_session_id: str | None,
        run_markers: dict[str, ConsoleRunMarker] | None,
    ) -> ConsoleRunMarker:
        """Return the fleet marker to render for a tab.

        ``run_markers`` (keyed by session id, sourced from ``ConsoleChat
        Controller.run_marker_for`` -- parallel-agents spec PA-T8) takes
        precedence when the caller supplies it. Falls back to the legacy
        single-session ``streaming_session_id`` cursor so unit tests that
        drive ``ConsoleSessionSurface.sync_sessions`` directly (without a
        controller) keep working unchanged.
        """
        if run_markers is not None:
            return run_markers.get(session_id, ConsoleRunMarker.NONE)
        if session_id == streaming_session_id:
            return ConsoleRunMarker.RUNNING
        return ConsoleRunMarker.NONE

    def _build_close_tab_button(self, session: ConsoleChatSession) -> Button:
        """Build the compact close control for a Console session tab."""
        close_button = Button(
            GLYPH_CLOSE,
            id=f"console-close-session-tab-{session.id}",
            classes="console-session-close-button",
            compact=True,
        )
        close_button.tooltip = "Close Console tab"
        close_button.styles.width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.min_width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.max_width = CONSOLE_CLOSE_TAB_BUTTON_WIDTH
        close_button.styles.height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        close_button.styles.min_height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        close_button.styles.max_height = CONSOLE_CLOSE_TAB_BUTTON_HEIGHT
        return close_button

    def _desired_tab_child_ids(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
    ) -> list[str]:
        """Return the expected child ID sequence for the session tab strip."""
        desired_ids: list[str] = []
        for session in sessions:
            desired_ids.append(f"console-session-tab-{session.id}")
            desired_ids.append(f"console-close-session-tab-{session.id}")
        desired_ids.append("console-new-chat-tab")
        desired_ids.append("console-new-temporary-tab")
        return desired_ids

    def _update_existing_tab_strip(
        self,
        *,
        tab_strip: HorizontalScroll,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
        streaming_session_id: str | None = None,
        run_markers: dict[str, ConsoleRunMarker] | None = None,
    ) -> None:
        """Update labels, tooltips, and active state without stealing focus."""
        session_by_id = {session.id: session for session in sessions}
        for child in tab_strip.children:
            child_id = child.id or ""
            if child_id.startswith("console-session-tab-"):
                session_id = child_id.removeprefix("console-session-tab-")
                session = session_by_id.get(session_id)
                if session is None or not isinstance(child, Button):
                    continue
                marker = self._resolve_tab_marker(
                    session_id,
                    streaming_session_id=streaming_session_id,
                    run_markers=run_markers,
                )
                child.label = self._tab_label(session.title, marker=marker)
                child.tooltip = _session_tab_tooltip(
                    session,
                    active=session.id == active_session_id,
                    marker=marker,
                )
                child.set_class(
                    session.id == active_session_id,
                    "console-session-tab-active",
                )

    def _record_mount_churn(self, *, mounted: int = 0, removed: int = 0) -> None:
        """Best-effort tab churn diagnostic hook."""
        try:
            monitor = getattr(self.app_instance, "ui_responsiveness_monitor", None)
            if monitor is not None:
                monitor.record_mounts(
                    "console-tabs",
                    mounted=mounted,
                    removed=removed,
                )
        except Exception:
            return

    async def sync_sessions(
        self,
        *,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
        streaming_session_id: str | None = None,
        run_markers: dict[str, ConsoleRunMarker] | None = None,
    ) -> None:
        """Render native Console session tabs from controller-owned state.

        Args:
            run_markers: Per-session fleet run marker (parallel-agents spec
                PA-T8), keyed by session id and sourced from
                ``ConsoleChatController.run_marker_for``. When ``None`` (unit
                tests that drive this surface directly, without a
                controller), falls back to the legacy ``streaming_session_id``
                single-session cursor for the RUNNING glyph only.
        """
        active_session = next(
            (session for session in sessions if session.id == active_session_id),
            None,
        )
        self.set_session_title(active_session.title if active_session else None)
        async with self._session_sync_lock:
            tab_strip = self.query_one("#console-native-tab-strip", HorizontalScroll)
            desired_ids = self._desired_tab_child_ids(
                sessions=sessions,
                active_session_id=active_session_id,
            )
            existing_ids = [child.id for child in tab_strip.children]
            if existing_ids == desired_ids:
                self._update_existing_tab_strip(
                    tab_strip=tab_strip,
                    sessions=sessions,
                    active_session_id=active_session_id,
                    streaming_session_id=streaming_session_id,
                    run_markers=run_markers,
                )
                return

            removed_count = len(tab_strip.children)
            mounted_count = (len(sessions) * 2) + 2
            for child in list(tab_strip.children):
                await child.remove()
            for session in sessions:
                is_active = session.id == active_session_id
                marker = self._resolve_tab_marker(
                    session.id,
                    streaming_session_id=streaming_session_id,
                    run_markers=run_markers,
                )
                await tab_strip.mount(
                    self._build_session_tab_button(
                        session,
                        active=is_active,
                        marker=marker,
                    )
                )
                await tab_strip.mount(self._build_close_tab_button(session))
            await tab_strip.mount(self._build_new_tab_button())
            await tab_strip.mount(self._build_new_temporary_tab_button())
            self._record_mount_churn(mounted=mounted_count, removed=removed_count)
        if active_session_id is not None:
            try:
                self.call_after_refresh(
                    self._scroll_active_tab_into_view, active_session_id
                )
            except Exception:
                # Best effort only: surfaces built outside a running app
                # (test doubles via __new__) have no message pump to
                # schedule the scroll with.
                pass

    def _scroll_active_tab_into_view(self, session_id: str) -> None:
        """Bring the active tab into the strip's visible scroll window."""
        try:
            tab_strip = self.query_one("#console-native-tab-strip", HorizontalScroll)
            tab = tab_strip.query_one(f"#console-session-tab-{session_id}", Button)
        except Exception:
            return
        try:
            tab_strip.scroll_to_widget(tab, animate=False)
        except Exception:
            return

    def set_session_title(self, title: str | None) -> None:
        """Show the active conversation/session title in the transcript header.

        Falls back to the static section label when ``title`` is empty or
        ``None``. Best-effort: a no-op when the header is not mounted.

        Args:
            title: Active conversation/session title, or ``None`` to reset.
        """
        normalized = (title or "").strip()
        self._session_title = normalized or None
        try:
            header = self.query_one("#console-transcript-title", Static)
            header.update(self._render_transcript_title())
        except Exception:
            return

    def _render_transcript_title(self) -> Text:
        """Return the transcript header text for the current session title."""
        # ``getattr`` because tests sometimes drive instances built with
        # ``__new__`` that never ran ``__init__``.
        if getattr(self, "_session_title", None):
            return Text(f"{CONSOLE_TRANSCRIPT_TITLE} | {self._session_title}")
        return Text(CONSOLE_TRANSCRIPT_TITLE)

    def sync_inline_guidance(
        self,
        card_state: ConsoleSetupCardState,
        *,
        provider_action_label: str = "",
        provider_action_tooltip: str = "",
    ) -> None:
        """Keep guidance out of the title and sync the empty transcript card state."""
        try:
            title = self.query_one("#console-transcript-title", Static)
        except Exception:
            return
        title.update(self._render_transcript_title())

        try:
            transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
        except Exception:
            return
        transcript.sync_empty_state(
            card_state,
            provider_action_label=provider_action_label,
            provider_action_tooltip=provider_action_tooltip,
        )

    def sync_background_effect_settings(
        self,
        settings: ConsoleBackgroundEffectSettings,
    ) -> None:
        """Apply updated Console background settings to the mounted transcript surface."""
        self.background_effect_settings = settings
        try:
            surface = self.query_one(
                "#console-transcript-surface",
                ConsoleTranscriptSurface,
            )
        except NoMatches:
            return
        surface.update_settings(self._transcript_background_effect_settings(settings))

    @staticmethod
    def _transcript_background_effect_settings(
        settings: ConsoleBackgroundEffectSettings,
    ) -> ConsoleBackgroundEffectSettings:
        """Return settings safe for the transcript-scoped effect surface."""
        if settings.scope == "transcript":
            return settings
        return replace(settings, enabled=False)
