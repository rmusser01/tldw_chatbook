"""Console-native chat session surface."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import HorizontalScroll, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_glyphs import GLYPH_CLOSE, GLYPH_IN_PROGRESS
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


def _session_tab_tooltip(
    session: ConsoleChatSession,
    *,
    active: bool,
    streaming: bool = False,
) -> str:
    """Return action copy for a Console session tab."""
    suffix = " Run in progress." if streaming else ""
    if active:
        return f"Active Console tab: {session.title}.{suffix} Click again to rename."
    return f"Switch to Console tab: {session.title}.{suffix}"


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
        yield ChatTaskCards(id="console-task-surface")
        yield ConsoleTranscriptSurface(
            self._transcript_background_effect_settings(
                self.background_effect_settings
            ),
            id="console-transcript-surface",
            classes="console-transcript-surface",
        )

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

    @classmethod
    def _display_title(cls, title: str) -> str:
        """Return a tab label that preserves space for close/rename controls.

        TASK-375: middle-truncate with a single-cell ellipsis rather than an
        end "...". The old end-truncation was defeated by the tab button's
        word-wrap (height-1 showed only the first word, never the mark); a
        middle ellipsis sits early in the label (well inside the button width),
        so with the button's nowrap it is always visible, and it preserves the
        distinguishing words at BOTH ends so two conversations sharing a first
        word are not reduced to the same fragment.
        """
        normalized_title = title.strip() or "Untitled"
        if len(normalized_title) <= CONSOLE_SESSION_TAB_DISPLAY_CHARS:
            return normalized_title
        keep = CONSOLE_SESSION_TAB_DISPLAY_CHARS - 1  # room for the ellipsis cell
        head = (keep + 1) // 2
        tail = keep - head
        head_text = normalized_title[:head].rstrip()
        tail_text = normalized_title[len(normalized_title) - tail:].lstrip()
        return f"{head_text}…{tail_text}"

    def _build_session_tab_button(
        self,
        session: ConsoleChatSession,
        *,
        active: bool,
        streaming: bool = False,
    ) -> Button:
        """Build a stable-width Console session tab title button."""
        classes = "console-session-tab"
        if active:
            classes = f"{classes} console-session-tab-active"
        button = ConsoleSessionTabButton(
            self._tab_label(session.title, streaming=streaming),
            id=f"console-session-tab-{session.id}",
            classes=classes,
            compact=True,
            session_id=session.id,
        )
        button.tooltip = _session_tab_tooltip(
            session, active=active, streaming=streaming
        )
        button.styles.width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.min_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.max_width = CONSOLE_SESSION_TAB_WIDTH
        button.styles.height = 1
        button.styles.min_height = 1
        button.styles.max_height = 1
        return button

    @classmethod
    def _tab_label(cls, title: str, *, streaming: bool) -> str:
        """Return the tab label, prefixed with a run-state glyph when streaming."""
        label = cls._display_title(title)
        if streaming:
            return f"{GLYPH_IN_PROGRESS} {label}"
        return label

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
        return desired_ids

    def _update_existing_tab_strip(
        self,
        *,
        tab_strip: HorizontalScroll,
        sessions: list[ConsoleChatSession],
        active_session_id: str | None,
        streaming_session_id: str | None = None,
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
                streaming = session.id == streaming_session_id
                child.label = self._tab_label(session.title, streaming=streaming)
                child.tooltip = _session_tab_tooltip(
                    session,
                    active=session.id == active_session_id,
                    streaming=streaming,
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
    ) -> None:
        """Render native Console session tabs from controller-owned state."""
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
                )
                return

            removed_count = len(tab_strip.children)
            mounted_count = (len(sessions) * 2) + 1
            for child in list(tab_strip.children):
                await child.remove()
            for session in sessions:
                is_active = session.id == active_session_id
                await tab_strip.mount(
                    self._build_session_tab_button(
                        session,
                        active=is_active,
                        streaming=session.id == streaming_session_id,
                    )
                )
                await tab_strip.mount(self._build_close_tab_button(session))
            await tab_strip.mount(self._build_new_tab_button())
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
