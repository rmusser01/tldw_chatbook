"""Read-only conversation transcript for the Personas workbench.

Replaces ``CCPConversationViewWidget`` on the Personas screen only, rendering
the flat transcript-line pattern the preview pane established: one Static per
message, ``role: content``, no per-message chrome.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static


class PersonasConversationTranscriptWidget(Container):
    """Flat, read-only transcript of a saved conversation."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so BUNDLED_CSS must not reference them).
    # height: 1fr (not 100%): the detail stack is a VerticalScroll (task-2231)
    # and this view is always shown WITH the 9-line conversation-actions block;
    # 100% would overflow the viewport by exactly that block, making the stack
    # scrollable in conversation view - focus auto-scroll then hid the action
    # buttons. 1fr resolves against the viewport MINUS the actions row, so
    # the pair exactly fills the viewport like the pre-scroll layout did.
    BUNDLED_CSS = """
    PersonasConversationTranscriptWidget {
        width: 100%;
        height: 1fr;
    }

    PersonasConversationTranscriptWidget #personas-transcript-scroll {
        height: 1fr;
    }

    PersonasConversationTranscriptWidget #personas-transcript-preview-note {
        height: auto;
    }

    PersonasConversationTranscriptWidget .personas-transcript-line {
        height: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-conversation-transcript-view")
        super().__init__(**kwargs)
        self._render_attempt: object | None = None
        self._render_lock = asyncio.Lock()

    def compose(self) -> ComposeResult:
        # markup=False: the title carries a user-provided conversation title,
        # which must render literally (an unmatched [/tag] would raise
        # MarkupError at render time with markup enabled).
        yield Static(
            "Conversation",
            id="personas-transcript-title",
            classes="destination-section",
            markup=False,
        )
        yield Static(
            "Preview shows up to 200 messages. Resume opens the saved chat in Console.",
            id="personas-transcript-preview-note",
            markup=False,
        )
        yield VerticalScroll(id="personas-transcript-scroll")

    def set_title(self, title: str) -> None:
        """Update the transcript header line."""
        self.query_one("#personas-transcript-title", Static).update(
            str(title or "Conversation")
        )

    async def show_loading(self, render_attempt: object | None = None) -> bool:
        """Replace the transcript with a loading placeholder.

        Gives the conversation click instant feedback while the message
        worker fetches; ``load_messages`` replaces it with the content.

        Args:
            render_attempt: Ownership token for the new preview. A standalone
                caller may omit it to claim a fresh token automatically.

        Returns:
            True when the loading state was committed by the current owner.
        """
        token = render_attempt if render_attempt is not None else object()
        self._render_attempt = token
        return await self._replace_scroll_contents(
            [Static("Loading transcript...", id="personas-transcript-loading")],
            token,
        )

    async def show_error(self, render_attempt: object | None = None) -> bool:
        """Replace the scroll contents with the recoverable preview error.

        Args:
            render_attempt: Existing preview ownership token. A standalone
                caller may omit it to claim a fresh token automatically.

        Returns:
            True when the error state was committed by the current owner.
        """
        token = self._standalone_or_existing_attempt(render_attempt)
        return await self._replace_scroll_contents(
            [
                Static(
                    "Couldn't load this preview. You can still resume the saved chat.",
                    id="personas-transcript-error",
                )
            ],
            token,
        )

    async def show_unavailable(
        self, render_attempt: object | None = None
    ) -> bool:
        """Hide transcript content after its authority or revision changes."""

        token = self._standalone_or_existing_attempt(render_attempt)
        return await self._replace_scroll_contents(
            [
                Static(
                    "This conversation changed or moved. Refresh conversations to retry.",
                    id="personas-transcript-unavailable",
                )
            ],
            token,
        )

    async def load_messages(
        self,
        messages: List[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]] = None,
        render_attempt: object | None = None,
    ) -> bool:
        """Replace the transcript with ``messages`` (async-safe replace).

        ``speaker_names`` maps roles to display names (e.g. ``{"user":
        "You", "assistant": "Detective Sam"}``); unmapped roles render as
        the raw role. Role CSS classes are unaffected by the mapping.

        Args:
            messages: Transcript messages to render.
            speaker_names: Optional role-to-display-name mapping.
            render_attempt: Existing preview ownership token. A standalone
                caller may omit it to claim a fresh token automatically.

        Returns:
            True when the transcript was committed by the current owner.
        """
        names = speaker_names or {}
        widgets: list[Static] = []
        for message in messages or []:
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "")
            speaker = str(names.get(role) or role)
            # Role styling is intentionally binary: "user" vs assistant-style
            # for every other role (assistant, system, tool, unknown, ...).
            role_class = (
                "personas-transcript-line-user"
                if role == "user"
                else "personas-transcript-line-assistant"
            )
            widgets.append(
                # markup=False: message content must render literally, never
                # as Rich markup (unmatched tags raise MarkupError at render).
                Static(
                    f"{speaker}: {content}",
                    classes=f"personas-transcript-line {role_class}",
                    markup=False,
                )
            )
        if not widgets:
            widgets.append(
                Static("No messages to display.", id="personas-transcript-empty")
            )
        token = self._standalone_or_existing_attempt(render_attempt)
        return await self._replace_scroll_contents(widgets, token)

    def invalidate_render(self, render_attempt: object | None = None) -> None:
        """Invalidate one matching render owner, or the current owner when omitted.

        Args:
            render_attempt: Token to invalidate. When omitted, any current
                render owner is invalidated.
        """
        if render_attempt is None or self._render_attempt is render_attempt:
            self._render_attempt = None

    def _standalone_or_existing_attempt(self, render_attempt: object | None) -> object:
        """Return an explicit token or claim a fresh one for standalone use."""
        if render_attempt is not None:
            return render_attempt
        token = object()
        self._render_attempt = token
        return token

    async def _replace_scroll_contents(
        self, widgets: list[Static], render_attempt: object
    ) -> bool:
        """Serialize a guarded remove-and-mount transcript replacement."""
        async with self._render_lock:
            if self._render_attempt is not render_attempt:
                return False
            scroll = self.query_one("#personas-transcript-scroll", VerticalScroll)
            await scroll.remove_children()
            if self._render_attempt is not render_attempt:
                return False
            await scroll.mount_all(widgets)
            return self._render_attempt is render_attempt


__all__ = ["PersonasConversationTranscriptWidget"]
