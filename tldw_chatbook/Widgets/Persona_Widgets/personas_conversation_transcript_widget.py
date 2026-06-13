"""Read-only conversation transcript for the Personas workbench.

Replaces ``CCPConversationViewWidget`` on the Personas screen only, rendering
the flat transcript-line pattern the preview pane established: one Static per
message, ``role: content``, no per-message chrome.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Static


class PersonasConversationTranscriptWidget(Container):
    """Flat, read-only transcript of a saved conversation."""

    # Structure only: colors come from the app stylesheet ($ds-* tokens do not
    # resolve in bare-App harnesses, so DEFAULT_CSS must not reference them).
    DEFAULT_CSS = """
    PersonasConversationTranscriptWidget {
        width: 100%;
        height: 100%;
    }

    PersonasConversationTranscriptWidget #personas-transcript-scroll {
        height: 1fr;
    }

    PersonasConversationTranscriptWidget .personas-transcript-line {
        height: auto;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("id", "personas-conversation-transcript-view")
        super().__init__(**kwargs)

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
        yield VerticalScroll(id="personas-transcript-scroll")

    def set_title(self, title: str) -> None:
        """Update the transcript header line."""
        self.query_one("#personas-transcript-title", Static).update(
            str(title or "Conversation")
        )

    async def show_loading(self) -> None:
        """Replace the transcript with a loading placeholder.

        Gives the conversation click instant feedback while the message
        worker fetches; ``load_messages`` replaces it with the content.
        """
        scroll = self.query_one("#personas-transcript-scroll", VerticalScroll)
        await scroll.remove_children()
        await scroll.mount(
            Static("Loading transcript...", id="personas-transcript-loading")
        )

    async def load_messages(
        self,
        messages: List[Dict[str, Any]],
        speaker_names: Optional[Dict[str, str]] = None,
    ) -> None:
        """Replace the transcript with ``messages`` (async-safe replace).

        ``speaker_names`` maps roles to display names (e.g. ``{"user":
        "You", "assistant": "Detective Sam"}``); unmapped roles render as
        the raw role. Role CSS classes are unaffected by the mapping.
        """
        names = speaker_names or {}
        scroll = self.query_one("#personas-transcript-scroll", VerticalScroll)
        await scroll.remove_children()
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
        await scroll.mount_all(widgets)


__all__ = ["PersonasConversationTranscriptWidget"]
