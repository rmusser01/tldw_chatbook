"""Read-only conversation transcript for the Personas workbench.

Replaces ``CCPConversationViewWidget`` on the Personas screen only, rendering
the flat transcript-line pattern the preview pane established: one Static per
message, ``role: content``, no per-message chrome.
"""

from __future__ import annotations

from typing import Any, Dict, List

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
        yield Static(
            "Conversation", id="personas-transcript-title", classes="destination-section"
        )
        yield VerticalScroll(id="personas-transcript-scroll")

    def set_title(self, title: str) -> None:
        """Update the transcript header line."""
        self.query_one("#personas-transcript-title", Static).update(
            str(title or "Conversation")
        )

    async def load_messages(self, messages: List[Dict[str, Any]]) -> None:
        """Replace the transcript with ``messages`` (async-safe replace)."""
        scroll = self.query_one("#personas-transcript-scroll", VerticalScroll)
        await scroll.remove_children()
        widgets: list[Static] = []
        for message in messages or []:
            role = str(message.get("role") or "unknown")
            content = str(message.get("content") or "")
            role_class = (
                "personas-transcript-line-user"
                if role == "user"
                else "personas-transcript-line-assistant"
            )
            widgets.append(
                Static(
                    f"{role}: {content}",
                    classes=f"personas-transcript-line {role_class}",
                )
            )
        if not widgets:
            widgets.append(
                Static("No messages to display.", id="personas-transcript-empty")
            )
        await scroll.mount_all(widgets)


__all__ = ["PersonasConversationTranscriptWidget"]
