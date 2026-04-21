"""Combined chat shell bar with explicit fallback labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from textual.app import ComposeResult
from textual.containers import Container
from textual.css.query import NoMatches
from textual.widgets import Static


@dataclass
class ChatShellLabelResolver:
    """Optional live label overrides for workspace/persona/character names."""

    workspace_name: Optional[str] = None
    persona_label: Optional[str] = None
    character_label: Optional[str] = None


@dataclass
class ChatShellContext:
    """Text labels for the combined chat shell."""

    backend_label: str
    scope_label: str
    assistant_label: str
    session_label: str

    @classmethod
    def from_tab_state(
        cls,
        tab_state: Any,
        resolver: Optional[ChatShellLabelResolver] = None,
    ) -> "ChatShellContext":
        return cls.from_session_data(tab_state, resolver=resolver)

    @classmethod
    def from_session_data(
        cls,
        session_data: Any,
        resolver: Optional[ChatShellLabelResolver] = None,
    ) -> "ChatShellContext":
        backend_value = getattr(session_data, "runtime_backend", "local") or "local"
        backend = "Server" if str(backend_value).lower() == "server" else "Local"

        scope_type = getattr(session_data, "scope_type", None) or "global"
        workspace_id = getattr(session_data, "workspace_id", None)
        workspace_name = getattr(resolver, "workspace_name", None) if resolver else None
        if scope_type == "workspace":
            scope_target = workspace_name or workspace_id
            scope = f"Workspace: {scope_target}" if scope_target else "Global"
        else:
            scope = "Global"

        assistant_kind = getattr(session_data, "assistant_kind", None)
        if assistant_kind == "character":
            character_name = getattr(resolver, "character_label", None) if resolver else None
            character_name = (
                character_name
                or getattr(session_data, "character_name", None)
                or getattr(session_data, "character_id", None)
            )
            assistant = f"Character: {character_name}" if character_name else "Assistant: General"
        elif assistant_kind == "persona":
            persona_label = getattr(resolver, "persona_label", None) if resolver else None
            persona_value = persona_label or getattr(session_data, "assistant_id", None)
            assistant = f"Persona: {persona_value}" if persona_value else "Assistant: General"
        else:
            assistant = "Assistant: General"

        title = getattr(session_data, "title", None) or "New chat"
        return cls(backend, scope, assistant, f"Session: {title}")

    def prioritized_segments(self, max_width: int) -> list[str]:
        primary = [self.backend_label, self.scope_label, self.assistant_label]
        session = self.session_label

        while len(" | ".join(primary + [session])) > max_width and len(session) > len("Session:"):
            session = session[:-1]

        if len(" | ".join(primary + [session])) <= max_width:
            return primary + [session]
        return primary

    def format(self, max_width: int) -> str:
        return " | ".join(self.prioritized_segments(max_width))


class ChatShellBar(Container):
    """Container for the session labels and embedded compact controls."""

    DEFAULT_CSS = """
    ChatShellBar {
        width: 100%;
        layout: grid;
        grid-size: 2 1;
        grid-columns: 1fr auto;
    }

    #chat-shell-context {
        width: 100%;
        min-width: 0;
    }

    .chat-shell-controls {
        width: auto;
    }
    """

    def __init__(
        self,
        session_data: Any = None,
        *,
        tab_state: Any = None,
        resolver: Optional[ChatShellLabelResolver] = None,
        app_instance: Any = None,
        on_sidebar_toggle_requested: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.session_data = session_data if session_data is not None else tab_state
        self.resolver = resolver
        self.app_instance = app_instance
        self.on_sidebar_toggle_requested = on_sidebar_toggle_requested
        self.context = ChatShellContext.from_session_data(self.session_data, resolver=resolver)

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.compact_model_bar import CompactModelBar

        yield Static(
            self.context.session_label,
            id="chat-shell-context",
            classes="chat-shell-context",
            expand=True,
            shrink=True,
        )
        yield CompactModelBar(
            self.app_instance,
            on_sidebar_toggle_requested=self.on_sidebar_toggle_requested,
            classes="chat-shell-controls",
        )

    def on_mount(self) -> None:
        self.call_after_refresh(self.refresh_context_label)

    def on_resize(self, event: Any) -> None:
        self.call_after_refresh(self.refresh_context_label)

    def sync_from_session_data(
        self,
        session_data: Any,
        resolver: Optional[ChatShellLabelResolver] = None,
    ) -> None:
        self.session_data = session_data
        self.resolver = resolver
        self.context = ChatShellContext.from_session_data(self.session_data, resolver=self.resolver)
        self.refresh_context_label()

    def sync_from_tab_state(
        self,
        tab_state: Any,
        resolver: Optional[ChatShellLabelResolver] = None,
    ) -> None:
        self.session_data = tab_state
        self.resolver = resolver
        self.context = ChatShellContext.from_tab_state(self.session_data, resolver=self.resolver)
        self.refresh_context_label()

    def refresh_context_label(self) -> None:
        try:
            label = self.query_one("#chat-shell-context", Static)
        except NoMatches:
            return

        available_width = label.size.width or self.size.width
        if available_width <= 0:
            return
        label.update(self.context.format(available_width))
