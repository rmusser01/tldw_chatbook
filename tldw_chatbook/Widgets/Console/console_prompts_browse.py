"""Source-scoped Browse renderer for :class:`ConsolePromptsModal`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Select, Static

from .console_prompts_state import PromptBrowseResult, PromptSource


def _row_token(identifier: str) -> str:
    if identifier and all(char.isalnum() or char in "_-" for char in identifier):
        return identifier
    return f"encoded-{identifier.encode('utf-8').hex()}"


def _identifier(item: Mapping[str, Any]) -> str:
    value = (
        item.get("source_id")
        or item.get("id")
        or item.get("uuid")
        or item.get("name", "")
    )
    return str(value)


class ConsolePromptsBrowse(Vertical):
    """Render Prompt rows and emit source/search/pagination intent only."""

    class ImproveRequested(Message):
        """Open model-assisted prompt choices."""

    class ConfigureProviderRequested(Message):
        """Open the host Console's provider/model recovery surface."""

    class SourceChanged(Message):
        def __init__(self, source: PromptSource) -> None:
            self.source = source
            super().__init__()

    class QueryChanged(Message):
        def __init__(self, query: str) -> None:
            self.query = query
            super().__init__()

    class PageRequested(Message):
        def __init__(self, page: int) -> None:
            self.page = page
            super().__init__()

    class RetryRequested(Message):
        """Retry the active list/search operation."""

    class ArtifactSelected(Message):
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier
            super().__init__()

    def __init__(
        self,
        *,
        source: PromptSource,
        query: str,
        page: int,
        improve_unavailable_reason: str = "",
        can_configure_provider: bool = False,
        manual_improve_available: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._source = source
        self._query = query
        self._page = page
        self._improve_unavailable_reason = improve_unavailable_reason.strip()
        self._can_configure_provider = bool(can_configure_provider)
        self._manual_improve_available = bool(manual_improve_available)
        self._row_ids: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        improve = Button(
            "Improve My Prompt",
            id="console-prompts-improve",
            classes="console-prompts-primary-action",
            disabled=bool(
                self._improve_unavailable_reason and not self._manual_improve_available
            ),
        )
        if self._improve_unavailable_reason:
            improve.tooltip = self._improve_unavailable_reason
        yield improve
        yield Static(
            (
                (
                    f"Model improvement unavailable — {self._improve_unavailable_reason} "
                    "Browse and structured Recipe editing remain available."
                    if self._manual_improve_available
                    else f"Improve unavailable — {self._improve_unavailable_reason} "
                    "Browse and manual editing remain available."
                )
                if self._improve_unavailable_reason
                else "Browse saved Prompts and Recipes by one source at a time."
            ),
            id="console-prompts-model-status",
            markup=False,
        )
        if self._improve_unavailable_reason and self._can_configure_provider:
            yield Button(
                "Configure provider / model",
                id="console-prompts-configure-provider",
            )
        with Horizontal(id="console-prompts-browse-controls"):
            yield Select(
                (("Local", "local"), ("Server", "server")),
                value=self._source,
                allow_blank=False,
                id="console-prompts-source",
            )
            yield Input(
                value=self._query,
                placeholder="Search names, tags, details, and Prompt content",
                id="console-prompts-search",
            )
        yield Static(
            "Loading Prompt Library…",
            id="console-prompts-browse-status",
            markup=False,
        )
        retry = Button("Retry", id="console-prompts-retry")
        retry.display = False
        yield retry
        with VerticalScroll(id="console-prompts-results"):
            yield Vertical(id="console-prompts-result-list")
        with Horizontal(id="console-prompts-pagination"):
            yield Button("Previous", id="console-prompts-previous", disabled=True)
            yield Static("Page 1 of 1", id="console-prompts-page", markup=False)
            yield Button("Next", id="console-prompts-next", disabled=True)

    def show_loading(self, *, source: PromptSource, query: str) -> None:
        action = "Searching" if query.strip() else "Loading"
        self.show_status(
            f"{action} {source.title()} Prompt Library…",
            retry=False,
        )

    def show_status(self, message: str, *, retry: bool = False) -> None:
        self.query_one("#console-prompts-browse-status", Static).update(message)
        self.query_one("#console-prompts-retry", Button).display = retry

    async def show_result(self, result: PromptBrowseResult, *, query: str) -> None:
        self._source = result.source
        self._query = query
        self._page = result.page
        rows = self.query_one("#console-prompts-result-list", Vertical)
        await rows.remove_children()
        self._row_ids.clear()
        for item in result.items:
            identifier = _identifier(item)
            token = _row_token(str(item.get("id") or identifier))
            self._row_ids[token] = identifier
            artifact_type = str(item.get("artifact_type") or "prompt").title()
            source = str(item.get("backend") or result.source).title()
            has_system = bool(item.get("has_system_prompt"))
            has_user = bool(item.get("has_user_prompt"))
            lanes = (
                "System + User"
                if has_system and has_user
                else "System"
                if has_system
                else "User"
                if has_user
                else "No compiled lanes"
            )
            updated = item.get("updated_at") or item.get("last_modified")
            metadata = f"{artifact_type} · {source} · {lanes}"
            if updated:
                metadata += f" · Updated {updated}"
            label = Text()
            label.append(str(item.get("name") or "Untitled Prompt"), style="bold")
            label.append(f"\n{metadata}")
            await rows.mount(
                Button(
                    label,
                    id=f"console-prompts-result-{token}",
                    classes="console-prompts-result",
                )
            )

        if result.items:
            self.show_status(
                f"{result.total_items} item{'s' if result.total_items != 1 else ''} "
                f"in {result.source.title()} · Prompt and Recipe types are labeled below."
            )
        elif query.strip():
            self.show_status(
                f'No matches for "{query.strip()}" — Change the query or switch source.'
            )
        else:
            self.show_status(
                f"{result.source.title()} Prompt Library is empty — "
                "Create or save a Prompt, then Retry."
            )
        self.query_one("#console-prompts-page", Static).update(
            f"Page {result.page} of {result.total_pages}"
        )
        self.query_one("#console-prompts-previous", Button).disabled = result.page <= 1
        self.query_one("#console-prompts-next", Button).disabled = (
            bool(query.strip()) or result.page >= result.total_pages
        )

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if not button_id.startswith("console-prompts-"):
            return
        if button_id == "console-prompts-improve":
            event.stop()
            self.post_message(self.ImproveRequested())
        elif button_id == "console-prompts-configure-provider":
            event.stop()
            self.post_message(self.ConfigureProviderRequested())
        elif button_id == "console-prompts-retry":
            event.stop()
            self.post_message(self.RetryRequested())
        elif button_id == "console-prompts-previous":
            event.stop()
            self.post_message(self.PageRequested(max(1, self._page - 1)))
        elif button_id == "console-prompts-next":
            event.stop()
            self.post_message(self.PageRequested(self._page + 1))
        elif button_id.startswith("console-prompts-result-"):
            event.stop()
            token = button_id.removeprefix("console-prompts-result-")
            identifier = self._row_ids.get(token)
            if identifier is not None:
                self.post_message(self.ArtifactSelected(identifier))

    @on(Select.Changed, "#console-prompts-source")
    def _source_changed(self, event: Select.Changed) -> None:
        value = str(event.value)
        if value in {"local", "server"} and value != self._source:
            event.stop()
            self.post_message(self.SourceChanged(value))  # type: ignore[arg-type]

    @on(Input.Changed, "#console-prompts-search")
    def _query_changed(self, event: Input.Changed) -> None:
        if event.value == self._query:
            return
        event.stop()
        self.post_message(self.QueryChanged(event.value))


__all__ = ["ConsolePromptsBrowse"]
