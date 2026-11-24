"""Inline and narrow-screen lookup across one workspace authority."""

from contextlib import aclosing

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static

from ...Research_Workspace.controller import ResearchWorkspaceController
from ...Research_Workspace.workspace_search import (
    MAX_SEARCH_QUERY,
    WorkspaceSearchHit,
    search_workspace,
)
from ...Widgets.modal_dismissal import SafeModalDismissMixin


SEARCH_RESULT_PAGE_SIZE = 12


class ResearchWorkspaceSearch(Vertical):
    class ResultChosen(Message):
        def __init__(self, hit: WorkspaceSearchHit, context_revision: int) -> None:
            super().__init__()
            self.hit = hit
            self.context_revision = context_revision

    def __init__(self, controller: ResearchWorkspaceController, **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self._query = ""
        self._offset = 0
        self._revision = None

    def compose(self) -> ComposeResult:
        yield Input(
            id="research-global-search-field",
            placeholder="Search workspace names, sources, notes, and outputs",
            max_length=MAX_SEARCH_QUERY,
        )
        with Vertical(id="research-global-search-panel"):
            yield Static("", id="research-global-search-status", markup=False)
            yield Select(
                [],
                prompt="Search results",
                id="research-global-search-results",
                compact=True,
            )
            with Horizontal(id="research-global-search-actions"):
                yield Button(
                    "Open",
                    id="research-global-search-open",
                    compact=True,
                    disabled=True,
                )
                yield Button(
                    "Previous",
                    id="research-global-search-back",
                    compact=True,
                    disabled=True,
                )
                yield Button(
                    "Next",
                    id="research-global-search-next",
                    compact=True,
                    disabled=True,
                )
                yield Button("Clear", id="research-global-search-clear", compact=True)

    def on_mount(self) -> None:
        self.query_one("#research-global-search-panel").display = False

    def focus_search(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted, "#research-global-search-field")
    def submit(self, event: Input.Submitted) -> None:
        event.stop()
        self._query = event.value.strip()
        self._offset = 0
        self._start_search()

    @on(Input.Changed, "#research-global-search-field")
    def changed(self, event: Input.Changed) -> None:
        if event.value.strip() != self._query:
            self._revision = None
            self.query_one("#research-global-search-open", Button).disabled = True
            if self.query_one("#research-global-search-panel").display:
                self.query_one("#research-global-search-status", Static).update(
                    "Press Enter to search with the updated text."
                )

    def _start_search(self) -> None:
        self.query_one("#research-global-search-panel").display = True
        self._set_results((), has_more=False)
        self.query_one("#research-global-search-status", Static).update(
            "Searching the selected authority…"
        )
        self.run_worker(
            self._find(self._query, self._offset),
            group="research-workspace-search",
            exclusive=True,
            exit_on_error=False,
        )

    async def _find(self, query: str, offset: int) -> None:
        controller = self.controller
        revision = controller.context_revision
        authority = controller.selected_data_source
        catalog = controller.catalog_state
        port = controller.port_for_data_source(authority)
        matches = []
        message = f"No matches in {authority.value.title()} workspace data."
        incomplete = False
        try:
            if catalog is None or port is None or catalog.recovery is not None:
                raise ValueError(
                    f"The {authority.value.title()} workspace catalog is unavailable."
                )
            async with aclosing(search_workspace(port, catalog, query)) as hits:
                index = 0
                async for hit in hits:
                    if index >= offset:
                        matches.append(hit)
                    if len(matches) > SEARCH_RESULT_PAGE_SIZE:
                        break
                    index += 1
        except (ValueError, RuntimeError, OSError) as exc:
            incomplete = True
            message = (
                str(exc)
                if type(exc) is ValueError
                else "Search could not read all selected owners. Try a specific category."
            )
        except Exception:
            incomplete = True
            message = "Search could not read the selected authority. Refresh the workspace and retry."
        if (
            not self.is_mounted
            or self._query != query
            or self.query_one(Input).value.strip() != query
            or self._offset != offset
            or controller.context_revision != revision
            or controller.selected_data_source is not authority
        ):
            return
        self._revision = revision
        if matches:
            message = f"{authority.value.title()} · Results {offset + 1}–{offset + min(len(matches), SEARCH_RESULT_PAGE_SIZE)}"
            if incomplete:
                message += " · Some owners could not be read"
        self.query_one("#research-global-search-status", Static).update(message)
        self._set_results(
            matches[:SEARCH_RESULT_PAGE_SIZE],
            has_more=len(matches) > SEARCH_RESULT_PAGE_SIZE,
        )

    def _set_results(self, matches, *, has_more: bool) -> None:
        select = self.query_one(Select)
        select.set_options(
            (
                Text(
                    hit.title
                    if hit.kind == "workspace"
                    else f"{hit.kind.title()} · {hit.title} · {hit.workspace_name}"
                ),
                hit,
            )
            for hit in matches
        )
        select.value = matches[0] if matches else Select.NULL
        self.query_one("#research-global-search-open", Button).disabled = not matches
        self.query_one(
            "#research-global-search-back", Button
        ).disabled = not self._offset
        self.query_one("#research-global-search-next", Button).disabled = not has_more

    @on(Button.Pressed, "#research-global-search-back")
    @on(Button.Pressed, "#research-global-search-next")
    def page(self, event: Button.Pressed) -> None:
        self._offset = max(
            0,
            self._offset
            + (
                SEARCH_RESULT_PAGE_SIZE
                if event.button.id.endswith("next")
                else -SEARCH_RESULT_PAGE_SIZE
            ),
        )
        self._start_search()

    @on(Button.Pressed, "#research-global-search-open")
    def open_result(self) -> None:
        hit = self.query_one(Select).value
        if (
            isinstance(hit, WorkspaceSearchHit)
            and self._revision == self.controller.context_revision
            and self.query_one(Input).value.strip() == self._query
        ):
            self.post_message(self.ResultChosen(hit, self._revision))
            self.clear()

    @on(Button.Pressed, "#research-global-search-clear")
    def clear(self) -> None:
        self._query = ""
        self._offset = 0
        self._revision = None
        self._set_results((), has_more=False)
        self.query_one(Input).value = ""
        self.query_one("#research-global-search-panel").display = False


class ResearchWorkspaceSearchModal(
    SafeModalDismissMixin, ModalScreen[ResearchWorkspaceSearch.ResultChosen | None]
):
    BINDINGS = [Binding("escape", "request_safe_cancel", "Close", show=False)]
    SAFE_MODAL_CONTENT = "#research-global-search-dialog"
    AUTO_FOCUS = "#research-global-search-field"

    def __init__(self, controller: ResearchWorkspaceController) -> None:
        super().__init__()
        self.controller = controller

    def compose(self) -> ComposeResult:
        with Vertical(
            id="research-global-search-dialog", classes="research-transfer-dialog"
        ):
            yield Static(
                f"Search {self.controller.selected_data_source.value.title()} workspace data",
                classes="research-pane-title",
            )
            yield ResearchWorkspaceSearch(self.controller)
            yield Button("Close", id="research-global-search-close", compact=True)

    @on(ResearchWorkspaceSearch.ResultChosen)
    def open_result(self, message: ResearchWorkspaceSearch.ResultChosen) -> None:
        message.stop()
        self.dismiss(message)

    @on(Button.Pressed, "#research-global-search-close")
    def close(self) -> None:
        self.dismiss(None)
