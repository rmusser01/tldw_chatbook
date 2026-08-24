"""Authority-explicit intake dialog for Research Workspace sources."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from ...Research_Workspace import (
    BoundedPageResult,
    ResearchCatalogItem,
    WorkspaceDataSource,
)
from ...Third_Party.textual_fspicker import FileOpen
from ...Widgets.modal_dismissal import SafeModalDismissMixin


ResearchIntakeKind = Literal["file", "url", "paste", "existing", "catalog"]


@dataclass(frozen=True, slots=True)
class ResearchSourceIntakeRequest:
    """Validated modal result; the screen captures the qualified owner."""

    kind: ResearchIntakeKind
    values: tuple[str, ...]
    title: str = ""


_TYPE_OPTIONS = (
    ("All types", ""),
    ("PDF", "pdf"),
    ("Video", "video"),
    ("Audio", "audio"),
    ("Website", "website"),
    ("Document", "document"),
    ("Text", "text"),
)
_SORT_OPTIONS = (
    ("Newest", "updated_desc"),
    ("Oldest", "updated_asc"),
    ("Title A-Z", "title_asc"),
    ("Title Z-A", "title_desc"),
)


class ResearchAddSourceModal(
    SafeModalDismissMixin, ModalScreen[ResearchSourceIntakeRequest | None]
):
    """Five-path intake modal whose authority never changes while open."""

    BINDINGS = [Binding("escape", "request_safe_cancel", "Cancel", show=False)]
    SAFE_MODAL_CONTENT = "#research-add-source-dialog"

    def __init__(
        self,
        data_source: WorkspaceDataSource,
        *,
        catalog_search: Callable[..., Awaitable[BoundedPageResult | None]]
        | None = None,
    ) -> None:
        super().__init__(id="research-add-source-modal")
        self.data_source = WorkspaceDataSource(data_source)
        self._catalog_search = catalog_search
        self._catalog_offsets = {"existing": 0, "search": 0}
        self._catalog_pages: dict[str, BoundedPageResult[ResearchCatalogItem]] = {}

    def compose(self) -> ComposeResult:
        local = self.data_source is WorkspaceDataSource.LOCAL
        with Vertical(id="research-add-source-dialog"):
            with Horizontal(id="research-add-source-heading-row"):
                yield Static("Add Sources", id="research-add-source-title")
                yield Static(
                    f"Workspace data: {self.data_source.value.title()}",
                    id="research-add-authority",
                )
                yield Button("Close", id="research-add-close", compact=True)
            yield Static("", id="research-add-error", markup=False)
            with TabbedContent(id="research-add-source-tabs"):
                with TabPane(
                    "Import Files" if local else "Upload", id="research-add-tab-upload"
                ):
                    with VerticalScroll(classes="research-add-tab-body"):
                        yield Static(
                            "Choose one source file. It creates one durable workspace receipt.",
                            id="research-add-upload-scope",
                            classes="research-add-help",
                        )
                        yield Input(
                            placeholder="Choose one source file",
                            id="research-add-upload-path",
                        )
                        with Horizontal(classes="research-add-actions"):
                            yield Button("Browse", id="research-add-upload-browse")
                            yield Button(
                                "Import" if local else "Upload",
                                id="research-add-upload-submit",
                                variant="primary",
                            )
                        yield Static(
                            "Progress and sanitized errors remain in Sources receipts after this dialog closes.",
                            id="research-add-upload-progress",
                        )
                with TabPane(
                    "Local Library" if local else "My Media",
                    id="research-add-tab-existing",
                ):
                    yield from self._catalog_controls(prefix="existing")
                with TabPane("URL", id="research-add-tab-url"):
                    with VerticalScroll(classes="research-add-tab-body"):
                        with Horizontal(classes="research-add-actions"):
                            yield Button(
                                "Single URL",
                                id="research-add-url-mode-single",
                                classes="is-active",
                            )
                            yield Button(
                                "Batch (one per line)", id="research-add-url-mode-batch"
                            )
                        yield Input(
                            placeholder="https://example.com/article",
                            id="research-add-url-single",
                        )
                        yield TextArea(id="research-add-url-batch")
                        yield Button(
                            "Add URL", id="research-add-url-submit", variant="primary"
                        )
                        yield Static(
                            "Batch intake creates one qualified operation per URL.",
                            id="research-add-url-progress",
                        )
                with TabPane("Paste", id="research-add-tab-paste"):
                    with VerticalScroll(classes="research-add-tab-body"):
                        yield Input(
                            placeholder="Title (optional)",
                            id="research-add-paste-title",
                        )
                        yield TextArea(id="research-add-paste-body")
                        yield Button(
                            "Add pasted text",
                            id="research-add-paste-submit",
                            variant="primary",
                        )
                        yield Static(
                            "Your draft stays here when intake cannot start.",
                            id="research-add-paste-progress",
                        )
                with TabPane(
                    "Search Local" if local else "Search Server",
                    id="research-add-tab-search",
                ):
                    yield from self._catalog_controls(prefix="search")

    def _catalog_controls(self, *, prefix: str) -> ComposeResult:
        with VerticalScroll(classes="research-add-tab-body"):
            if prefix == "existing":
                yield Static(
                    "Choose one existing item per Add Sources action.",
                    id="research-add-existing-selection-scope",
                    classes="research-add-help",
                    markup=False,
                )
            if prefix == "search" and self.data_source is WorkspaceDataSource.SERVER:
                yield Static(
                    "[Unavailable] Web search is unavailable here. Configure a "
                    "web-search provider, then add selected result URLs through URL intake.",
                    id="research-add-search-unavailable",
                    markup=False,
                )
            yield Input(
                placeholder=(
                    "Search Local Library"
                    if self.data_source is WorkspaceDataSource.LOCAL
                    else "Search Server Media"
                ),
                id=f"research-add-{prefix}-query",
            )
            with Horizontal(classes="research-add-filters"):
                yield Select(
                    _TYPE_OPTIONS,
                    value="",
                    allow_blank=False,
                    id=f"research-add-{prefix}-type",
                )
                yield Select(
                    _SORT_OPTIONS,
                    value="updated_desc",
                    allow_blank=False,
                    id=f"research-add-{prefix}-sort",
                )
            yield Button("Search", id=f"research-add-{prefix}-search", compact=True)
            yield Select(
                (("No catalog results loaded", ""),),
                value="",
                allow_blank=True,
                id=f"research-add-{prefix}-results",
            )
            with Horizontal(classes="research-add-actions"):
                yield Button(
                    "Previous", id=f"research-add-{prefix}-prev", disabled=True
                )
                yield Button("Next", id=f"research-add-{prefix}-next", disabled=True)
                yield Button(
                    "Select result", id=f"research-add-{prefix}-select", disabled=True
                )
                yield Button(
                    "Add selected",
                    id=f"research-add-{prefix}-submit",
                    variant="primary",
                    disabled=True,
                )
            yield Static("Page 1 · no results", id=f"research-add-{prefix}-page")

    def _show_error(self, message: str) -> None:
        error = self.query_one("#research-add-error", Static)
        error.update(message)
        error.display = bool(message)

    @on(Button.Pressed, "#research-add-close")
    def close_modal(self) -> None:
        self.dismiss(None)

    async def _perform_safe_cancel(self, *, source: str) -> None:
        """Dismiss safely while leaving the mounted draft untouched for recovery."""

        del source
        self.dismiss_safe_once(None)

    @on(Button.Pressed, "#research-add-upload-browse")
    def browse_source(self) -> None:
        def selected(path: Path | None) -> None:
            if path is not None:
                self.query_one("#research-add-upload-path", Input).value = str(path)

        self.app.push_screen(
            FileOpen(title="Choose one source file"), callback=selected
        )

    @on(Button.Pressed, "#research-add-upload-submit")
    def submit_file(self) -> None:
        raw = self.query_one("#research-add-upload-path", Input).value.strip()
        if not raw:
            self._show_error("Choose one source file before continuing.")
            return
        self.dismiss(ResearchSourceIntakeRequest("file", (raw,)))

    @on(Button.Pressed, "#research-add-url-submit")
    def submit_urls(self) -> None:
        batch_widget = self.query_one("#research-add-url-batch", TextArea)
        single_widget = self.query_one("#research-add-url-single", Input)
        raw_values = (
            batch_widget.text.splitlines()
            if batch_widget.display
            else (single_widget.value,)
        )
        values = tuple(
            dict.fromkeys(value.strip() for value in raw_values if value.strip())
        )
        if not values or any(
            not value.startswith(("http://", "https://")) for value in values
        ):
            self._show_error("Enter HTTP or HTTPS URLs, one per line for batch intake.")
            return
        self.dismiss(ResearchSourceIntakeRequest("url", values))

    @on(Button.Pressed, "#research-add-paste-submit")
    def submit_paste(self) -> None:
        body = self.query_one("#research-add-paste-body", TextArea).text.strip()
        if not body:
            self._show_error("Paste source text before continuing.")
            return
        title = self.query_one("#research-add-paste-title", Input).value.strip()
        self.dismiss(ResearchSourceIntakeRequest("paste", (body,), title=title))

    @on(Button.Pressed, "#research-add-url-mode-single")
    @on(Button.Pressed, "#research-add-url-mode-batch")
    def switch_url_mode(self, event: Button.Pressed) -> None:
        batch = str(event.button.id).endswith("-batch")
        self.query_one("#research-add-url-single", Input).display = not batch
        self.query_one("#research-add-url-batch", TextArea).display = batch
        self.query_one("#research-add-url-mode-single", Button).set_class(
            not batch, "is-active"
        )
        self.query_one("#research-add-url-mode-batch", Button).set_class(
            batch, "is-active"
        )

    @on(Button.Pressed, "#research-add-existing-search")
    @on(Button.Pressed, "#research-add-search-search")
    def search_catalog(self, event: Button.Pressed) -> None:
        prefix = "existing" if "-existing-" in str(event.button.id) else "search"
        self._catalog_offsets[prefix] = 0
        self._start_catalog_search(prefix)

    @on(Button.Pressed, "#research-add-existing-prev")
    @on(Button.Pressed, "#research-add-existing-next")
    @on(Button.Pressed, "#research-add-search-prev")
    @on(Button.Pressed, "#research-add-search-next")
    def page_catalog(self, event: Button.Pressed) -> None:
        widget_id = str(event.button.id)
        prefix = "existing" if "-existing-" in widget_id else "search"
        delta = -25 if widget_id.endswith("-prev") else 25
        self._catalog_offsets[prefix] = max(0, self._catalog_offsets[prefix] + delta)
        self._start_catalog_search(prefix)

    @on(Select.Changed, "#research-add-existing-results")
    @on(Select.Changed, "#research-add-search-results")
    def select_catalog_result(self, event: Select.Changed) -> None:
        prefix = "existing" if "-existing-" in str(event.select.id) else "search"
        selected = bool(str(event.value or ""))
        self.query_one(f"#research-add-{prefix}-select", Button).disabled = not selected
        self.query_one(f"#research-add-{prefix}-submit", Button).disabled = not selected

    @on(Button.Pressed, "#research-add-existing-select")
    @on(Button.Pressed, "#research-add-search-select")
    def confirm_catalog_result(self, event: Button.Pressed) -> None:
        prefix = "existing" if "-existing-" in str(event.button.id) else "search"
        selected = str(
            self.query_one(f"#research-add-{prefix}-results", Select).value or ""
        )
        if selected:
            self.query_one(f"#research-add-{prefix}-submit", Button).disabled = False

    @on(Button.Pressed, "#research-add-existing-submit")
    @on(Button.Pressed, "#research-add-search-submit")
    def submit_catalog_result(self, event: Button.Pressed) -> None:
        prefix = "existing" if "-existing-" in str(event.button.id) else "search"
        selected = str(
            self.query_one(f"#research-add-{prefix}-results", Select).value or ""
        )
        if not selected:
            self._show_error("Select one catalog result before continuing.")
            return
        kind: ResearchIntakeKind = "existing" if prefix == "existing" else "catalog"
        self.dismiss(ResearchSourceIntakeRequest(kind, (selected,)))

    def _start_catalog_search(self, prefix: str) -> None:
        if prefix == "search" and self.data_source is WorkspaceDataSource.SERVER:
            self._show_error(
                "Web search is unavailable here. Configure a web-search provider, "
                "then add selected result URLs through URL intake."
            )
            return
        if self._catalog_search is None:
            self._show_error("Catalog search is unavailable for this authority.")
            return
        self.run_worker(
            self._search_catalog(prefix),
            group=f"research-add-{prefix}-catalog",
            exclusive=True,
        )

    async def _search_catalog(self, prefix: str) -> None:
        callback = self._catalog_search
        if callback is None:
            return
        query = self.query_one(f"#research-add-{prefix}-query", Input).value
        source_type = str(
            self.query_one(f"#research-add-{prefix}-type", Select).value or ""
        )
        sort_by = str(
            self.query_one(f"#research-add-{prefix}-sort", Select).value
            or "updated_desc"
        )
        try:
            page = await callback(
                query=query,
                source_types=(source_type,) if source_type else (),
                sort_by=sort_by,
                limit=25,
                offset=self._catalog_offsets[prefix],
            )
        except Exception:
            self._show_error("Catalog search failed for the selected authority.")
            return
        if page is None:
            self._show_error("Catalog context changed; reopen Add Sources.")
            return
        self._catalog_pages[prefix] = page
        results = self.query_one(f"#research-add-{prefix}-results", Select)
        results.set_options(
            [
                (f"{item.title} · {item.source_type}", item.catalog_item_id)
                for item in page.items
            ]
            or [("No results", "")]
        )
        results.clear()
        self.query_one(f"#research-add-{prefix}-prev", Button).disabled = (
            page.offset == 0
        )
        self.query_one(
            f"#research-add-{prefix}-next", Button
        ).disabled = not page.has_more
        self.query_one(f"#research-add-{prefix}-page", Static).update(
            f"Page {page.offset // page.limit + 1} · {page.total if page.total is not None else 'bounded results'}"
        )
        self._show_error("")

    def on_mount(self) -> None:
        self.query_one("#research-add-error", Static).display = False
        self.query_one("#research-add-url-batch", TextArea).display = False
        if self.data_source is WorkspaceDataSource.SERVER:
            for suffix in (
                "query",
                "type",
                "sort",
                "search",
                "results",
                "prev",
                "next",
                "select",
                "submit",
            ):
                control = self.query_one(f"#research-add-search-{suffix}")
                control.disabled = True
                control.tooltip = (
                    "Web search unavailable; configure a web-search provider and use URL intake."
                )
