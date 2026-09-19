"""Library region and safe compact selection/confirmation overlays."""

import asyncio
import sqlite3
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from tldw_chatbook.Workflows.catalog import discover
from tldw_chatbook.Workflows.document_service import PAGE_SIZE


class WorkflowButton(Button):
    """Button-local Space activation, never a printable screen shortcut."""

    BINDINGS: ClassVar = [("space", "press", "Activate")]


def compact_button(label: str, identifier: str, **kwargs: Any) -> Button:
    """Create a button styled as a compact one-row workflow action.

    Args:
        label: Literal button text, also used as its tooltip.
        identifier: Textual widget ID.
        **kwargs: Additional Button constructor options, such as ``disabled``.
            Label, ID and tooltip are supplied by this helper.

    Returns:
        A WorkflowButton with the ``workflow-compact`` CSS class.
    """
    button = WorkflowButton(Text(label), id=identifier, tooltip=label, **kwargs)
    button.add_class("workflow-compact")
    return button


class ChoiceModal(ModalScreen[str | None]):
    """Choose an option ID or entered name; cancellation returns None.

    Escape and Cancel dismiss without a choice. Callers own any resulting
    mutation and restore focus to their opener.

    Attributes:
        title_text: Dialog heading.
        choices: Ordered ``(label, option_id)`` pairs with unique option IDs.
        detail: Literal explanatory text.
        text_input: Whether choices submit the name input instead of an ID.
        scroll_detail: Whether explanatory text has a scrollable container.
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        title: str,
        choices: tuple[tuple[str, str], ...],
        *,
        detail: str = "",
        text_input: bool = False,
        scroll_detail: bool = False,
    ) -> None:
        """Initialize a selection or name-entry dialog.

        Args:
            title: Dialog heading.
            choices: Ordered ``(label, option_id)`` pairs with unique IDs.
            detail: Literal explanatory text.
            text_input: Include a name input and return its value on selection.
            scroll_detail: Place the detail text in a scrollable container.
        """
        super().__init__()
        self.title_text, self.choices, self.detail = title, choices, detail
        self.text_input = text_input
        self.scroll_detail = scroll_detail
        self.set_class(scroll_detail, "has-scroll-detail")

    def compose(self) -> ComposeResult:
        """Build the heading, detail, optional name input and choice controls.

        Yields:
            Widgets for the configured dialog content.

        Raises:
            DuplicateID: Multiple choices use the same option ID.
        """
        with Vertical():
            yield Label(self.title_text)
            if self.scroll_detail:
                with VerticalScroll(id="workflow-dialog-detail"):
                    yield Static(self.detail, markup=False)
            else:
                yield Static(self.detail, markup=False, classes="choice-copy")
            if self.text_input:
                yield Input(placeholder="Name", id="workflow-dialog-name")
            yield OptionList(
                *(Option(Text(label), id=key) for label, key in self.choices),
                id="workflow-dialog-choices",
            )
            yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self) -> None:
        """Focus the name input when present, otherwise the choice list.

        Returns:
            None.
        """
        self.query_one(Input if self.text_input else OptionList).focus()

    def action_cancel(self) -> None:
        """Dismiss without selecting a value.

        Returns:
            None.
        """
        self.dismiss(None)

    @on(Button.Pressed, "#workflow-dialog-cancel")
    def cancel(self, event: Button.Pressed) -> None:
        """Consume a Cancel-button press and dismiss without a value.

        Args:
            event: Press of the dialog's Cancel button.

        Returns:
            None.
        """
        event.stop()
        self.action_cancel()

    @on(OptionList.OptionSelected)
    def choose(self, event: OptionList.OptionSelected) -> None:
        """Consume a choice event and dispatch to the dialog's selection policy.

        Args:
            event: Selected option from the dialog's choice list.

        Returns:
            None.
        """
        event.stop()
        self.select_option(event.option)

    def select_option(self, option: Option) -> None:
        """Dismiss with the entered name or the selected option's ID.

        Args:
            option: Chosen list option. Its ID is used unless name entry is on.

        Returns:
            None.

        Raises:
            NoMatches: Name entry is enabled but its Input is absent.
        """
        self.dismiss(self.query_one(Input).value if self.text_input else option.id)

    @on(Input.Submitted)
    def named(self, event: Input.Submitted) -> None:
        """Submit a nonblank name, preserving the entered whitespace.

        Args:
            event: Submitted input text. Ignored unless name entry is enabled
                and the value contains a non-whitespace character.

        Returns:
            None.
        """
        if self.text_input and event.value.strip():
            event.stop()
            self.dismiss(event.value)


class PagedChoiceModal(ChoiceModal):
    """Select from bounded pages loaded by cancellable read-only workers.

    Attributes:
        loader: Read-only callback taking an offset and query, returning ordered
            ``(label, option_id)`` pairs with at most one lookahead row.
        searchable: Whether to display a search input.
        new_workflow: Whether to offer the ``__new_workflow__`` result.
        page_offset: Offset of the last successfully loaded page.
        search_query: Query of the last successfully loaded page.
    """

    def __init__(
        self,
        title: str,
        loader: Callable[[int, str], tuple[tuple[str, str], ...]],
        *,
        detail: str = "",
        searchable: bool = False,
        new_workflow: bool = False,
    ) -> None:
        """Initialize a selector with optional search and workflow creation.

        Args:
            title: Dialog heading.
            loader: Read-only callable run in a thread with ``(offset, query)``.
                Return up to ``PAGE_SIZE + 1`` pairs, using the extra row to
                signal another page. IDs must be unique and must not use
                ``__previous_page__`` or ``__next_page__``.
            detail: Literal explanation shown alongside the page number.
            searchable: Add an input that loads the first page for each query.
            new_workflow: Add a button returning ``__new_workflow__``.
        """
        super().__init__(title, (), detail=detail)
        self.loader = loader
        self.searchable = searchable
        self.new_workflow = new_workflow
        self.page_offset = 0
        self.search_query = ""

    def compose(self) -> ComposeResult:
        """Build the search, status and paged-selection controls.

        Yields:
            Widgets for loading and choosing a page item.
        """
        with Vertical():
            yield Label(self.title_text)
            if self.searchable:
                yield Input(
                    placeholder="Search all workflows", id="workflow-page-search"
                )
            yield Static(
                self.detail,
                markup=False,
                classes="choice-copy",
                id="workflow-page-status",
            )
            yield OptionList(id="workflow-dialog-choices")
            if self.new_workflow:
                yield compact_button("New workflow", "workflow-dialog-new")
            yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self) -> None:
        """Focus search or choices and schedule the initial unfiltered page.

        Returns:
            None.
        """
        self.query_one(Input if self.searchable else OptionList).focus()
        self.load_page(0, "")

    @work(exclusive=True, group="workflow-choice-page")
    async def load_page(self, offset: int, query: str) -> None:
        """Load a bounded page in an exclusive Textual worker.

        The loader runs off the app loop. OSError, RuntimeError, ValueError and
        sqlite3.Error are displayed as a retry-by-reopening message, leaving
        the list disabled. Successful loads replace rows and paging controls.

        Args:
            offset: Nonnegative row offset supplied to the loader.
            query: Search text supplied unchanged to the loader.

        Returns:
            None from the coroutine; the ``work`` decorator returns a Worker
            when the method is called to schedule it.

        Raises:
            NoMatches: The choice list or page-status widget is absent.
            DuplicateID: Loaded option IDs collide with each other or a paging
                control. This occurs after the loader exception handler.
        """
        listing = self.query_one(OptionList)
        listing.disabled = True
        status = self.query_one("#workflow-page-status", Static)
        status.update("Loading…")
        try:
            rows = await asyncio.to_thread(self.loader, offset, query)
        except (OSError, RuntimeError, ValueError, sqlite3.Error):
            status.update("Unable to load this page. Reopen the selector to retry.")
            return
        self.page_offset, self.search_query = offset, query
        options = [Option(Text(label), id=key) for label, key in rows[:PAGE_SIZE]]
        if offset:
            options.append(Option("Previous page", id="__previous_page__"))
        if len(rows) > PAGE_SIZE:
            options.append(Option("Next page", id="__next_page__"))
        listing.clear_options()
        listing.add_options(options)
        listing.highlighted = 0 if options else None
        listing.disabled = False
        status.update(
            f"Page {offset // PAGE_SIZE + 1} · {self.detail}"
            if rows
            else "No matching items"
        )

    @on(Input.Changed, "#workflow-page-search")
    def search_page(self, event: Input.Changed) -> None:
        """Consume a search edit and schedule its first result page.

        Args:
            event: Change to the page-search input.

        Returns:
            None.
        """
        event.stop()
        self.load_page(0, event.value)

    def select_option(self, option: Option) -> None:
        """Load a neighboring page or dismiss with the selected item's ID.

        Args:
            option: Item or reserved previous/next-page control from the list.

        Returns:
            None.
        """
        if option.id == "__next_page__":
            self.load_page(self.page_offset + PAGE_SIZE, self.search_query)
        elif option.id == "__previous_page__":
            self.load_page(max(0, self.page_offset - PAGE_SIZE), self.search_query)
        else:
            self.dismiss(option.id)

    @on(Button.Pressed, "#workflow-dialog-new")
    def create_workflow(self, event: Button.Pressed) -> None:
        """Dismiss with the workflow-creation sentinel for the caller to handle.

        Args:
            event: Press of the New workflow button.

        Returns:
            None.
        """
        event.stop()
        self.dismiss("__new_workflow__")


class StepChooser(ChoiceModal):
    """Review an available step type before confirming its insertion.

    Choosing a row shows its requirements; Insert returns the canonical type.
    Unavailable catalog entries are visible only under Show all and disabled.

    Attributes:
        show_all: Whether discovery includes unavailable catalog entries.
    """

    def __init__(self) -> None:
        """Initialize the chooser with only locally authorable step subsets."""
        super().__init__("Add step · Local subsets", ())
        self.show_all = False

    def compose(self) -> ComposeResult:
        """Build catalog search, detail and explicit insertion controls.

        Yields:
            Widgets for reviewing and confirming an available step type.
        """
        with Vertical():
            yield Label(self.title_text)
            yield Input(
                placeholder="Search actions or canonical types",
                id="workflow-type-search",
            )
            yield compact_button("Show all", "workflow-show-all")
            yield OptionList(id="workflow-dialog-choices")
            yield Static(
                "Select an action to review its requirements.",
                id="workflow-type-detail",
                markup=False,
                classes="choice-copy",
            )
            with Horizontal(classes="workflow-dialog-actions"):
                yield compact_button(
                    "Insert step", "workflow-insert-confirm", disabled=True
                )
                yield compact_button("Cancel · Esc", "workflow-dialog-cancel")

    def on_mount(self) -> None:
        """Populate supported types, apply chooser styling and focus search.

        Returns:
            None.
        """
        self.fill()
        self.add_class("workflow-step-chooser")
        self.query_one(Input).focus()

    def fill(self, query: str = "") -> None:
        """Refresh catalog choices and require a new insertion confirmation.

        Args:
            query: Text matched against the bundled discovery catalog.

        Returns:
            None.

        Raises:
            NoMatches: The choice list or insertion button is absent.
        """
        entries = discover(show_all=self.show_all, query=query)
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            Option(
                f"{item.family} · {item.label} ({item.step_type})\n{item.example} · "
                + (
                    "Local subset; setup checked before run"
                    if item.available
                    else "Unavailable · " + item.disposition
                ),
                id=item.step_type,
                disabled=not item.available,
            )
            for item in entries
        )
        listing.highlighted = 0 if entries else None
        self.query_one("#workflow-insert-confirm", Button).disabled = True

    @on(Input.Changed, "#workflow-type-search")
    def search(self, event: Input.Changed) -> None:
        """Filter the catalog using the current search text.

        Args:
            event: Change to the step-type search input.

        Returns:
            None.
        """
        self.fill(event.value)

    @on(Button.Pressed, "#workflow-show-all")
    def all_types(self, event: Button.Pressed) -> None:
        """Toggle unavailable catalog entries while retaining the search text.

        Args:
            event: Press of the Show all / Supported subsets button.

        Returns:
            None.
        """
        event.stop()
        self.show_all = not self.show_all
        event.button.label = "Supported subsets" if self.show_all else "Show all"
        self.fill(self.query_one(Input).value)

    def select_option(self, option: Option) -> None:
        """Show the chosen type's detail and focus its insertion confirmation.

        Args:
            option: Enabled catalog option selected for review.

        Returns:
            None.

        Raises:
            NoMatches: The detail widget or insertion button is absent.
        """
        self.query_one("#workflow-type-detail", Static).update(
            str(option.prompt)
            + "\nInserts a draft step. Bind file/model/actor/Notes resources before execution."
        )
        self.query_one("#workflow-insert-confirm", Button).disabled = False
        self.query_one("#workflow-insert-confirm", Button).focus()

    @on(Button.Pressed, "#workflow-insert-confirm")
    def insert(self, event: Button.Pressed) -> None:
        """Dismiss with the highlighted type, if any, for the caller to insert.

        Args:
            event: Press of the Insert step confirmation button.

        Returns:
            None.
        """
        event.stop()
        listing = self.query_one(OptionList)
        if listing.highlighted is not None:
            self.dismiss(listing.get_option_at_index(listing.highlighted).id)


class WorkflowLibrary(Vertical):
    """Render one library page and send paging/selection requests to its owner.

    Filtering and fetching remain with the document owner; this widget retains
    exact workflow and revision IDs independently of display labels.

    Attributes:
        rows: Current ``(label, workflow_id, revision_id)`` rows.
        page_offset: Offset of the displayed page within the owner's results.
    """

    class PageRequested(Message):
        """Request a library page without changing the selected workflow.

        Attributes:
            offset: Requested nonnegative row offset.
            query: Search text used to filter the complete library.
        """

        def __init__(self, offset: int, query: str) -> None:
            """Initialize a library-page request.

            Args:
                offset: Requested nonnegative row offset.
                query: Search text passed unchanged to the document owner.
            """
            super().__init__()
            self.offset, self.query = offset, query

    class Selected(Message):
        """Request the exact workflow revision represented by a library row.

        Attributes:
            workflow_id: Selected workflow's stable identity.
            revision_id: Saved revision identity carried by the selected row.
        """

        def __init__(self, workflow_id: str, revision_id: str) -> None:
            """Initialize a workflow-revision selection request.

            Args:
                workflow_id: Workflow identity from the selected row.
                revision_id: Revision identity from the same row.
            """
            super().__init__()
            self.workflow_id, self.revision_id = workflow_id, revision_id

    def __init__(self, **kwargs: Any) -> None:
        """Initialize an empty library at the first page.

        Args:
            **kwargs: Options forwarded to Textual's Vertical constructor.
        """
        super().__init__(**kwargs)
        self.rows: tuple[tuple[str, str, str], ...] = ()
        self.page_offset = 0

    def compose(self) -> ComposeResult:
        """Build library search, selection, paging and creation controls.

        Yields:
            Widgets for browsing a bounded library page.
        """
        yield Label("Workflow library", id="workflow-library-heading")
        yield Input(placeholder="Search workflows", id="workflow-library-search")
        yield OptionList(id="workflow-library-list")
        with Horizontal(id="workflow-library-pages"):
            yield compact_button("Previous", "workflow-library-previous", disabled=True)
            yield compact_button("Next", "workflow-library-next", disabled=True)
        yield compact_button("New workflow", "workflow-new")

    def show_rows(
        self,
        rows: Iterable[tuple[str, str, str]],
        *,
        offset: int = 0,
        has_next: bool = False,
    ) -> None:
        """Replace the visible page and update its paging controls.

        Args:
            rows: Bounded page of ``(label, workflow_id, revision_id)`` tuples
                with unique workflow IDs. Materialized once without filtering
                or truncation; labels are rendered as literal text.
            offset: Nonnegative offset of this page. Zero disables Previous.
            has_next: Whether the owner found another page, enabling Next.

        Returns:
            None. The first row is highlighted, or no row for an empty page.

        Raises:
            NoMatches: The list or paging buttons have not been composed.
            DuplicateID: Multiple rows contain the same workflow ID.
        """
        self.rows = tuple(rows)
        self.page_offset = offset
        listing = self.query_one(OptionList)
        listing.clear_options()
        listing.add_options(
            Option(Text(label + "\nLocal · saved revision"), id=wid)
            for label, wid, rid in self.rows
        )
        listing.highlighted = 0 if listing.option_count else None
        self.query_one("#workflow-library-previous", Button).disabled = offset == 0
        self.query_one("#workflow-library-next", Button).disabled = not has_next

    @on(Input.Changed)
    def search(self, event: Input.Changed) -> None:
        """Request the first page for the updated library query.

        Args:
            event: Change to the library's search input.

        Returns:
            None.
        """
        event.stop()
        self.post_message(self.PageRequested(0, event.value))

    @on(Button.Pressed, "#workflow-library-previous, #workflow-library-next")
    def page(self, event: Button.Pressed) -> None:
        """Request an adjacent page using the current search text.

        Args:
            event: Press of the Previous or Next page button. The requested
                offset advances by PAGE_SIZE and is clamped at zero.

        Returns:
            None.
        """
        event.stop()
        delta = PAGE_SIZE if event.button.id == "workflow-library-next" else -PAGE_SIZE
        self.post_message(
            self.PageRequested(
                max(0, self.page_offset + delta), self.query_one(Input).value
            )
        )

    @on(OptionList.OptionSelected)
    def select(self, event: OptionList.OptionSelected) -> None:
        """Post the workflow and revision IDs belonging to a selected row.

        Args:
            event: Selection of a workflow ID present in the current rows.

        Returns:
            None.

        Raises:
            StopIteration: The selected option ID is absent from current rows.
        """
        event.stop()
        row = next(row for row in self.rows if row[1] == event.option.id)
        self.post_message(self.Selected(row[1], row[2]))
