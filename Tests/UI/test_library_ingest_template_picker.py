"""Task 11 (PR D, AC 39): the Library ingest chunking-template picker.

Spec §9.3: the ingest flow gains a template ``Select``, populated from the DB
via the scope service, defaulting to "None (manual settings)" (today's
behavior exactly). Four contract properties pinned here:

* lists the DB's live templates (via the app's RAG admin scope service);
* the default option is the None label and the empty-string value;
* option labels are ``escape_markup``-ed (template names are user-authored
  free text and ``Select`` parses its labels as markup);
* the populate call is OFF the mount path (mount-time DB populate is the
  documented "(0) count bug" cause) and the control is hidden in server mode
  (the template travels in local chunk options only).
"""

from __future__ import annotations

import inspect

import pytest
from textual import on
from textual.app import ComposeResult
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Select

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestCanvasState,
    LibraryIngestFormState,
    build_library_ingest_state,
)
from tldw_chatbook.Widgets.Library.library_ingest_canvas import LibraryIngestCanvas

PICKER_ID = "opt-generic-chunk_template"
NONE_LABEL = "None (manual settings)"
NONE_VALUE = ""


class _FakeScopeService:
    """Stands in for the app's ``rag_admin_scope_service`` (async surface)."""

    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self.calls: list[dict] = []

    async def list_templates(self, **kwargs) -> list[dict]:
        self.calls.append(kwargs)
        return self._records


class _PickerHost(ConsolidatedCSSApp):
    def __init__(
        self,
        state: LibraryIngestCanvasState,
        service: _FakeScopeService | None = None,
    ) -> None:
        super().__init__()
        self._state = state
        self._service = service
        self.option_changes: list[LibraryIngestCanvas.OptionValueChanged] = []

    @property
    def rag_admin_scope_service(self) -> _FakeScopeService | None:
        return self._service

    def compose(self) -> ComposeResult:
        yield LibraryIngestCanvas(self._state, id="library-ingest-canvas")

    @on(LibraryIngestCanvas.OptionValueChanged)
    def _record_option_change(
        self, event: LibraryIngestCanvas.OptionValueChanged
    ) -> None:
        self.option_changes.append(event)


def _local_state() -> LibraryIngestCanvasState:
    return build_library_ingest_state((), form=LibraryIngestFormState(path="/tmp/x.txt"))


def _server_state() -> LibraryIngestCanvasState:
    return build_library_ingest_state(
        (),
        form=LibraryIngestFormState(path="/tmp/x.txt"),
        runtime_source="server",
        server_ingest_available=True,
        ingest_backend="server",
    )


async def _wait_for_picker_options(
    pilot, expected_count: int, *, attempts: int = 60
) -> list:
    options: list = []
    for _ in range(attempts):
        await pilot.pause()
        options = pilot.app.query_one(f"#{PICKER_ID}", Select)._options
        if len(options) >= expected_count:
            break
    return options


@pytest.mark.asyncio
async def test_picker_lists_db_templates_via_scope_service():
    service = _FakeScopeService(
        [
            {"name": "tiny-words", "is_builtin": True},
            {"name": "big-words", "is_builtin": False},
        ]
    )
    app = _PickerHost(_local_state(), service)
    async with app.run_test() as pilot:
        options = await _wait_for_picker_options(pilot, 3)

    assert [value for _label, value in options] == ["", "tiny-words", "big-words"]
    assert service.calls, "the picker never consulted the scope service"
    assert service.calls[0].get("mode") == "local"


@pytest.mark.asyncio
async def test_picker_default_is_none_manual_settings():
    service = _FakeScopeService([{"name": "tiny-words"}])
    app = _PickerHost(_local_state(), service)
    async with app.run_test() as pilot:
        select = pilot.app.query_one(f"#{PICKER_ID}", Select)
        await pilot.pause()

    assert select.value == NONE_VALUE
    first_label, first_value = select._options[0]
    assert first_value == NONE_VALUE
    assert Content.from_markup(first_label).plain == NONE_LABEL


@pytest.mark.asyncio
async def test_picker_escapes_markup_in_labels():
    """A template name with markup chars must survive the Select's parser."""
    adversarial = "chapter [red] bold"
    service = _FakeScopeService([{"name": adversarial}])
    app = _PickerHost(_local_state(), service)
    async with app.run_test() as pilot:
        options = await _wait_for_picker_options(pilot, 2)

    labels = {value: label for label, value in options}
    assert adversarial in labels, f"options were {options}"
    # Unescaped, "[red]" would be eaten as a style tag; escaped it renders
    # literally -- the plain text of the parsed label is the original name.
    assert Content.from_markup(labels[adversarial]).plain == adversarial


@pytest.mark.asyncio
async def test_picker_populates_off_the_mount_path():
    """The DB populate must not run from ``on_mount`` (the "(0) count" trap)."""
    on_mount_source = inspect.getsource(LibraryIngestCanvas.on_mount)
    assert "chunk_template" not in on_mount_source, (
        "the template populate must be scheduled off the mount path, not "
        f"from on_mount: {on_mount_source}"
    )
    # And the populate genuinely runs after mount, from the visibility event:
    service = _FakeScopeService([{"name": "tiny-words"}])
    app = _PickerHost(_local_state(), service)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert service.calls, "the off-mount populate never ran"


@pytest.mark.asyncio
async def test_picker_hidden_in_server_mode():
    service = _FakeScopeService([{"name": "tiny-words"}])
    app = _PickerHost(_server_state(), service)
    async with app.run_test() as pilot:
        with pytest.raises(NoMatches):
            pilot.app.query_one(f"#{PICKER_ID}", Select)
        # Hidden means not consulted either: server-mode snapshots never
        # carry a template (Task 10's strip is the defensive half).
        for _ in range(10):
            await pilot.pause()

    assert service.calls == []


@pytest.mark.asyncio
async def test_picker_choice_flows_into_the_chunk_template_slot():
    service = _FakeScopeService([{"name": "tiny-words"}])
    app = _PickerHost(_local_state(), service)
    async with app.run_test() as pilot:
        await _wait_for_picker_options(pilot, 2)
        select = pilot.app.query_one(f"#{PICKER_ID}", Select)
        select.value = "tiny-words"
        await pilot.pause()
        select.value = NONE_VALUE
        await pilot.pause()

    chunk_template_events = [
        event
        for event in app.option_changes
        if event.group == "generic" and event.name == "chunk_template"
    ]
    assert [event.value for event in chunk_template_events] == [
        "tiny-words",
        NONE_VALUE,
    ]


@pytest.mark.asyncio
async def test_picker_without_scope_service_stays_default_only():
    """A host with no service (or a failing one) degrades to the None option."""
    app = _PickerHost(_local_state(), service=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        select = pilot.app.query_one(f"#{PICKER_ID}", Select)

    assert [value for _label, value in select._options] == [NONE_VALUE]
