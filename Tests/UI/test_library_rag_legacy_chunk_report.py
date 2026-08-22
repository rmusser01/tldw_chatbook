"""Task 12 (PR E, AC 41): the legacy-chunk report line's first renderer.

Spec §10.0-§10.1: the Library Search/RAG panel (the Library RAG surface
ADR-003 names as the owner of RAG execution) renders the legacy-chunk line
sourced through ``RAGAdminScopeService.get_template_diagnostics`` -- the
``rag.admin.observe.local`` action. Contract pinned here:

* the line renders when the diagnostics payload carries a NON-EMPTY
  ``legacy_chunk_report`` (fetch off the mount path, like the ingest
  template picker);
* the line is OMITTED when the payload omits the key or carries ``""``
  (a fully stamped library shows nothing, not a zero);
* the renderer consumes ONLY ``legacy_chunk_report`` -- the same payload's
  ``capability`` / ``missing_methods`` / ``fallback_enabled`` are hardcoded
  upstream (spec §11 item 4) and must never surface as a health claim;
* a missing scope service (or a failing fetch) degrades quietly.
"""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
    LibrarySearchRagPanel,
)

REPORT_LINE_ID = "library-rag-legacy-chunk-line"
REPORT_COPY = "Chunked by an older engine: 3 items"

#: The full diagnostics payload shape the local service really returns
#: (`LocalRAGAdminService.get_template_diagnostics`): the three hardcoded
#: sibling keys ride the same payload and are never rendered.
_FULL_PAYLOAD = {
    "db_class": "some.module.ChunkingInterop",
    "capability": "native",
    "missing_methods": ["spoofed-method-name"],
    "fallback_enabled": False,
    "hint": "Local chunking templates use the bundled chunking interop service.",
    "backend": "local",
    "legacy_chunk_report": REPORT_COPY,
}


class _FakeDiagnosticsScopeService:
    """Stands in for the app's ``rag_admin_scope_service`` (observe action)."""

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload
        self.calls: list[dict] = []

    async def get_template_diagnostics(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return self._payload


class _ReportHost(ConsolidatedCSSApp):
    """Mount one Search/RAG panel over a stubbed scope service."""

    def __init__(
        self,
        state: LibraryRagPanelState,
        service: _FakeDiagnosticsScopeService | None = None,
    ) -> None:
        super().__init__()
        self._state = state
        self._service = service

    @property
    def rag_admin_scope_service(self) -> _FakeDiagnosticsScopeService | None:
        return self._service

    def compose(self) -> ComposeResult:
        yield LibrarySearchRagPanel(self._state, id="library-search-rag-panel")


def _panel_state() -> LibraryRagPanelState:
    return LibraryRagPanelState.from_values(
        source_counts={"media": 3},
        query="",
        mode="search",
    )


async def _wait_for_report(pilot, *, shown: bool, attempts: int = 80) -> Static:
    """Pause until the report line settles into ``shown`` visibility.

    The fetch is a worker scheduled off the mount path, so the line's
    display flip lands a few pauses after mount -- the same wait shape the
    ingest template picker's tests use.
    """
    line = pilot.app.query_one(f"#{REPORT_LINE_ID}", Static)
    for _ in range(attempts):
        if line.display == shown:
            await pilot.pause()
            return line
        await pilot.pause()
    raise AssertionError(
        f"report line never settled to display={shown}; "
        f"display={line.display!r} text={str(line.renderable)!r}"
    )


@pytest.mark.asyncio
async def test_report_line_renders_the_payload_string_exactly():
    service = _FakeDiagnosticsScopeService(_FULL_PAYLOAD)
    app = _ReportHost(_panel_state(), service)
    async with app.run_test() as pilot:
        line = await _wait_for_report(pilot, shown=True)

    assert str(line.renderable) == REPORT_COPY
    assert service.calls, "the renderer never consulted the scope service"
    assert service.calls[0].get("mode") == "local"


@pytest.mark.asyncio
async def test_report_line_omitted_when_key_absent_from_payload():
    """A fully stamped library omits the key entirely -- nothing renders."""
    payload = {key: value for key, value in _FULL_PAYLOAD.items()
               if key != "legacy_chunk_report"}
    service = _FakeDiagnosticsScopeService(payload)
    app = _ReportHost(_panel_state(), service)
    async with app.run_test() as pilot:
        line = await _wait_for_report(pilot, shown=False)

    assert line.display is False
    assert str(line.renderable) == ""


@pytest.mark.asyncio
async def test_report_line_omitted_when_payload_reports_empty_string():
    service = _FakeDiagnosticsScopeService(
        {**_FULL_PAYLOAD, "legacy_chunk_report": ""}
    )
    app = _ReportHost(_panel_state(), service)
    async with app.run_test() as pilot:
        line = await _wait_for_report(pilot, shown=False)

    assert line.display is False
    assert str(line.renderable) == ""


@pytest.mark.asyncio
async def test_hardcoded_diagnostics_keys_never_render():
    """Only ``legacy_chunk_report`` is consumed -- the hardcoded siblings
    (capability/missing_methods/fallback_enabled) must never surface, with
    or without a report to show."""
    service = _FakeDiagnosticsScopeService(_FULL_PAYLOAD)
    app = _ReportHost(_panel_state(), service)
    async with app.run_test() as pilot:
        line = await _wait_for_report(pilot, shown=True)
        visible_text = " ".join(
            str(widget.renderable)
            for widget in pilot.app.query(Static)
            if widget.display
        )

    assert str(line.renderable) == REPORT_COPY
    for forbidden in ("native", "spoofed-method-name", "fallback"):
        assert forbidden not in visible_text, (
            f"hardcoded diagnostics key leaked into the rendered surface: "
            f"{forbidden!r}"
        )

    # And the omission direction: no report key at all means NOTHING from
    # this payload renders, including the siblings.
    empty_service = _FakeDiagnosticsScopeService(
        {key: value for key, value in _FULL_PAYLOAD.items()
         if key != "legacy_chunk_report"}
    )
    app = _ReportHost(_panel_state(), empty_service)
    async with app.run_test() as pilot:
        line = await _wait_for_report(pilot, shown=False)
        visible_text = " ".join(
            str(widget.renderable)
            for widget in pilot.app.query(Static)
            if widget.display
        )

    assert str(line.renderable) == ""
    for forbidden in ("native", "spoofed-method-name", "fallback"):
        assert forbidden not in visible_text


@pytest.mark.asyncio
async def test_report_fetch_degrades_quietly_without_a_scope_service():
    app = _ReportHost(_panel_state(), service=None)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.pause()
        line = await _wait_for_report(pilot, shown=False)

    assert line.display is False
    assert str(line.renderable) == ""
