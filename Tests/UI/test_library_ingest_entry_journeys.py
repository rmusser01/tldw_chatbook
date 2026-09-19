"""Import recovery text must remain readable where the user decides to start."""

from typing import ClassVar

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_library_ingest_canvas import _CanvasHost
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.ingest_types import PreflightResult
from tldw_chatbook.Library.library_ingest_jobs import LibraryIngestJobRegistry
from tldw_chatbook.Library.library_ingest_state import (
    LibraryIngestFormState,
    build_library_ingest_state,
)


class _IngestHost(_CanvasHost):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]


def _painted(host, widget):
    region = widget.region
    strips = list(host.screen._compositor.render_strips())
    return " ".join(
        " ".join(
            strips[y].crop(max(0, region.x), min(host.size.width, region.right)).text
            for y in range(max(0, region.y), min(len(strips), region.bottom))
        ).split()
    )


def _gate_state(case):
    form = LibraryIngestFormState(path="/inbox/source.txt")
    preflight = PreflightResult({}, [], [], 0, False, 0)
    kwargs = {}
    if case == "missing":
        preflight = PreflightResult(
            {}, [], ["Path not found"], 0, False, 0, path_invalid=True
        )
    elif case == "unsupported":
        preflight = PreflightResult(
            {"unsupported": ["/inbox/source.unknown"]}, [], [], 20, False, 1
        )
    elif case == "option":
        preflight = PreflightResult({"generic": [form.path]}, [], [], 20, False, 1)
        form.type_options = {"generic": {"chunk": True, "chunk_size": "abc"}}
    elif case == "consent":
        preflight = PreflightResult({"generic": [form.path]}, [], [], 20, False, 1)
        kwargs = {
            "start_confirm_armed": True,
            "start_confirm_line": (
                "⚠ Press Start again to import anyway — "
                "1 file will fail without more tooling."
            ),
        }
    return build_library_ingest_state((), form=form, preflight=preflight, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (72, 18)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize(
    "case", ["missing", "empty", "unsupported", "option", "consent"]
)
async def test_import_gate_paints_complete_recovery_above_start(
    case, size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    state = _gate_state(case)
    host = _IngestHost(state)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await pilot.pause()
        gate = host.query_one("#library-ingest-start-quiet-line", Static)
        start = host.query_one("#library-ingest-start", Button)
        # Compare the UI's full input message to actual compositor paint:
        # a height clamp loses the recovery suffix even when the widget exists.
        assert state.start_quiet_line
        assert state.start_quiet_line in _painted(host, gate)
        assert gate.region.bottom <= start.region.y
        assert start in host.screen._compositor.visible_widgets
        assert start.disabled is (case != "consent")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_keyboard_clear_recovers_missing_source_and_keeps_metadata(
    tmp_path, size, theme, monkeypatch
):
    monkeypatch.delenv("NO_COLOR", raising=False)
    source = tmp_path / "ready.txt"
    source.write_text("A small local document for preflight only.")
    app = _build_test_app()
    _seed_conversations(app, [], media=[])
    app.media_db = MediaDatabase(str(tmp_path / "media.db"), client_id="ingest-entry")
    app.library_ingest_jobs = LibraryIngestJobRegistry()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = host.screen
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("i")
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await _focus(screen, host, pilot, "#library-ingest-path", "Path to a local")
        title = screen.query_one("#library-ingest-title", Input)
        title.focus()
        await pilot.press("d", "r", "a", "f", "t")
        field = screen.query_one("#library-ingest-path", Input)
        field.focus()
        field.value = str(tmp_path / "missing.txt")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._ingest_state.form.preflight is not None
                and screen._ingest_state.form.preflight.path_invalid
            ),
            message="Missing-source preflight did not settle",
        )
        await pilot.pause()
        gate = screen.query_one("#library-ingest-start-quiet-line", Static)
        assert "file or folder." in _painted(host, gate)
        assert screen.query_one("#library-ingest-start", Button).disabled
        await pilot.press("tab")
        await _focus(screen, host, pilot, "#library-ingest-browse", "Browse")
        await pilot.press("tab")
        await _focus(screen, host, pilot, "#library-ingest-clear-path", "Clear")
        await pilot.press("enter")
        await _focus(screen, host, pilot, "#library-ingest-path", "Path to a local")
        await pilot.press("x")
        assert screen.query_one("#library-ingest-path", Input).value == "x"
        assert screen._ingest_state.form.title == "draft"
        field = screen.query_one("#library-ingest-path", Input)
        field.value = str(source)
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._ingest_state.form.preflight is not None
                and screen._ingest_state.form.preflight.type_groups.get("generic")
                and not screen.query_one("#library-ingest-start", Button).disabled
            ),
            message="Valid-source preflight did not enable Start",
        )
        await pilot.pause()
        assert screen.query_one("#library-ingest-title") is title
        assert title.value == "draft"
        gate = screen.query_one("#library-ingest-start-quiet-line", Static)
        assert gate.region.height >= 1
        assert _painted(host, gate) == ""
        assert app.library_ingest_jobs.jobs() == ()
