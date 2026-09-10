"""Library copy, glyph and dialog grammar (critique #9, wave tasks 32235/32236/32221/32229).

One meaning per state glyph (32235), a blocked RAG Answer panel that speaks
the Media reader's one-line grammar (32236), and file dialogs whose one
typed field is a real path field (32229).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _wait_for_selector,
)
from Tests.UI.test_library_content_hub import _wait_for_library_shell_ready

_CREDENTIAL_RECOVERY = (
    "Set OPENAI_API_KEY or add api_key under [api_settings.openai]."
)

#: What the blocked RAG Answer panel is allowed to say, in full.
_BLOCKED_REASON = (
    "No analysis provider is configured · Set one in Settings ▸ Providers & Models."
)


# ---------------------------------------------------------------------------
# task-32235: one meaning per glyph
# ---------------------------------------------------------------------------


def test_one_meaning_per_library_glyph() -> None:
    """The three literals that used to share ``○`` no longer do."""
    from tldw_chatbook.Library.library_ingest_state import (
        _GLYPH_DONE,
        _GLYPH_FAILED,
        _GLYPH_SKIPPED,
    )
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_DISABLED_ACTION_MARKER,
        LIBRARY_GLYPH_OUTCOME_DONE,
        LIBRARY_GLYPH_OUTCOME_FAILED,
        LIBRARY_GLYPH_OUTCOME_SKIPPED,
        LIBRARY_GLYPH_SELECTED,
        LIBRARY_GLYPH_UNSELECTED,
    )
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        scope_toggle_label,
    )
    from tldw_chatbook.Library.library_rag_state import LibraryRagSourceOption

    assert LIBRARY_DISABLED_ACTION_MARKER == "○"
    assert (LIBRARY_GLYPH_UNSELECTED, LIBRARY_GLYPH_SELECTED) == ("☐", "☑")
    assert _GLYPH_SKIPPED == LIBRARY_GLYPH_OUTCOME_SKIPPED == "–"
    # The other two outcome glyphs already agreed with the legend; pinned so
    # the queue and the legend cannot drift apart later.
    assert (_GLYPH_DONE, _GLYPH_FAILED) == (
        LIBRARY_GLYPH_OUTCOME_DONE,
        LIBRARY_GLYPH_OUTCOME_FAILED,
    )
    assert LIBRARY_DISABLED_ACTION_MARKER not in {
        LIBRARY_GLYPH_SELECTED,
        LIBRARY_GLYPH_UNSELECTED,
        LIBRARY_GLYPH_OUTCOME_SKIPPED,
    }
    def _option(selected: bool) -> LibraryRagSourceOption:
        return LibraryRagSourceOption(
            source_type="media",
            label="Media",
            count=0,
            selected=selected,
            status="empty",
        )

    assert scope_toggle_label(_option(False)) == "☐ Media (0)"
    assert scope_toggle_label(_option(True)) == "☑ Media (0)"


def test_import_type_toggle_and_queue_row_read_from_the_legend() -> None:
    """The Import canvas's own two glyph sites follow the same legend."""
    from tldw_chatbook.Library.library_ingest_jobs import IngestJobState, LibraryIngestJob
    from tldw_chatbook.Library.library_ingest_state import (
        LibraryIngestFormState,
        build_library_ingest_state,
    )
    from tldw_chatbook.Widgets.Library.library_ingest_canvas import _toggle_label

    assert _toggle_label(enabled=False, text="Chunk") == "☐ Chunk"
    assert _toggle_label(enabled=True, text="Chunk") == "☑ Chunk"

    row = build_library_ingest_state(
        (
            LibraryIngestJob(
                job_id="ingest-job-1",
                source_path="/tmp/weird.xyz",
                state=IngestJobState.SKIPPED,
                submitted_at=100.0,
                error="Unsupported file type: .xyz Supported types: .pdf",
            ),
        ),
        form=LibraryIngestFormState(),
    ).queue_rows[0]
    assert row.line.startswith("– skipped · weird.xyz")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
async def test_the_sources_panel_paints_checkboxes_at_both_sizes(size) -> None:
    """AC#3: the selection glyph is a checkbox on screen, at both widths."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=size) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        notes = screen.query_one("#library-rag-scope-toggle-notes", Button)
        assert str(notes.label) == "☑ Notes (1)"
        notes.press()
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-rag-scope-toggle-notes")
        notes = screen.query_one("#library-rag-scope-toggle-notes", Button)
        assert str(notes.label) == "☐ Notes (1)"


# ---------------------------------------------------------------------------
# task-32236: the blocked RAG Answer panel speaks the Media grammar
# ---------------------------------------------------------------------------


def _blocked_state():
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState

    return LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="anything",
        mode="rag",
        provider_credential_recovery=_CREDENTIAL_RECOVERY,
    )


def test_a_missing_provider_key_blocks_in_the_media_grammar() -> None:
    """AC#1: one line, the Media reader's sentence, no env var, no TOML."""
    state = _blocked_state()

    assert state.query_state.run_action.enabled is False
    assert state.query_state.run_action.disabled_reason == _BLOCKED_REASON
    for banned in ("OPENAI_API_KEY", "[api_settings.", "Owner:", "Recovery:", "Why:"):
        assert banned not in state.query_state.run_action.disabled_reason


def test_the_blocked_panel_paints_one_line_and_an_action() -> None:
    """AC#1/#2: bare reason + an Open Settings action; no recovery dump."""
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        library_rag_query_status_children,
    )

    children = library_rag_query_status_children(_blocked_state())
    by_id = {child.id: child for child in children}

    callout = by_id["library-rag-query-blocked-callout"]
    assert isinstance(callout, Static)
    assert str(callout.renderable) == _BLOCKED_REASON
    assert "library-rag-query-recovery" not in by_id
    assert isinstance(by_id["library-rag-open-provider-settings"], Button)

    painted = "\n".join(
        str(getattr(child, "renderable", getattr(child, "label", ""))) for child in children
    )
    for banned in ("Owner:", "Recovery:", "Why:", "OPENAI_API_KEY", "[api_settings."):
        assert banned not in painted, painted


def test_the_structured_record_survives_in_the_log() -> None:
    """AC#2: what left the screen is still recorded once for diagnosis."""
    from loguru import logger

    from tldw_chatbook.Widgets.Library import library_search_rag_panel as panel

    state = _blocked_state()
    assert "Owner: LLM provider credential." in state.query_state.recovery_copy

    panel._last_logged_query_recovery = ""
    records: list[str] = []
    sink = logger.add(lambda message: records.append(message), level="INFO")
    try:
        panel.library_rag_query_status_children(state)
        logged_once = len(records)
        # The builder runs on every keystroke; the record is logged once.
        panel.library_rag_query_status_children(state)
    finally:
        logger.remove(sink)

    assert any("LLM provider credential" in record for record in records), records
    assert len(records) == logged_once, records


def test_a_quiet_gate_still_renders_no_callout_at_all() -> None:
    """The empty-query gate keeps its single quiet line (A1 stays true)."""
    from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
    from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
        library_rag_query_status_children,
    )

    children = library_rag_query_status_children(
        LibraryRagPanelState.from_values(source_counts={"notes": 1}, query="")
    )
    assert [child.id for child in children] == ["library-rag-query-quiet-line"]


# ---------------------------------------------------------------------------
# task-32229: the file dialog's one field is a path field
# ---------------------------------------------------------------------------


def _picker_app(default_location):
    from textual.app import App

    from tldw_chatbook.Third_Party.textual_fspicker import FileSave

    class _PickerApp(App[None]):
        def on_mount(self) -> None:
            self.push_screen(FileSave(location=str(default_location)))

    return _PickerApp()


async def _open_picker(app, pilot):
    from tldw_chatbook.Third_Party.textual_fspicker.base_dialog import InputBar

    await pilot.pause()
    for _ in range(50):
        bar = app.screen.query(InputBar)
        if bar:
            return bar.first().query_one(Input)
        await pilot.pause(0.02)
    raise AssertionError("File dialog never mounted")


@pytest.mark.asyncio
async def test_a_pasted_absolute_path_moves_the_dialog_tree(tmp_path) -> None:
    """AC#1: typing an absolute directory jumps the listing to it."""
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

    target = tmp_path / "deep" / "nested"
    target.mkdir(parents=True)
    app = _picker_app(tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        field = await _open_picker(app, pilot)
        field.value = str(target)
        await pilot.pause()
        await pilot.pause()
        assert (
            app.screen.query_one(DirectoryNavigation).location.resolve()
            == target.resolve()
        )


@pytest.mark.asyncio
async def test_a_tilde_path_expands(tmp_path, monkeypatch) -> None:
    """AC#1: "~" is a path the dialog understands, not a filename."""
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

    home = tmp_path / "home"
    (home / "sub").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    app = _picker_app(tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        field = await _open_picker(app, pilot)
        field.value = "~/sub"
        await pilot.pause()
        await pilot.pause()
        assert (
            app.screen.query_one(DirectoryNavigation).location.resolve()
            == (home / "sub").resolve()
        )


@pytest.mark.asyncio
async def test_a_typed_file_name_never_moves_the_tree(tmp_path) -> None:
    """The field is still a file-name field: a bare name browses nowhere."""
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation

    (tmp_path / "deep").mkdir()
    app = _picker_app(tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        field = await _open_picker(app, pilot)
        field.value = "deep"
        await pilot.pause()
        await pilot.pause()
        assert (
            app.screen.query_one(DirectoryNavigation).location.resolve()
            == tmp_path.resolve()
        )


@pytest.mark.asyncio
async def test_ctrl_a_selects_the_whole_file_name(tmp_path) -> None:
    """AC (32229): Ctrl+A selects the field instead of moving to its start."""
    app = _picker_app(tmp_path)

    async with app.run_test(size=(120, 40)) as pilot:
        field = await _open_picker(app, pilot)
        field.focus()
        await pilot.pause()
        field.value = "report.zip"
        await pilot.pause()
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert tuple(field.selection) == (0, len("report.zip"))
