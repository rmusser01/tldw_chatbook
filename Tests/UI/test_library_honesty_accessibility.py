"""task-4023 AC#1-#4 (re-critique 2026-08-09 RC-07/RC-09/RC-10).

RC-07: the Library's disabled controls were colour-only and below every
contrast floor (measured live at HEAD: select-mode bulk buttons 1.39:1,
empty-list Select 1.45:1, Export submit 1.44:1, Collections' three form
buttons 2.30:1). These tests pin the NON-COLOUR half of the fix -- the
``○`` disabled marker (extending the product's existing ``✓/○``
vocabulary) and the F-018 reason tooltips; the contrast floor itself is
app-tier CSS, proven by live ANSI measurement (see task-4023's notes) and
pinned here only at the source level.

RC-09: the Details disclosure's DB-sizes line rendered once from the
app-level cache and was never refreshed -- live, the disclosure kept
reporting ``Prompts 180.0KB`` while disk (incl. sidecars) held 4.8MB,
even after closing and reopening Details. Opening Details now recomputes
the sizes and patches the line in place.

RC-10: F1's remaining gaps at HEAD -- the Search/RAG set omitted F6
though the footer's global cluster advertises it; a second F1 left the
help panel open (the app-level F1 delegate finds no handler on the panel
itself); repeated same-key binding extras had no intra-set dedupe; and
the panel never named the surface it was describing.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_DISABLED_ACTION_MARKER,
    LIBRARY_ROW_BROWSE_COLLECTIONS,
    LIBRARY_ROW_BROWSE_SEARCH,
    LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP,
    library_disabled_action_label,
)
from tldw_chatbook.Library.library_media_state import LibraryMediaCanvasState
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Library.library_notes_state import (
    DatabaseNoteDraft,
    LibraryNoteSessionSnapshot,
    NormalizedDatabaseNote,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    LibraryNotePresentationState,
    LibraryNotesCanvas,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
from tldw_chatbook.Widgets.Library.library_export_canvas import LibraryExportCanvas
from tldw_chatbook.Widgets.Library.library_collections_panel import (
    LibraryCollectionsPanel,
)
from tldw_chatbook.Library.library_collections_state import (
    LibraryCollectionActionState,
    LibraryCollectionsPanelState,
)
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_export_state import (
    LibraryExportFormState,
    build_library_export_form_state,
)

_CSS_DIR = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"


# ---------------------------------------------------------------------------
# AC#1 (RC-07): non-colour disabled marker + reason at the control.
# ---------------------------------------------------------------------------


def test_disabled_action_label_helper_prefixes_marker_only_when_disabled():
    assert library_disabled_action_label("Export selected", True) == (
        f"{LIBRARY_DISABLED_ACTION_MARKER} Export selected"
    )
    assert library_disabled_action_label("Export selected", False) == (
        "Export selected"
    )


def _select_mode_zero_selected_state() -> LibraryMediaCanvasState:
    return LibraryMediaCanvasState(
        rows=(),
        type_options=("All",),
        active_type="All",
        status_copy="",
        empty_copy="No media in your Library yet.",
        selected_id="",
        preview_lines=(),
        count=0,
        select_mode=False,
        selected_count=0,
    )


class _EmptyMediaCanvasApp(ConsolidatedCSSApp):
    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_select_mode_zero_selected_state(), id="library-media-canvas"
        )


@pytest.mark.asyncio
async def test_media_select_toggle_empty_list_carries_marker_and_reason():
    """RC-07: the empty-list Select toggle measured 1.45:1 with no marker
    and no reason -- 'click does nothing, says nothing'. Disabled now
    carries the ``○`` marker in its label and an F-018 reason tooltip."""
    app = _EmptyMediaCanvasApp()
    async with app.run_test() as pilot:
        toggle = pilot.app.query_one("#library-media-select-toggle", Button)
        assert toggle.disabled is True
        assert str(toggle.label) == f"{LIBRARY_DISABLED_ACTION_MARKER} Select"
        assert str(toggle.tooltip) == LIBRARY_SELECT_TOGGLE_DISABLED_TOOLTIP


def _select_mode_state(selected_count: int) -> LibraryMediaCanvasState:
    from tldw_chatbook.Library.library_media_state import LibraryMediaRow

    rows = tuple(
        LibraryMediaRow(
            media_id=f"m{i}",
            title=f"Item {i}",
            media_type="document",
            secondary="document · 1m",
            selected=False,
            checked=i < selected_count,
        )
        for i in range(2)
    )
    return LibraryMediaCanvasState(
        rows=rows,
        type_options=("All",),
        active_type="All",
        status_copy="",
        empty_copy="",
        selected_id="",
        preview_lines=(),
        count=2,
        select_mode=True,
        selected_count=selected_count,
    )


class _SelectModeApp(ConsolidatedCSSApp):
    def __init__(self, selected_count: int):
        super().__init__()
        self._selected_count = selected_count

    def compose(self):
        yield LibraryMediaCanvas(
            canvas=_select_mode_state(self._selected_count),
            id="library-media-canvas",
        )


@pytest.mark.asyncio
async def test_select_mode_bulk_buttons_carry_marker_at_zero_selection():
    """RC-07's headline: 'the very buttons the user entered Select mode
    looking for' were colour-only. At 0 selected both bulk actions carry
    the marker; with a selection neither does."""
    async with _SelectModeApp(0).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-media-export-selected", Button)
        delete_btn = pilot.app.query_one("#library-media-delete-selected", Button)
        assert export_btn.disabled and delete_btn.disabled
        assert str(export_btn.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")
        assert str(delete_btn.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")

    async with _SelectModeApp(1).run_test() as pilot:
        export_btn = pilot.app.query_one("#library-media-export-selected", Button)
        delete_btn = pilot.app.query_one("#library-media-delete-selected", Button)
        assert not export_btn.disabled and not delete_btn.disabled
        assert not str(export_btn.label).startswith(LIBRARY_DISABLED_ACTION_MARKER)
        assert not str(delete_btn.label).startswith(LIBRARY_DISABLED_ACTION_MARKER)


class _ExportCanvasApp(ConsolidatedCSSApp):
    def __init__(self, state: LibraryExportFormState):
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryExportCanvas(self._state, id="library-export-canvas")


def _export_state(*, destination: str) -> LibraryExportFormState:
    return build_library_export_form_state(
        scope=ExportScope(kind="notes"),
        counts={"media": 0, "conversations": 0, "notes": 3},
        name="Library export",
        description="",
        media_quality="original",
        destination=destination,
    )


@pytest.mark.asyncio
async def test_export_submit_disabled_carries_marker_and_keeps_reason():
    state = _export_state(destination="")  # no destination -> gate closed
    assert state.export_enabled is False
    async with _ExportCanvasApp(state).run_test() as pilot:
        submit = pilot.app.query_one("#library-export-submit", Button)
        assert submit.disabled is True
        assert str(submit.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")
        assert str(submit.tooltip)  # F-018 reason survives

    enabled_state = _export_state(destination="/tmp/out.zip")
    assert enabled_state.export_enabled is True
    async with _ExportCanvasApp(enabled_state).run_test() as pilot:
        submit = pilot.app.query_one("#library-export-submit", Button)
        assert submit.disabled is False
        assert not str(submit.label).startswith(LIBRARY_DISABLED_ACTION_MARKER)


def _collections_state(*, create_enabled: bool) -> LibraryCollectionsPanelState:
    def action(widget_id: str, label: str, enabled: bool, reason: str):
        return LibraryCollectionActionState(
            widget_id=widget_id,
            label=label,
            enabled=enabled,
            disabled_reason="" if enabled else reason,
        )

    return LibraryCollectionsPanelState(
        status="empty",
        collections=(),
        selected_collection_id=None,
        selected_collection=None,
        empty_copy="No stored collection items are available locally yet.",
        create_action=action(
            "library-create-collection",
            "Create Collection",
            create_enabled,
            "Enter a Collection name.",
        ),
        rename_action=action(
            "library-rename-collection",
            "Rename Collection",
            False,
            "Select a Collection before renaming it.",
        ),
        delete_action=action(
            "library-delete-collection",
            "Delete Collection",
            False,
            "Select a Collection before deleting it.",
        ),
    )


class _CollectionsPanelApp(ConsolidatedCSSApp):
    def __init__(self, state: LibraryCollectionsPanelState):
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryCollectionsPanel(self._state, id="library-collections-panel")


@pytest.mark.asyncio
async def test_collections_disabled_actions_carry_marker_enabled_do_not():
    """RC-07: Collections' three form buttons measured 2.30:1 disabled with
    colour as the only state carrier."""
    async with _CollectionsPanelApp(
        _collections_state(create_enabled=False)
    ).run_test() as pilot:
        for widget_id in (
            "library-create-collection",
            "library-rename-collection",
            "library-delete-collection",
        ):
            button = pilot.app.query_one(f"#{widget_id}", Button)
            assert button.disabled is True
            assert str(button.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} "), (
                widget_id
            )
            assert str(button.tooltip), widget_id  # reason at the control

    async with _CollectionsPanelApp(
        _collections_state(create_enabled=True)
    ).run_test() as pilot:
        create = pilot.app.query_one("#library-create-collection", Button)
        assert create.disabled is False
        assert not str(create.label).startswith(LIBRARY_DISABLED_ACTION_MARKER)


def test_library_disabled_contrast_rules_live_in_source_and_bundle():
    """The 3:1 floor itself is proven by live ANSI measurement (task-4023
    notes); this pins that the Legible Disabled escape rules exist at the
    app tier (the only tier that outranks both ``Button:disabled`` layers,
    per DESIGN.md's TASK-1801 section) in the SOURCE tcss and the built
    bundle, and that the export-submit rule no longer states its label in
    ``$ds-text-disabled`` (the alpha-blend token that measured 1.44:1)."""
    source = (_CSS_DIR / "components" / "_agentic_terminal.tcss").read_text(
        encoding="utf-8"
    )
    bundle = (_CSS_DIR.parent / "css" / "tldw_cli_modular.tcss").read_text(
        encoding="utf-8"
    )
    for haystack in (source, bundle):
        assert "Button.library-canvas-action:disabled" in haystack
        assert "Button.library-source-action:disabled" in haystack
    # The export-submit disabled rule must not re-introduce the compound
    # alpha token; it states a colour that clears 3:1 on its own surface.
    for haystack in (source, bundle):
        start = haystack.index("#library-export-submit:disabled")
        block = haystack[start : haystack.index("}", start)]
        assert "$ds-text-disabled" not in block


# ---------------------------------------------------------------------------
# AC#3 (RC-09): Details DB sizes refresh on disclosure open.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_details_open_recomputes_db_sizes_and_patches_line():
    """RC-09 observed effect: the disclosure kept showing the compose-time
    sizes even after close/reopen while disk had grown. Opening Details
    must recompute through the app's DBStatusManager and patch the mounted
    line in place."""
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _wait_for_library_shell,
    )
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.db_sizes_status = {
        "prompts": "148.0KB",
        "chachanotes": "1.0MB",
        "media": "476.0KB",
    }

    class _StubManager:
        def __init__(self, target):
            self.target = target
            self.calls = 0

        async def update_db_sizes(self):
            self.calls += 1
            self.target.db_sizes_status = {
                "prompts": "180.0KB",
                "chachanotes": "1.1MB",
                "media": "508.0KB",
            }

    app.db_status_manager = _StubManager(app)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)

        # Ensure a deterministic starting point: details closed.
        screen._set_library_rail_section("details", False)
        await pilot.pause()

        screen._set_library_rail_section("details", True)
        for _ in range(300):
            widgets = list(screen.query("#library-details-db-sizes"))
            if widgets and "180.0KB" in str(widgets[0].render()):
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "Opening Details did not refresh the DB-sizes line "
                f"(manager calls={app.db_status_manager.calls})."
            )
        assert app.db_status_manager.calls >= 1
        rendered = str(screen.query_one("#library-details-db-sizes", Static).render())
        assert "180.0KB" in rendered and "508.0KB" in rendered


@pytest.mark.asyncio
async def test_details_open_mounts_db_sizes_line_when_compose_had_none():
    """Recompose discipline: the compose branch renders the sizes line only
    when the cache exists -- the in-place updater must own the same
    conditional and MOUNT the line when the first reading lands on open."""
    from Tests.UI.test_library_shell import (
        LIBRARY_TEST_SIZE,
        LibraryHarness,
        _seed_conversations,
        _two_conversations,
        _wait_for_library_shell,
    )
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    if hasattr(app, "db_sizes_status"):
        delattr(app, "db_sizes_status")

    class _StubManager:
        def __init__(self, target):
            self.target = target

        async def update_db_sizes(self):
            self.target.db_sizes_status = {
                "prompts": "10.0KB",
                "chachanotes": "20.0KB",
                "media": "30.0KB",
            }

    app.db_status_manager = _StubManager(app)

    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = host.screen_stack[-1]
        await _wait_for_library_shell(screen, pilot)
        screen._set_library_rail_section("details", False)
        await pilot.pause()
        assert not list(screen.query("#library-details-db-sizes"))

        screen._set_library_rail_section("details", True)
        for _ in range(300):
            widgets = list(screen.query("#library-details-db-sizes"))
            if widgets and "10.0KB" in str(widgets[0].render()):
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "Opening Details with an empty cache did not mount the "
                "freshly computed DB-sizes line."
            )


# ---------------------------------------------------------------------------
# AC#4 (RC-10): F1 -- F6 on Search/RAG, one entry per key, surface label.
# ---------------------------------------------------------------------------


def test_search_rag_shortcuts_include_f6_like_every_sibling_set():
    """RC-10: Search/RAG's footer globals advertise 'F6 panes' but F1 (fed
    by LIBRARY_SHORTCUTS, which has no F6 entry) never listed the key --
    the only Library per-mode set without it."""
    assert ("F6", "next pane") in LibraryScreen.LIBRARY_SHORTCUTS


def _f1_fake(footer_shortcuts, binding_extras, row_id=""):
    from types import MethodType

    pushed = []
    fake = SimpleNamespace(
        _library_footer_shortcuts_for_current_state=lambda: footer_shortcuts,
        _active_library_binding_shortcuts=lambda: binding_extras,
        _library_selected_row_id=row_id,
        app=SimpleNamespace(push_screen=pushed.append),
    )
    # The real surface-label method (the mapping under test), bound to the
    # fake -- the types.MethodType idiom the multiselect fakes use.
    fake._library_help_surface_label = MethodType(
        LibraryScreen._library_help_surface_label, fake
    )
    fake._pushed = pushed
    return fake


def test_f1_lists_each_key_once_even_across_repeated_binding_extras():
    """(RC-10) The task-3312 dedupe only guarded footer-vs-extras
    collisions; two simultaneously active BINDINGS extras sharing a key
    still rendered twice. The first active entry wins -- the same order
    Textual resolves same-key bindings in."""
    fake = _f1_fake(
        footer_shortcuts=(("/", "focus search"),),
        binding_extras=(
            ("escape", "Back"),
            ("escape", "Focus rail"),
            ("ctrl+s", "Save skill"),
        ),
    )
    LibraryScreen.action_show_workbench_help(fake)
    (panel,) = fake._pushed
    keys = [key for key, _label in panel.state.shortcuts]
    assert keys.count("escape") == 1
    labels = dict(panel.state.shortcuts)
    assert labels["escape"] == "Back"


def test_f1_title_names_the_current_surface():
    """(RC-10) 'Collections' panel said nothing about Collections -- the
    panel now names the surface it describes."""
    fake = _f1_fake(
        footer_shortcuts=(("/", "focus search"), ("F6", "next pane")),
        binding_extras=(),
        row_id=LIBRARY_ROW_BROWSE_COLLECTIONS,
    )
    LibraryScreen.action_show_workbench_help(fake)
    (panel,) = fake._pushed
    assert panel.state.title == "Library Shortcuts — Collections"

    fake = _f1_fake(
        footer_shortcuts=LibraryScreen.LIBRARY_SHORTCUTS,
        binding_extras=(),
        row_id=LIBRARY_ROW_BROWSE_SEARCH,
    )
    LibraryScreen.action_show_workbench_help(fake)
    (panel,) = fake._pushed
    assert panel.state.title == "Library Shortcuts — Search / RAG"

    fake = _f1_fake(
        footer_shortcuts=LibraryScreen.LIBRARY_LANDING_SHORTCUTS,
        binding_extras=(),
        row_id="",
    )
    LibraryScreen.action_show_workbench_help(fake)
    (panel,) = fake._pushed
    assert panel.state.title == "Library Shortcuts — Landing"


# ---------------------------------------------------------------------------
# AC#5: one interaction grammar -- footer sets, marker vocabulary, cyclers.
# ---------------------------------------------------------------------------


def _all_library_footer_sets() -> dict[str, tuple[tuple[str, str], ...]]:
    """Every footer shortcut set constant LibraryScreen declares."""
    return {
        name: value
        for name, value in vars(LibraryScreen).items()
        if name.startswith("LIBRARY_") and name.endswith(("_SHORTCUTS", "_COMPACT"))
    }


def test_library_footer_sets_share_one_grammar():
    """AC#5: four footer dialects at HEAD -- the Notes workflow crammed
    several keys into ONE (key, label) pair with '·' separators and
    TitleCase key names ("Ctrl+N", "New · / Find · Esc Library") while
    every sibling set rendered per-key pairs with lowercase keys. One
    grammar: per-key pairs; keys lowercase (F-keys and the pgup/pgdn
    range spelled as-is); labels are plain phrases with no embedded
    separator."""
    import re

    sets = _all_library_footer_sets()
    assert len(sets) >= 15  # every per-mode set is under this contract
    for name, shortcuts in sets.items():
        for key, label in shortcuts:
            assert "·" not in label, f"{name} embeds a run-on separator: {label!r}"
            assert key == key.lower() or re.fullmatch(r"F\d+", key), (
                f"{name} spells a key off-grammar: {key!r}"
            )


def test_notes_footer_states_use_per_key_grammar_and_never_advertise_dead_keys():
    """The inline notes-workflow sets (confirm-delete, select, sort,
    create, sync) follow the same grammar, and a state whose keys are all
    locked advertises NOTHING instead of a dead 'Esc Locked' entry."""
    fake = SimpleNamespace(
        _library_notes_compact=False,
        _library_notes_stage="notes",
        _library_notes_workflow_active=lambda: True,
        _library_note_session=SimpleNamespace(
            snapshot=None, conflict_resolution_running=False
        ),
        _library_note_confirming_delete=True,
        _library_notes_select_mode=False,
        _library_notes_sort_choices_visible=False,
        _library_note_create_running=False,
        _library_notes_sync_active_token=None,
        _library_notes_focus_region=lambda: "navigator",
    )
    from types import MethodType

    fake._notes_footer_tier = MethodType(LibraryScreen._notes_footer_tier, fake)
    for name in vars(LibraryScreen):
        if name.startswith("LIBRARY_NOTES_"):
            setattr(fake, name, getattr(LibraryScreen, name))
    shortcuts = LibraryScreen._library_notes_footer_shortcuts(fake)
    assert shortcuts == (("enter", "confirm delete"), ("esc", "cancel delete"))

    # A running conflict resolution locks every key -> nothing advertised.
    fake._library_note_confirming_delete = False
    fake._library_note_session = SimpleNamespace(
        snapshot=SimpleNamespace(in_conflict=True),
        conflict_resolution_running=True,
    )
    assert LibraryScreen._library_notes_footer_shortcuts(fake) == ()

    # Compact tier: same keys, compressed labels (the footer's own
    # FULL/COMPACT global-tier idiom, not a second dialect).
    fake._library_note_session = SimpleNamespace(
        snapshot=None, conflict_resolution_running=False
    )
    fake._library_notes_compact = True
    fake._library_notes_focus_region = lambda: "editor"
    compact = LibraryScreen._library_notes_footer_shortcuts(fake)
    assert compact == LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS_COMPACT
    assert [key for key, _ in compact] == [
        key for key, _ in LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS
    ]


def _binding_parts(entry) -> tuple[str, str, str]:
    """Normalize Textual Binding objects and legacy binding tuples."""
    if hasattr(entry, "key"):
        return entry.key, entry.action, entry.description
    return str(entry[0]), str(entry[1]), str(entry[2])


def test_notes_ctrl_s_is_absent_from_binding_footer_and_f1_while_skill_keeps_it():
    """Notes uses visible Save/autosave; only the Skill editor owns Ctrl+S."""
    bindings = tuple(_binding_parts(entry) for entry in LibraryScreen.BINDINGS)
    assert not any(action == "library_notes_save" for _key, action, _label in bindings)
    assert any(
        key == "ctrl+s" and action == "library_skill_save"
        for key, action, _label in bindings
    )
    assert ("ctrl+s", "save skill") in LibraryScreen.LIBRARY_SKILL_EDITOR_SHORTCUTS
    for shortcuts in (
        LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS,
        LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS_COMPACT,
    ):
        assert all(key != "ctrl+s" for key, _label in shortcuts)

    fake = _f1_fake(
        footer_shortcuts=LibraryScreen.LIBRARY_NOTES_EDITOR_SHORTCUTS,
        binding_extras=tuple(
            (key, label)
            for key, action, label in bindings
            if action == "library_notes_save"
        ),
        row_id="browse-notes",
    )
    LibraryScreen.action_show_workbench_help(fake)
    (panel,) = fake._pushed
    assert all(key != "ctrl+s" for key, _label in panel.state.shortcuts)


class _DatabaseNoteEditorApp(ConsolidatedCSSApp):
    """Mount one Database Note editor without the Library service layer."""

    def compose(self):
        baseline = NormalizedDatabaseNote(
            note_id="note-1",
            title="Visible Save",
            body="Body",
            keywords=(),
            version=1,
            created_at="2026-08-27T00:00:00Z",
            modified_at="2026-08-27T00:00:00Z",
        )
        snapshot = LibraryNoteSessionSnapshot(
            baseline=baseline,
            draft=DatabaseNoteDraft(
                note_id="note-1",
                title="Visible Save",
                body="Body",
                keywords_text="",
                revision=1,
            ),
            session_generation=1,
            saved_revision=1,
            dirty=False,
            saving=False,
            in_conflict=False,
            conflict_generation=0,
            status_message="Saved",
        )
        yield LibraryNotesCanvas(
            mode="editor",
            presentation_state=LibraryNotePresentationState(
                snapshot=snapshot,
                metadata_line="",
                status_line="Saved",
            ),
        )


@pytest.mark.asyncio
async def test_database_save_remains_visible_focusable_and_in_normal_pane_order():
    """Removing the accelerator never removes the ordinary Save affordance."""
    async with _DatabaseNoteEditorApp().run_test(size=(120, 40)) as pilot:
        save = pilot.app.query_one("#library-note-save", Button)
        assert save.display and not save.disabled and save.can_focus
        save.focus()
        await pilot.pause()
        assert save.has_focus

    (work_pane,) = tuple(
        target
        for target in LibraryScreen._WORKBENCH_FOCUS_TARGETS
        if target.pane_id == "library-note-work-pane"
    )
    assert work_pane.preferred_focus_ids[0] == "library-note-save"


@pytest.mark.asyncio
async def test_compact_notes_editor_footer_context_actually_displays_at_60_cols():
    """The compact tier exists for the DISPLAYED footer: at 60 cols the
    responsive ladder drops an over-wide context whole ("… F1 …" taught a
    compact user nothing). The compressed editor context must survive in
    the rendered footer text, not merely in the stored full string."""
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _active_library_screen,
        _build_test_app,
        _open_note_editor,
        _seed_conversations,
        _two_conversations,
        _two_notes,
        _wait_for_library_notes_compact,
        _wait_for_library_shell,
    )
    from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_library_notes_compact(screen, pilot, True)
        await _open_note_editor(screen, pilot)
        footer = screen.query_one(AppFooterStatus)
        for _ in range(300):
            displayed = str(footer._shortcut_display.renderable)
            if displayed.startswith("esc notes"):
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "Compact editor context was not DISPLAYED at 60 cols: "
                f"{str(footer._shortcut_display.renderable)!r}"
            )


def test_cycle_controls_use_the_cycle_glyph_not_the_disclosure_glyph():
    """AC#5: '▸' meant three things (selected row, collapsed disclosure,
    silent value-cycler). task-14902 evolved the vocabulary: chooser-openers
    (press opens a direct-pick strip) are glyph-free ``name: value`` labels
    via ``library_choice_label``; the surviving cyclers are genuine
    two-option TOGGLES whose ``⇄`` sits between the fully-enumerated
    options with the ``✓`` marker on the active one. Neither shape ever
    uses '▸'."""
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_CYCLE_MARKER,
        library_choice_label,
        library_choice_tooltip,
        library_toggle_label,
    )
    from tldw_chatbook.Widgets.Library.library_skills_canvas import (
        skill_context_toggle_label,
        skill_disable_model_label,
        skill_user_invocable_label,
    )

    assert library_choice_label("type", "All") == "type: All"
    assert library_choice_tooltip("media type", ("All", "video")) == (
        "Press to pick media type: All · video."
    )
    assert library_toggle_label("mode", ("Search", "RAG Answer"), 0) == (
        "mode: ✓ Search ⇄ RAG Answer"
    )
    for label in (
        skill_user_invocable_label(True),
        skill_disable_model_label(True),
        skill_context_toggle_label("inline"),
    ):
        # Kept toggles: the glyph between the options, the full option
        # set on the label, the active option ✓-marked.
        assert LIBRARY_CYCLE_MARKER in label, label
        assert "✓" in label, label
        assert "▸" not in label, label


def test_no_widget_module_still_builds_a_cycler_with_the_disclosure_glyph():
    """Source-level sweep: no Library widget (or the screen's in-place
    patchers) may build a '... ▸' cycle label again. Leading row markers
    (`marker = "▸"`) and prose mentions are untouched by this pattern."""
    widgets_dir = (
        Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Widgets" / "Library"
    )
    offenders = []
    sources = list(widgets_dir.glob("*.py")) + [
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "library_screen.py"
    ]
    for path in sources:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if (
                ' ▸"' in line
                and "marker" not in line
                and not line.lstrip().startswith("#")
            ):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


@pytest.mark.asyncio
async def test_media_canvas_actions_share_one_toolbar_row():
    """AC#5: three toolbar layouts -- Media stacked its actions vertically
    (one full-width button per line) while Notes/Prompts/Skills use
    horizontal ds-toolbar rows. Media's type filter, Export… and Select now
    share one ds-toolbar Horizontal like its siblings."""
    from textual.containers import Horizontal

    async with _SelectModeApp(0).run_test() as pilot:
        buttons = [
            pilot.app.query_one(selector, Button)
            for selector in (
                "#library-media-type-filter",
                "#library-media-export",
                "#library-media-select-toggle",
            )
        ]
        parents = {button.parent for button in buttons}
        assert len(parents) == 1
        (parent,) = parents
        assert isinstance(parent, Horizontal)
        assert parent.has_class("ds-toolbar")


@pytest.mark.asyncio
async def test_selected_collection_row_carries_the_selected_marker():
    """AC#5: the selected Collections row was colour-only (`is-active`);
    every other Library list marks its selected row with a leading '▸ '."""
    import dataclasses

    from tldw_chatbook.Library.library_collections_state import (
        LibraryCollectionSummary,
    )

    def summary(collection_id: str, name: str, selected: bool):
        return LibraryCollectionSummary(
            collection_id=collection_id,
            name=name,
            description="",
            item_count=2,
            source_authority="local",
            sync_status="local-only",
            sync_status_detail="",
            sync_status_label_override="",
            created_at="",
            updated_at="",
            selected=selected,
        )

    rows = (summary("c-1", "Research", True), summary("c-2", "Queue", False))
    state = dataclasses.replace(
        _collections_state(create_enabled=True), status="ready", collections=rows
    )
    async with _CollectionsPanelApp(state).run_test() as pilot:
        selected = pilot.app.query_one("#library-collection-select-0", Button)
        unselected = pilot.app.query_one("#library-collection-select-1", Button)
        assert str(selected.label).startswith("▸ ")
        assert not str(unselected.label).startswith("▸")


# ---------------------------------------------------------------------------
# AC#6 (RC-08): search honesty -- results visible at the point of action,
# one query value across BOTH mounted inputs, executed-only Recents.
# (Enter-runs-the-search and executed-only history were already pinned by
# test_library_shell's rail-submit/history tests -- dissolved at HEAD.)
# ---------------------------------------------------------------------------


def _search_canvas_harness():
    from Tests.UI.test_library_shell import (
        LibraryHarness,
        _StaticLibraryRagSearchService,
        _active_library_screen,
        _build_test_app,
        _seed_conversations,
        _two_conversations,
    )

    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Tides research note", "id": "note-1"}],
        media=[{"title": "Ocean survey transcript", "id": "media-1"}],
    )
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": f"Result {index}",
                    "snippet": "A snippet about the tides research corpus.",
                    "source_id": f"note-{index}",
                    "chunk_id": f"chunk-{index}",
                    "provenance": {"source_type": "note"},
                }
                for index in range(1, 6)
            ]
        }
    )
    return app, LibraryHarness(app), _active_library_screen


@pytest.mark.asyncio
async def test_typing_in_either_search_input_updates_the_other_in_place():
    """RC-08: 'two search inputs are live with different values'. The STATE
    was already single-source, but the sibling WIDGET only re-seeded on
    recompose -- live at HEAD the canvas held 'terminals render' while the
    rail box still showed 'terminals'. Typing in either now patches the
    other mounted input in place, both directions."""
    from textual.widgets import Input

    from Tests.UI.test_library_shell import (
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app, host, active_screen = _search_canvas_harness()
    async with host.run_test(size=(170, 50)) as pilot:
        screen = active_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        canvas_input = screen.query_one("#library-rag-query-input", Input)
        rail_input = screen.query_one("#library-search-input", Input)

        canvas_input.value = "terminals render"
        await pilot.pause()
        assert rail_input.value == "terminals render"

        rail_input.value = "tide charts"
        await pilot.pause()
        assert canvas_input.value == "tide charts"


@pytest.mark.asyncio
async def test_run_reveals_the_evidence_region_instead_of_leaving_the_fold_intact():
    """RC-08: results landed ~30 rows below the fold behind the
    configuration region -- pressing Run left the visible half of the
    canvas pixel-identical. Running a query must scroll the panel so the
    Evidence region is inside the panel's visible window."""
    from textual.widgets import Input

    from Tests.UI.test_library_shell import (
        _wait_for_library_rag_query_ready,
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app, host, active_screen = _search_canvas_harness()
    # A short terminal keeps the Evidence region genuinely below the fold
    # behind the query/scope regions, reproducing the observed geometry.
    async with host.run_test(size=(170, 24)) as pilot:
        screen = active_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        panel = screen.query_one("#library-search-rag-panel")
        # Precondition: the heading starts below the panel's visible window
        # (screen-space regions; otherwise this test cannot fail
        # meaningfully). Wait for layout to settle first.
        for _ in range(300):
            heading = screen.query_one("#library-rag-results-heading")
            if heading.region.y > 0 and panel.region.height > 0:
                break
            await pilot.pause(0.01)
        assert heading.region.y >= panel.region.bottom, (
            "Evidence region was not below the fold at this size: "
            f"heading={heading.region}, panel={panel.region}."
        )

        screen.query_one("#library-rag-query-input", Input).value = "tides"
        await _wait_for_library_rag_query_ready(screen, pilot, "tides")
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-card-0")

        # Assert the SETTLED state, not a transient frame mid-scroll: wait
        # for the panel's scroll offset to hold still across frames first
        # (a mutation that drops the reveal still passed a frame-catching
        # loop here -- an unrelated scroll passes THROUGH the visible
        # window on its way to the bottom).
        last_offset = None
        stable = 0
        for _ in range(300):
            offset = panel.scroll_offset
            stable = stable + 1 if offset == last_offset else 0
            last_offset = offset
            if stable >= 10:
                break
            await pilot.pause(0.01)
        heading = screen.query_one("#library-rag-results-heading")
        assert panel.region.y <= heading.region.y < panel.region.bottom, (
            "Run did not reveal the Evidence region at settle: heading at "
            f"{heading.region}, panel window {panel.region}, "
            f"scroll {panel.scroll_offset}."
        )


@pytest.mark.asyncio
async def test_blocked_run_records_nothing_in_recent_searches():
    """RC-08: 'never-executed strings still enter Recent searches'. The
    gate-blocked path (blank query) must leave history untouched."""
    from Tests.UI.test_library_shell import (
        _wait_for_library_shell,
        _wait_for_selector,
    )

    app, host, active_screen = _search_canvas_harness()
    async with host.run_test(size=(170, 50)) as pilot:
        screen = active_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        await screen._start_library_rag_query()  # blank query -> gate blocked
        await pilot.pause()
        assert screen._library_search_history == ()
        assert not list(screen.query(".library-rag-history-row"))


# ---------------------------------------------------------------------------
# AC#7: layout/copy honesty -- canvas title budget, viewer Type line, and
# the three surfaces whose Escape was inert (Export, Collections, staging).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_media_canvas_rows_do_not_inherit_the_rail_title_budget():
    """AC#7: media/conversations canvas rows truncated titles at 17 chars
    (the RAIL's 20-cell cap) on a 170-column terminal, leaving ~115 blank
    columns. Canvas rows render the full title; CSS ellipsis handles real
    overflow at the rendered edge."""
    import dataclasses

    from tldw_chatbook.Library.library_media_state import LibraryMediaRow

    long_title = "Quarterly planning notes for the Atlas migration project 2026"
    state = dataclasses.replace(
        _select_mode_zero_selected_state(),
        rows=(
            LibraryMediaRow(
                media_id="m1",
                title=long_title,
                media_type="document",
                secondary="document · 1m",
                selected=False,
                checked=False,
            ),
        ),
        count=1,
    )

    class _App(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryMediaCanvas(canvas=state, id="library-media-canvas")

    async with _App().run_test(size=(170, 40)) as pilot:
        row = pilot.app.query_one("#library-media-row-0", Button)
        assert long_title in str(row.label)


def test_viewer_type_line_names_rendered_markdown_honestly():
    """AC#7: 'Type: plaintext' for a .md the viewer renders as markdown.
    The metadata line now says what the user is looking at while still
    naming the stored type."""
    from tldw_chatbook.Library.library_media_viewer_state import (
        build_library_media_viewer_state,
    )

    markdown_detail = {
        "id": "1",
        "title": "Notes",
        "type": "plaintext",
        "content": "# Heading\n\n- a real markdown bullet\n- another\n",
    }
    state = build_library_media_viewer_state(markdown_detail)
    assert state.is_markdown
    assert "Type: markdown (stored as plaintext)" in state.metadata_lines

    plain_detail = {
        "id": "2",
        "title": "Log",
        "type": "plaintext",
        "content": "just words\nno structure at all\n" * 3,
    }
    plain_state = build_library_media_viewer_state(plain_detail)
    assert not plain_state.is_markdown
    assert "Type: plaintext" in plain_state.metadata_lines


@pytest.mark.asyncio
async def test_escape_works_on_export_collections_and_staging_canvases():
    """AC#7: Escape was inert on Export, Collections, and the Study
    staging canvas (and 'Export…' from within Media navigated away with
    no return path). Escape now: Export -> back to the canvas that opened
    it (or the hub from the rail), Collections -> focus rail, staging ->
    hub. The footer advertises each via the shared seam."""
    from textual.widgets import Input as _Input

    from Tests.UI.test_product_maturity_phase39_library_collections import (
        DestinationHarness,
        FakeLibraryCollectionsService,
        _active_destination_screen,
        _seed_library_sources,
        _wait_for_library_snapshot,
        _wait_for_selector,
    )
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_INGEST_EXPORT,
    )

    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        # --- Export… from within Media returns to Media on Escape.
        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_MEDIA}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-export")
        screen.query_one("#library-media-export", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-submit")
        assert screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT
        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("esc") == "back to Media"
        await pilot.press("escape")
        for _ in range(300):
            if screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA:
                break
            await pilot.pause(0.01)
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_MEDIA

        # --- Export entered from the rail returns to the hub landing.
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-submit")
        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("esc") == "back to hub"
        await pilot.press("escape")
        for _ in range(300):
            if screen._library_selected_row_id == "":
                break
            await pilot.pause(0.01)
        assert screen._library_selected_row_id == ""

        # --- Collections: Escape is the list-canvas focus hop to the rail.
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")
        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("esc") == "focus rail"
        await pilot.press("escape")
        for _ in range(300):
            focused = screen.focused
            if isinstance(focused, _Input) and focused.id == "library-search-input":
                break
            await pilot.pause(0.01)
        assert getattr(screen.focused, "id", None) == "library-search-input"

        # --- Study staging canvas: Escape returns to the hub.
        screen.query_one("#library-row-create-study", Button).press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-actions")
        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("esc") == "back to hub"
        await pilot.press("escape")
        for _ in range(300):
            if screen._library_selected_row_id == "":
                break
            await pilot.pause(0.01)
        assert screen._library_selected_row_id == ""


# ---------------------------------------------------------------------------
# AC#1 follow-up (Task 2 review M-1): the IN-PLACE marker patchers had zero
# automated coverage -- removing `_patch_library_disabled_marker_label` from
# `_apply_library_row_toggle` (mutation C) or the collections patcher's label
# rebuild (mutation D) survived every existing suite. These pins drive the
# real patch paths across the disabled boundary in both directions.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_row_toggle_patcher_rebuilds_marker_label_both_directions():
    """Mutation C's survival: `_apply_library_row_toggle` flips `.disabled`
    on the bulk buttons in place, so it must rebuild the "○" marker label
    too. Crossing 0->1 must strip the marker AND enable; 1->0 must restore
    both. The host app recomposes to the 0-selected state, so the
    patcher's full-recompose fallback cannot mask a missing label patch."""
    from tldw_chatbook.Library.row_selection import RowSelection
    from tldw_chatbook.UI.Screens.library_screen import _apply_library_row_toggle

    app = _SelectModeApp(0)
    async with app.run_test() as pilot:
        app._library_media_row_selection = RowSelection("media")
        row_button = pilot.app.query_one("#library-media-row-0", Button)
        export_btn = pilot.app.query_one("#library-media-export-selected", Button)
        delete_btn = pilot.app.query_one("#library-media-delete-selected", Button)
        assert (
            str(export_btn.label) == f"{LIBRARY_DISABLED_ACTION_MARKER} Export selected"
        )

        # 0 -> 1 selected through the real patch path.
        app._library_media_row_selection.toggle("m0")
        _apply_library_row_toggle(app, "media", row_button, "m0")
        await pilot.pause()
        assert export_btn.disabled is False
        assert str(export_btn.label) == "Export selected"
        assert delete_btn.disabled is False
        assert str(delete_btn.label) == "Delete selected"
        assert str(row_button.label).startswith("☑")

        # 1 -> 0: the marker must come back with `disabled`.
        app._library_media_row_selection.toggle("m0")
        _apply_library_row_toggle(app, "media", row_button, "m0")
        await pilot.pause()
        assert export_btn.disabled is True
        assert str(export_btn.label) == (
            f"{LIBRARY_DISABLED_ACTION_MARKER} Export selected"
        )
        assert delete_btn.disabled is True
        assert str(delete_btn.label) == (
            f"{LIBRARY_DISABLED_ACTION_MARKER} Delete selected"
        )
        assert str(row_button.label).startswith("☐")


@pytest.mark.asyncio
async def test_collections_patcher_rebuilds_marker_label_both_directions():
    """Mutation D's survival: `_refresh_collections_panel_action_state_widgets`
    flips `.disabled` on the three form actions in place (no recompose), so
    it must rebuild the "○" marker label alongside. Driven through the real
    Input.Changed path on the mounted Library screen."""
    from textual.widgets import Input

    from Tests.UI.test_product_maturity_phase39_library_collections import (
        DestinationHarness,
        FakeLibraryCollectionsService,
        _active_destination_screen,
        _seed_library_sources,
        _wait_for_library_snapshot,
        _wait_for_selector,
    )
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        create_btn = screen.query_one("#library-create-collection", Button)
        assert create_btn.disabled is True
        assert str(create_btn.label).startswith(f"{LIBRARY_DISABLED_ACTION_MARKER} ")

        name_input = screen.query_one("#library-collection-name-input", Input)
        name_input.value = "Research"
        await pilot.pause()
        assert create_btn.disabled is False
        assert str(create_btn.label) == "Create Collection"

        name_input.value = ""
        await pilot.pause()
        assert create_btn.disabled is True
        assert str(create_btn.label) == (
            f"{LIBRARY_DISABLED_ACTION_MARKER} Create Collection"
        )


# ---------------------------------------------------------------------------
# AC#1 follow-up (whole-branch review IMPORTANT-1): the compact width patcher
# (`apply_compact_presentation`) rewrote the notes Export-selected label plain
# on every compact-boundary crossing -- stripping the "○" marker while
# `disabled=True` and leaving `_library_disabled_marker_base` at the
# wrong-tier spelling, so the next in-place patch rebuilt the wrong-width
# label. Both the Task-2 sweep (grep for `.disabled =`) and the Task-3
# grammar sweep missed it because this patcher rewrites the LABEL without
# touching `disabled`.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_presentation_keeps_marker_and_retiers_the_stash():
    """IMPORTANT-1: crossing the compact boundary with the button disabled
    must preserve the "○" marker (and the F-018 reason), and must re-tier
    the stashed base so the row-toggle patcher rebuilds the tier-correct
    spelling in BOTH directions (enable while compact, disable while wide).
    """
    from tldw_chatbook.Library.library_notes_state import LibraryNotesListState
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP,
    )
    from tldw_chatbook.UI.Screens.library_screen import (
        _patch_library_disabled_marker_label,
    )
    from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

    state = LibraryNotesListState(
        rows=(),
        header_copy="Notes (0)",
        status_copy="",
        empty_copy="",
        select_mode=True,
        selected_count=0,
    )

    class _NotesSelectApp(ConsolidatedCSSApp):
        def compose(self):
            yield LibraryNotesCanvas(list_state=state, id="library-notes-canvas")

    async with _NotesSelectApp().run_test() as pilot:
        canvas = pilot.app.query_one(LibraryNotesCanvas)
        export_btn = pilot.app.query_one("#library-notes-export-selected", Button)
        assert export_btn.disabled is True
        assert str(export_btn.label) == (
            f"{LIBRARY_DISABLED_ACTION_MARKER} Export selected"
        )

        # Wide -> compact while disabled: marker + reason survive at the
        # compact spelling, and the stash re-tiers with the label.
        canvas.apply_compact_presentation(True)
        assert export_btn.disabled is True
        assert str(export_btn.label) == f"{LIBRARY_DISABLED_ACTION_MARKER} Export"
        assert export_btn._library_disabled_marker_base == "Export"
        assert str(export_btn.tooltip) == LIBRARY_EXPORT_SELECTED_DISABLED_TOOLTIP

        # Enable while compact through the shared patcher: the re-tiered
        # stash must yield the COMPACT spelling, not "Export selected".
        export_btn.disabled = False
        _patch_library_disabled_marker_label(export_btn)
        assert str(export_btn.label) == "Export"

        # Compact -> wide while enabled: wide spelling, still no marker,
        # stash re-tiers back.
        canvas.apply_compact_presentation(False)
        assert str(export_btn.label) == "Export selected"
        assert export_btn._library_disabled_marker_base == "Export selected"

        # Disable while wide through the shared patcher: the marker returns
        # at the wide tier.
        export_btn.disabled = True
        _patch_library_disabled_marker_label(export_btn)
        assert str(export_btn.label) == (
            f"{LIBRARY_DISABLED_ACTION_MARKER} Export selected"
        )


# ---------------------------------------------------------------------------
# AC#7 follow-up (whole-branch review M-A): the rail-switch export-origin
# clear (`_select_library_rail_row`'s `_library_export_origin_row_id = ""`)
# had zero coverage -- mutating it to a no-op left the full honesty file
# green. The escape pin's rail-entry leg only reaches the rail AFTER
# `action_library_export_back` already cleared the origin, so the guarded
# path (Export-from-Media -> rail-switch AWAY -> rail-enter Export fresh)
# was unpinned: with the clear gone, the footer would lie "esc back to
# Media" and Escape would navigate there.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rail_entry_to_export_after_media_origin_does_not_claim_media():
    """M-A: Export reached from Media arms a Media back-origin; a plain
    rail switch to another canvas must clear it, so a later fresh rail
    entry into Export is hub-origined -- footer says "back to hub" and
    Escape lands on the hub, never Media."""
    from Tests.UI.test_product_maturity_phase39_library_collections import (
        DestinationHarness,
        FakeLibraryCollectionsService,
        _active_destination_screen,
        _seed_library_sources,
        _wait_for_library_snapshot,
        _wait_for_selector,
    )
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Library.library_shell_state import (
        LIBRARY_ROW_BROWSE_MEDIA,
        LIBRARY_ROW_INGEST_EXPORT,
    )

    app = _build_test_app()
    _seed_library_sources(app)
    app.library_collections_service = FakeLibraryCollectionsService()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        # Arm the Media origin through the real bypass seam.
        screen.query_one(f"#library-row-{LIBRARY_ROW_BROWSE_MEDIA}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-export")
        screen.query_one("#library-media-export", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-submit")
        assert screen._library_export_origin_row_id == LIBRARY_ROW_BROWSE_MEDIA

        # Rail-switch AWAY (a plain route boundary, not Export's own back).
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        # Fresh rail entry into Export: the stale Media origin must be gone.
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}", Button).press()
        await _wait_for_selector(screen, pilot, "#library-export-submit")
        shortcuts = dict(screen._library_footer_shortcuts_for_current_state())
        assert shortcuts.get("esc") == "back to hub"

        await pilot.press("escape")
        for _ in range(300):
            if screen._library_selected_row_id == "":
                break
            await pilot.pause(0.01)
        assert screen._library_selected_row_id == ""
