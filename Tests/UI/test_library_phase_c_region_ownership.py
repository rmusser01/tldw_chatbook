"""Phase-C region ownership (media): which `@on` rows the canvas now owns.

The spec's phase-C rule
(``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md``,
"Phase C — region ownership") is an ORIGIN rule, not a name rule: an `@on`
handler may leave the screen's routing table only if the message it catches
is POSTED inside the region widget's own subtree, because Textual dispatches
a bubbling message along the DOM path from sender to screen. A message posted
by a sibling pane (the Reader, the Trash canvas) or by an ANCESTOR (the
adaptive shell) never passes through the media canvas at all, so those rows
are permanent -- "phase C shrinks the table, it cannot empty it".

This file pins all three halves of that census so a later change cannot drift
any of them silently:

1. ``_MEDIA_MIGRATED_ROWS`` (16) -- canvas-origin AND their behaviour already
   lives in ``LibraryMediaController``. These moved to ``LibraryMediaCanvas``.
2. ``_MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS`` (20) -- canvas-origin, but the
   handler BODY is still screen-native (outside the media series'
   ``_MEDIA_CLUSTER_METHOD_NAMES``). Moving the `@on` without the body would
   force the canvas to reach back through the screen, which the
   named-constructor-dependency canon retired. They stay until a phase-A
   extraction gives them a controller home; pinned here so the reason is
   recorded rather than remembered.
3. ``_MEDIA_PERMANENT_SCREEN_ROWS`` (44) -- NOT canvas-origin. Permanent.

The origin of every row below was established by reading where the control is
COMPOSED (or, for the three message classes, where the message is POSTED), not
from its name: ``#library-media-back`` and ``#library-media-review`` differ by
origin, not by vocabulary.
"""
from __future__ import annotations

import ast
import inspect
import pathlib
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from textual.widgets import Button, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_library_shell,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_BROWSE_NOTES,
)

@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [False, True])
async def test_export_dispatch_queues_on_the_surviving_screen(admitted: bool) -> None:
    """Release the canvas message pump before Export can remove its ancestor.

    Args:
        admitted: Whether the existing resident-canvas gate accepts the press.
    """
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas

    queued = []
    stopped = []
    export = AsyncMock()
    actions = SimpleNamespace(
        call_next=queued.append, handle_library_media_export=export
    )
    canvas = SimpleNamespace(
        _media_actions_for_press=lambda event: actions if admitted else None
    )
    event = SimpleNamespace(stop=lambda: stopped.append(True))
    result = LibraryMediaCanvas.handle_library_media_export(canvas, event)
    if inspect.isawaitable(result):
        await result

    export.assert_not_awaited()
    assert stopped == ([True] if admitted else [])
    assert len(queued) == int(admitted)
    if admitted:
        await queued[0]()
        export.assert_awaited_once_with(event)


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCREEN_PATH = _REPO_ROOT / "tldw_chatbook/UI/Screens/library_screen.py"
_CANVAS_PATH = _REPO_ROOT / "tldw_chatbook/Widgets/Library/library_media_canvas.py"

#: (message expression, selector, handler name). The 16 rows that moved:
#: canvas-origin, and their body already lives on ``LibraryMediaController``,
#: so the canvas can own the ROUTING without owning the behaviour (the One
#: Rule: a region widget owns pixels; the controller owns the work).
_MEDIA_MIGRATED_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "LibraryMediaRowGeometryChanged",
        "",
        "_handle_library_media_row_geometry_changed",
    ),
    ("Button.Pressed", "#library-media-export", "handle_library_media_export"),
    ("Input.Changed", "#library-media-filter", "handle_library_media_filter_changed"),
    (
        "Input.Submitted",
        "#library-media-filter",
        "handle_library_media_filter_submitted",
    ),
    (
        "Button.Pressed",
        "#library-media-filter-clear",
        "handle_library_media_filter_clear",
    ),
    ("Button.Pressed", "#library-media-next", "handle_library_media_next"),
    (
        "Button.Pressed",
        "#library-media-open-viewer",
        "handle_library_media_open_viewer",
    ),
    ("Button.Pressed", "#library-media-previous", "handle_library_media_previous"),
    ("Button.Pressed", "#library-media-retry", "handle_library_media_retry"),
    ("Button.Pressed", "#library-media-review", "handle_library_media_review_these"),
    (
        "Button.Pressed",
        "#library-media-review-selected",
        "handle_library_media_review_selected",
    ),
    (
        "Button.Pressed",
        "#library-media-review-sets",
        "handle_library_media_review_sets",
    ),
    ("Button.Pressed", "#library-media-select-all", "handle_library_media_select_all"),
    (
        "Button.Pressed",
        "#library-media-select-clear",
        "handle_library_media_select_clear",
    ),
    ("Button.Pressed", "#library-media-sort", "handle_library_media_sort"),
    (
        "OptionList.OptionSelected",
        "#library-media-sort-choices",
        "handle_library_media_sort_choice",
    ),
)

#: Canvas-origin, but BLOCKED: the handler body is screen-native. Migrating
#: the `@on` alone would leave the canvas calling back into the screen.
_MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS: frozenset[str] = frozenset(
    {
        "handle_library_media_analyze_overwrite",
        "handle_library_media_analyze_receipt_dismiss",
        "handle_library_media_analyze_retry",
        "handle_library_media_analyze_selected",
        "handle_library_media_analyze_skip",
        "handle_library_media_bulk_delete_cancel",
        "handle_library_media_bulk_delete_confirm",
        "handle_library_media_bulk_delete_receipt_dismiss",
        "handle_library_media_bulk_delete_undo",
        "handle_library_media_delete_selected",
        "handle_library_media_empty_clear_type",
        "handle_library_media_empty_import",
        "handle_library_media_export_selected",
        "handle_library_media_review_dismiss_receipt_close",
        "handle_library_media_review_dismiss_undo",
        "handle_library_media_row",
        "handle_library_media_select_toggle",
        "handle_library_media_trash_open",
        "handle_library_media_type_choice",
        "handle_library_media_type_filter_pressed",
    }
)

#: NOT canvas-origin: 26 posted by the Reader (``LibraryMediaViewer``, 25
#: controls + its ``SpeakerRenamed`` message), 13 by ``LibraryMediaTrash
#: Canvas``, 3 by ``LibraryMediaContent`` inside the Reader, 1
#: (``MediaShellResized``) by the adaptive shell, which is the canvas's
#: ANCESTOR -- a message travelling up from there can never reach a
#: descendant -- and 1 (``handle_library_media_rail_return``) the task-32065
#: "‹ Library" grip-return control, which ``LibraryScreen._build_library_
#: media_rail_return`` composes and mounts on the canvas HOST (a sibling of
#: the media canvas, not a child), its handler posting the screen-level
#: ``PaneToggleRequested`` -- screen-origin, so permanent. Arrived in the
#: ``origin/dev`` reconciliation merge (44, was 43). Permanent delegator rows.
_MEDIA_PERMANENT_SCREEN_ROWS: frozenset[str] = frozenset(
    {
        "_handle_library_media_speaker_renamed",
        "_resize_library_media_reader_shell",
        "handle_library_media_analysis_cancel",
        "handle_library_media_analysis_edit",
        "handle_library_media_analysis_generate",
        "handle_library_media_analysis_save",
        "handle_library_media_back",
        "handle_library_media_content_mode_raw",
        "handle_library_media_content_mode_rendered",
        "handle_library_media_content_search_next",
        "handle_library_media_content_search_prev",
        "handle_library_media_content_search_submitted",
        "handle_library_media_delete",
        "handle_library_media_delete_cancel",
        "handle_library_media_delete_confirm",
        "handle_library_media_edit",
        "handle_library_media_edit_cancel",
        "handle_library_media_edit_save",
        "handle_library_media_highlight_add",
        "handle_library_media_highlight_delete",
        "handle_library_media_image_preview_retry",
        "handle_library_media_image_preview_toggle",
        "handle_library_media_open",
        "handle_library_media_open_original",
        "handle_library_media_rail_return",
        "handle_library_media_read_later",
        "handle_library_media_reader_find",
        "handle_library_media_reader_mode",
        "handle_library_media_reader_more",
        "handle_library_media_reader_retry",
        "handle_library_media_trash_back",
        "handle_library_media_trash_delete",
        "handle_library_media_trash_delete_cancel",
        "handle_library_media_trash_delete_confirm",
        "handle_library_media_trash_next",
        "handle_library_media_trash_previous",
        "handle_library_media_trash_restore",
        "handle_library_media_trash_retry",
        "handle_library_media_trash_row",
        "handle_library_media_trash_search_changed",
        "handle_library_media_trash_search_submitted",
        "handle_library_media_trash_type_choice",
        "handle_library_media_trash_type_filter",
        "use_media_in_chat",
    }
)


def _on_rows(path: pathlib.Path) -> dict[str, tuple[str, str]]:
    """Map every ``@on``-decorated method name to its (message, selector).

    Args:
        path: Module to read.

    Returns:
        Handler name -> (unparsed message expression, selector or "").
    """
    tree = ast.parse(path.read_text())
    rows: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if getattr(decorator.func, "id", None) != "on":
                continue
            message = ast.unparse(decorator.args[0])
            selector = (
                ast.unparse(decorator.args[1]).strip("'\"")
                if len(decorator.args) > 1
                else ""
            )
            rows[node.name] = (message, selector)
    return rows


@pytest.mark.unit
def test_the_census_covers_every_media_row_on_the_screen_exactly_once() -> None:
    """The three sets partition the screen's media routing table.

    Without this, a media `@on` row added later would belong to no set and
    every other test here would pass while saying nothing about it.
    """
    screen_rows = _on_rows(_SCREEN_PATH)
    media_rows = {
        name
        for name, (message, selector) in screen_rows.items()
        if "media" in selector.lower() or "Media" in message
    }
    migrated = {name for _, _, name in _MEDIA_MIGRATED_ROWS}
    classified = (
        migrated | _MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS | _MEDIA_PERMANENT_SCREEN_ROWS
    )
    assert len(migrated) == 16
    assert len(_MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS) == 20
    assert len(_MEDIA_PERMANENT_SCREEN_ROWS) == 44
    assert not migrated & _MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS
    assert not migrated & _MEDIA_PERMANENT_SCREEN_ROWS
    assert not _MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS & _MEDIA_PERMANENT_SCREEN_ROWS
    unclassified = media_rows - classified
    assert not unclassified, (
        "media `@on` row(s) on LibraryScreen that this census does not "
        f"classify: {sorted(unclassified)!r}. Establish the ORIGIN (where "
        "the control is composed / the message posted) and add it to the "
        "migrated, deferred, or permanent set."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "selector", "name"),
    _MEDIA_MIGRATED_ROWS,
    ids=[f"{name}" for _, _, name in _MEDIA_MIGRATED_ROWS],
)
def test_a_migrated_row_is_caught_at_the_canvas_and_gone_from_the_screen(
    message: str, selector: str, name: str
) -> None:
    """Each migration, pinned individually: same row, new receiver.

    Three assertions per row, because two of them alone would pass on a
    half-done migration: the canvas carries the SAME ``@on`` arguments, the
    canvas's body forwards to the SAME-NAMED controller method (the wiring
    suites' proven same-name-forwarding shape, not a loose "mentions the
    controller" check), and the screen no longer carries the row at all.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas

    canvas_rows = _on_rows(_CANVAS_PATH)
    assert name in canvas_rows, (
        f"{name!r} has not moved to LibraryMediaCanvas yet"
    )
    assert canvas_rows[name] == (message, selector), (
        f"{name!r} moved but its @on arguments changed: "
        f"{canvas_rows[name]!r} != {(message, selector)!r}"
    )

    method = getattr(LibraryMediaCanvas, name, None)
    assert method is not None, f"{name!r} is decorated but not defined"
    body = inspect.getsource(method)
    assert re.search(rf"\.{re.escape(name)}\(", body), (
        f"{name!r} on the canvas does not forward to the same-named "
        "controller method"
    )

    assert getattr(LibraryScreen, name, None) is None, (
        f"{name!r} is still on LibraryScreen -- a migrated row must leave "
        "the screen's routing table, or the message is handled twice"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_MEDIA_PERMANENT_SCREEN_ROWS))
def test_a_permanent_row_stays_on_the_screen(name: str) -> None:
    """Scope honesty: a non-canvas-origin row may NOT be migrated.

    These messages are posted by the Reader, the Trash canvas, the Reader's
    content pane, or the adaptive shell. None of those is inside the media
    canvas's subtree, so an `@on` row moved here would simply stop firing --
    a silent dead control, which is exactly the failure this pin exists to
    turn into a red test.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
    from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas

    assert name in _on_rows(_SCREEN_PATH), (
        f"{name!r} left the screen's routing table, but its message is not "
        "posted inside the media canvas -- it can only be caught at the "
        "screen. Phase C shrinks the table; it cannot empty it."
    )
    assert getattr(LibraryScreen, name, None) is not None
    assert getattr(LibraryMediaCanvas, name, None) is None, (
        f"{name!r} appeared on LibraryMediaCanvas, which never sees its "
        "message"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_MEDIA_DEFERRED_CANVAS_ORIGIN_ROWS))
def test_a_deferred_canvas_origin_row_stays_until_its_body_moves(name: str) -> None:
    """Canvas-origin, but its behaviour is not the controller's yet.

    The blocker is recorded, not hidden: these 20 bodies are screen-native
    and outside the media series' ``_MEDIA_CLUSTER_METHOD_NAMES``, so the
    canvas would have to call back through the screen to run them. When a
    later extraction gives one of them a controller home, move it to
    ``_MEDIA_MIGRATED_ROWS`` -- this test failing is the reminder.
    """
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    assert name in _on_rows(_SCREEN_PATH), (
        f"{name!r} left the screen. If its body moved to "
        "LibraryMediaController first, promote it into _MEDIA_MIGRATED_ROWS."
    )
    assert getattr(LibraryScreen, name, None) is not None


@pytest.mark.unit
def test_both_media_canvas_construction_sites_bind_the_actions_collaborator() -> None:
    """The canvas can only own routing if it was handed something to call.

    ``actions`` defaults to ``None`` so the ~5 test files that build a bare
    canvas keep working -- which means a production construction site that
    forgot it would produce a canvas whose toolbar is silently inert. Both
    sites are pinned by source, because there is no runtime symptom to
    assert against short of pressing every control.
    """
    from tldw_chatbook.UI.Library_Modules import library_media_controller
    from tldw_chatbook.UI.Screens import library_screen

    for module, expected in (
        (library_media_controller, "actions=self,"),
        (library_screen, "actions=self._media_controller,"),
    ):
        source = pathlib.Path(module.__file__).read_text()
        construction = source.split("LibraryMediaCanvas(")
        assert len(construction) == 2, (
            f"{module.__name__} no longer has exactly one "
            "LibraryMediaCanvas construction site -- re-derive this pin"
        )
        assert expected in construction[1].split(")")[0], (
            f"{module.__name__} builds LibraryMediaCanvas without binding "
            f"{expected!r}"
        )


async def _settle(pilot, passes: int = 40, delay: float = 0.01) -> None:
    for _ in range(passes):
        await pilot.pause(delay)


async def _press_rail_row(screen, pilot, row_id: str) -> None:
    screen.query_one(f"#library-row-{row_id}", Button).press()
    await _settle(pilot)


async def _enter_media(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _settle(pilot, passes=30)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
        await _settle(pilot, passes=30)
    await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_MEDIA)
    return screen


@pytest.mark.asyncio
async def test_a_migrated_press_still_reaches_the_controller_end_to_end() -> None:
    """The live half: a real press, dispatched at the canvas, still works.

    The structural pins above prove the rows moved; this proves the move did
    not break dispatch. Sort is the control chosen because its whole effect
    is observable without touching the database: the chooser opens.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media(host, pilot)
        canvas = screen.query_one("#library-media-canvas")
        assert canvas.display, "precondition: the media canvas is showing"
        assert not screen._media_state.sort_choices_visible

        canvas.query_one("#library-media-sort", Button).press()
        await _settle(pilot)

        assert screen._media_state.sort_choices_visible, (
            "the migrated Sort handler did not fire at the canvas"
        )
        assert canvas.query("#library-media-sort-choices"), (
            "the chooser did not paint after the migrated handler ran"
        )


@pytest.mark.asyncio
async def test_a_migrated_input_row_still_reaches_the_controller() -> None:
    """The filter Input rows migrated too; `Input.Changed` still lands.

    Distinct from the Button case on purpose: the residency refusal below
    mirrors ``on_button_pressed``'s scope exactly (presses only), so the
    Input rows must keep working with no guard in front of them.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media(host, pilot)
        canvas = screen.query_one("#library-media-canvas")

        canvas.query_one("#library-media-filter", Input).value = "zzz-no-match"
        await _settle(pilot)

        assert screen._media_state.filter_timer is not None, (
            "the migrated filter handler did not arm the debounce timer"
        )


@pytest.mark.asyncio
async def test_a_hidden_resident_canvas_refuses_its_own_migrated_presses() -> None:
    """The hazard the migration creates, and the reason for the guard seam.

    Textual 8.2.8 dispatches a node's ``@on``-DECORATED handlers BEFORE that
    node's ``on_<message>`` convention method (``MessagePump._get_dispatch_
    methods`` yields decorated handlers first per MRO class), and
    ``event.stop()`` only stops BUBBLING -- it cannot un-run a handler on the
    same node. So task 2's residency gate (``LibraryMediaCanvas.
    on_button_pressed``) no longer protects anything that moved onto the
    canvas: a press aimed at the hidden, off-route resident canvas would run
    the migrated handler first and act on a route the user has left.

    Measured before the guard existed: this test failed with the sort chooser
    opening on the hidden Media canvas while the user was on Notes.
    """
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=None, media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _enter_media(host, pilot)
        await _press_rail_row(screen, pilot, LIBRARY_ROW_BROWSE_NOTES)

        canvas = screen.query_one("#library-media-canvas")
        assert not canvas.display, (
            "precondition: the media canvas must be resident and hidden"
        )
        before = screen._media_state.sort_choices_visible

        canvas.query_one("#library-media-sort", Button).press()
        await _settle(pilot)

        assert screen._media_state.sort_choices_visible == before, (
            "a press on the hidden resident canvas ran its migrated handler"
        )
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen.query_one("#library-notes-canvas").display


# ---------------------------------------------------------------------------
# The census-escape guard. Phase C fused the two per-route reader shells into
# one resident ``LibraryBrowseReaderShell`` (id ``#library-browse-reader-
# shell``) and renamed their grips onto the ``library-browse`` prefix. The
# ``#`` -> ``.`` census that carried the ~35 "is this route active?" DOM probes
# across missed one LIVE ``query_one("#library-media-reader-shell")`` (a width
# probe in ``_select_library_rail_row_after_source_admission``): the retired id
# never matches, ``NoMatches`` is caught, the width reads 0, and every "Browse
# Media" entry silently cleared the user's ``priority_pane`` -- the exact
# regression the surrounding comment block documents guarding against (Qodo
# #9). A ``sqlite_master``-is-green-for-a-dead-index situation: the code
# looked fine and ran, it just queried a name nothing answers to.
#
# This guard closes the CLASS, not the instance: it fails if any production
# module reintroduces a live ``query``/``query_one`` against a retired
# shell/grip id. It parses the AST rather than grepping the text on purpose --
# a retired id is legitimately NAMED in prose (this module's own comment above,
# and ``library_browse_reader_shell``'s module docstring), and only a string
# literal handed to a query call is the bug.
_RETIRED_LIBRARY_QUERY_IDS: frozenset[str] = frozenset(
    {
        # The two fused per-route reader shells.
        "library-media-reader-shell",
        "library-notes-reader-shell",
        # Their grips, retired with the per-route ``id_prefix``es
        # (``library-media`` / ``library-notes``) for the shared
        # ``library-browse`` prefix.
        "library-media-library-grip",
        "library-media-items-grip",
        "library-notes-library-grip",
        "library-notes-items-grip",
    }
)

#: The Textual node-query methods whose first positional arg is a CSS selector.
_QUERY_METHOD_NAMES: frozenset[str] = frozenset(
    {"query", "query_one", "query_exactly_one", "query_children"}
)

_PRODUCTION_ROOT = _REPO_ROOT / "tldw_chatbook"


def _live_retired_id_queries(
    path: pathlib.Path,
) -> list[tuple[int, str, str]]:
    """Every ``query``/``query_one`` call in ``path`` whose selector literal
    names a retired shell/grip id.

    Args:
        path: A production ``.py`` module.

    Returns:
        ``(lineno, retired_id, selector_literal)`` for each offending call.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in _QUERY_METHOD_NAMES:
            continue
        for arg in node.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            for retired in _RETIRED_LIBRARY_QUERY_IDS:
                if retired in arg.value:
                    hits.append((node.lineno, retired, arg.value))
    return hits


@pytest.mark.unit
def test_no_production_module_queries_a_retired_library_shell_or_grip_id() -> None:
    """A live query of a retired phase-C id is a silent dead probe.

    ``query_one`` of a name nothing mounts raises ``NoMatches``, and every one
    of these sites already wraps that in a ``try``/``except`` that swallows it
    into a benign-looking default -- so the failure is invisible until a user
    notices the behaviour it drives is wrong. This guard turns the whole class
    into a red test at the source, the way ``_on_rows`` pins the routing table:
    the moment a retired id reappears inside a query, it names the file, line,
    and selector.
    """
    offenders: dict[str, list[tuple[int, str, str]]] = {}
    for path in sorted(_PRODUCTION_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hits = _live_retired_id_queries(path)
        if hits:
            offenders[str(path.relative_to(_REPO_ROOT))] = hits
    assert not offenders, (
        "Live query of a retired phase-C shell/grip id (these ids are no "
        "longer mounted -- the query silently raises NoMatches). Retarget to "
        "the resident id (#library-browse-reader-shell) or the shared grips "
        "(#library-browse-library-grip / #library-browse-items-grip), or the "
        f"route marker class (.library-media-route): {offenders!r}"
    )
