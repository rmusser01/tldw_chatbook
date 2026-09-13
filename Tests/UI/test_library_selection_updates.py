"""TASK-252: Library targeted (non-recompose) updates for selection interactions.

Covers the two-tier design decided for the audit's staged SELECTION
interaction class (Docs/Design/2026-07-16-performance-audit.md §P1 B2):

* Tier 1 -- select-mode checkbox toggles patch the pressed row's marker,
  the "N selected" Static, and the export-selected button's disabled
  state in place, instead of ``self.refresh(recompose=True)`` (a
  whole-screen remove/remount of the nav bar, footer, ~20-row rail, and
  50-100-row canvas).
* Tier 2 -- browse-mode row selection (the ``▸`` highlight + reader
  change) and select-mode enter/exit/select-all/clear call the mounted
  canvas's own ``sync_state`` -- a canvas-scoped recompose that rebuilds
  only the canvas's own children, skipping the nav bar, footer, and rail.

Reuses the established Library mounted-test harness from
``test_library_shell.py`` (``LibraryHarness`` / ``_build_test_app`` /
``_seed_conversations`` / the ``_wait_for_*`` pollers).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Library.library_conversations_canvas import (
    LibraryConversationsCanvas,
)
from tldw_chatbook.Widgets.Library.library_conversation_reader import (
    LibraryConversationReader,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app


def _spy_screen_recomposes(monkeypatch) -> list:
    """Patch ``BaseAppScreen.refresh`` to record every ``recompose=True`` call.

    Args:
        monkeypatch: The active pytest ``monkeypatch`` fixture (restores the
            original method automatically at test teardown).

    Returns:
        A list that accumulates the ``self`` (screen instance) of every
        ``BaseAppScreen.refresh(recompose=True)`` call made after this spy
        is installed.
    """
    calls: list = []
    original = BaseAppScreen.refresh

    def spy(self, *args, **kwargs):
        if kwargs.get("recompose"):
            calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(BaseAppScreen, "refresh", spy)
    return calls


async def _enter_conversations_select_mode(screen, pilot):
    """Drive the harness to the conversations canvas with select mode on."""
    screen.query_one("#library-row-browse-conversations").press()
    await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

    screen.query_one("#library-conversations-select-toggle").press()
    await _wait_for_selector(screen, pilot, "#library-conversations-select-all")
    await pilot.pause()


@pytest.mark.asyncio
async def test_checkbox_toggle_does_not_recompose_screen(monkeypatch):
    """A select-mode checkbox press updates in place, not via a screen
    recompose.

    Pre-fix RED: the select-mode branch of
    ``handle_library_conversation_row`` called
    ``self.refresh(recompose=True)`` directly, so the screen-level
    recompose count rose by 1 on every checkbox press.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        recompose_calls = _spy_screen_recomposes(monkeypatch)

        row = screen.query_one("#library-conversation-row-0", Button)
        assert str(row.label).startswith("☐")
        count_static = screen.query_one("#library-conversations-selected-count", Static)
        assert str(count_static.renderable) == "0 selected"
        export_button = screen.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_button.disabled is True

        row.press()
        await pilot.pause()

        assert recompose_calls == []  # no screen-level recompose
        assert str(row.label).startswith("☑")  # marker flipped in place
        assert (
            str(
                screen.query_one(
                    "#library-conversations-selected-count", Static
                ).renderable
            )
            == "1 selected"
        )
        assert (
            screen.query_one("#library-conversations-export-selected", Button).disabled
            is False
        )

        # Toggling back off is symmetric.
        row.press()
        await pilot.pause()
        assert recompose_calls == []
        assert str(row.label).startswith("☐")
        assert (
            str(
                screen.query_one(
                    "#library-conversations-selected-count", Static
                ).renderable
            )
            == "0 selected"
        )
        assert (
            screen.query_one("#library-conversations-export-selected", Button).disabled
            is True
        )


@pytest.mark.asyncio
async def test_checkbox_toggle_leaves_rail_untouched():
    """AC #2: selection/checkbox interactions never change rail counts, so
    the mounted rail row must survive a toggle unchanged -- proven by
    object identity (a screen recompose would tear down and rebuild a
    fresh ``LibraryRail``, minting a new row-button instance)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        rail_row_before = screen.query_one("#library-row-browse-conversations")
        label_before = str(rail_row_before.label)

        screen.query_one("#library-conversation-row-0", Button).press()
        await pilot.pause()

        rail_row_after = screen.query_one("#library-row-browse-conversations")
        assert rail_row_after is rail_row_before
        assert str(rail_row_after.label) == label_before


@pytest.mark.asyncio
async def test_browse_row_selection_routes_through_canvas_sync_state(monkeypatch):
    """Clicking a conversation row outside select mode (choosing which row
    is read -- the ``▸`` marker + permanent reader pane) calls the mounted
    canvas's own ``sync_state`` (a canvas-scoped recompose rebuilding only
    the canvas's own children), never the screen-level
    ``self.refresh(recompose=True)``.

    Pre-fix RED: ``LibraryConversationsCanvas.sync_state`` had zero
    callers (audit §P1 B2); this interaction went through
    ``self.refresh(recompose=True)`` instead.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-1")
        await pilot.pause()

        sync_calls: list = []
        original_sync = LibraryConversationsCanvas.sync_state

        def spy_sync(self, canvas):
            sync_calls.append(canvas)
            return original_sync(self, canvas)

        monkeypatch.setattr(LibraryConversationsCanvas, "sync_state", spy_sync)
        recompose_calls = _spy_screen_recomposes(monkeypatch)

        # The service owns row order: entering the canvas auto-previews the
        # first returned row (chat-1).
        reader = screen.query_one(
            "#library-conversation-reader", LibraryConversationReader
        )
        assert screen._selected_conversation_id == "chat-1"
        assert reader.state.selected_id == "chat-1"

        screen.query_one("#library-conversation-row-1", Button).press()
        await pilot.pause()

        assert len(sync_calls) == 1
        assert recompose_calls == []
        assert screen._selected_conversation_id == "chat-2"
        assert reader.state.selected_id == "chat-2"


@pytest.mark.asyncio
async def test_tier2_canvas_sync_releases_mouse_capture_first():
    """The shared tier-2 canvas-sync helper releases ``App.mouse_captured``
    before recomposing the canvas -- mirroring ``BaseAppScreen.refresh``'s
    guard (see its docstring for the full mouse-capture war story), since
    ``canvas.sync_state`` recomposes the canvas directly and bypasses that
    screen-level protection entirely."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")
        await pilot.pause()

        capture_calls: list = []
        original_capture = pilot.app.capture_mouse

        def recording_capture(widget):
            capture_calls.append(widget)
            return original_capture(widget)

        pilot.app.capture_mouse = recording_capture

        # Select-mode enter/exit is a Tier-2 canvas-sync interaction too.
        screen.query_one("#library-conversations-select-toggle").press()
        await pilot.pause()

        assert None in capture_calls


@pytest.mark.asyncio
async def test_tier1_toggle_falls_back_to_recompose_on_query_one_failure(monkeypatch):
    """If the tier-1 in-place helper's ``query_one`` raises (e.g. the
    select-mode action strip isn't mounted because the mode raced), the
    checkbox toggle falls back to the old full recompose instead of
    crashing -- and the underlying selection state still reflects the
    toggle."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _enter_conversations_select_mode(screen, pilot)

        row = screen.query_one("#library-conversation-row-0", Button)
        conversation_id = row.conversation_id

        original_query_one = type(screen).query_one

        def raising_query_one(self, selector, *args, **kwargs):
            if isinstance(selector, str) and "selected-count" in selector:
                raise RuntimeError("forced query_one failure (fallback test)")
            return original_query_one(self, selector, *args, **kwargs)

        monkeypatch.setattr(type(screen), "query_one", raising_query_one)
        recompose_calls = _spy_screen_recomposes(monkeypatch)

        row.press()
        await pilot.pause()

        assert len(recompose_calls) == 1  # fell back to a full recompose
        assert screen._conversations_state.row_selection.is_selected(conversation_id)


@pytest.mark.asyncio
async def test_toggle_preserves_markup_escaped_titles():
    """PR #665 review: the in-place marker flip must slice the RAW label,
    not round-trip through .plain — a bracketed user title like "[draft]"
    would otherwise lose its escape_markup() and restyle/raise on toggle."""
    from tldw_chatbook.UI.Screens.library_screen import _apply_library_row_toggle

    class _Selection:
        count = 1

        @staticmethod
        def is_selected(_row_id):
            return True

    class _Recorder:
        disabled = False

        @staticmethod
        def update(_text):
            return None

    class _ConversationsState:
        row_selection = _Selection()

    class _Screen:
        _conversations_state = _ConversationsState()

        @staticmethod
        def query_one(selector, _cls=None):
            return _Recorder()

        @staticmethod
        def refresh(**_kwargs):
            raise AssertionError("fallback recompose must not fire")

    escaped_title = "\\[draft] weird title"  # as the canvas escapes it
    label_rest = f" {escaped_title}\n    2 messages"
    button = Button(f"☐{label_rest}")
    button._library_row_label_rest = label_rest  # as the canvas stashes it
    _apply_library_row_toggle(_Screen(), "conversations", button, "conv-1")
    # The label is rebuilt from the RAW stash — the escaped form must
    # survive verbatim (reading the mounted label back would un-escape it).
    assert f"☑{label_rest}" == f"☑{button._library_row_label_rest}"
    rendered = str(button.label)
    assert rendered.startswith("☑")
    assert "[draft] weird title" in rendered  # renders literally, not as markup
    assert "weird title" in rendered


@pytest.mark.parametrize("receiver_kind", ("screen", "controller"))
def test_media_row_toggle_resolves_the_dotted_state_path(receiver_kind: str):
    """wave-7 task 3: the media row-selection object moved to
    ``screen._media_state.row_selection``, so the dispatcher's COMPUTED name
    (``f"_library_{kind}_row_selection"``) can no longer reach it and the
    "media" branch must take the DOTTED form -- exactly as "conversations"
    already does one line above it.

    This is the guard the fifth census spelling needs (recipe §3): a name
    built at runtime appears nowhere in the source, so no reference census
    can see it go stale, and the failure is SILENT -- the ``attrgetter``
    raises, ``_apply_library_row_toggle``'s own ``except Exception`` swallows
    it, and the targeted patch degrades into the full-screen recompose the
    Tier-1 design exists to avoid. Making ``refresh`` raise is what turns
    that silent degradation into a red test (the
    ``test_toggle_preserves_markup_escaped_titles`` precedent above).

    **The ``controller`` leg was added at the wave-8 close, and it shipped
    RED.** Wave-7's dotted retarget was written against the SCREEN receiver
    only, and ``LibraryMediaController`` never declared a ``_media_state``
    accessor property (it holds ``_media_state_accessor`` and nothing else).
    Because ``handle_library_media_select_all`` /
    ``handle_library_media_select_clear`` hand the sibling
    ``_sync_library_canvas`` dispatcher a bare controller ``self`` -- and that
    dispatcher's media leg assigns through ``screen._media_state.
    selected_media_id`` -- every media "Select all"/"Clear" press raised
    ``AttributeError`` into the dispatcher's own ``except Exception`` and took
    the whole-screen recompose, live and silently. The notes series' own
    dual-receiver guard (below) is what found it, one function along.

    So: census a shared dispatcher's RECEIVERS, not just its spellings. A
    dotted retarget is only correct for the receiver it was written against,
    and a screen-only guard passes while the controller receiver degrades.
    """
    from tldw_chatbook.UI.Library_Modules.library_media_controller import (
        LibraryMediaController,
    )
    from tldw_chatbook.UI.Screens.library_screen import _apply_library_row_toggle

    class _Selection:
        count = 1

        @staticmethod
        def is_selected(_row_id):
            return True

    class _Recorder:
        disabled = True
        tooltip = None

        @staticmethod
        def update(_text):
            return None

    class _MediaState:
        row_selection = _Selection()

    def _query_one(_selector, _cls=None):
        return _Recorder()

    def _query(_selector):
        # dev's task-32045 added `screen.query("#library-media-select-bulk-
        # reason")` to this leg during the wave's review window, and it is
        # guarded by `if bulk_reason:` -- so an EMPTY result is a state the
        # production code already handles (the reason line simply absent).
        # An empty tuple is therefore the minimal stub that keeps this test
        # about the thing it guards: whether the DOTTED `_media_state` path
        # resolves on this receiver. Without it the `AttributeError` for the
        # missing `query` is swallowed by the dispatcher's own
        # `except Exception`, `refresh` fires, and BOTH legs go red for a
        # reason that has nothing to do with the state path.
        return ()

    def _analyze_reason():
        return ""

    def _refresh(**_kwargs):
        raise AssertionError("fallback recompose must not fire")

    label_rest = " Quarterly review\n    audio"
    button = Button(f"☐{label_rest}")
    button._library_row_label_rest = label_rest  # as the canvas stashes it

    if receiver_kind == "screen":

        class _Screen:
            _media_state = _MediaState()
            query_one = staticmethod(_query_one)
            query = staticmethod(_query)
            refresh = staticmethod(_refresh)
            _library_media_analyze_reason = staticmethod(_analyze_reason)

        receiver = _Screen()
    else:

        class _Controller(LibraryMediaController):
            # `_media_state` is deliberately NOT overridden -- the accessor
            # property on the REAL `LibraryMediaController` is the thing under
            # test. `query_one`/`query`/`refresh` are framework-service
            # properties on that class, so they can only be stubbed by
            # overriding them here, never by instance assignment.
            def __init__(self, state):
                self._media_state_accessor = lambda: state

            query_one = staticmethod(_query_one)
            query = staticmethod(_query)
            refresh = staticmethod(_refresh)
            _library_media_analyze_reason = staticmethod(_analyze_reason)

        receiver = _Controller(_MediaState())

    _apply_library_row_toggle(receiver, "media", button, "local:media:1")

    # Reached the real patch path: marker flipped in place, and the media-only
    # checked flag the canvas reads back was set.
    assert str(button.label).startswith("☑")
    assert button._library_media_checked is True


@pytest.mark.parametrize("receiver_kind", ("screen", "controller"))
def test_notes_row_toggle_resolves_the_dotted_state_path(receiver_kind: str):
    """wave-8 task 3: the notes row-selection object moved to
    ``screen._notes_state.row_selection``, so the dispatcher's COMPUTED name
    (``f"_library_{kind}_row_selection"``) can no longer reach it and the
    "notes" branch must take the DOTTED form -- the third and last kind to
    need it, after "conversations" and "media" above.

    This is the guard the fifth census spelling needs (recipe §3): a name
    built at runtime appears nowhere in the source, so no reference census
    can see it go stale, and the failure is SILENT -- the ``attrgetter``
    raises, ``_apply_library_row_toggle``'s own ``except Exception`` swallows
    it, and the targeted patch degrades into the full-screen recompose the
    Tier-1 design exists to avoid. Making ``refresh`` raise is what turns
    that silent degradation into a red test.

    **Why this one is parametrized over the RECEIVER, unlike its two
    siblings.** Notes' cluster hands the sibling ``_sync_library_canvas``
    dispatcher a bare ``self`` from 26 moved bodies (31 call sites), so a
    ``screen`` argument is a ``LibraryNotesController`` as often as it is a
    ``LibraryScreen``. A dotted spelling that resolves only on the screen is
    therefore only half a fix, and a screen-only guard would pass while the
    controller leg silently took the fallback. The ``controller`` leg builds
    the REAL ``LibraryNotesController`` class (subclassed only to stub the
    three framework-service properties a double cannot assign over) so the
    ``_notes_state`` accessor under test is the production one.
    """
    from tldw_chatbook.UI.Library_Modules.library_notes_controller import (
        LibraryNotesController,
    )
    from tldw_chatbook.UI.Screens.library_screen import _apply_library_row_toggle

    class _Selection:
        count = 1

        @staticmethod
        def is_selected(_row_id):
            return True

    class _Recorder:
        disabled = True
        tooltip = None

        @staticmethod
        def update(_text):
            return None

    class _NotesState:
        row_selection = _Selection()

    def _query_one(_selector, _cls=None):
        return _Recorder()

    def _query(selector):
        # `.library-notes-row` -> the sibling rows sharing this note id;
        # `#library-note-work-pane` -> absent, so the work-pane leg no-ops.
        return (button,) if selector == ".library-notes-row" else ()

    def _refresh(**_kwargs):
        raise AssertionError("fallback recompose must not fire")

    label_rest = " Shared note\n    today"
    button = Button(f"☐ {label_rest}")
    button._library_row_label_rest = label_rest  # as the canvas stashes it
    button.note_id = "n1"

    if receiver_kind == "screen":

        class _Screen:
            _notes_state = _NotesState()
            query_one = staticmethod(_query_one)
            query = staticmethod(_query)
            refresh = staticmethod(_refresh)

        receiver = _Screen()
    else:

        class _Controller(LibraryNotesController):
            # `_notes_state` is deliberately NOT overridden -- the real
            # accessor property on `LibraryNotesController` is the thing
            # under test. `query_one`/`query`/`refresh` are framework-service
            # properties on that class, so they can only be stubbed by
            # overriding them here, never by instance assignment.
            def __init__(self, state):
                self._notes_state_accessor = lambda: state

            query_one = staticmethod(_query_one)
            query = staticmethod(_query)
            refresh = staticmethod(_refresh)

        receiver = _Controller(_NotesState())

    _apply_library_row_toggle(receiver, "notes", button, "n1")

    # Reached the real patch path: the notes glyph (marker + space, unlike
    # media/conversations) was flipped in place on the matching row.
    assert str(button.label).startswith("☑ ")
