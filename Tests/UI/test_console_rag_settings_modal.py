"""The Library-search chip's Library search settings modal: gating, results, dismissal.

User request 2026-08-01: clicking "Library search: off" in the status strip
opens a modal that lets the user set the retrieval query and run it --
instead of the query living only in a rail input that may not even be on
screen.
"""

from unittest.mock import Mock

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rag_state import (
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LIBRARY_RAG_SOURCE_TYPES,
)
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.Widgets.Console.console_rag_settings_modal import (
    CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
    CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX,
    ConsoleRagSettingsModal,
    ConsoleRagSettingsResult,
    console_rag_source_toggle_label,
    normalize_console_rag_source_types,
)


def _retrieval_for(screen) -> ConsoleRetrievalController:
    """Bind the controller's pure settings edges to a lightweight screen."""
    owner = object.__new__(ConsoleRetrievalController)
    owner._library_rag_source_scope = lambda: normalize_console_rag_source_types(
        getattr(screen, "_console_library_rag_source_types", None)
    )
    owner._set_library_rag_source_scope = screen._set_console_library_rag_source_scope
    owner._set_library_rag_query = screen._set_console_library_rag_query
    owner._run_library_rag_action = screen._run_console_library_rag_from_visible_action
    return owner


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_returns_the_query_and_is_gated_on_non_blank_text():
    """Run is disabled while blank, enables on typing, and returns the query."""

    class RagHost(ConsolidatedCSSApp):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(source_types=("notes", "media")),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleRagSettingsModal)

        run = modal.query_one("#console-rag-settings-run", Button)
        assert run.disabled is True, "blank query must not be runnable"

        query_input = modal.query_one("#console-rag-settings-query", Input)
        query_input.value = "incident retro notes"
        await pilot.pause()
        assert run.disabled is False

        await pilot.click("#console-rag-settings-run")
        await pilot.pause()
        await pilot.pause()

    assert received == [
        ConsoleRagSettingsResult(
            query="incident retro notes",
            run=True,
            source_types=("notes", "media"),
        )
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_prefilled_query_is_runnable_and_enter_submits():
    """A prefilled modal is one keypress from retrieval (Enter submits)."""

    class RagHost(ConsolidatedCSSApp):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(query="what changed in auth"),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#console-rag-settings-run", Button).disabled is False

        modal.query_one("#console-rag-settings-query", Input).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

    assert received == [
        ConsoleRagSettingsResult(query="what changed in auth", run=True)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_escape_and_backdrop_all_dismiss_without_changes():
    """Every no-action exit returns None; inside clicks keep it open."""

    class RagHost(ConsolidatedCSSApp):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(query="draft text"),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen

        # A click inside the box (its header line) must not dismiss.
        box = modal.query_one("#console-rag-settings")
        await pilot.click(offset=(box.region.x + 2, box.region.y + 1))
        await pilot.pause()
        assert app.screen is modal

        # Backdrop click dismisses with no changes.
        await pilot.click(offset=(1, 1))
        await pilot.pause()
        await pilot.pause()
        assert app.screen is not modal

    assert received == [None]


def _static_plain_text(widget: Static) -> str:
    rendered = widget.render()
    return str(getattr(rendered, "plain", rendered))


def _toggle_labels(modal: ConsoleRagSettingsModal) -> list[str]:
    return [
        str(button.label)
        for button in modal.query(".console-rag-settings-source-toggle").results(Button)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modal_shows_one_toggle_per_library_source_with_display_labels():
    """(a) RAG-44: the read-only "Scope: notes, media, conversations" line
    is now a toggle per Library source, in Library's four-source
    vocabulary, with Library's display-cased labels -- the checked ones
    are exactly the current selection."""

    class RagHost(ConsolidatedCSSApp):
        pass

    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(source_types=("notes", "conversations"))
        )
        await pilot.pause()
        modal = app.screen

        assert _toggle_labels(modal) == [
            "✓ Notes",
            "○ Media",
            "✓ Conversations",
            "○ Prompts",
        ]
        # No fifth vocabulary: the ids are Library's source-type keys and
        # the labels come from Library's one label table.
        assert [
            str(button.id or "").removeprefix(CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX)
            for button in modal.query(".console-rag-settings-source-toggle").results(
                Button
            )
        ] == list(LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_every_source_toggle_fits_inside_the_modal_box():
    """Four toggles on one row inside a 64-column modal: each must land
    inside the box AND have room to actually draw its label.

    The content-box assertions are the load-bearing ones: the first cut
    of this row rendered every toggle as two Button border rows with a
    ZERO-height content area -- clickable, correctly labelled in the
    widget tree, and completely invisible on screen. Region geometry
    alone would have passed that.
    """

    class RagHost(ConsolidatedCSSApp):
        pass

    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(ConsoleRagSettingsModal())
        await pilot.pause()
        modal = app.screen
        box = modal.query_one("#console-rag-settings")
        toggles = list(
            modal.query(".console-rag-settings-source-toggle").results(Button)
        )

        assert len(toggles) == 4
        for toggle in toggles:
            assert toggle.content_size.height >= 1, f"{toggle.id} draws no rows"
            assert toggle.content_size.width >= len(str(toggle.label))
            assert toggle.region.right <= box.region.right


@pytest.mark.integration
@pytest.mark.asyncio
async def test_toggling_a_source_off_returns_the_reduced_source_types():
    """(b) modal half: switching Media off and running returns a
    ``source_types`` without media (the screen then sends exactly that to
    retrieval)."""

    class RagHost(ConsolidatedCSSApp):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(
                query="what changed in auth",
                source_types=CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
            ),
            callback=received.append,
        )
        await pilot.pause()
        modal = app.screen

        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}media")
        await pilot.pause()
        assert _toggle_labels(modal)[1] == "○ Media"

        await pilot.click("#console-rag-settings-run")
        await pilot.pause()
        await pilot.pause()

    assert received == [
        ConsoleRagSettingsResult(
            query="what changed in auth",
            run=True,
            source_types=("notes", "conversations"),
        )
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancel_discards_toggle_changes():
    """(c) modal half: toggles changed then cancelled return nothing at
    all, so the screen's stored scope cannot move."""

    class RagHost(ConsolidatedCSSApp):
        pass

    received: list[ConsoleRagSettingsResult | None] = []
    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(
                query="what changed in auth",
                source_types=CONSOLE_RAG_DEFAULT_SOURCE_TYPES,
            ),
            callback=received.append,
        )
        await pilot.pause()

        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}prompts")
        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}notes")
        await pilot.pause()

        await pilot.click("#console-rag-settings-cancel")
        await pilot.pause()
        await pilot.pause()

    assert received == [None]


@pytest.mark.unit
def test_cancel_leaves_the_screens_stored_scope_untouched():
    """(c) screen half: the ``None`` callback writes neither the query nor
    the source scope."""
    screen = Mock()
    _retrieval_for(screen)._apply_console_rag_settings_choice(None)

    screen._set_console_library_rag_query.assert_not_called()
    screen._set_console_library_rag_source_scope.assert_not_called()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_is_gated_on_at_least_one_selected_source():
    """Running with every source switched off would retrieve from nothing;
    the Run action is gated on a selection the same way it is gated on a
    non-blank query."""

    class RagHost(ConsolidatedCSSApp):
        pass

    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(
                query="what changed in auth",
                source_types=("notes",),
            )
        )
        await pilot.pause()
        modal = app.screen
        run = modal.query_one("#console-rag-settings-run", Button)
        assert run.disabled is False

        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}notes")
        await pilot.pause()
        assert run.disabled is True

        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}media")
        await pilot.pause()
        assert run.disabled is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modal_summary_line_and_readiness_card_share_one_builder():
    """Two seams, one builder (the PR-2 scope-summary lesson): the modal's
    own summary line and the Inspector readiness card's label are the same
    string for the same selection, and both track a toggle."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class RagHost(ConsolidatedCSSApp):
        pass

    screen = ChatScreen.__new__(ChatScreen)
    screen._console_library_rag_source_types = ("notes", "conversations")

    app = RagHost()
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleRagSettingsModal(source_types=("notes", "conversations"))
        )
        await pilot.pause()
        modal = app.screen
        summary = modal.query_one("#console-rag-settings-scope", Static)

        assert _static_plain_text(summary) == (
            _retrieval_for(screen)._console_library_rag_scope_label()
        )
        assert _static_plain_text(summary) == (
            "Sources: Notes, Conversations (Media, Prompts off)"
        )

        await pilot.click(f"#{CONSOLE_RAG_SOURCE_TOGGLE_ID_PREFIX}media")
        await pilot.pause()
        screen._console_library_rag_source_types = ("notes", "media", "conversations")
        assert _static_plain_text(summary) == (
            _retrieval_for(screen)._console_library_rag_scope_label()
        )


@pytest.mark.unit
def test_readiness_card_label_uses_the_library_summary_grammar():
    """(d) the readiness-card label reflects the STORED scope in Library's
    summary grammar (selected in canonical order, deselected named as
    off), under Console's own "Sources" noun -- "Scope" is already spent
    on the Console's item scope ("Scope: 2 items")."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)

    screen._console_library_rag_source_types = ("notes", "conversations")
    assert (
        _retrieval_for(screen)._console_library_rag_scope_label()
        == "Sources: Notes, Conversations (Media, Prompts off)"
    )

    screen._console_library_rag_source_types = ("notes", "media", "conversations")
    assert (
        _retrieval_for(screen)._console_library_rag_scope_label()
        == "Sources: Notes, Media, Conversations (Prompts off)"
    )

    screen._console_library_rag_source_types = LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES
    assert (
        _retrieval_for(screen)._console_library_rag_scope_label()
        == "Sources: all local sources"
    )


@pytest.mark.unit
def test_default_console_source_scope_is_todays_three():
    """(e) zero behavior change until a toggle is touched: with nothing
    stored, retrieval still runs over exactly notes/media/conversations
    (prompts OFF), and the default is ONE tuple shared by the screen and
    the modal."""
    from tldw_chatbook.UI.Screens.chat_screen import (
        CONSOLE_LIBRARY_RAG_SOURCE_SCOPE,
        ChatScreen,
    )

    assert CONSOLE_RAG_DEFAULT_SOURCE_TYPES == ("notes", "media", "conversations")
    assert CONSOLE_LIBRARY_RAG_SOURCE_SCOPE is CONSOLE_RAG_DEFAULT_SOURCE_TYPES

    screen = Mock()
    screen._console_library_rag_query = "what changed in auth"
    ChatScreen._run_console_library_rag_from_visible_action(screen)

    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.source_types == ("notes", "media", "conversations")


@pytest.mark.unit
def test_stored_source_scope_is_what_retrieval_receives():
    """(b) screen half: the stored selection -- not the constant -- is what
    the retrieval request carries."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = "what changed in auth"
    screen._console_library_rag_source_types = ("notes", "conversations")

    ChatScreen._run_console_library_rag_from_visible_action(screen)

    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.source_types == ("notes", "conversations")


@pytest.mark.unit
def test_modal_choice_stores_the_source_scope_before_running():
    """The modal's Run stores the chosen sources through the one writer,
    then delegates the run -- so the retrieval that follows uses them."""
    screen = Mock()
    _retrieval_for(screen)._apply_console_rag_settings_choice(
        ConsoleRagSettingsResult(
            query="what changed",
            run=True,
            source_types=("notes", "prompts"),
        ),
    )

    screen._set_console_library_rag_source_scope.assert_called_once_with(
        ("notes", "prompts")
    )
    screen._run_console_library_rag_from_visible_action.assert_called_once()


@pytest.mark.unit
def test_source_scope_survives_a_screen_state_round_trip():
    """A customized source scope is Console-local state, but it must not
    evaporate on a tab switch: it round-trips through the native Console
    screen state next to the sessions themselves."""
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatSession,
        ConsoleChatStore,
    )
    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState

    from Tests.UI.console_controller_stubs import NO_APP, stub_message_controller

    def _bare_screen(store: ConsoleChatStore) -> ChatScreen:
        screen = ChatScreen.__new__(ChatScreen)
        screen._retrieval = Mock()
        screen._console_chat_store = store
        screen._session = ConsoleSessionController.__new__(ConsoleSessionController)
        screen._console_visible_draft_session_id = None
        screen._console_composer_or_none = lambda: None
        screen._task_resume_state = TaskResumeState()
        # `_restore_native_console_state` calls the three
        # `_rehydrate_console_message_*` helpers, which moved to
        # `ConsoleMessageController` (wave-3 console decomposition, task 1)
        # and are reached through `ChatScreen`'s delegations.
        # `ChatScreen.__new__` skips the construction `__init__` would do.
        # Those three read only `app_instance`, so nothing else is wired.
        stub_message_controller(
            screen,
            context="test_console_rag_settings_modal._bare_screen",
            # No harness app: this shell exercises the three rehydrate
            # helpers, which read `app_instance` only through
            # `getattr(..., None)`. Declared rather than inferred, so a
            # future helper needing a real app fails loudly here.
            app_instance=NO_APP,
        )
        return screen

    store = ConsoleChatStore()
    session = ConsoleChatSession(id="session-a", title="Chat 1")
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_screen(store)
    screen._console_library_rag_source_types = ("notes", "prompts")

    payload = screen._serialize_native_console_state()
    assert payload is not None

    restored = _bare_screen(ConsoleChatStore())
    restored._restore_native_console_state(payload)

    assert restored._console_library_rag_source_types == ("notes", "prompts")
    assert (
        _retrieval_for(restored)._console_library_rag_scope_label()
        == "Sources: Notes, Prompts (Media, Conversations off)"
    )


@pytest.mark.unit
def test_source_type_normalization_refuses_unknown_values_and_falls_back():
    """The Console's stored scope is normalized at every boundary: unknown
    values are dropped, order is canonical, and an unusable value (legacy
    screen state, an empty selection) falls back to the default rather
    than retrieving from nothing."""
    assert normalize_console_rag_source_types(["prompts", "notes"]) == (
        "notes",
        "prompts",
    )
    assert normalize_console_rag_source_types(["Notes", "notes"]) == ("notes",)
    assert (
        normalize_console_rag_source_types(["workspaces", "bogus"])
        == CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    )
    assert normalize_console_rag_source_types([]) == CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    assert normalize_console_rag_source_types(None) == CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    assert (
        normalize_console_rag_source_types(object()) == CONSOLE_RAG_DEFAULT_SOURCE_TYPES
    )


@pytest.mark.unit
def test_toggle_labels_come_from_the_one_library_label_table():
    """No fifth vocabulary: every toggle label is Library's display label
    for that source type."""
    labels = dict(LIBRARY_RAG_SOURCE_TYPES)
    for source_type in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES:
        assert console_rag_source_toggle_label(source_type, True) == (
            f"✓ {labels[source_type]}"
        )
        assert console_rag_source_toggle_label(source_type, False) == (
            f"○ {labels[source_type]}"
        )


@pytest.mark.unit
def test_status_copy_is_honest_about_what_on_means():
    """The modal explains that "on" == staged retrieved evidence."""
    off = ConsoleRagSettingsModal()
    assert "Library search is off" in off._status_copy()
    assert "staged" in off._status_copy()

    on = ConsoleRagSettingsModal(rag_active=True, staged_title="Incident Review")
    assert "Library search is on" in on._status_copy()
    assert "Incident Review" in on._status_copy()


@pytest.mark.unit
def test_screen_callback_stores_sanitized_query_and_delegates_run():
    """The screen-side callback owns sanitization and the run delegation."""
    from textual.css.query import QueryError

    screen = Mock()
    screen._console_library_rag_query = ""
    screen.query_one = Mock(side_effect=QueryError("not mounted"))

    # The bare state contract: sanitize, store through the one query
    # writer (`_set_console_library_rag_query`), delegate when run=True.
    _retrieval_for(screen)._apply_console_rag_settings_choice(
        ConsoleRagSettingsResult(query="  spaced   query  ", run=True)
    )
    screen._set_console_library_rag_query.assert_called_once_with("spaced query")
    screen._run_console_library_rag_from_visible_action.assert_called_once()

    screen._set_console_library_rag_query.reset_mock()
    screen._run_console_library_rag_from_visible_action.reset_mock()
    _retrieval_for(screen)._apply_console_rag_settings_choice(
        ConsoleRagSettingsResult(query="no run", run=False)
    )
    screen._set_console_library_rag_query.assert_called_once_with("no run")
    screen._run_console_library_rag_from_visible_action.assert_not_called()

    screen._set_console_library_rag_query.reset_mock()
    _retrieval_for(screen)._apply_console_rag_settings_choice(None)
    screen._set_console_library_rag_query.assert_not_called()


@pytest.mark.unit
def test_visible_run_action_falls_back_to_the_composer_draft():
    """User decision (2026-08-02): with no dedicated query set, the visible
    Search Library action retrieves with the composer draft instead of
    demanding a query the collapsed rail gives no place to type. The
    fallback is STORED so the rail input and this modal agree with what
    actually ran.

    RAG-41/42 (2026-08-04): with no query anywhere -- no dedicated query
    AND an empty composer draft -- this used to toast at an invisible
    input; it now opens the RAG settings modal instead (the one place a
    query can actually be typed), and does not toast."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = ""
    composer = Mock()
    composer.draft_text.return_value = "  what   changed in auth  "
    screen._console_composer_or_none.return_value = composer

    ChatScreen._run_console_library_rag_from_visible_action(screen)

    screen._set_console_library_rag_query.assert_called_once_with(
        "what changed in auth"
    )
    screen.app_instance.notify.assert_not_called()
    screen._retrieval._stage_console_library_rag_launch.assert_called_once()
    launch = screen._retrieval._stage_console_library_rag_launch.call_args.args[0]
    assert launch.payload["query"] == "what changed in auth"
    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.query == "what changed in auth"

    # No dedicated query AND an empty composer: the settings modal opens
    # instead of toasting, and no retrieval runs underneath it.
    empty = Mock()
    empty._console_library_rag_query = ""
    empty_composer = Mock()
    empty_composer.draft_text.return_value = "   "
    empty._console_composer_or_none.return_value = empty_composer

    ChatScreen._run_console_library_rag_from_visible_action(empty)

    empty.app_instance.notify.assert_not_called()
    empty._open_console_rag_settings.assert_called_once()
    empty._retrieval._stage_console_library_rag_launch.assert_not_called()


@pytest.mark.unit
def test_dedicated_query_still_wins_over_the_composer_draft():
    """A query set through the rail or the modal is what runs; the draft
    fallback only fills a VACANT query, never overrides an explicit one."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = "explicit query"
    composer = Mock()
    composer.draft_text.return_value = "draft text"
    screen._console_composer_or_none.return_value = composer

    ChatScreen._run_console_library_rag_from_visible_action(screen)

    screen._set_console_library_rag_query.assert_not_called()
    request = screen._execute_console_library_rag_search.call_args.args[0]
    assert request.query == "explicit query"


@pytest.mark.unit
def test_modal_open_prefills_a_normal_question_draft():
    """Sanity companion to the guard tests below: an ordinary question
    draft still prefills the RAG settings modal (the chip-open site,
    ``_open_console_rag_settings`` -- the run-fallback site's equivalent
    is already covered by ``test_visible_run_action_falls_back_to_the_
    composer_draft``)."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = Mock()
    screen._console_library_rag_query = ""
    screen._pending_console_launch_context = None
    composer = Mock()
    composer.draft_text.return_value = "  what   changed in auth  "
    screen._console_composer_or_none.return_value = composer

    ChatScreen._open_console_rag_settings(screen)

    modal = screen.app.push_screen.call_args.args[0]
    assert modal._query == "what changed in auth"


@pytest.mark.unit
@pytest.mark.parametrize(
    "unsafe_draft",
    [
        pytest.param("/Users/x/notes.md", id="absolute-path"),
        pytest.param("file:///Users/x/notes.md", id="file-uri"),
        pytest.param("https://example.com/incident-notes", id="bare-url"),
        pytest.param("x" * 201, id="oversized-201-chars"),
    ],
)
def test_prefill_guards_reject_paths_urls_and_oversized_drafts_at_both_sites(
    unsafe_draft,
):
    """RAG-43: a composer draft that IS (in its entirety) a dropped file
    path, a ``file://`` URI, a bare URL, or longer than 200 chars must
    never silently become the retrieval query -- live UAT saw a fixture
    path prefill verbatim into the query field. Both prefill sites --
    the RAG chip's modal-open prefill and the visible Run Library RAG
    action's queryless fallback -- share one guard, so both must refuse
    these drafts the same way.

    Post-Task-2, the run-fallback's empty branch opens the settings
    modal instead of toasting, so a guarded draft must land there too
    (queryless, exactly like an empty draft) rather than being silently
    stored and run.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    # Site 1: the RAG chip's modal-open prefill.
    modal_screen = Mock()
    modal_screen._console_library_rag_query = ""
    modal_screen._pending_console_launch_context = None
    modal_composer = Mock()
    modal_composer.draft_text.return_value = unsafe_draft
    modal_screen._console_composer_or_none.return_value = modal_composer

    ChatScreen._open_console_rag_settings(modal_screen)

    modal = modal_screen.app.push_screen.call_args.args[0]
    assert modal._query == ""

    # Site 2: the visible Run Library RAG action's queryless fallback.
    run_screen = Mock()
    run_screen._console_library_rag_query = ""
    run_composer = Mock()
    run_composer.draft_text.return_value = unsafe_draft
    run_screen._console_composer_or_none.return_value = run_composer

    ChatScreen._run_console_library_rag_from_visible_action(run_screen)

    run_screen._set_console_library_rag_query.assert_not_called()
    run_screen._retrieval._stage_console_library_rag_launch.assert_not_called()
    run_screen.app_instance.notify.assert_not_called()
    run_screen._open_console_rag_settings.assert_called_once()


@pytest.mark.unit
def test_prefill_allows_a_question_that_merely_mentions_a_url():
    """Borderline ruling (RAG-43): only drafts that ARE a path/URL in
    their *entirety* are guarded, not drafts that merely contain one
    alongside other text. A question like this is still exactly the
    text the user is about to send -- retrieval should look for it too,
    same as any other question draft -- so it is deliberately NOT
    guarded even though it embeds a URL."""
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    draft = "check out https://example.com/incident-notes for context"

    # Site 1: modal-open prefill.
    modal_screen = Mock()
    modal_screen._console_library_rag_query = ""
    modal_screen._pending_console_launch_context = None
    modal_composer = Mock()
    modal_composer.draft_text.return_value = draft
    modal_screen._console_composer_or_none.return_value = modal_composer

    ChatScreen._open_console_rag_settings(modal_screen)

    modal = modal_screen.app.push_screen.call_args.args[0]
    assert modal._query == draft

    # Site 2: run-fallback stores and runs with the draft as-is.
    run_screen = Mock()
    run_screen._console_library_rag_query = ""
    run_composer = Mock()
    run_composer.draft_text.return_value = draft
    run_screen._console_composer_or_none.return_value = run_composer

    ChatScreen._run_console_library_rag_from_visible_action(run_screen)

    run_screen._set_console_library_rag_query.assert_called_once_with(draft)
    run_screen._open_console_rag_settings.assert_not_called()
