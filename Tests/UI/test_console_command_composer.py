"""Composer command interception + unknown-command Enter-again (Task 10);
`/prompt` resolution + insertion + Library-insert consumption (Task 12)."""

from unittest.mock import AsyncMock, Mock

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_console_native_chat_flow import (
    CapturingGateway,
    _configure_native_ready_console,
    _wait_for_text,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import ConsoleHarness
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.Prompts_DB import PromptsDatabase
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_prompt_picker_modal import (
    FILTER_INPUT_ID,
    ROW_ID_PREFIX,
    SEARCH_DEBOUNCE_SECONDS,
)


def _real_prompt_scope_service(tmp_path):
    """Build a real ``PromptsDatabase`` + ``PromptScopeService`` (mirrors
    ``Tests/UI/test_library_prompts_canvas.py``'s helper of the same name)."""
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    service = PromptScopeService(local_service=ScopeLocalPromptService(db), server_service=None)
    return db, service


async def _wait_for_picker_search(pilot) -> None:
    """Advance past the picker's debounce timer and let its search settle."""
    await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()


UNKNOWN_NOPE_HINT = (
    "Unknown command /nope — available: /prompt, /system. "
    "Press Enter again to send as text."
)
UNKNOWN_NADA_HINT = (
    "Unknown command /nada — available: /prompt, /system. "
    "Press Enter again to send as text."
)


def _system_message_contents(console) -> list[str]:
    store = console._ensure_console_chat_store()
    if store.active_session_id is None:
        return []
    messages = store.messages_for_session(store.active_session_id)
    return [message.content for message in messages if message.role is ConsoleMessageRole.SYSTEM]


async def _spy_submit_draft(console) -> AsyncMock:
    """Wrap the active controller's ``submit_draft`` so real sends still work."""
    controller = console._ensure_console_chat_controller()
    spy = AsyncMock(wraps=controller.submit_draft)
    controller.submit_draft = spy
    return spy


@pytest.mark.asyncio
async def test_console_unknown_command_first_enter_renders_hint_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)

        assert composer.draft_text() == "/nope x"
        submit_spy.assert_not_called()
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_unknown_command_second_unmodified_enter_sends_as_text():
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        submit_spy.assert_not_called()

        send_button.press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_called_once_with("/nope x")
        assert gateway.sent_messages[-1][-1]["content"] == "/nope x"
        assert console._console_unknown_send_armed is None


@pytest.mark.asyncio
async def test_console_unknown_command_edit_between_enters_re_hints_and_does_not_send():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        # Edit the draft to a different unknown command between Enters.
        composer.load_draft("/nada y")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NADA_HINT)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nada y"
        assert console._console_unknown_send_armed == "/nada y"
        contents = _system_message_contents(console)
        assert contents.count(UNKNOWN_NOPE_HINT) == 1
        assert contents.count(UNKNOWN_NADA_HINT) == 1


@pytest.mark.asyncio
async def test_console_unknown_command_roundtrip_edit_back_to_armed_text_requires_fresh_arm():
    """Editing away and back to the armed text still disarms (Task 10 hardening).

    Comparing the armed snapshot to the current draft text alone would let a
    user edit away from an armed unknown draft and back to the exact same
    text, then have an unrelated second Enter silently send it. The composer
    change subscription must disarm on *any* edit, not just a text mismatch.
    """
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/nope x")
        submit_spy = await _spy_submit_draft(console)
        send_button = console.query_one("#console-send-message", Button)

        send_button.press()
        await _wait_for_text(console, pilot, UNKNOWN_NOPE_HINT)
        assert console._console_unknown_send_armed == "/nope x"

        composer.load_draft("/nope xy")
        composer.load_draft("/nope x")
        await pilot.pause()
        assert console._console_unknown_send_armed is None

        send_button.press()
        await pilot.pause(0.1)

        submit_spy.assert_not_called()
        assert composer.draft_text() == "/nope x"
        assert console._console_unknown_send_armed == "/nope x"


@pytest.mark.asyncio
async def test_console_collapsed_paste_starting_with_slash_sends_normally():
    gateway = CapturingGateway()
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_provider_gateway_factory = lambda: gateway
    host = ConsoleHarness(app)
    pasted_text = "/nope " + ("x" * 80)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.insert_pasted_text(pasted_text)
        assert composer.has_paste_segments()
        submit_spy = await _spy_submit_draft(console)

        console.query_one("#console-send-message", Button).press()
        await _wait_for_text(console, pilot, "accepted")

        submit_spy.assert_called_once_with(pasted_text)
        assert gateway.sent_messages[-1][-1]["content"] == pasted_text
        assert console._console_unknown_send_armed is None
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_prompt_command_dispatches_insert_prompt_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt")
        submit_spy = await _spy_submit_draft(console)
        insert_prompt_spy = AsyncMock()
        console._console_command_insert_prompt = insert_prompt_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        insert_prompt_spy.assert_called_once()
        called_parse = insert_prompt_spy.call_args.args[0]
        assert called_parse.name == "prompt"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/prompt"


@pytest.mark.asyncio
async def test_console_system_command_dispatches_apply_system_stub():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/system helpful")
        submit_spy = await _spy_submit_draft(console)
        apply_system_spy = AsyncMock()
        console._console_command_apply_system = apply_system_spy

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.1)

        apply_system_spy.assert_called_once()
        called_parse = apply_system_spy.call_args.args[0]
        assert called_parse.name == "system"
        assert called_parse.args == "helpful"
        submit_spy.assert_not_called()
        assert composer.draft_text() == "/system helpful"


# ---------------------------------------------------------------------------
# Task 12: `/prompt` resolution + insertion + Library-insert consumption.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_prompt_command_unique_exact_name_replaces_draft(tmp_path):
    """A unique exact (case-insensitive) name match REPLACES the draft with
    the resolved prompt's ``user_prompt``, via paste semantics (a short body
    inserts inline, unchanged)."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="Alice",
        details="",
        system_prompt="",
        user_prompt="Please summarize the following text.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Case-insensitive: typed lowercase, stored mixed-case.
        composer.load_draft("/prompt summarize")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Please summarize the following text."
        assert len(host.screen_stack) == baseline_depth, "no picker should have opened for a unique match"


@pytest.mark.asyncio
async def test_console_prompt_command_large_user_prompt_collapses_to_paste_token(tmp_path):
    """An oversized resolved body collapses to a paste token for DISPLAY,
    while the canonical (sent) draft text keeps the full body -- exactly
    like a real paste."""
    large_body = "y" * 200
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Big",
        author="",
        details="",
        system_prompt="",
        user_prompt=large_body,
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Big")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == large_body
        assert "Pasted Text:" in composer._display_draft_text()
        assert large_body not in composer._display_draft_text()


@pytest.mark.asyncio
async def test_console_prompt_command_unique_prefix_match_resolves(tmp_path):
    """No exact match, but a unique case-insensitive name PREFIX match,
    still resolves without opening the picker."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Summarize body.",
        keywords=[],
    )
    db.add_prompt(
        name="Translate",
        author="",
        details="",
        system_prompt="",
        user_prompt="Translate body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Summ")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Summarize body."
        assert len(host.screen_stack) == baseline_depth


@pytest.mark.asyncio
async def test_console_prompt_command_exact_match_wins_over_ambiguous_prefix(tmp_path):
    """Resolution order: an exact name match resolves immediately even when
    the query is ALSO an ambiguous prefix of another prompt's name."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Sum",
        author="",
        details="",
        system_prompt="",
        user_prompt="Exact body.",
        keywords=[],
    )
    db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Prefix body.",
        keywords=[],
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Sum")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert composer.draft_text() == "Exact body."
        assert len(host.screen_stack) == baseline_depth


@pytest.mark.asyncio
async def test_console_prompt_command_ambiguous_exact_match_opens_picker(tmp_path):
    """Two prompts differing only by name CASE (the DB's UNIQUE constraint on
    ``name`` is case-sensitive, so both can exist) means the ci exact-match
    stage itself is ambiguous -- must fall through to the picker rather than
    guessing, with the typed args prefilled into the picker's filter."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Foo", author="", details="", system_prompt="", user_prompt="Upper body.", keywords=[]
    )
    db.add_prompt(
        name="foo", author="", details="", system_prompt="", user_prompt="Lower body.", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1, "the picker must have opened"
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == "Foo"
        # The draft is left exactly as typed -- the picker is a detour, not
        # a replacement, until the user actually picks something.
        assert composer.draft_text() == "/prompt Foo"


@pytest.mark.asyncio
async def test_console_prompt_command_no_args_opens_picker_with_empty_query(tmp_path):
    """`/prompt` with no args at all skips resolution entirely and opens the
    picker to browse, rather than attempting a meaningless empty-name match."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Summarize", author="", details="", system_prompt="", user_prompt="Body.", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]
        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        assert filter_input.value == ""


@pytest.mark.asyncio
async def test_console_prompt_command_picker_uses_real_bound_search_and_selection_replaces_draft(
    tmp_path,
):
    """The picker's ``prompt_search`` is really bound to the scope service
    (fresh reads, not a boot-time snapshot): typing further into the filter
    narrows results from the live DB, and picking a row replaces the draft
    with paste semantics -- same as a directly-resolved `/prompt <name>`."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="",
        details="",
        system_prompt="",
        user_prompt="Summarize body.",
        keywords=[],
    )
    db.add_prompt(
        name="Sundial", author="", details="", system_prompt="", user_prompt="Sundial body.", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        # Ambiguous prefix: both "Summarize" and "Sundial" start with "Su".
        composer.load_draft("/prompt Su")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1
        picker = host.screen_stack[-1]

        filter_input = picker.query_one(f"#{FILTER_INPUT_ID}", Input)
        filter_input.value = "Summariz"
        await _wait_for_picker_search(pilot)

        row = picker.query_one(f"#{ROW_ID_PREFIX}{prompt_id}", Button)
        row.press()
        await pilot.pause(0.2)

        assert len(host.screen_stack) == baseline_depth, "the picker must have dismissed"
        assert composer.draft_text() == "Summarize body."


@pytest.mark.asyncio
async def test_console_prompt_command_picker_escape_leaves_draft_untouched(tmp_path):
    """Escaping the picker dismisses with ``None`` and never touches the
    composer draft."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(
        name="Foo", author="", details="", system_prompt="", user_prompt="Upper body.", keywords=[]
    )
    db.add_prompt(
        name="foo", author="", details="", system_prompt="", user_prompt="Lower body.", keywords=[]
    )
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.prompt_scope_service = service
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("/prompt Foo")

        console.query_one("#console-send-message", Button).press()
        await pilot.pause(0.2)
        assert len(host.screen_stack) == baseline_depth + 1

        await pilot.press("escape")
        await pilot.pause(0.1)

        assert len(host.screen_stack) == baseline_depth
        assert composer.draft_text() == "/prompt Foo"


# -- Library "Use in Console" consumption (ChatScreen-side gating/insertion) --


@pytest.mark.asyncio
async def test_console_pending_prompt_insert_is_consumed_automatically_on_mount():
    """The staged field is consumed by the real ``on_mount`` wiring itself
    (not just the private method called directly) -- proves the Library
    hand-off actually lands without any test-only shortcut."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.pending_console_prompt_insert = "staged on mount"
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        for _ in range(40):
            if composer.draft_text() == "staged on mount":
                break
            await pilot.pause(0.05)

        assert composer.draft_text() == "staged on mount"
        assert app.pending_console_prompt_insert is None


@pytest.mark.asyncio
async def test_console_pending_prompt_insert_is_consumed_automatically_on_resume():
    """Same as the ``on_mount`` variant above, but exercises the real
    ``on_screen_resume`` timer path -- the finding this regression guards
    against is specific to resume, where (unlike ``on_mount``) nothing
    schedules an equivalent ``_sync_native_console_chat_ui`` pass ahead of
    the 0.15s consumption timer."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        app.pending_console_prompt_insert = "staged on resume"
        console.on_screen_resume()

        for _ in range(40):
            if composer.draft_text() == "staged on resume":
                break
            await pilot.pause(0.05)

        assert composer.draft_text() == "staged on resume"
        assert app.pending_console_prompt_insert is None


@pytest.mark.asyncio
async def test_console_resume_triggered_prompt_insert_survives_stale_session_switch():
    """Regression for the resume wipe-race: if a session switch races ahead
    of a resume-triggered insert, ``_console_visible_draft_session_id`` can
    be stale relative to the store's active session when the insert's own
    0.15s timer fires. Without the fix, a *later* call to
    ``_sync_console_session_draft`` (as several real call sites make, e.g.
    the periodic transcript poller or any other action routed through
    ``_sync_native_console_chat_ui``) would then unconditionally reload the
    composer from the newly-active session's stale stored draft, silently
    discarding the insert -- with no retry, since the pending field is
    already cleared once the insert lands. This must not happen: the insert
    consumption itself has to settle the draft tracker before inserting, so
    a later sync pass is a no-op instead of a clobber."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)

        store = console._ensure_console_chat_store()
        first_session = store.ensure_session()
        # Let the mount-time sync pass settle the visible-draft tracker onto
        # the first session before simulating a session switch that races
        # ahead of the not-yet-fired resume consumption below.
        await pilot.pause(0.1)
        assert console._console_visible_draft_session_id == first_session.id

        second_session = store.create_session(title="Second")
        store.set_session_draft(second_session.id, "stale leftover draft")
        assert store.active_session_id == second_session.id
        # The tracker has NOT caught up with the switch yet -- this is the
        # exact staleness the finding describes.
        assert console._console_visible_draft_session_id == first_session.id

        app.pending_console_prompt_insert = "resume-triggered insert"
        console.on_screen_resume()
        await pilot.pause(0.25)  # past the 0.15s consumption timer

        assert app.pending_console_prompt_insert is None
        assert "resume-triggered insert" in composer.draft_text()

        # Simulate a later, unrelated sync pass -- any of several real call
        # sites (periodic transcript polling, another send/stop cycle, a
        # settings-modal callback) route through this same method. It must
        # not retroactively wipe the insert by reloading the stale draft.
        console._sync_console_session_draft()
        assert "resume-triggered insert" in composer.draft_text()
        assert console._console_visible_draft_session_id == second_session.id


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_empty_draft_is_clean_insert():
    """An empty composer draft gets a clean insert -- no separator noise."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        assert composer.draft_text() == ""

        app.pending_console_prompt_insert = "inserted body"
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "inserted body"
        assert app.pending_console_prompt_insert is None


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_appends_to_existing_draft():
    """Library's insert-in-console NEVER clobbers an in-progress draft --
    it appends onto it instead."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")

        app.pending_console_prompt_insert = "inserted body"
        await console._consume_pending_console_prompt_insert()

        draft = composer.draft_text()
        assert draft.startswith("abc")
        assert "inserted body" in draft
        assert draft == "abc\ninserted body"
        assert app.pending_console_prompt_insert is None


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_large_body_appends_as_collapsed_token():
    """An oversized appended body still collapses to a display token, exactly
    like a real paste onto an existing draft."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    large_body = "z" * 200

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")

        app.pending_console_prompt_insert = large_body
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == f"abc\n{large_body}"
        assert "Pasted Text:" in composer._display_draft_text()
        assert large_body not in composer._display_draft_text()


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_blocked_shows_exact_toast():
    """First-run setup blocked (no provider/model configured): the insert
    shows the exact toast copy, leaves the draft completely untouched, and
    still clears the pending field (no stale re-fire on a later mount)."""
    app = _build_test_app()  # deliberately NOT _configure_native_ready_console
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")
        notify_spy = Mock()
        app.notify = notify_spy

        app.pending_console_prompt_insert = "inserted body"
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "abc"
        notify_spy.assert_called_once_with(
            "Finish provider setup to insert prompts.", severity="warning"
        )
        assert app.pending_console_prompt_insert is None


@pytest.mark.asyncio
async def test_console_consumes_pending_prompt_insert_noop_when_nothing_pending():
    """Nothing pending: no-op, no notify, draft untouched."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        baseline_depth = len(host.screen_stack)
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("abc")
        notify_spy = Mock()
        app.notify = notify_spy

        assert app.pending_console_prompt_insert is None
        await console._consume_pending_console_prompt_insert()

        assert composer.draft_text() == "abc"
        notify_spy.assert_not_called()
