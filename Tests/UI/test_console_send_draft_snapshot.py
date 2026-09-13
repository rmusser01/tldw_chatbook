"""TASK-340: Enter must snapshot the draft synchronously at the keypress.

The Console Enter branch posts a ``Button.Pressed`` message and the send
handler used to read ``composer.draft_text()`` only when that message was
finally processed — printable keys handled in between mutated the draft and
were folded into the sent message (UX review finding
j6-send-captures-late-keystrokes). These tests deliver Enter synchronously
via ``ChatScreen.on_key`` and interleave typing before the message pump runs,
which is exactly the interleave a fast typist produces.
"""

import asyncio
from dataclasses import replace

import pytest
from textual.events import Key
from textual.widgets import Button

from Tests.UI.test_console_native_chat_flow import (
    BlockedGateway,
    _build_console_send_test_app,
    _persist_console_provider_config,
    _select_llamacpp_console,
)
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
    _visible_text,
)
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"


async def _wait_for_text(console, pilot, needle: str, tries: int = 40) -> None:
    for _ in range(tries):
        if needle in _visible_text(console):
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {needle!r}")


def _ready_openai_app(monkeypatch, reply: str):
    app = _build_console_send_test_app()
    # This in-memory composer harness has no durable trace repository.
    # Exercise ordinary sending with the supported capture-off setting.
    from tldw_chatbook import config as config_module

    assert config_module.save_settings_to_cli_config(
        {"console": {"exchange_capture": False}}
    )
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": DUMMY_OPENAI_API_KEY},
    )

    def fake_chat_api_call(**_kwargs):
        return reply

    monkeypatch.setattr(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call",
        fake_chat_api_call,
    )
    return app


def _press_enter_synchronously(console) -> None:
    """Deliver Enter to the screen key handler without pumping messages."""
    console.on_key(Key(key="enter", character="\r"))


def test_composer_captured_revision_is_non_destructive_until_custody_commit() -> None:
    composer = ConsoleComposerBar()
    composer.load_draft("accepted revision")

    captured = composer.capture_draft_for_send()

    assert captured is not None
    assert captured.text == "accepted revision"
    assert composer.draft_text() == "accepted revision"

    composer.insert_text("newer typing")
    composer.commit_captured_draft(captured)

    assert composer.draft_text() == "newer typing"


def test_composer_commit_refuses_a_retyped_identical_captured_revision() -> None:
    """A replacement draft must not be mistaken for the accepted revision."""
    composer = ConsoleComposerBar()
    composer.load_draft("accepted revision")

    captured = composer.capture_draft_for_send()
    assert captured is not None

    composer.clear_draft()
    composer.insert_text("accepted revision")

    assert composer.commit_captured_draft(captured) is False
    assert composer.draft_text() == "accepted revision"


@pytest.mark.asyncio
async def test_console_custody_snapshots_draft_before_late_keystrokes(monkeypatch):
    app = _ready_openai_app(monkeypatch, "snapshot reply")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("line one")
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        controller = console._ensure_console_chat_controller()
        received: dict[str, object] = {}

        async def submit_spy(draft, **kwargs):
            received["draft"] = draft
            received.update(kwargs)
            kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        async def run_chain(*, session_id, initial_turn):
            received["chain_session_id"] = session_id
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", submit_spy)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)
        stash = composer.capture_draft_for_send()
        assert await console._dispatch_console_draft_send(stash.text, stash=stash)
        # Keystrokes arriving before the Button.Pressed message is processed:
        composer.insert_text("line two")

        assert turn_ids
        await console._console_runtime().wait_for_turn(turn_ids[0])

        assert received["draft"] == "line one"
        assert received["chain_session_id"] == received["session_id"]
        # The late keystrokes belong to the NEXT draft — and the
        # accepted-submit clear must not eat them either.
        assert composer.draft_text() == "line two"


@pytest.mark.asyncio
async def test_mouse_send_completion_preserves_text_typed_after_acceptance(monkeypatch):
    """The real Send button commits its capture without erasing newer typing."""

    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        store = controller.store
        session = store.ensure_session()
        store.switch_session(session.id)
        console._session._sync_console_session_draft()
        composer.load_draft("original mouse draft")
        store.set_session_draft(session.id, composer.draft_text())

        async def accept_custody(draft, **kwargs):
            assert draft == "original mouse draft"
            assert kwargs["session_id"] == session.id
            kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        async def run_chain(*, session_id, initial_turn):
            assert session_id == session.id
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", accept_custody)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, target_session_id: str) -> str:
            turn_id = original_launch(draft, target_session_id)
            # Typing can land after the mouse capture but before the
            # synchronous custody callback commits that exact revision.
            composer.insert_text("newer draft")
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)

        dispatch_sessions: list[str | None] = []
        original_dispatch = console._prompt_queue.dispatch

        async def record_dispatch(draft, *, session_id=None, stash=None):
            dispatch_sessions.append(session_id)
            return await original_dispatch(draft, session_id=session_id, stash=stash)

        monkeypatch.setattr(console._prompt_queue, "dispatch", record_dispatch)

        send_button = console.query_one("#console-send-message", Button)
        assert await console.handle_console_send_message(Button.Pressed(send_button))
        assert turn_ids
        await console._console_runtime().wait_for_turn(turn_ids[0])

        assert composer.draft_text() == "newer draft"
        assert store.session_draft(session.id) == "newer draft"
        assert dispatch_sessions == [session.id]


@pytest.mark.asyncio
async def test_console_pre_durable_failure_keeps_newer_typing_and_recovery():
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("keep me")

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)
        stash = composer.capture_draft_for_send()
        assert await console._dispatch_console_draft_send(stash.text, stash=stash)
        composer.insert_text("!")
        with pytest.raises(RuntimeError, match="refused before durable"):
            await console._console_runtime().wait_for_turn(turn_ids[0])

        recovery = console._console_runtime().recoveries_for_session(
            console._ensure_console_chat_store().active_session_id
        )
        assert [entry.draft for entry in recovery] == ["keep me"]
        assert composer.draft_text() == "!"
        monkeypatch.undo()


@pytest.mark.asyncio
async def test_console_unknown_command_hint_restores_draft(monkeypatch):
    app = _ready_openai_app(monkeypatch, "never sent")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.load_draft("/nosuchcommand")

        _press_enter_synchronously(console)
        await pilot.pause()
        await pilot.pause()

        # The unknown-command hint path must put the draft back so the
        # armed second-Enter flow still compares against the same text.
        assert composer.draft_text() == "/nosuchcommand"


@pytest.mark.asyncio
async def test_console_armed_unknown_mouse_send_snapshots_before_skill_await(
    monkeypatch,
):
    """Typing during the unknown-command check belongs to the next draft."""

    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        console._session._sync_console_session_draft()
        armed = "/nosuchcommand"
        composer.load_draft(armed)
        store.set_session_draft(session.id, armed)
        console._console_unknown_send_armed = armed

        entered = asyncio.Event()
        release = asyncio.Event()

        async def delayed_skill_context():
            entered.set()
            await release.wait()
            return object()

        async def no_blocked_match(_name, _summaries):
            return False

        monkeypatch.setattr(
            console._skill, "_fetch_console_skill_context", delayed_skill_context
        )
        monkeypatch.setattr(
            console._skill, "_console_skill_blocked_summaries", lambda _context: ()
        )
        monkeypatch.setattr(
            console._skill,
            "_console_skill_blocked_match_response",
            no_blocked_match,
        )
        requests: list[object] = []

        def record_accept(request):
            requests.append(request)
            return request.turn_id

        monkeypatch.setattr(console._console_runtime(), "accept_turn", record_accept)
        send_button = console.query_one("#console-send-message", Button)
        send_task = asyncio.create_task(
            console.handle_console_send_message(Button.Pressed(send_button))
        )

        await entered.wait()
        composer.insert_text(" suffix")
        release.set()

        assert await send_task
        assert len(requests) == 1
        assert requests[0].draft == armed
        assert composer.draft_text() == " suffix"
        assert store.session_draft(session.id) == " suffix"


@pytest.mark.asyncio
async def test_console_blocked_send_retains_exact_recovery_after_runtime_custody():
    app = _build_console_send_test_app()
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "test-model"
    app.console_provider_gateway_factory = BlockedGateway
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        _select_llamacpp_console(console)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        composer.insert_text_as_paste("pasted payload " * 20)
        composer.insert_text(" tail")
        # Unfurl the token fully (collapsed -> confirm -> expanded); Enter
        # only sends once no token is awaiting the unfurl flow, and expanded
        # segments retain their paste provenance.
        assert composer.activate_focused_paste_token()
        assert composer.activate_focused_paste_token()
        assert not composer.activate_focused_paste_token()
        assert composer.has_paste_segments()
        expected = composer.draft_text()
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        console._prompt_queue._launch_chain = record_launch

        stash = composer.capture_draft_for_send()
        assert stash is not None
        assert await console._dispatch_console_draft_send(stash.text, stash=stash)
        assert turn_ids
        with pytest.raises(RuntimeError, match="refused before durable"):
            await console._console_runtime().wait_for_turn(turn_ids[0])

        session_id = console._ensure_console_chat_store().active_session_id
        recovery = console._console_runtime().recoveries_for_session(session_id)
        assert [entry.draft for entry in recovery] == [expected]
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_double_enter_sends_once_and_loses_nothing(monkeypatch):
    """A second Enter before the first Pressed handler runs must not
    overwrite the pending stash with None (that ate the message)."""
    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(console, "_should_capture_console_input", lambda _composer: True)
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        submitted: list[str] = []

        async def submit_once(draft, **kwargs):
            submitted.append(draft)
            entered.set()
            await release.wait()
            kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        async def run_chain(*, session_id, initial_turn):
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", submit_once)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)
        composer.load_draft("line one")

        _press_enter_synchronously(console)
        _press_enter_synchronously(console)

        for _ in range(20):
            if entered.is_set():
                break
            await pilot.pause(0.01)
        assert entered.is_set()
        assert len(turn_ids) == 1

        release.set()
        await console._console_runtime().wait_for_turn(turn_ids[0])

        store = console._ensure_console_chat_store()
        assert submitted == ["line one"]
        assert store.active_session_id is not None
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_submit_exception_retains_exact_recovery_and_keeps_app_alive(
    monkeypatch,
):
    """A post-custody exception retains the draft without overwriting UI."""
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        controller = console._ensure_console_chat_controller()

        async def exploding_submit(draft, **kwargs):
            raise RuntimeError("provider imploded")

        monkeypatch.setattr(controller, "submit_draft", exploding_submit)
        composer.load_draft("precious draft")

        _press_enter_synchronously(console)
        for _ in range(10):
            await pilot.pause(0.05)

        assert composer.draft_text() == ""
        runtime = console._console_runtime()
        recovery, = runtime.recoveries_for_session(controller.store.active_session_id)
        assert recovery.draft == "precious draft"
        runtime.restore_turn_recovery(recovery.turn_id)
        console._prompt_queue._load_recovered_turn(recovery.session_id)
        assert composer.draft_text() == "precious draft"
        # App survived the worker exception (queries still work).
        assert console.query_one("#console-native-composer", ConsoleComposerBar)


# ---------------------------------------------------------------------------
# TASK-4 (D2 fix wave): the swallowed send -- resolve the session at
# dispatch, no silent refusals, guard the no-op press.
#
# Fresh-profile arrangement below: no active session before the send, the
# gap the historical "" sentinel exploited. Additive beside the six tests
# above, which all assume an active session already exists.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_fresh_profile_first_send_resolves_real_session_not_sentinel(
    monkeypatch,
):
    """A first send pins the real session represented by the composer."""
    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(console, "_should_capture_console_input", lambda _composer: True)
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        store = console._ensure_console_chat_store()
        owning_session_id = console._console_visible_draft_session_id
        assert owning_session_id is not None
        # Fresh-profile arrangement: no active session at the moment Enter
        # is pressed (whatever the mount-time sync worker created so far is
        # deliberately orphaned here -- the point is the STORE STATE at
        # Enter, not how a real user would reach it).
        store.active_session_id = None

        groups: list[str] = []
        real_run_worker = console.run_worker

        def spying_run_worker(work, **kwargs):
            groups.append(kwargs.get("group", ""))
            return real_run_worker(work, **kwargs)

        monkeypatch.setattr(console, "run_worker", spying_run_worker)
        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        submitted: list[tuple[str, str]] = []

        async def submit_once(draft, **kwargs):
            submitted.append((draft, kwargs["session_id"]))
            entered.set()
            await release.wait()
            kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        async def run_chain(*, session_id, initial_turn):
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", submit_once)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)

        composer.load_draft("hello fresh")
        _press_enter_synchronously(console)
        for _ in range(20):
            if entered.is_set():
                break
            await pilot.pause(0.01)
        assert entered.is_set()
        assert len(turn_ids) == 1
        release.set()
        await console._console_runtime().wait_for_turn(turn_ids[0])

        assert store.active_session_id is not None
        # Normal sends now run in app-owned runtime tasks. An unrelated
        # periodic "console-sync" worker may fire, but admission itself
        # must never create a screen-owned console-run worker.
        console_run_groups = [g for g in groups if g.startswith("console-run-")]
        assert console_run_groups == []
        assert submitted == [("hello fresh", owning_session_id)]


@pytest.mark.asyncio
async def test_console_active_run_queues_without_a_screen_worker(monkeypatch):
    """An accepted active turn routes the next draft to its runtime-owned queue."""
    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")

        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        store = controller.store
        session = store.ensure_session()
        initial_snapshot = controller.prompt_queue_registry.snapshot(session.id)
        began = controller.prompt_queue_registry.begin_chain(
            session.id,
            context_epoch=store.conversation_context_epoch(session.id),
            expected_revision=initial_snapshot.revision,
        )
        assert began.applied
        accepted_activity = replace(
            controller.activity_for(session.id),
            occupies_slot=True,
            accepted_live_turn=True,
        )
        monkeypatch.setattr(
            controller,
            "activity_for",
            lambda target_session_id: (
                accepted_activity
                if target_session_id == session.id
                else controller.prompt_queue_coordinator.activity(target_session_id)
            ),
        )
        # Simulate the race `_active_run_rejection` defends against: by the
        # time the dispatched worker's `submit_draft` actually runs, another
        # send already put this session's run state into STREAMING.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "already streaming"),
            session_id=session.id,
        )

        result = await console._dispatch_console_draft_send("hello")

        assert result is True
        snapshot = controller.prompt_queue_registry.snapshot(session.id)
        assert [entry.preview for entry in snapshot.entries] == ["hello"]


@pytest.mark.asyncio
async def test_console_synchronous_custody_refusal_notifies_and_keeps_draft(
    monkeypatch,
):
    """A refusal before runtime custody leaves the captured revision live."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep me")
        stash = composer.capture_draft_for_send()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )

        def refuse(_draft: str, _session_id: str) -> str:
            raise RuntimeError("Console session is closed.")

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", refuse)

        notices: list[tuple[str, str]] = []
        console.app_instance.notify = lambda message, **kwargs: notices.append(
            (str(message), kwargs.get("severity", ""))
        )

        sent = await console._dispatch_console_draft_send("keep me", stash=stash)

        assert sent is False
        assert notices == [("Console session is closed.", "warning")]
        assert composer.draft_text() == "keep me"


@pytest.mark.asyncio
async def test_console_deleted_owner_refuses_and_keeps_captured_inputs(monkeypatch):
    """A session deleted during admission becomes the normal custody refusal."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep deleted owner draft")
        stash = composer.capture_draft_for_send()
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        attachment = PendingAttachment(
            "/private/deleted-owner.png",
            "deleted-owner.png",
            "image",
            "attachment",
            data=b"private attachment",
        )
        assert store.add_pending_attachment(session.id, attachment)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        controller = console._ensure_console_chat_controller()

        def delete_owner(_session_id: str) -> None:
            store.close_session(session.id)
            return None

        monkeypatch.setattr(controller, "send_refusal_copy", delete_owner)
        notices: list[tuple[str, str]] = []
        console.app_instance.notify = lambda message, **kwargs: notices.append(
            (str(message), kwargs.get("severity", ""))
        )

        sent = await console._dispatch_console_draft_send(
            stash.text, stash=stash, session_id=session.id
        )

        assert sent is False
        assert notices == [("Console session is closed.", "warning")]
        assert composer.draft_text() == "keep deleted owner draft"
        assert session.pending_attachments == [attachment]


@pytest.mark.asyncio
async def test_console_valid_owner_internal_key_error_is_not_a_custody_refusal(
    monkeypatch,
):
    """A valid owner's unrelated admission defect must remain observable."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("keep valid owner draft")
        stash = composer.capture_draft_for_send()
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        attachment = PendingAttachment(
            "/private/valid-owner.png",
            "valid-owner.png",
            "image",
            "attachment",
            data=b"private attachment",
        )
        assert store.add_pending_attachment(session.id, attachment)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        monkeypatch.setattr(
            console._ensure_console_chat_controller(),
            "send_refusal_copy",
            lambda _session_id: None,
        )

        def fail_internal_context(_session_id: str) -> None:
            raise KeyError("internal settings lookup")

        monkeypatch.setattr(
            console._session,
            "_build_console_turn_execution_context",
            fail_internal_context,
        )

        with pytest.raises(KeyError, match="internal settings lookup"):
            await console._dispatch_console_draft_send(
                stash.text, stash=stash, session_id=session.id
            )

        assert composer.draft_text() == "keep valid owner draft"
        assert store.pending_attachments(session.id) == [attachment]


@pytest.mark.asyncio
async def test_console_hidden_send_keeps_draft_and_unblocks_next_send(
    monkeypatch,
):
    """A hidden Send keeps its capture live and does not latch Enter."""
    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(
            console, "_should_capture_console_input", lambda _composer: True
        )
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()
        controller = console._ensure_console_chat_controller()
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        submitted: list[str] = []

        async def submit_once(draft, **kwargs):
            submitted.append(draft)
            entered.set()
            await release.wait()
            kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(accepted=True, should_clear_draft=True)

        async def run_chain(*, session_id, initial_turn):
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", submit_once)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        turn_ids: list[str] = []
        original_launch = console._prompt_queue._launch_chain

        def record_launch(draft: str, session_id: str) -> str:
            turn_id = original_launch(draft, session_id)
            turn_ids.append(turn_id)
            return turn_id

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", record_launch)
        composer.load_draft("no-op test")

        send_button = console.query_one("#console-send-message", Button)
        send_button.styles.display = "none"

        _press_enter_synchronously(console)
        await pilot.pause()

        assert composer.draft_text() == "no-op test"
        assert console._console_pending_send is None
        assert turn_ids == []

        send_button.styles.display = "block"
        _press_enter_synchronously(console)
        for _ in range(20):
            if entered.is_set():
                break
            await pilot.pause(0.01)
        assert entered.is_set()
        assert len(turn_ids) == 1

        release.set()
        await console._console_runtime().wait_for_turn(turn_ids[0])

        assert submitted == ["no-op test"]
        assert composer.draft_text() == ""


@pytest.mark.asyncio
async def test_console_deferred_enter_refuses_after_owning_session_changes(
    monkeypatch,
):
    """A delayed A callback must never borrow B's config or attachments."""

    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(
            console, "_should_capture_console_input", lambda _composer: True
        )
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        store = console._ensure_console_chat_store()
        session_a = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("draft for A")
        store.set_session_draft(session_a.id, "draft for A")

        scheduled: list[object] = []
        monkeypatch.setattr(
            console.app,
            "call_later",
            lambda callback, *_args, **_kwargs: scheduled.append(callback),
        )
        context_sessions: list[str] = []
        original_context = console._session._build_console_turn_execution_context

        def record_context(session_id: str):
            context_sessions.append(session_id)
            return original_context(session_id)

        monkeypatch.setattr(
            console._session, "_build_console_turn_execution_context", record_context
        )
        accepted_sessions: list[str] = []
        runtime = console._console_runtime()
        original_accept = runtime.accept_turn

        def record_accept(request):
            accepted_sessions.append(request.session_id)
            return original_accept(request)

        monkeypatch.setattr(runtime, "accept_turn", record_accept)

        _press_enter_synchronously(console)
        assert len(scheduled) == 1

        session_b = store.create_session(title="Session B")
        store.set_session_draft(session_b.id, "draft for B")
        console._session._sync_console_session_draft()
        assert composer.draft_text() == "draft for B"

        assert await scheduled[0]() is False

        assert context_sessions == []
        assert accepted_sessions == []
        assert console._console_pending_send is None
        assert store.session_draft(session_a.id) == "draft for A"
        assert store.session_draft(session_b.id) == "draft for B"
        assert composer.draft_text() == "draft for B"


@pytest.mark.asyncio
async def test_console_same_session_send_waits_for_pending_enter_token(monkeypatch):
    """A tokenless A send cannot race A's already-scheduled Enter send."""

    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(
            console, "_should_capture_console_input", lambda _composer: True
        )
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("one A draft")
        store.set_session_draft(session.id, composer.draft_text())

        scheduled: list[object] = []
        monkeypatch.setattr(
            console.app,
            "call_later",
            lambda callback, *_args, **_kwargs: scheduled.append(callback),
        )
        context_sessions: list[str] = []
        original_context = console._session._build_console_turn_execution_context

        def record_context(session_id: str):
            context_sessions.append(session_id)
            return original_context(session_id)

        monkeypatch.setattr(
            console._session, "_build_console_turn_execution_context", record_context
        )
        requests: list[object] = []
        runtime = console._console_runtime()
        original_accept = runtime.accept_turn

        def record_accept(request):
            requests.append(request)
            return original_accept(request)

        monkeypatch.setattr(runtime, "accept_turn", record_accept)

        _press_enter_synchronously(console)
        assert len(scheduled) == 1
        pending = console._console_pending_send
        send_button = console.query_one("#console-send-message", Button)

        assert not await console.handle_console_send_message(
            Button.Pressed(send_button)
        )
        assert context_sessions == []
        assert requests == []
        assert console._console_pending_send is pending

        assert await scheduled[0]()
        assert context_sessions == [session.id]
        assert len(requests) == 1
        assert requests[0].draft == "one A draft"
        assert console._console_pending_send is None


@pytest.mark.asyncio
async def test_console_mouse_send_cannot_claim_another_sessions_pending_enter(
    monkeypatch,
):
    """B sends its live draft while A's scheduled capture remains A-owned."""

    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(
            console, "_should_capture_console_input", lambda _composer: True
        )
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        store = console._ensure_console_chat_store()
        session_a = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("draft for A")
        store.set_session_draft(session_a.id, "draft for A")

        scheduled: list[object] = []
        monkeypatch.setattr(
            console.app,
            "call_later",
            lambda callback, *_args, **_kwargs: scheduled.append(callback),
        )
        context_sessions: list[str] = []
        original_context = console._session._build_console_turn_execution_context

        def record_context(session_id: str):
            context_sessions.append(session_id)
            return original_context(session_id)

        monkeypatch.setattr(
            console._session, "_build_console_turn_execution_context", record_context
        )
        attachment_sessions: list[str] = []
        original_pending_attachments = store.pending_attachments

        def record_pending_attachments(session_id: str):
            attachment_sessions.append(session_id)
            return original_pending_attachments(session_id)

        monkeypatch.setattr(store, "pending_attachments", record_pending_attachments)
        requests: list[object] = []
        runtime = console._console_runtime()
        original_accept = runtime.accept_turn

        def record_accept(request):
            requests.append(request)
            return original_accept(request)

        monkeypatch.setattr(runtime, "accept_turn", record_accept)

        _press_enter_synchronously(console)
        assert len(scheduled) == 1

        session_b = store.create_session(title="Session B")
        store.set_session_draft(session_b.id, "draft for B")
        console._session._sync_console_session_draft()
        send_button = console.query_one("#console-send-message", Button)

        assert await console.handle_console_send_message(Button.Pressed(send_button))
        assert len(requests) == 1
        request = requests[0]
        assert request.session_id == session_b.id
        assert request.draft == "draft for B"
        assert context_sessions == [session_b.id]
        # Current dev's question/image send guard also reads attachments;
        # every read must still use B, never the pending Enter's owner A.
        assert attachment_sessions and set(attachment_sessions) == {session_b.id}
        assert store.session_draft(session_a.id) == "draft for A"
        assert console._console_pending_send is not None

        assert await scheduled[0]() is False
        assert len(requests) == 1
        assert console._console_pending_send is None

        # The A callback claimed only A's token and released the Enter gate.
        composer.load_draft("next draft for B")
        store.set_session_draft(session_b.id, composer.draft_text())
        _press_enter_synchronously(console)
        assert len(scheduled) == 2


@pytest.mark.asyncio
async def test_console_session_switch_during_admission_does_not_commit_into_new_view(
    monkeypatch,
):
    app = _ready_openai_app(monkeypatch, "unused")
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(
            console._prompt_queue, "_blocked_reason_accessor", lambda: ""
        )
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        store = console._ensure_console_chat_store()
        session_a = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("accepted A revision")
        store.set_session_draft(session_a.id, "accepted A revision")
        session_b = store.create_session(title="Session B", activate=False)
        store.set_session_draft(session_b.id, "untouched B draft")
        stash = composer.capture_draft_for_send()
        assert stash is not None

        def admit_then_switch(draft: str, session_id: str) -> str:
            assert draft == "accepted A revision"
            assert session_id == session_a.id
            store.switch_session(session_b.id)
            console._session._sync_console_session_draft()
            return "turn-a"

        monkeypatch.setattr(console._prompt_queue, "_launch_chain", admit_then_switch)

        assert await console._dispatch_console_draft_send(
            stash.text, stash=stash, session_id=session_a.id
        )

        assert console._console_visible_draft_session_id == session_b.id
        assert composer.draft_text() == "untouched B draft"
        assert store.session_draft(session_b.id) == "untouched B draft"


@pytest.mark.asyncio
async def test_console_enter_schedules_visible_send_without_a_watchdog(monkeypatch):
    """Enter uses the app pump directly; no screen timer owns the capture."""
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        monkeypatch.setattr(console, "_console_setup_modal_blocking", lambda: False)
        monkeypatch.setattr(
            console, "_should_capture_console_input", lambda _composer: True
        )
        monkeypatch.setattr(console, "_console_command_popup_or_none", lambda: None)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.focus()
        await pilot.pause()

        composer.load_draft("scheduled directly")
        send_button = console.query_one("#console-send-message", Button)
        send_button.styles.display = "block"
        send_button.disabled = False
        scheduled: list[object] = []
        timers: list[object] = []
        monkeypatch.setattr(
            console.app,
            "call_later",
            lambda callback, *_args, **_kwargs: scheduled.append(callback),
        )
        monkeypatch.setattr(
            console,
            "set_timer",
            lambda *_args, **_kwargs: timers.append(object()),
        )

        _press_enter_synchronously(console)

        assert len(scheduled) == 1
        assert timers == []
        assert console._console_pending_send is not None
        assert composer.draft_text() == "scheduled directly"


async def _finish_test_custody(console):
    runtime = console._console_runtime()
    record = next(iter(runtime._turn_custody.values()))
    try:
        await runtime.wait_for_turn(record.turn_id)
    except RuntimeError as exc:
        assert "refused before durable acceptance" in str(exc)
    await asyncio.sleep(0)
    return runtime, record.turn_id


@pytest.mark.asyncio
@pytest.mark.parametrize("keyboard,late_text", [(False, ""), (False, "new draft"), (True, ""), (True, "new draft")])
async def test_setup_refusal_after_custody_preserves_exact_recovery(monkeypatch, keyboard, late_text):
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("original draft")
        stash = composer.capture_draft_for_send()

        async def refuse_after_setup(*, session_id, initial_turn):
            if late_text:
                composer.insert_text(late_text)
                store.set_session_draft(session_id, late_text)
            return ConsoleSubmitResult(False, False, "Setup cancelled")

        monkeypatch.setattr(controller, "run_prompt_chain", refuse_after_setup)
        if keyboard:
            assert await console._dispatch_console_draft_send("original draft", stash=stash, session_id=session.id)
        else:
            assert await console.handle_console_send_message(Button.Pressed(console.query_one("#console-send-message", Button)))
        runtime, turn_id = await _finish_test_custody(console)
        assert composer.draft_text() == late_text
        recovery, = runtime.recoveries_for_session(session.id)
        assert recovery.turn_id == turn_id
        assert recovery.draft == "original draft"
        if late_text:
            with pytest.raises(RuntimeError, match="live draft changed"):
                runtime.restore_turn_recovery(turn_id)
            assert composer.draft_text() == late_text
            assert runtime.recoveries_for_session(session.id) == (recovery,)
        else:
            runtime.restore_turn_recovery(turn_id)
            console._prompt_queue._load_recovered_turn(session.id)
            assert composer.draft_text() == "original draft"
            assert runtime.recoveries_for_session(session.id) == ()


@pytest.mark.asyncio
async def test_refusal_recovery_restore_never_overwrites_another_visible_owner(monkeypatch):
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        console._session._sync_console_session_draft()
        composer.load_draft("A private draft")
        stash = composer.capture_draft_for_send()
        other = None

        async def refuse_after_setup(*, session_id, initial_turn):
            nonlocal other
            other = store.create_session(title="B")
            store.set_session_draft(other.id, "B private draft")
            console._session._sync_console_session_draft()
            return ConsoleSubmitResult(False, False, "Setup cancelled")

        monkeypatch.setattr(controller, "run_prompt_chain", refuse_after_setup)
        assert await console._dispatch_console_draft_send("A private draft", stash=stash, session_id=session.id)
        runtime, turn_id = await _finish_test_custody(console)
        assert composer.draft_text() == "B private draft"
        runtime.restore_turn_recovery(turn_id)
        assert store.session_draft(session.id) == "A private draft"
        assert store.session_draft(other.id) == "B private draft"
        assert composer.draft_text() == "B private draft"


@pytest.mark.asyncio
@pytest.mark.parametrize("keyboard", [False, True])
@pytest.mark.parametrize("durable", [False, True])
async def test_custody_refusal_recovery_respects_durable_acceptance_and_undo(monkeypatch, keyboard, durable):
    app = _ready_openai_app(monkeypatch, "never used")
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        session = controller.store.ensure_session()
        console._session._sync_console_session_draft()
        composer.insert_text("original")
        composer.insert_text(" draft")
        composer.insert_text(" extra")
        assert composer.undo()
        stash = composer.capture_draft_for_send()

        async def refuse(draft, **kwargs):
            if durable:
                kwargs["custody_acceptance_hook"]()
            return ConsoleSubmitResult(False, False, "Setup result")

        async def run_chain(*, session_id, initial_turn):
            return await initial_turn()

        monkeypatch.setattr(controller, "submit_draft", refuse)
        monkeypatch.setattr(controller, "run_prompt_chain", run_chain)
        if keyboard:
            assert await console._dispatch_console_draft_send("original draft", stash=stash, session_id=session.id)
        else:
            assert await console.handle_console_send_message(Button.Pressed(console.query_one("#console-send-message", Button)))
        runtime, turn_id = await _finish_test_custody(console)
        assert composer.draft_text() == ""
        assert not composer.undo()
        assert not composer.redo()
        if durable:
            assert runtime.recoveries_for_session(session.id) == ()
        else:
            runtime.restore_turn_recovery(turn_id)
            console._prompt_queue._load_recovered_turn(session.id)
            assert composer.draft_text() == "original draft"
            # Explicit recovery is a new draft; consumed edits never leak back
            # through the old send revision's undo/redo history.
            assert not composer.redo()
