"""The wake FIRES with no Console screen mounted (task-15860, plan Task 1 AC#1).

This is the slice every earlier landing of task-15860 built toward and
deliberately did not ship:

* the **ownership** landing made the app own the Console runtime;
* the **lifetime** landing made the runtime outlive `ChatScreen`, split
  `leave_console()` (end ONE visit) from `dispose()` (app exit), and made
  `_shutdown_requested` per-VISIT -- installing the fresh Event at
  `attach_view` rather than at the end of `leave_console` *specifically*
  so the flag stays set between visits and `ConsoleFleetWakeCoordinator.
  _attempt`'s gate kept refusing;
* the **viewless** landing gave every hook slot a semantically correct
  detached value (`wake_conversation_in_view` -> not in view, so the ◈
  FLEET_UNSEEN mark SURVIVES a delivery nobody could have watched);
* the **continuity** landing made the app-owned store the only home of
  Console history, so a turn that ran while Console was unmounted is
  there when the user comes back.

Everything above is merged. The one thing left is the gate: `_attempt`
refused whenever `controller._shutdown_requested` was set, which after the
lifetime landing means "a visit ended" *and* "the app is exiting". This
file drives the whole chain through the production path and asserts the
first meaning must not refuse.

Rig notes:

* the survivor settles from a plain child thread through the real fan-out
  (`on_fleet_drained`), exactly as `ConsoleAgentBridge` delivers it;
* Console is left through the REAL navigation API (`NavigateToScreen` +
  the real "Leave Console?" dialog), so `ChatScreen.on_unmount` runs and
  `leave_console_runtime` genuinely ends the visit -- nothing is
  monkeypatched and no gate is suppressed;
* every durable surface is a real on-disk DB (ChaChaNotes, `agent_runs`,
  the conversation marks store).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from Tests.Chat.test_console_fleet_wake import _drain, _quiet, _settle, _survivor
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_store_continuity import (
    CHILD_RESULT,
    _StallingWakeGateway,
    _db_chain,
    _drain_from_child_thread,
    _navigate,
    _rendered_text,
    _seed_console,
    _terminal_survivor_run,
    SEEDED_USER,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_fleet_wake import WAKE_NOTICE_HEADER
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.Chat.console_session_endpoint_policy import (
    ConsoleEndpointPolicyState,
    ConsoleEndpointRollbackOutcome,
    ConsoleEphemeralEndpointPolicy,
)
from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


WAKE_REPLY = "HEADLESS-WAKE-REPLY"
CONFIGURED_VLLM_URL = "http://127.0.0.1:9098/v1"
LIVE_VLLM_URL = "http://127.0.0.1:9188/v1"


def _build_console_app(tmp_path):
    """A real app with real DBs and a recording provider gateway."""
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    gateway = _StallingWakeGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    return app, gateway


def _recording_vllm_gateway(app):
    """Production resolver/adapter with observable selections and dispatches."""

    calls: list[dict[str, object]] = []
    selections: list[object] = []
    resolutions: list[object] = []

    gateway = ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
    )
    resolve = gateway.resolve_for_send

    async def record(selection):
        selections.append(selection)
        resolution = await resolve(selection)
        resolutions.append(resolution)
        return resolution

    async def record_stream(resolution, messages, **_kwargs):
        calls.append(
            {
                "base_url": resolution.base_url,
                "messages": str(getattr(messages, "messages_payload", messages)),
            }
        )
        yield WAKE_REPLY

    gateway.resolve_for_send = record
    gateway.stream_chat = record_stream
    return gateway, selections, resolutions, calls


def _adopt_live_vllm(store, session_id: str):
    """Install the same endpoint-safe settings/live policy pair as handoff."""

    prior_settings = store.session_settings(session_id)
    assert prior_settings is not None
    settings = replace(
        prior_settings,
        provider="vllm",
        model="wake-vllm-model",
        base_url=None,
        streaming=False,
    )
    policy = ConsoleEphemeralEndpointPolicy(
        provider=settings.provider,
        model=settings.model,
        base_url=LIVE_VLLM_URL,
    )
    prior_policy = store.session_ephemeral_endpoint_policy(session_id)
    prior_has_user_work = store.ensure_session().has_user_work
    receipt = store.adopt_session_ephemeral_endpoint(
        session_id,
        settings=settings,
        policy=policy,
    )
    return (
        prior_settings,
        prior_policy,
        prior_has_user_work,
        settings,
        policy,
        receipt,
    )


@pytest.mark.asyncio
async def test_a_survivor_settling_with_no_console_mounted_wakes_the_supervisor(
    tmp_path,
):
    """AC#1, end to end, through the production path.

    RED before the gate change: the survivor settles, the fan-out records
    it, the coordinator hops to the app loop -- and `_attempt` returns at
    the `_shutdown_requested` gate because the visit ended. Nothing
    reaches the provider, nothing is persisted, the ledger is unstamped.

    Asserted here (the whole chain, not just the send):

    1. exactly ONE wake turn reaches the provider, carrying the child's
       result as the payload's trailing machine-labelled `user` entry;
    2. the machine-origin SYSTEM notice and the assistant reply are in the
       app-owned store AND in ChaChaNotes;
    3. `agent_runs.wake_delivered_at` is stamped exactly once (a
       re-delivered drain and an extra retry change nothing);
    4. NO new USER row exists anywhere -- store or DB (a wake is never
       user input);
    5. the ◈ FLEET_UNSEEN mark SURVIVES, because a runtime with no view
       reports the conversation as NOT watched;
    6. navigating back shows the delivered turn in the rendered
       transcript.
    """
    app, gateway = _build_console_app(tmp_path)

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, gateway
        )
        wake = controller.fleet_wake
        runs_db = controller._agent_bridge.runs_db
        run_id = _terminal_survivor_run(runs_db, conversation_id)
        marks = app.conversation_local_marks_service
        marks.set_mark(conversation_id, ConversationLocalMarksService.FLEET_UNSEEN)
        rows_before = _db_chain(app.chachanotes_db, conversation_id)
        assert len(rows_before) == 2, (
            "harness precondition: the seeded turn must have persisted two rows; "
            f"got {[(r[1], r[2][:24]) for r in rows_before]}"
        )

        # -- leave Console through the real navigation ---------------------
        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack, "Console must actually unmount"
        assert controller is app.console_runtime.chat_controller, (
            "harness precondition: the runtime must OUTLIVE the screen"
        )
        assert controller._shutdown_requested.is_set(), (
            "harness precondition: leaving Console must have ended the visit -- "
            "that set Event is exactly the state the old gate refused, so a "
            "test that never reached it would prove nothing"
        )
        assert controller._disposed is False, (
            "harness precondition: a navigation is not an app exit"
        )

        # -- NOW the survivor settles, with nothing mounted ----------------
        gateway.reply = WAKE_REPLY
        before = len(gateway.payloads)
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )

        # (1) exactly one wake turn reached the provider.
        assert await _settle(
            lambda: len(gateway.payloads) > before, seconds=10.0
        ), (
            "a background sub-agent settled while NO Console screen was mounted "
            "and no wake turn ever reached the provider"
        )
        assert await _quiet(lambda: len(gateway.payloads) > before + 1), (
            "one settle delivered more than one wake turn"
        )
        payload = gateway.payloads[-1]
        trailing = payload[-1]
        assert trailing["role"] == ConsoleMessageRole.USER.value, (
            "the notice must be the payload's trailing user-role entry (a "
            f"payload ending on an assistant row is a prefill); got {trailing['role']!r}"
        )
        assert WAKE_NOTICE_HEADER in str(trailing["content"]), (
            "the machine marking is missing from the wake payload"
        )
        assert CHILD_RESULT in str(trailing["content"]), (
            "the child's result never reached the supervisor"
        )

        # (2) both rows landed in the app-owned store AND in ChaChaNotes.
        assert await _settle(
            lambda: any(
                m.content.startswith(WAKE_NOTICE_HEADER)
                for m in store.messages_for_session(session_id)
            )
            and any(
                m.content == WAKE_REPLY
                for m in store.messages_for_session(session_id)
            ),
            seconds=10.0,
        ), (
            "the headless wake turn never landed in the app-owned store: "
            f"{[(m.role.value, m.content[:28]) for m in store.messages_for_session(session_id)]}"
        )
        notice_rows = [
            m
            for m in store.messages_for_session(session_id)
            if getattr(m.metadata, "origin", "") == "agent_wake"
        ]
        assert len(notice_rows) == 1, (
            f"expected exactly one machine-origin notice row, got {len(notice_rows)}"
        )
        assert notice_rows[0].role is ConsoleMessageRole.SYSTEM, (
            f"the wake notice must be a SYSTEM row, got {notice_rows[0].role}"
        )

        assert await _settle(
            lambda: len(_db_chain(app.chachanotes_db, conversation_id)) == 4,
            seconds=10.0,
        ), (
            "the headless wake turn never PERSISTED: "
            f"{[(r[1], r[2][:28]) for r in _db_chain(app.chachanotes_db, conversation_id)]}"
        )
        db_rows = _db_chain(app.chachanotes_db, conversation_id)
        senders = [row[1] for row in db_rows]
        contents = [row[2] for row in db_rows]
        assert senders == ["user", "assistant", "system", "assistant"], (
            f"unexpected persisted row shape: {list(zip(senders, [c[:24] for c in contents]))}"
        )
        assert contents[2].startswith(WAKE_NOTICE_HEADER)
        assert contents[3] == WAKE_REPLY

        # (3) the ledger is stamped exactly once, and stays that way.
        stamp = (runs_db.get_run(run_id) or {}).get("wake_delivered_at")
        assert stamp, "the headless delivery never stamped agent_runs.wake_delivered_at"
        _drain_from_child_thread(
            wake, _drain(conversation_id, _survivor(run_id, session_id=session_id))
        )
        wake.retry_soon()
        assert await _quiet(lambda: len(gateway.payloads) > before + 1, seconds=1.0), (
            "a redelivered drain woke the supervisor a second time"
        )
        assert (runs_db.get_run(run_id) or {}).get("wake_delivered_at") == stamp, (
            "the delivered ledger stamp moved on a redelivered drain"
        )
        assert len(_db_chain(app.chachanotes_db, conversation_id)) == 4, (
            "a redelivered drain persisted more rows"
        )

        # (4) no USER row anywhere -- a wake is never user input.
        user_rows = [
            m
            for m in store.messages_for_session(session_id)
            if m.role is ConsoleMessageRole.USER
        ]
        assert [m.content for m in user_rows] == [SEEDED_USER], (
            "the headless wake wrote a USER transcript row: "
            f"{[m.content[:32] for m in user_rows]}"
        )
        assert senders.count("user") == 1, (
            f"the headless wake persisted a USER row: {senders}"
        )

        # (5) the ◈ mark SURVIVES: nobody could have watched this delivery.
        assert marks.has_mark(
            conversation_id, ConversationLocalMarksService.FLEET_UNSEEN
        ), (
            "a wake delivered with no Console mounted cleared the ◈ mark -- the "
            "user has no way to learn the supervisor turn ever ran"
        )

        # (6) returning to Console shows the delivered turn.
        chat2 = await _navigate(app, pilot, "chat", expect="ChatScreen")
        assert isinstance(chat2, ChatScreen), type(chat2).__name__
        assert chat2 is not chat, "screens are never cached"
        await pilot.pause()
        assert await _settle(
            lambda: WAKE_REPLY in _rendered_text(chat2), seconds=10.0
        ), (
            "the headless wake turn is not in the transcript the returning user "
            "sees"
        )
        rendered = _rendered_text(chat2)
        assert WAKE_NOTICE_HEADER in rendered, (
            "the wake notice never rendered for the returning user"
        )


@pytest.mark.asyncio
async def test_headless_wake_uses_active_live_only_vllm_endpoint(tmp_path):
    """An unmounted wake resolves the process-local endpoint, not config."""

    app, seed_gateway = _build_console_app(tmp_path)
    app.app_config.setdefault("api_settings", {}).setdefault("vllm", {})[
        "api_url"
    ] = CONFIGURED_VLLM_URL

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, seed_gateway
        )
        _adopt_live_vllm(store, session_id)
        gateway, selections, resolutions, calls = _recording_vllm_gateway(app)
        controller.provider_gateway = gateway
        run_id = _terminal_survivor_run(
            controller._agent_bridge.runs_db,
            conversation_id,
        )

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack
        assert controller._turn_context_provider is None

        _drain_from_child_thread(
            controller.fleet_wake,
            _drain(conversation_id, _survivor(run_id, session_id=session_id)),
        )

        assert await _settle(lambda: bool(selections), seconds=10.0)
        selection = selections[-1]
        assert selection.base_url == LIVE_VLLM_URL
        assert selection.base_url != CONFIGURED_VLLM_URL
        assert selection.configured_endpoint_fallback_allowed is False
        assert await _settle(lambda: bool(calls), seconds=10.0), repr(resolutions)
        assert calls[-1].get("base_url") == LIVE_VLLM_URL
        assert CHILD_RESULT in str(calls[-1].get("messages"))
        await gateway.aclose()


@pytest.mark.asyncio
async def test_headless_wake_blocks_after_real_vllm_rollback_conflict(tmp_path):
    """A real metadata winner prevents an unmounted wake from misrouting."""

    app, seed_gateway = _build_console_app(tmp_path)
    app.app_config.setdefault("api_settings", {}).setdefault("vllm", {})[
        "api_url"
    ] = CONFIGURED_VLLM_URL

    async with app.run_test(size=(160, 48)) as pilot:
        chat, controller, store, session_id, conversation_id = await _seed_console(
            app, pilot, seed_gateway
        )
        (
            prior_settings,
            prior_policy,
            prior_has_user_work,
            settings,
            policy,
            receipt,
        ) = _adopt_live_vllm(store, session_id)
        assert receipt is not None
        current = app.chachanotes_db.get_conversation_by_id(conversation_id)
        assert current is not None
        winner_metadata = '{"headless_concurrent_owner":"winner"}'
        assert app.chachanotes_db.update_conversation(
            conversation_id,
            {"metadata": winner_metadata},
            expected_version=current["version"],
        )
        assert (
            store.rollback_session_ephemeral_endpoint_adoption(
                session_id,
                expected_settings=settings,
                expected_policy=policy,
                prior_settings=prior_settings,
                prior_policy=prior_policy,
                prior_has_user_work=prior_has_user_work,
                receipt=receipt,
            )
            is ConsoleEndpointRollbackOutcome.BLOCKED_DURABLE_RESTORE
        )
        blocked_policy = store.session_ephemeral_endpoint_policy(session_id)
        assert blocked_policy is not None
        assert blocked_policy.state is ConsoleEndpointPolicyState.BLOCKED

        gateway, selections, _resolutions, calls = _recording_vllm_gateway(app)
        controller.provider_gateway = gateway
        run_id = _terminal_survivor_run(
            controller._agent_bridge.runs_db,
            conversation_id,
        )
        rows_before = _db_chain(app.chachanotes_db, conversation_id)

        await _navigate(app, pilot, "library", expect="LibraryScreen")
        assert chat not in app.screen_stack
        assert controller._turn_context_provider is None

        _drain_from_child_thread(
            controller.fleet_wake,
            _drain(conversation_id, _survivor(run_id, session_id=session_id)),
        )

        assert await _settle(lambda: bool(selections), seconds=10.0)
        selection = selections[-1]
        assert selection.base_url is None
        assert selection.configured_endpoint_fallback_allowed is False
        assert calls == []
        assert await _quiet(lambda: bool(calls), seconds=1.0)
        assert _db_chain(app.chachanotes_db, conversation_id) == rows_before
        durable = app.chachanotes_db.get_conversation_by_id(conversation_id)
        assert durable is not None
        assert durable["metadata"] == winner_metadata
        assert all(
            message.content != CHILD_RESULT
            for message in store.all_messages_for_session(session_id)
        )
        await gateway.aclose()
