"""Committed reports enter the existing wake scheduler without reading bodies."""

import asyncio
import threading
from uuid import uuid4

import pytest

from Tests.Chat.test_automatic_wake_budget import close_rig, queue_result, result_for
from Tests.Chat.test_console_fleet_wake import _controller_rig, _settle
from Tests.private_profile import private_profile_test
from tldw_chatbook.Agents.fleet_messages import MessageIdentity
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge


def progress_rig(tmp_path):
    rig = _controller_rig(tmp_path)
    bridge = ConsoleAgentBridge(
        agent_runs_db=rig[2], store=rig[3], provider_gateway=rig[5]
    )
    rig[7].update_agent_runtime(bridge=bridge, enabled=False)
    return (*rig[:6], bridge, rig[7])


def reporter(rig, *, chain_id=None, conversation_id=None):
    db, session, bridge = rig[2], rig[4], rig[6]
    conversation_id = conversation_id or session.id
    chain_id = chain_id or db.automatic_work.create_chain(
        conversation_id, root_submission_id=f"progress-{conversation_id}"
    )
    parent = db.create_run(
        conversation_id=conversation_id, agent_kind="primary", work_chain_id=chain_id
    )
    db.set_status(parent, "done", "parent ended")
    child = db.create_run(
        conversation_id=conversation_id, parent_run_id=parent, agent_kind="subagent"
    )
    owner = rig[3].progress_owner_id(session.id)
    store = bridge.message_store
    rig[3].prepare_progress_inbox(session.id, message_store=store)
    inbox = store.get_inbox(owner) or store.open_inbox(owner)
    identity = MessageIdentity(f"handle-{child}", child, parent, chain_id, "worker")
    return inbox.sender(identity), identity, chain_id


@pytest.mark.asyncio
@private_profile_test
async def test_committed_progress_wakes_once_with_ids_and_fresh_read_notice(
    tmp_path, request
):
    rig = progress_rig(tmp_path)
    try:
        sender, identity, chain_id = reporter(rig)
        message = sender.send("BODY_MUST_NEVER_ENTER_WAKE_NOTICE")
        assert await _settle(lambda: len(rig[5].payloads) == 1)
        assert await _settle(lambda: not rig[7].fleet_wake.delivering_session_ids())
        notice = rig[5].payloads[0][-1]["content"]
        assert message in notice
        assert "read_agent_messages" in notice
        assert "BODY_MUST_NEVER_ENTER_WAKE_NOTICE" not in notice
        assert rig[2].get_run(identity.run_id)["wake_delivered_at"] is None
        assert rig[6].progress_pending_metadata(rig[4].id)[0][0] == message
        assert rig[2].automatic_work.snapshot(chain_id).used["generation"] == 1
        rig[7].fleet_wake.on_progress_enqueued(
            rig[3].progress_owner_id(rig[4].id), message, identity
        )
        await asyncio.sleep(0.35)
        assert len(rig[5].payloads) == 1
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_progress_and_completion_coalesce_but_collection_before_claim_cancels_hint(
    tmp_path, request
):
    rig = progress_rig(tmp_path)
    try:
        sender, _, chain_id = reporter(rig)
        first = sender.send("collected before admission")
        rig[6].discard_progress(rig[3].progress_owner_id(rig[4].id), [first])
        await asyncio.sleep(0.35)
        assert rig[5].payloads == []
        second = sender.send("pending private body")
        completed = result_for(rig, chain_id)
        queue_result(rig, completed)
        assert await _settle(lambda: rig[2].get_run(completed)["wake_delivered_at"])
        assert len(rig[5].payloads) == 1
        notice = rig[5].payloads[0][-1]["content"]
        assert second in notice and completed in notice
        with rig[2].connection() as conn:
            row = conn.execute(
                "SELECT cause FROM automatic_wake_attempts WHERE state='completed'"
            ).fetchone()
            assert row[0] == "mixed"
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_old_native_owner_hint_does_not_wake_replacement(tmp_path, request):
    rig = progress_rig(tmp_path)
    try:
        sender, identity, _ = reporter(rig)
        old_owner = rig[3].progress_owner_id(rig[4].id)
        message = sender.send("late report")
        rig[6].close_progress(rig[4].id)
        rig[7].fleet_wake.on_progress_enqueued(old_owner, message, identity)
        await asyncio.sleep(0.35)
        assert rig[5].payloads == []
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_temporary_report_save_keeps_live_claim_and_never_wakes_again(
    tmp_path, request
):
    rig = list(progress_rig(tmp_path))
    rig[4] = rig[3].create_session(ephemeral=True)
    rig = tuple(rig)
    try:
        sender, identity, chain_id = reporter(rig)
        owner = rig[3].progress_owner_id(rig[4].id)
        message_id = sender.send("temporary report survives explicit Save")
        assert await _settle(lambda: len(rig[5].payloads) == 1)
        assert await _settle(lambda: not rig[7].fleet_wake.delivering_session_ids())
        with rig[2].connection() as conn:
            assert (
                conn.execute(
                    "SELECT message_ids_json FROM automatic_wake_attempts WHERE cause='progress'"
                ).fetchone()[0]
                == "[]"
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_progress_wake_claims"
                ).fetchone()[0]
                == 0
            )
        saved = rig[3].promote_ephemeral_session(rig[4].id)
        assert saved and rig[3].progress_owner_id(rig[4].id) == owner
        rig[7].fleet_wake.seed_progress_hints()
        await asyncio.sleep(0.35)
        assert len(rig[5].payloads) == 1
        assert rig[2].automatic_work.snapshot(chain_id).used["generation"] == 1
        assert rig[6].progress_pending_metadata(rig[4].id) == ((message_id, identity),)
        assert (
            rig[0]
            .get_connection()
            .execute(
                "SELECT message_id FROM fleet_progress_messages WHERE conversation_id=?",
                (saved,),
            )
            .fetchone()[0]
            == message_id
        )
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", [False, True])
@private_profile_test
async def test_save_keeps_temporary_sources_and_admits_new_saved_chain_progress(
    tmp_path, request, agent
):
    rig = list(progress_rig(tmp_path))
    rig[4] = rig[3].create_session(ephemeral=True)
    if agent:
        from Tests.Chat.test_automatic_wake_dispatch import real_gateway

        gateway, calls = real_gateway(tuple(rig), agent=True)
        rig[5], rig[6] = gateway, rig[7]._agent_bridge

        def notices():
            return [call["messages_payload"][-1]["content"] for call, _ in calls]
    else:

        def notices():
            return [payload[-1]["content"] for payload in rig[5].payloads]

    rig = tuple(rig)
    try:
        temporary_sender, temporary_identity, temporary_chain = reporter(rig)
        temporary_id = temporary_sender.send("temporary work before Save")
        assert await _settle(lambda: len(notices()) == 1, seconds=20)
        assert await _settle(
            lambda: not rig[7].fleet_wake.delivering_session_ids(), seconds=20
        )
        saved = rig[3].promote_ephemeral_session(rig[4].id)
        assert saved and saved != rig[4].id
        canvas_scopes = []
        if agent:
            from tldw_chatbook.Canvas.models import CanvasScope
            from tldw_chatbook.Canvas.native_authority import (
                NativeConsoleCanvasAuthority,
            )
            from tldw_chatbook.Chat.console_canvas_controller import (
                ConsoleCanvasController,
            )

            canvas = ConsoleCanvasController()
            rig[3].canvas_turn_controller = canvas

            def live_canvas_scope(session_id):
                scope = CanvasScope(
                    session_id=session_id,
                    conversation_id=rig[4].persisted_conversation_id or session_id,
                    active_message_ids=rig[3].canvas_active_path_message_ids(
                        session_id
                    ),
                    selected_canvas_id=None,
                    selected_revision_id=None,
                    run_id=uuid4().hex,
                )
                canvas_scopes.append(scope)
                return scope

            canvas_authority = NativeConsoleCanvasAuthority(
                scope_resolver=live_canvas_scope,
                run_scope_resolver=live_canvas_scope,
                canvas_controller=canvas,
            )
            rig[7]._canvas_enabled_reader = lambda: True
            rig[7]._canvas_disabled_reader = lambda: False
        saved_sender, saved_identity, saved_chain = reporter(rig, conversation_id=saved)
        saved_id = saved_sender.send("new saved manual chain progress")
        assert await _settle(lambda: len(notices()) == 2, seconds=20)
        assert await _settle(
            lambda: not rig[7].fleet_wake.delivering_session_ids(), seconds=20
        )
        assert saved_id in notices()[1]
        assert temporary_id not in notices()[1]

        # The earlier child can still report on its immutable temporary chain;
        # its terminal survivor shares that old causal bucket after Save.
        blocked = [True]
        rig[7].wake_user_priority_probe = lambda _session: blocked[0]
        late_id = temporary_sender.send("temporary child survives Save")
        assert await _settle(
            lambda: late_id in rig[7].fleet_wake._pending_progress.get(rig[4].id, {})
        )
        completed = result_for(rig, temporary_chain)
        queue_result(rig, completed)
        assert await _settle(
            lambda: completed in rig[7].fleet_wake._pending.get(rig[4].id, {})
        )
        blocked[0] = False
        rig[7].fleet_wake.retry_soon()
        assert await _settle(lambda: len(notices()) == 3, seconds=20)
        assert await _settle(
            lambda: not rig[7].fleet_wake.delivering_session_ids(), seconds=20
        )
        assert late_id in notices()[2]
        assert completed in notices()[2]
        assert rig[2].get_run(completed)["wake_delivered_at"] is not None
        assert rig[2].get_run(temporary_identity.run_id)["wake_delivered_at"] is None
        assert rig[2].get_run(saved_identity.run_id)["wake_delivered_at"] is None
        assert rig[2].automatic_work.snapshot(temporary_chain).used["generation"] == 2
        assert rig[2].automatic_work.snapshot(saved_chain).used["generation"] == 1
        if agent:
            assert calls[1][1] == saved_chain
            assert calls[2][1] == temporary_chain
            assert canvas_scopes and all(
                scope.conversation_id == saved for scope in canvas_scopes
            )
            assert canvas_authority is not None
        rig[7].fleet_wake.seed_progress_hints()
        await asyncio.sleep(0.35)
        assert len(notices()) == 3
        assert {mid for mid, _ in rig[6].progress_pending_metadata(rig[4].id)} == {
            temporary_id,
            saved_id,
            late_id,
        }
    finally:
        await close_rig(rig)
        if agent:
            await gateway.aclose()


@pytest.mark.asyncio
@private_profile_test
async def test_stale_completion_keeps_both_pending_progress_chains_schedulable(
    tmp_path, request
):
    rig = progress_rig(tmp_path)
    blocked = [True]
    rig[7].wake_user_priority_probe = lambda _session: blocked[0]
    try:
        chains = tuple(
            rig[2].automatic_work.create_chain(
                rig[4].id, root_submission_id=f"manual-{name}"
            )
            for name in ("a", "b")
        )
        reports = []
        for chain in chains:
            sender, _, _ = reporter(rig, chain_id=chain)
            reports.append(sender.send("pending child progress"))
        wake = rig[7].fleet_wake
        assert await _settle(
            lambda: len(wake._pending_progress.get(rig[4].id, {})) == 2
        )
        stale = result_for(rig, chains[0])
        rig[2].mark_wake_delivered((stale,))
        queue_result(rig, stale)
        blocked[0] = False
        wake.retry_soon()
        assert await _settle(lambda: len(rig[5].payloads) == 2)
        assert await _settle(lambda: not wake.delivering_session_ids())
        notices = [payload[-1]["content"] for payload in rig[5].payloads]
        assert all(
            sum(report in notice for notice in notices) == 1 for report in reports
        )
        assert all(
            rig[2].automatic_work.snapshot(chain).used["generation"] == 1
            for chain in chains
        )
        assert not wake.has_pending(rig[4].id)
        assert wake.pending_conversation_ids() == ()
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_temporary_and_saved_source_buckets_share_one_native_delivery(
    tmp_path, request
):
    rig = list(progress_rig(tmp_path))
    rig[4] = rig[3].create_session(ephemeral=True)
    rig = tuple(rig)
    gate = asyncio.Event()
    try:
        temporary_sender, _, temporary_chain = reporter(rig)
        temporary_sender.send("temporary first wake")
        assert await _settle(lambda: len(rig[5].payloads) == 1)
        assert await _settle(lambda: not rig[7].fleet_wake.delivering_session_ids())
        saved = rig[3].promote_ephemeral_session(rig[4].id)
        saved_sender, _, saved_chain = reporter(rig, conversation_id=saved)
        rig[5].stream_gate = gate
        temporary_id = temporary_sender.send("old temporary source still active")
        saved_id = saved_sender.send("fresh saved source")
        wake = rig[7].fleet_wake
        assert await _settle(lambda: len(rig[5].payloads) == 2)
        await asyncio.sleep(0.35)
        assert len(rig[5].payloads) == 2
        assert wake.delivering_session_ids() == (rig[4].id,)
        gate.set()
        assert await _settle(lambda: len(rig[5].payloads) == 3)
        assert await _settle(lambda: not wake.delivering_session_ids())
        notices = [payload[-1]["content"] for payload in rig[5].payloads[1:]]
        assert all(
            sum(report in notice for notice in notices) == 1
            for report in (temporary_id, saved_id)
        )
        assert rig[2].automatic_work.snapshot(temporary_chain).used["generation"] == 2
        assert rig[2].automatic_work.snapshot(saved_chain).used["generation"] == 1
    finally:
        gate.set()
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign_token", [False, True])
@pytest.mark.parametrize("agent", [False, True])
@private_profile_test
async def test_manual_preparation_cannot_dispatch_a_foreign_automatic_chain(
    tmp_path, request, monkeypatch, foreign_token, agent
):
    from Tests.Chat.test_automatic_wake_dispatch import real_gateway
    from tldw_chatbook.Agents.agent_models import WorkOrigin
    from tldw_chatbook.Chat import console_fleet_wake

    rig = list(progress_rig(tmp_path))
    rig[4] = rig[7].new_session()
    rig = tuple(rig)
    gateway, calls = real_gateway(rig, agent=agent)
    attempted = []
    refusals = []
    try:
        chain = rig[2].automatic_work.create_chain(
            "foreign-conversation", root_submission_id="foreign-manual"
        )
        token = None
        if foreign_token:
            foreign = console_fleet_wake.ConsoleFleetWakeCoordinator(rig[7])
            token = console_fleet_wake.AgentWakeAuthorization(
                foreign,
                rig[4].id,
                _key=console_fleet_wake._WAKE_AUTHORIZATION_KEY,
                conversation_id="foreign-conversation",
                work_chain_id=chain,
            )
            token.accepted = True
        original = rig[7]._stream_assistant_response

        async def unowned_automatic_dispatch(**kwargs):
            attempted.append(True)
            kwargs.update(
                work_origin=WorkOrigin.AUTOMATIC,
                work_chain_id=chain,
                wake_authorization=token,
            )
            try:
                return await original(**kwargs)
            except PermissionError as exc:
                refusals.append(str(exc))
                raise

        monkeypatch.setattr(
            rig[7], "_stream_assistant_response", unowned_automatic_dispatch
        )
        result = await rig[7].submit_draft("Manual preparation", session_id=rig[4].id)
        assert attempted == [True], result
        assert refusals == ["Automatic source requires live wake authority."]
        assert result.provider_started is False
        assert calls == []
        assert rig[2].automatic_work.snapshot(chain).used["generation"] == 0
        assert rig[4].persisted_conversation_id != "foreign-conversation"
    finally:
        await close_rig(rig)
        await gateway.aclose()


@pytest.mark.asyncio
@private_profile_test
async def test_completed_temporary_wake_save_then_restart_requires_review(
    tmp_path, request
):
    rig = list(progress_rig(tmp_path))
    rig[4] = rig[3].create_session(ephemeral=True)
    rig = tuple(rig)
    closed = False
    try:
        sender, identity, chain_id = reporter(rig)
        message_id = sender.send("explicitly saved private temporary report")
        assert await _settle(lambda: len(rig[5].payloads) == 1)
        assert await _settle(lambda: not rig[7].fleet_wake.delivering_session_ids())
        with rig[2].connection() as conn:
            row = conn.execute(
                "SELECT state, message_ids_json FROM automatic_wake_attempts WHERE cause='progress'"
            ).fetchone()
            assert tuple(row) == ("completed", "[]")
        saved = rig[3].promote_ephemeral_session(rig[4].id)
        assert saved
        await close_rig(rig)
        closed = True

        # Reopen both databases and restore the saved inbox under a fresh native
        # owner. No process-local claim dictionary survives this transition.
        restarted = list(progress_rig(tmp_path))
        restarted[4] = restarted[3].restore_persisted_session(
            title="Saved temporary work",
            workspace_id=None,
            persisted_conversation_id=saved,
            all_nodes=(),
        )
        rig = tuple(restarted)
        closed = False
        rig[7].fleet_wake.start_recovery()
        _ = rig[6].message_store  # Bind and load the restored saved inbox.
        assert await rig[7].fleet_wake.wait_for_recovery()
        await asyncio.sleep(0.35)
        assert rig[6].progress_pending_metadata(rig[4].id) == ((message_id, identity),)
        assert rig[2].automatic_work.snapshot(chain_id).status == "review_required"
        assert rig[5].payloads == []
        owner = rig[3].progress_owner_id(rig[4].id)
        assert (
            rig[6].progress_snapshot(owner)[0].body
            == "explicitly saved private temporary report"
        )
        assert rig[2].get_run(identity.run_id)["wake_delivered_at"] is None
    finally:
        if not closed:
            await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_collection_after_claim_refunds_before_provider_acceptance(
    tmp_path, request, monkeypatch
):
    rig = progress_rig(tmp_path)
    try:
        sender, _, chain_id = reporter(rig)
        owner = rig[3].progress_owner_id(rig[4].id)
        inbox = rig[6].message_store.get_inbox(owner)
        original = rig[2].automatic_work.claim_wake
        claimed = threading.Event()

        def collect_after_claim(*args, **kwargs):
            attempt = original(*args, **kwargs)
            inbox.discard(attempt.message_ids)
            claimed.set()
            return attempt

        monkeypatch.setattr(rig[2].automatic_work, "claim_wake", collect_after_claim)
        sender.send("collected after reservation")
        assert await _settle(claimed.is_set)
        assert await _settle(
            lambda: (
                rig[2].automatic_work.snapshot(chain_id).available["generation"] == 3
                and not rig[7].fleet_wake.has_pending(rig[4].id)
            )
        )
        with rig[2].connection() as conn:
            assert (
                conn.execute("SELECT state FROM automatic_wake_attempts").fetchone()[0]
                == "aborted"
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_progress_wake_claims"
                ).fetchone()[0]
                == 0
            )
        assert rig[5].payloads == []
        assert rig[2].automatic_work.snapshot(chain_id).status == "active"
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["fence", "dispose"])
@pytest.mark.parametrize("cause", ["completion", "progress"])
@private_profile_test
async def test_close_after_claim_commit_settles_worker_and_refunds_prepared_attempt(
    tmp_path, request, monkeypatch, close, cause
):
    rig = progress_rig(tmp_path)
    release = threading.Event()
    worker_finished = threading.Event()
    try:
        sender, identity, chain_id = reporter(rig)
        wake = rig[7].fleet_wake
        original = rig[2].automatic_work.claim_wake
        committed = threading.Event()

        def hold_committed_claim(*args, **kwargs):
            attempt = original(*args, **kwargs)
            committed.set()
            try:
                assert release.wait(10), "test did not release the committed claim"
                return attempt
            finally:
                worker_finished.set()

        monkeypatch.setattr(rig[2].automatic_work, "claim_wake", hold_committed_claim)
        if cause == "progress":
            checked_run = identity.run_id
            sender.send("private report during close")
        else:
            checked_run = result_for(rig, chain_id)
            queue_result(rig, checked_run)
        assert await _settle(committed.is_set)
        delivery = next(iter(wake._delivery_tasks))
        if close == "fence":
            wake.fence_conversation(rig[4].id, generation=1)
        else:
            wake.dispose()
        # Close cancellation runs while the committed claim's worker is held.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not delivery.done()
        assert rig[4].id in wake.delivering_session_ids()
        release.set()
        assert await _settle(worker_finished.is_set)
        await asyncio.wait_for(asyncio.shield(delivery), timeout=10)
        with rig[2].connection() as conn:
            assert (
                conn.execute("SELECT state FROM automatic_wake_attempts").fetchone()[0]
                == "aborted"
            )
            assert (
                conn.execute(
                    "SELECT count(*) FROM automatic_progress_wake_claims"
                ).fetchone()[0]
                == 0
            )
        assert rig[2].automatic_work.snapshot(chain_id).available["generation"] == 3
        assert rig[2].automatic_work.snapshot(chain_id).status == "active"
        assert rig[2].get_run(checked_run)["wake_delivered_at"] is None
        assert rig[5].payloads == []
    finally:
        release.set()
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
@private_profile_test
async def test_restart_progress_attempt_is_review_only_and_report_stays_readable(
    tmp_path, request, monkeypatch, accepted
):
    from tldw_chatbook.Chat import console_fleet_wake

    rig = progress_rig(tmp_path)
    monkeypatch.setattr(console_fleet_wake, "autowake_enabled", lambda: False)
    try:
        sender, identity, chain_id = reporter(rig)
        message_id = sender.send("saved pending progress")
        rig[7].fleet_wake.dispose()
        rig[2].automatic_work.claim_wake(
            chain_id,
            attempt_id="old-progress",
            owner_id="old-runtime",
            session_id=rig[4].id,
            run_ids=(),
            progress_messages=((message_id, identity.run_id),),
        )
        if accepted:
            rig[2].automatic_work.accept_wake("old-progress", owner_id="old-runtime")
        wake = console_fleet_wake.ConsoleFleetWakeCoordinator(rig[7])
        rig[7]._fleet_wake = wake
        wake.wire(app=rig[1])
        rig[7]._register_fleet_wake(rig[6])
        monkeypatch.setattr(console_fleet_wake, "autowake_enabled", lambda: True)
        await wake.recover()
        await asyncio.sleep(0.35)
        assert rig[5].payloads == []
        assert rig[2].automatic_work.snapshot(chain_id).status == "review_required"
        assert (
            rig[2]
            .automatic_work.read_attempt("old-progress", owner_id="old-runtime")
            .state
            == "review_required"
        )
        assert rig[6].progress_pending_metadata(rig[4].id) == ((message_id, identity),)
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("defer", ["user", "capacity"])
@private_profile_test
async def test_progress_waits_for_manual_priority_and_reserved_slot(
    tmp_path, request, monkeypatch, defer
):
    rig = progress_rig(tmp_path)
    controller = rig[7]
    blocked = [True]
    if defer == "user":
        controller.wake_user_priority_probe = lambda _session: blocked[0]
    else:
        monkeypatch.setattr(
            type(controller),
            "max_parallel_runs",
            property(lambda _self: 1 if blocked[0] else 3),
        )
    try:
        sender, _, _ = reporter(rig)
        sender.send("deferred private progress")
        await asyncio.sleep(0.35)
        assert rig[5].payloads == []
        assert controller.fleet_wake.has_pending(rig[4].id)
        blocked[0] = False
        controller.fleet_wake.retry_soon()
        assert await _settle(lambda: len(rig[5].payloads) == 1)
        assert await _settle(lambda: not controller.fleet_wake.delivering_session_ids())
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@private_profile_test
async def test_fourth_progress_report_stays_pending_when_shared_generations_exhaust(
    tmp_path, request
):
    rig = progress_rig(tmp_path)
    try:
        sender, identity, chain_id = reporter(rig)
        for index in range(3):
            sender.send(f"report {index}")
            assert await _settle(lambda index=index: len(rig[5].payloads) == index + 1)
            assert await _settle(lambda: not rig[7].fleet_wake.delivering_session_ids())
        fourth = sender.send("fourth stays available for explicit read")
        assert await _settle(
            lambda: (
                rig[2].automatic_work.snapshot(chain_id).pause_reason
                == "generation_budget"
            )
        )
        assert len(rig[5].payloads) == 3
        assert (fourth, identity) in rig[6].progress_pending_metadata(rig[4].id)
        assert rig[2].get_run(identity.run_id)["wake_delivered_at"] is None
    finally:
        await close_rig(rig)
