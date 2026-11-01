"""Confirmation custody, native-call settlement and cancellation barriers."""

import asyncio
import threading

import pytest

from Tests.Chat.test_archive_recovery_operations import (
    recovery_stores as recovery_stores,  # noqa: PLC0414 - pytest fixture re-export
)
from tldw_chatbook.Chat.archive_recovery.contracts import (
    ArchiveSearchRequest,
    RecoveryCallKey,
    RecoveryConfirmation,
)
from tldw_chatbook.Chat.archive_recovery.coordinator import ArchiveRecoveryCoordinator
from tldw_chatbook.Chat.archive_recovery.result_store import ArchiveResultStore
from tldw_chatbook.Chat.archive_recovery.search_service import ArchiveSearchService


def setup(app, *, clock=None, ui=True):
    store = ArchiveResultStore()
    coordinator = ArchiveRecoveryCoordinator(
        app, store, **({"monotonic": clock} if clock else {})
    )
    if ui:
        coordinator.set_presenter(lambda _confirmation: True, lambda _session: True)
    coordinator.open_run("source", "run")
    page = ArchiveSearchService(
        app.local_chat_conversation_service, app.workspace_registry_service
    ).search(ArchiveSearchRequest())
    projection = store.publish(
        "source", ArchiveSearchRequest(), page, model_visible=True
    )
    return coordinator, store, projection["items"][0]["result_ref"]


def key(call="call", digest="args", run="run"):
    return RecoveryCallKey("source", run, call, digest)


@pytest.mark.asyncio
async def test_no_ui_deny_expiry_and_answerability_at_decision(recovery_stores):
    app, _, _ = recovery_stores
    now = [0.0]
    c, _, ref = setup(app, clock=lambda: now[0], ui=False)
    assert (
        await c.request("source", ref, call_key=key(), scope=None)
    ).status == "unavailable"
    c.set_presenter(lambda _: True, lambda _: True)
    card = await c.request("source", ref, call_key=key("deny"), scope=None)
    assert isinstance(card, RecoveryConfirmation)
    assert (
        await c.decide("source", card.request_id, allow=False, scope=None)
    ).status == "declined"
    assert (
        await c.request("source", ref, call_key=key("deny"), scope=None)
    ).status == "declined"
    card = await c.request("source", ref, call_key=key("expire"), scope=None)
    now[0] = 301
    c.expire_pending()
    assert (await c.wait_result(card.operation_id)).status == "expired"
    assert c.snapshot("source") is None
    card = await c.request(
        "source", ref, call_key=key("lost-ui"), scope="chat_and_workspace"
    )
    c.set_presenter(lambda _: True, lambda _: False)
    assert (
        await c.decide("source", card.request_id, allow=True, scope=None)
    ).status == "unavailable"
    assert app.workspace_registry_service.get_workspace("project").archived
    await c.shutdown()


@pytest.mark.asyncio
async def test_duplicate_native_manual_and_changed_args(recovery_stores):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    assert (
        await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
        == card
    )
    assert await c.request("source", ref, call_key=None, scope=None) == card
    assert (
        await c.request(
            "source", ref, call_key=key(digest="changed"), scope="chat_only"
        )
    ).status == "stale"
    assert (
        await c.request("source", ref, call_key=key("new"), scope="chat_and_workspace")
    ).status == "busy"
    assert (
        await c.request("source", ref, call_key=None, scope="chat_only")
    ).status == "busy"
    a, b = await asyncio.gather(
        c.decide("source", card.request_id, allow=True, scope=None),
        c.decide("source", card.request_id, allow=True, scope=None),
    )
    assert a == b and a.status == "restored"
    assert app.workspace_registry_service.get_workspace("project").archive_revision == 2
    assert (
        await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    ) == a
    await c.shutdown()


@pytest.mark.asyncio
async def test_run_reuse_eviction_consumed_request_and_late_callbacks(recovery_stores):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    first = await c.request("source", ref, call_key=key(), scope=None)
    await c.decide("source", first.request_id, allow=False, scope=None)
    for n in range(21):
        card = await c.request("source", ref, call_key=key(str(n)), scope=None)
        await c.decide("source", card.request_id, allow=False, scope=None)
    assert (
        await c.request("source", ref, call_key=key(), scope=None)
    ).status == "expired"
    assert (
        await c.decide("source", first.request_id, allow=True, scope="chat_only")
    ).status == "expired"
    c.open_run("source", "other")
    card = await c.request("source", ref, call_key=key(run="other"), scope=None)
    assert isinstance(card, RecoveryConfirmation)
    await c.close_run("source", "other")
    assert (
        await c.request("source", ref, call_key=key(run="other"), scope=None)
    ).status == "expired"
    assert not c.open_run("source", "other")
    await c.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", ["waiter", "session", "shutdown"])
async def test_actual_write_cancellation_is_joined_before_release(
    recovery_stores, monkeypatch, closing
):
    app, cid, db = recovery_stores
    c, store, ref = setup(app)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    original = app.workspace_registry_service.unarchive_workspace

    def gated(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), "test did not release actual workspace write"
        return original(*args, **kwargs)

    monkeypatch.setattr(app.workspace_registry_service, "unarchive_workspace", gated)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    waiter = asyncio.create_task(
        c.decide("source", card.request_id, allow=True, scope=None)
    )
    await asyncio.wait_for(entered.wait(), 5)
    assert c.snapshot("source").phase == "workspace_write"
    assert c.snapshot("source").confirmation is None
    assert not c.present_pending("source")
    if closing == "waiter":
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        closing_task = None
    else:
        started = asyncio.Event()
        original_close = store.close_session

        def close_results(session):
            assert not app.workspace_registry_service.get_workspace("project").archived
            original_close(session)

        monkeypatch.setattr(store, "close_session", close_results)

        async def close():
            started.set()
            if closing == "session":
                await c.close_session("source")
            else:
                await c.shutdown()
            # Runtime may release handles only once coordinator close returns.
            assert not app.workspace_registry_service.get_workspace("project").archived

        closing_task = asyncio.create_task(close())
        await started.wait()
        assert not closing_task.done()
    release.set()
    if closing_task:
        await closing_task
        result = await waiter
    else:
        result = await c.wait_result(card.operation_id)
    assert result.status == "partial"
    assert result.workspace_changed and not result.chat_changed
    assert db.get_conversation_by_id(cid)["archived"]
    assert not app.workspace_registry_service.get_workspace("project").archived
    await c.shutdown()


@pytest.mark.asyncio
async def test_name_revision_new_confirmation_and_cas_name(recovery_stores):
    app, cid, db = recovery_stores
    c, _, ref = setup(app)
    app.workspace_registry_service.create_workspace(
        workspace_id="other", name="Project"
    )
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    result = await c.decide("source", card.request_id, allow=True, scope=None)
    assert result.status == "name_conflict"
    receipt = c.receipts("source")[-1]
    assert receipt.phase == "settled"
    assert receipt.confirmation == card
    revised = await c.revise_workspace_name("source", card.request_id, "Recovered")
    assert (
        revised.request_id != card.request_id
        and revised.replacement_name == "Recovered"
    )
    assert (
        await c.decide("source", card.request_id, allow=True, scope=None)
    ).status == "name_conflict"
    assert db.get_conversation_by_id(cid)["archived"]
    result = await c.decide("source", revised.request_id, allow=True, scope=None)
    assert result.status == "restored"
    assert app.workspace_registry_service.get_workspace("project").name == "Recovered"
    await c.shutdown()


@pytest.mark.asyncio
async def test_page_eviction_pinned_target_and_background_remount(recovery_stores):
    app, _, _ = recovery_stores
    c, store, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope="chat_only")
    store.close_session("source")
    c.set_presenter(lambda _: True, lambda _: True)
    assert c.snapshot("source").confirmation == card
    assert c.present_pending("source")
    revised = await c.revise_workspace_name("source", card.request_id, "Reviewed")
    assert (
        await c.decide("source", card.request_id, allow=True, scope="chat_only")
    ).status == "expired"
    assert (
        await c.decide("source", revised.request_id, allow=False, scope=None)
    ).status == "declined"
    await c.shutdown()


@pytest.mark.asyncio
async def test_revised_card_is_current_for_native_duplicate_and_stale_revision_refused(
    recovery_stores,
):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError

    app, cid, db = recovery_stores
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    revised = await c.revise_workspace_name("source", card.request_id, "Reviewed")
    assert (
        await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
        == revised
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET version = version + 1 WHERE id = ?", (cid,)
        )
    with pytest.raises(ArchiveRecoveryError) as failure:
        await c.revise_workspace_name("source", revised.request_id, "Again")
    assert failure.value.code == "stale"
    assert c.snapshot("source") is None
    await c.shutdown()


@pytest.mark.asyncio
async def test_native_cannot_join_manual_and_scope_preselection_can_join_unselected(
    recovery_stores,
):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=None, scope=None)
    assert (
        await c.request("source", ref, call_key=None, scope="chat_and_workspace")
        == card
    )
    assert (await c.request("source", ref, call_key=key(), scope=None)).status == "busy"
    await c.decide("source", card.request_id, allow=False, scope=None)
    # Busy is a settled native call too; it cannot later acquire authorization.
    assert (await c.request("source", ref, call_key=key(), scope=None)).status == "busy"
    await c.shutdown()


@pytest.mark.asyncio
async def test_close_waiter_cancellation_still_joins_started_storage(
    recovery_stores, monkeypatch
):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    close_started = asyncio.Event()
    release = threading.Event()
    original = app.workspace_registry_service.unarchive_workspace

    def gated(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(app.workspace_registry_service, "unarchive_workspace", gated)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    decision = asyncio.create_task(
        c.decide("source", card.request_id, allow=True, scope=None)
    )
    await asyncio.wait_for(entered.wait(), 5)

    async def close():
        close_started.set()
        try:
            await c.shutdown()
        finally:
            assert not app.workspace_registry_service.get_workspace("project").archived

    shutdown = asyncio.create_task(close())
    await close_started.wait()
    shutdown.cancel()
    # A scheduler callback gives cancellation a deterministic chance to arrive.
    cancellation_delivered = asyncio.Event()
    loop.call_soon(cancellation_delivered.set)
    await cancellation_delivered.wait()
    try:
        assert not shutdown.done()
    finally:
        release.set()
        await decision
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    await c.shutdown()


@pytest.mark.asyncio
async def test_real_timer_expires_background_confirmation_without_remount(
    recovery_stores, monkeypatch
):
    import tldw_chatbook.Chat.archive_recovery.coordinator as module

    app, _, _ = recovery_stores
    monkeypatch.setattr(module, "CONFIRMATION_TTL_SECONDS", 0.01)
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope=None)
    c.set_presenter(None, None)
    result = await asyncio.wait_for(c.wait_result(card.operation_id), 2)
    assert result.status == "expired"
    assert not c.present_pending("source")
    await c.shutdown()


@pytest.mark.asyncio
async def test_settled_expiry_releases_pinned_confirmation(recovery_stores):
    app, _, _ = recovery_stores
    now = [0.0]
    c, _, ref = setup(app, clock=lambda: now[0])
    card = await c.request("source", ref, call_key=key(), scope=None)
    now[0] = 301
    c.expire_pending()
    op = c._operations[card.operation_id]
    assert op.target is None and op.confirmation is None
    assert op.ready.result() == op.result.result()
    await c.shutdown()


@pytest.mark.asyncio
async def test_stale_decision_wrong_session_and_closed_callbacks(recovery_stores):
    app, cid, db = recovery_stores
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    assert (
        await c.decide("other", card.request_id, allow=True, scope=None)
    ).status == "expired"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET version = version + 1 WHERE id = ?", (cid,)
        )
    assert (
        await c.decide("source", card.request_id, allow=True, scope=None)
    ).status == "stale"
    assert app.workspace_registry_service.get_workspace("project").archived
    await c.close_session("source")
    assert not c.open_run("source", "later")
    assert (
        await c.request("source", ref, call_key=None, scope=None)
    ).status == "expired"
    await c.shutdown()


@pytest.mark.asyncio
async def test_shutdown_joins_read_started_before_confirmation(
    recovery_stores, monkeypatch
):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    original = app.local_chat_conversation_service.get_conversation_metadata

    def gated(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10)
        return original(*args)

    monkeypatch.setattr(
        app.local_chat_conversation_service, "get_conversation_metadata", gated
    )
    request = asyncio.create_task(c.request("source", ref, call_key=key(), scope=None))
    await asyncio.wait_for(entered.wait(), 5)
    closing_started = asyncio.Event()

    async def close():
        closing_started.set()
        await c.shutdown()

    close_task = asyncio.create_task(close())
    await closing_started.wait()
    assert not close_task.done()
    release.set()
    await close_task
    assert (await request).status == "cancelled"
    assert app.workspace_registry_service.get_workspace("project").archived


@pytest.mark.asyncio
async def test_teardown_joins_all_storage_even_when_cancelled_name_read_fails(
    recovery_stores, monkeypatch
):
    app, _, _ = recovery_stores
    c, _, ref = setup(app)
    card = await c.request("source", ref, call_key=key(), scope="chat_and_workspace")
    loop = asyncio.get_running_loop()
    reading = asyncio.Event()
    writing = asyncio.Event()
    read_release = threading.Event()
    write_release = threading.Event()
    metadata = app.local_chat_conversation_service.get_conversation_metadata
    restore = app.workspace_registry_service.unarchive_workspace
    reads = [0]

    def read(*args):
        reads[0] += 1
        if reads[0] == 1:
            loop.call_soon_threadsafe(reading.set)
            assert read_release.wait(10)
            raise RuntimeError("private read failed")
        return metadata(*args)

    def write(*args, **kwargs):
        loop.call_soon_threadsafe(writing.set)
        assert write_release.wait(10)
        return restore(*args, **kwargs)

    monkeypatch.setattr(
        app.local_chat_conversation_service, "get_conversation_metadata", read
    )
    monkeypatch.setattr(app.workspace_registry_service, "unarchive_workspace", write)
    revision = asyncio.create_task(
        c.revise_workspace_name("source", card.request_id, "New name")
    )
    await asyncio.wait_for(reading.wait(), 5)
    revision.cancel()
    with pytest.raises(asyncio.CancelledError):
        await revision
    decision = asyncio.create_task(
        c.decide("source", card.request_id, allow=True, scope=None)
    )
    await asyncio.wait_for(writing.wait(), 5)
    close_joining = asyncio.Event()
    finish_session = c._finish_session

    async def finish(session_id):
        close_joining.set()
        await finish_session(session_id)

    monkeypatch.setattr(c, "_finish_session", finish)
    shutdown = asyncio.create_task(c.shutdown())
    await close_joining.wait()
    read_task = next(task for task in c._tasks if task is not c._active["source"].task)
    read_release.set()
    with pytest.raises(RuntimeError):
        await asyncio.shield(read_task)
    try:
        # Explicit loop barrier lets the join see the failed read.
        delivered = asyncio.Event()
        loop.call_soon(delivered.set)
        await delivered.wait()
        assert not shutdown.done()
    finally:
        write_release.set()
        assert (await decision).status == "partial"
    await shutdown


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "title",
    ["x" * 1100, "", "  spaced  ", "😀" * 1024],
    ids=["long", "empty", "padded", "byte_fitted"],
)
@pytest.mark.parametrize("mutate", [False, True], ids=["unchanged", "version_changed"])
async def test_display_title_fitting_preserves_exact_recovery_identity(
    recovery_stores, title, mutate
):
    """Real search references restore unchanged titles, but never newer versions."""
    import json

    from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryRuntime

    app, cid, db = recovery_stores
    db.update_conversation(
        cid, {"title": title}, db.get_conversation_by_id(cid)["version"]
    )
    if title.startswith("😀"):
        for index in range(19):
            extra = db.add_conversation(
                {
                    "title": title + str(index),
                    "scope_type": "workspace",
                    "workspace_id": "project",
                }
            )
            db.set_conversations_archived(
                [extra], archived=True, expected_versions={extra: 1}
            )
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _sid: True
    runtime.present_confirmation = lambda _card: True
    try:
        projection = await runtime.search("source", {"limit": 20}, model_visible=False)
        assert len(json.dumps(projection, ensure_ascii=False).encode()) <= 32 * 1024
        if title.startswith("😀"):
            row = next(
                item for item in projection["items"] if len(item["title"]) < 1024
            )
            cid = row["conversation_id"]
        else:
            row = next(
                item for item in projection["items"] if item["conversation_id"] == cid
            )
        if mutate:
            db.update_conversation(
                cid,
                {"title": "Intervening title"},
                db.get_conversation_by_id(cid)["version"],
            )
        card = await runtime.coordinator.request(
            "source", row["result_ref"], call_key=None, scope=None
        )
        if mutate:
            assert card.status == "stale"
            assert db.get_conversation_by_id(cid)["archived"]
            assert app.workspace_registry_service.get_workspace("project").archived
            return
        assert isinstance(card, RecoveryConfirmation), card
        assert (
            card.target.title
            == app.local_chat_conversation_service.get_conversation_metadata(cid)[
                "title"
            ]
        )
        outcome = await runtime.coordinator.decide(
            "source", card.request_id, allow=True, scope="chat_and_workspace"
        )
        assert (
            outcome.status == "restored"
            and outcome.chat_changed
            and outcome.workspace_changed
        )
        assert not db.get_conversation_by_id(cid)["archived"]
        assert not app.workspace_registry_service.get_workspace("project").archived
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("eviction", ["fifo", "ttl"])
async def test_runtime_retry_retains_pending_and_completed_outcomes_after_page_eviction(
    recovery_stores, eviction  # noqa: F811 - imported pytest fixture parameter
):
    from tldw_chatbook.Chat.archive_recovery.contracts import ArchiveRecoveryError
    from tldw_chatbook.Chat.archive_recovery.coordinator import MAX_SESSION_RECEIPTS
    from tldw_chatbook.Chat.archive_recovery.result_store import (
        MAX_SESSION_PAGES,
        RESULT_TTL_SECONDS,
    )
    from tldw_chatbook.Chat.archive_recovery.runtime import ArchiveRecoveryRuntime

    app, cid, db = recovery_stores
    runtime = ArchiveRecoveryRuntime(app)
    runtime.is_answerable = lambda _: True
    shown = asyncio.Event()
    runtime.present_confirmation = lambda _: (shown.set() or True)

    async def acknowledge(_):
        return True

    runtime.on_settled = acknowledge
    coordinator = runtime.coordinator
    assert coordinator.open_run("source", "run")
    now = [0.0]
    runtime.results._monotonic = lambda: now[0]
    page = ArchiveSearchService(
        app.local_chat_conversation_service, app.workspace_registry_service
    ).search(ArchiveSearchRequest())

    def publish(*, model_visible=True):
        return runtime.results.publish(
            "source", ArchiveSearchRequest(), page, model_visible=model_visible
        )["items"][0]["result_ref"]

    ref = publish()
    arguments = {"result_ref": ref, "scope": "chat_only"}
    first = asyncio.create_task(
        runtime.request_restore("source", "run", "call", arguments)
    )
    tasks = [first]
    try:
        await asyncio.wait_for(shown.wait(), 5)
        original = coordinator.snapshot("source").confirmation
        if eviction == "fifo":
            for _ in range(MAX_SESSION_PAGES):
                publish()
        else:
            now[0] = RESULT_TTL_SECONDS + 1
        with pytest.raises(ArchiveRecoveryError, match="expired"):
            runtime.results.resolve("source", ref, for_agent=True)

        retry = asyncio.create_task(
            runtime.request_restore("source", "run", "call", arguments)
        )
        tasks.append(retry)
        await asyncio.sleep(0)
        assert (
            not retry.done()
        ), "a retained pending call must still await its original decision"
        assert coordinator.snapshot("source").confirmation is original
        changed = await runtime.request_restore(
            "source", "run", "call", {**arguments, "scope": "chat_and_workspace"}
        )
        assert changed.status == "stale"
        outcome = await coordinator.decide(
            "source", original.request_id, allow=True, scope="chat_only"
        )
        assert outcome.status == "restored"
        assert await first is outcome
        assert await retry is outcome
        assert (
            await runtime.request_restore("source", "run", "call", arguments) is outcome
        )
        assert not db.get_conversation_by_id(cid)["archived"]
        assert app.workspace_registry_service.get_workspace("project").archived

        # New agent calls still cannot select manual-only evidence.
        manual_ref = publish(model_visible=False)
        refused = await runtime.request_restore(
            "source", "run", "manual", {"result_ref": manual_ref}
        )
        assert refused.status == "unavailable"
        assert coordinator.snapshot("source") is None

        # Existing bounded receipt eviction cannot revive consumed authorization.
        for index in range(MAX_SESSION_RECEIPTS):
            expired = await runtime.request_restore(
                "source", "run", f"expired-{index}", {"result_ref": "missing"}
            )
            assert expired.status == "expired"
        assert len(coordinator.receipts("source")) == MAX_SESSION_RECEIPTS
        assert len(coordinator._operations) == MAX_SESSION_RECEIPTS
        assert (
            await runtime.request_restore("source", "run", "call", arguments)
        ).status == "expired"
        assert (
            await coordinator.decide(
                "source", original.request_id, allow=True, scope="chat_only"
            )
        ).status == "expired"
        assert coordinator.snapshot("source") is None
    finally:
        await runtime.shutdown()
        await asyncio.gather(*tasks, return_exceptions=True)
