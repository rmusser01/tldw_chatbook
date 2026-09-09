"""Startup discovers saved results without granting uncertain work a replay."""

import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from Tests.DB.test_automatic_wake_attempts import claim, survivor
from Tests.DB.test_automatic_work_budget import chain
from tldw_chatbook.Chat import console_launch_wake as launch
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.private_sqlite import connect_private_sqlite
from tldw_chatbook.Utils.private_paths import PrivatePathError


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "autowake_enabled", lambda: True)
    db = AgentRunsDB(tmp_path / "agent_runs.db")
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(
            db_path=tmp_path / "chacha.sqlite",
            get_conversation_by_id=lambda cid: {"id": cid},
        ),
        app_config={},
    )
    yield app, db
    db.close()


def _forbid_runtime(app):
    pytest.fail("startup with no pending native results constructed a runtime")


def _controller(app, db):
    runtime = ConsoleRuntime(app)
    store = runtime.ensure_chat_store()
    controller = runtime.ensure_chat_controller(
        store=store,
        provider_gateway=object(),
        agent_bridge=SimpleNamespace(runs_db=db),
    )
    return runtime, controller


def test_discovery_reads_unmarked_claimed_results_without_recovery(native):
    app, db = native
    chain_id = chain(db)
    run_id = survivor(db, chain_id)
    claim(db, chain_id, [run_id])
    assert db.automatic_work.accept_wake("attempt", owner_id="owner")
    # A fresh read must find the claim while leaving recovery to runtime startup.
    db.close()
    assert launch.marked_conversations_at_launch(app) == ("conversation",)
    assert (
        db.automatic_work.read_attempt("attempt", owner_id="owner").state == "accepted"
    )
    assert db.automatic_work.snapshot(chain_id).status == "active"
    assert not hasattr(app, "_console_runtime")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", [None, ":memory:", "file::memory:?cache=shared", "file:native?mode=memory"]
)
async def test_memory_and_absent_native_paths_never_build_runtime(
    tmp_path, monkeypatch, path
):
    app = SimpleNamespace(chachanotes_db=SimpleNamespace(db_path=path))
    monkeypatch.setattr(launch, "autowake_enabled", lambda: True)
    monkeypatch.setattr(launch, "_ensure_launch_runtime", _forbid_runtime)
    assert launch.marked_conversations_at_launch(app) == ()
    assert await launch.deliver_launch_wakes(app, ("badge-only",)) == 0
    assert not (tmp_path / "agent_runs.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_missing_or_empty_ledger_ignores_badges_and_constructs_nothing(
    tmp_path, monkeypatch, existing
):
    db_path = tmp_path / "agent_runs.db"
    if existing:
        AgentRunsDB(db_path).close()
    app = SimpleNamespace(
        chachanotes_db=SimpleNamespace(db_path=tmp_path / "chacha.sqlite"),
        conversation_local_marks_service=SimpleNamespace(
            FLEET_UNSEEN="unseen",
            list_marked_conversation_ids=lambda _: ("badge-only",),
        ),
    )
    monkeypatch.setattr(launch, "autowake_enabled", lambda: True)
    monkeypatch.setattr(launch, "_ensure_launch_runtime", _forbid_runtime)
    assert launch.marked_conversations_at_launch(app) == ()
    assert await launch.deliver_launch_wakes(app, ("badge-only",)) == 0
    assert db_path.exists() is existing


def test_discovery_excludes_completed_delivery_and_within_turn_result(native):
    app, db = native
    chain_id = chain(db)
    delivered = survivor(db, chain_id)
    db.mark_wake_delivered([delivered])
    parent = db.create_run(
        conversation_id="conversation", agent_kind="primary", work_chain_id=chain_id
    )
    child = db.create_run(
        conversation_id="conversation", agent_kind="subagent", parent_run_id=parent
    )
    db.set_status(child, "done", "collected in turn")
    db.set_status(parent, "done", "parent")
    assert launch.marked_conversations_at_launch(app) == ()


def test_launch_sqlite_owner_allows_only_existing_private_readonly_files(
    native, tmp_path
):
    _app, db = native
    with pytest.raises(ValueError, match="read-only"):
        connect_private_sqlite("chat.launch_wake", db.db_path)
    connection = connect_private_sqlite(
        "chat.launch_wake", db.db_path, read_only=True, must_exist=True
    )
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("CREATE TABLE forbidden (id INTEGER)")
    finally:
        connection.close()
    missing = tmp_path / "missing.db"
    with pytest.raises((PrivatePathError, OSError, sqlite3.OperationalError)):
        connect_private_sqlite(
            "chat.launch_wake", missing, read_only=True, must_exist=True
        )
    assert not missing.exists()
    symlink = tmp_path / "linked.db"
    symlink.symlink_to(db.db_path)
    with pytest.raises(PrivatePathError):
        connect_private_sqlite(
            "chat.launch_wake", symlink, read_only=True, must_exist=True
        )


@pytest.mark.asyncio
async def test_runtime_recovers_once_and_remount_does_not_invalidate_live_owner(
    native, monkeypatch
):
    app, db = native
    old_chain = chain(db)
    claim(db, old_chain, [survivor(db, old_chain)])
    assert db.automatic_work.accept_wake("attempt", owner_id="owner")
    recover = db.automatic_work.recover
    recovered = []

    def record_recovery(**kwargs):
        recovered.append(kwargs["current_owner_id"])
        return recover(**kwargs)

    monkeypatch.setattr(db.automatic_work, "recover", record_recovery)
    runtime, controller = _controller(app, db)
    try:
        assert await controller.fleet_wake.wait_for_recovery()
        assert db.automatic_work.snapshot(old_chain).status == "review_required"
        owner = recovered[0]
        live_chain = chain(db, submission="live")
        claim(db, live_chain, [survivor(db, live_chain)], attempt="live", owner=owner)
        for _ in range(2):
            view = SimpleNamespace(console_view_hooks=dict)
            runtime.attach_view(view)
            runtime.detach_view(view)
            assert runtime.ensure_chat_controller() is controller
            db.close()
            assert db.automatic_work.snapshot(live_chain).status == "active"
        assert recovered == [owner]
        assert (
            db.automatic_work.read_attempt("live", owner_id=owner).state == "prepared"
        )
    finally:
        await runtime.dispose()


@pytest.mark.asyncio
async def test_launch_waits_for_recovery_before_hydrating_unmarked_claim(
    native, monkeypatch
):
    app, db = native
    chain_id = chain(db)
    claim(db, chain_id, [survivor(db, chain_id)])
    assert db.automatic_work.accept_wake("attempt", owner_id="owner")
    started, release = threading.Event(), threading.Event()
    recover = db.automatic_work.recover
    events = []

    def blocked_recovery(**kwargs):
        started.set()
        assert release.wait(5)
        result = recover(**kwargs)
        events.append("recovered")
        return result

    def tree(cid, **kwargs):
        events.append("hydrated")
        return {"conversation": {"id": cid, "title": "Saved"}, "root_threads": []}

    monkeypatch.setattr(db.automatic_work, "recover", blocked_recovery)
    app.chat_conversation_scope_service = SimpleNamespace(get_conversation_tree=tree)
    runtime, controller = _controller(app, db)
    monkeypatch.setattr(launch, "_ensure_launch_runtime", lambda _: controller)
    delivery = asyncio.create_task(launch.deliver_launch_wakes(app, ()))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.sleep(0)
        assert not delivery.done()
        assert not controller.store.sessions()
        assert events == []
        release.set()
        assert await asyncio.wait_for(delivery, 3) == 1
        assert events == ["recovered", "hydrated"]
        assert controller.fleet_wake.pause_reason("conversation")
        assert (
            db.automatic_work.read_attempt("attempt", owner_id="owner").state
            == "review_required"
        )
        assert db.automatic_work.snapshot(chain_id).used["generation"] == 1
    finally:
        release.set()
        await asyncio.gather(delivery, return_exceptions=True)
        await runtime.dispose()


@pytest.mark.asyncio
async def test_failed_recovery_prevents_hydration(native, monkeypatch):
    app, db = native
    chain_id = chain(db)
    survivor(db, chain_id)

    def fail_recovery(**kwargs):
        raise OSError("unavailable ledger")

    monkeypatch.setattr(db.automatic_work, "recover", fail_recovery)
    runtime, controller = _controller(app, db)
    monkeypatch.setattr(launch, "_ensure_launch_runtime", lambda _: controller)
    try:
        assert await launch.deliver_launch_wakes(app, ("conversation",)) == 0
        assert not controller.store.sessions()
        assert not await controller.fleet_wake.wait_for_recovery()
    finally:
        await runtime.dispose()


def test_remount_rearms_each_concurrent_delivery():
    seen = []
    wake = SimpleNamespace(
        delivering_session_ids=lambda: ("first", "second"), delivery_ui_hook=None
    )
    runtime = ConsoleRuntime(None)
    runtime.set_chat_controller(SimpleNamespace(fleet_wake=wake))
    view = SimpleNamespace(console_view_hooks=lambda: {"delivery_ui_hook": seen.append})
    runtime.attach_view(view)
    assert seen == ["first", "second"]


def test_synchronous_native_construction_defers_one_audit_until_loop_capture(native):
    app, db = native
    chain_id = chain(db)
    claim(db, chain_id, [survivor(db, chain_id)])
    runtime, controller = _controller(app, db)

    async def start_loop():
        try:
            controller.fleet_wake.wire(app=app)
            assert await controller.fleet_wake.wait_for_recovery()
            assert db.automatic_work.snapshot(chain_id).status == "review_required"
        finally:
            await runtime.dispose()

    asyncio.run(start_loop())
