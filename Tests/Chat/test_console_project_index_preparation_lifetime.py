"""Custody across the complete live and preview index preparation callback."""

import asyncio
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_console_unused_project_index import _console_index
from Tests.private_profile import private_profile_test
from tldw_chatbook import config
from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.participants import core_cached_connection
from tldw_chatbook.Chat import console_chat_controller as host
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_preparation_reads import (
    preparation_read_for,
    preparation_source_for,
)
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.project_index import runtime as index
from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService


class _UnexpectedProfileRead(dict):
    def __bool__(self):
        pytest.fail("custom profile container was coerced")

    def get(self, *args):
        pytest.fail("custom profile container was read")


@pytest.fixture
def composition(tmp_path, monkeypatch):
    db = WorkspaceDB(tmp_path / "workspaces.db")
    app = SimpleNamespace(workspace_registry_service=LocalWorkspaceRegistryService(db))
    runtime = ConsoleRuntime(app)
    app.console_runtime = runtime
    store = ConsoleChatStore()
    controller = host.ConsoleChatController(store=store, provider_gateway=object())
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    session = store.create_session(workspace_id="project")
    other = store.create_session(workspace_id="other", activate=False)
    consumed = []
    mcp_counts = []

    async def mcp(_session_id, *, publish_counts, **kwargs):
        mcp_counts.append(publish_counts)
        return None

    monkeypatch.setattr(controller, "_compose_mcp_provider", mcp)
    monkeypatch.setattr(
        controller,
        "_compose_local_provider",
        lambda **kwargs: (consumed.append(kwargs), None),
    )
    fixture = SimpleNamespace(
        app=app,
        runtime=runtime,
        store=store,
        controller=controller,
        session=session,
        other=other,
        consumed=consumed,
        mcp_counts=mcp_counts,
    )
    try:
        yield fixture
    finally:
        db.close()


def install_index_callback(
    monkeypatch,
    composition,
    tmp_path,
    *,
    guard=None,
    close=None,
    state="Building",
    cold=False,
):
    controller = composition.controller
    scope = object()
    events = []

    @contextmanager
    def registry_scope():
        try:
            yield
        finally:
            events.append("close")
            if close is not None:
                close()

    def building_coverage(candidate, floor):
        assert candidate is scope
        assert len(floor) == 1
        events.append("coverage")
        return True

    def status(service, key):
        events.append("status")
        if state == "unavailable":
            raise OSError("optional cache could not open")
        return SimpleNamespace(state=state, indexed=1, scope=scope)

    def root_guard(write):
        assert not write
        events.append("guard")
        if guard is not None:
            guard()
        return True

    def exclusions():
        events.append("exclusions")
        return (tmp_path / "hidden",)

    service = SimpleNamespace(
        registry_connection_scope=registry_scope, building_coverage=building_coverage
    )

    def get_or_build(app):
        if state == "construction-unavailable":
            raise OSError("optional index service could not open")
        if not hasattr(app, "_project_index_service"):
            app._project_index_service = service
        return app._project_index_service

    if not cold:
        composition.app._project_index_service = service
    monkeypatch.setattr(index, "get_project_index_service", get_or_build)
    monkeypatch.setattr(index, "project_index_key", lambda *args: "key")
    monkeypatch.setattr(index, "project_index_status_for_console", status)
    root = RunAdmittedWorkspaceRoot(
        root=tmp_path,
        guard=root_guard,
        exclusions_provider=exclusions,
        workspace_id=composition.session.workspace_id,
        binding_id="binding",
        alias="p",
        locator_fingerprint="synthetic locator",
        root_identity=(("synthetic root", 1, 1, 1),),
        allow_write=False,
    )
    root_read = []

    def capture(*, session, **kwargs):
        root_read.append(session)
        return (root,)

    monkeypatch.setattr(host, "capture_run_admitted_workspace_roots", capture)
    composition.roots, composition.scope = (root,), scope
    composition.events, composition.root_read = events, root_read
    return controller


def compose(fixture, route="dispatch", *, session_id=None):
    return fixture.controller._compose_agent_request_providers(
        session_id=fixture.session.id if session_id is None else session_id,
        project_selection=None,
        project_authority_guard=None,
        publish_mcp_counts=route == "dispatch",
        **({} if route == "personal" else {"admitted_roots": fixture.roots}),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, revoke_queued",
    [
        pytest.param("dispatch", False, id="dispatch"),
        pytest.param("next-send", False, id="next-send"),
        pytest.param("personal", False, id="personal"),
        pytest.param("dispatch", True, id="queued-source-revocation"),
    ],
)
async def test_shared_index_handoff_is_retained_while_queued_and_through_final_close(
    composition, monkeypatch, tmp_path, route, revoke_queued
):
    host_loop = asyncio.get_running_loop()
    queued, first, final, closing = [asyncio.Event() for _ in range(4)]
    unqueue, release_first, release_final, release_close = [
        threading.Event() for _ in range(4)
    ]
    seen = []

    def guard():
        read = preparation_read_for(controller)
        seen.append((read, preparation_source_for(controller)))
        at, release = (
            (first, release_first) if len(seen) == 1 else (final, release_final)
        )
        host_loop.call_soon_threadsafe(at.set)
        assert release.wait(15)

    def close():
        host_loop.call_soon_threadsafe(closing.set)
        assert release_close.wait(15)

    controller = install_index_callback(
        monkeypatch,
        composition,
        tmp_path,
        guard=guard,
        close=close,
        cold=route == "personal",
    )
    worker = ThreadPoolExecutor(max_workers=1)
    worker.submit(unqueue.wait, 15)
    original_submit = host_loop.run_in_executor
    before_submit = []

    def submit(executor, callback, *args):
        before_submit.append(
            (
                controller._preparation_reads.copy(),
                composition.runtime._preparation_reads.copy(),
            )
        )
        queued.set()
        return original_submit(worker, callback, *args)

    monkeypatch.setattr(host_loop, "run_in_executor", submit)
    if route == "personal":
        # The requested chat may already be in the background at submission.
        composition.store.switch_session(composition.other.id)
    task = asyncio.create_task(compose(composition, route))
    try:
        await asyncio.wait_for(queued.wait(), 10)
        [(controller_reads, runtime_reads)] = before_submit
        assert len(controller_reads) == 1 and controller_reads == runtime_reads
        (read,) = controller_reads
        assert read.session_id == composition.session.id
        assert preparation_source_for(controller) is None and not read.retired.done()
        if revoke_queued:
            monkeypatch.setattr(composition.app, "workspace_registry_service", object())
            unqueue.set()
            with pytest.raises(
                RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await task
            assert read.retired.done() and not read.callback_returned
            assert not composition.events and not composition.consumed
            assert not controller._preparation_reads
            assert not composition.runtime._preparation_reads
            return

        # A viewed-chat switch never revokes the requested session. Live and
        # Next Send both cancel but Personal Context proceeds to composition.
        composition.store.switch_session(composition.other.id)
        if route != "personal":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        unqueue.set()
        await asyncio.wait_for(first.wait(), 10)
        assert seen == [(read, read.source)] and read.source is not None
        release_first.set()
        await asyncio.wait_for(final.wait(), 10)
        if route != "personal":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        assert read in controller._preparation_reads
        assert read in composition.runtime._preparation_reads
        assert not read.retired.done()
        release_final.set()
        await asyncio.wait_for(closing.wait(), 10)
        assert not read.retired.done() and not read.callback_returned
        assert read in controller._preparation_reads
        assert read in composition.runtime._preparation_reads
        assert not composition.consumed
        release_close.set()
        if route == "personal":
            await task
            assert (
                composition.consumed[0]["project_index_scopes"]["p"]
                is composition.scope
            )
            assert composition.root_read == [composition.session]
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not composition.consumed
        assert composition.mcp_counts == [route == "dispatch"]
        assert read.retired.done() and read.callback_returned
        assert (
            not controller._preparation_reads
            and not composition.runtime._preparation_reads
        )
        assert len(before_submit) == 1
        assert composition.events == [
            "guard",
            "status",
            "exclusions",
            "coverage",
            "guard",
            "close",
        ]
    finally:
        for event in (unqueue, release_first, release_final, release_close):
            event.set()
        await asyncio.gather(task, return_exceptions=True)
        worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revocation",
    ["session", "profile", "profile-container", "runtime", "callback-body"],
)
async def test_revocation_after_original_callback_refuses_indexes(
    composition, monkeypatch, tmp_path, revocation
):
    loop = asyncio.get_running_loop()
    closing, release = asyncio.Event(), threading.Event()

    def close():
        loop.call_soon_threadsafe(closing.set)
        assert release.wait(15)

    install_index_callback(monkeypatch, composition, tmp_path, close=close)
    task = asyncio.create_task(compose(composition))
    try:
        await asyncio.wait_for(closing.wait(), 10)
        if revocation == "session":
            composition.store._sessions[composition.session.id] = replace(
                composition.session
            )
        elif revocation == "profile":
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "other.toml"))
        elif revocation == "profile-container":
            monkeypatch.setattr(config, "_CONFIG_CACHE", _UnexpectedProfileRead())
        elif revocation == "callback-body":
            callback = host.ConsoleChatController._prepare_project_index_scopes
            monkeypatch.setattr(callback, "__code__", callback.__code__.replace())
        else:
            composition.runtime._admission_fenced_sessions.add(composition.session.id)
        release.set()
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await task
        assert not composition.consumed
        assert not composition.controller._preparation_reads
        assert not composition.runtime._preparation_reads
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state", ["Disabled", "unavailable", "construction-unavailable"]
)
async def test_incompatible_or_unavailable_status_does_not_repeat_native_guard(
    composition, monkeypatch, tmp_path, state
):
    install_index_callback(monkeypatch, composition, tmp_path, state=state)
    await compose(composition)
    expected = (
        [] if state == "construction-unavailable" else ["guard", "status", "close"]
    )
    assert composition.events == expected
    assert "project_index_scopes" not in composition.consumed[0]


@pytest.mark.asyncio
async def test_index_result_is_rechecked_on_issuing_loop_before_local_composition(
    composition, monkeypatch, tmp_path
):
    from tldw_chatbook.Chat import console_project_index_preparation as preparation

    install_index_callback(monkeypatch, composition, tmp_path)
    original = preparation.run_preparation_read

    async def revoke_after_retirement(*args, **kwargs):
        result = await original(*args, **kwargs)
        assert not composition.controller._preparation_reads
        monkeypatch.setattr(composition.app, "workspace_registry_service", object())
        return result

    monkeypatch.setattr(preparation, "run_preparation_read", revoke_after_retirement)
    with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
        await compose(composition)
    assert not composition.consumed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback",
    [
        "custom",
        "unmanaged",
        "missing-session",
        "custom-profile",
        "custom-profile-section",
    ],
)
async def test_compatibility_routes_keep_original_callback_contract(
    composition, monkeypatch, tmp_path, fallback
):
    seen = []
    main = threading.get_ident()
    controller = install_index_callback(
        monkeypatch,
        composition,
        tmp_path,
        guard=lambda: seen.append(preparation_source_for(composition.controller)),
    )
    if fallback == "custom":

        def custom(roots):
            assert roots == composition.roots and threading.get_ident() != main
            seen.append(preparation_source_for(controller))
            return None, {}

        monkeypatch.setattr(controller, "_prepare_project_index_scopes", custom)
    elif fallback == "unmanaged":
        controller._hooks_v2_runtime = None
    elif fallback in {"custom-profile", "custom-profile-section"}:
        custom = _UnexpectedProfileRead()
        cache = {"paths": custom} if fallback == "custom-profile-section" else custom
        monkeypatch.setattr(config, "_CONFIG_CACHE", cache)
    await compose(
        composition,
        "next-send",
        session_id="missing" if fallback == "missing-session" else None,
    )
    assert seen and all(source is None for source in seen)
    assert (
        not controller._preparation_reads and not composition.runtime._preparation_reads
    )


@pytest.mark.asyncio
async def test_custom_controller_with_supplied_roots_does_not_probe_ownership():
    main = threading.get_ident()
    worker_seen = []

    class CustomController:
        app = SimpleNamespace()

        @property
        def store(self):
            pytest.fail("custom callback has no managed store")

        @property
        def _hooks_v2_runtime(self):
            pytest.fail("custom callback has no managed runtime")

        async def _compose_mcp_provider(self, session_id, **kwargs):
            return None

        def _prepare_project_index_scopes(self, roots):
            assert roots == () and threading.get_ident() != main
            worker_seen.append(preparation_source_for(self))
            return None, {}

        def _compose_local_provider(self, **kwargs):
            return None, None

    await host.ConsoleChatController._compose_agent_request_providers(
        CustomController(),
        session_id="custom",
        project_selection=None,
        project_authority_guard=None,
        admitted_roots=(),
        publish_mcp_counts=False,
    )
    assert worker_seen == [None]


@pytest.mark.parametrize("method", ["callback", "lifecycle"])
@private_profile_test
async def test_preimport_controller_change_uses_legacy_index_callback(
    composition, monkeypatch, tmp_path, request, method
):
    module_name = "tldw_chatbook.Chat.console_project_index_preparation"
    assert module_name not in sys.modules
    main = threading.get_ident()
    worker_reads = []
    replacement_calls = []

    def see_worker():
        worker_reads.append(
            (
                threading.get_ident() == main,
                preparation_read_for(controller),
                preparation_source_for(controller),
                bool(controller._preparation_reads),
                bool(composition.runtime._preparation_reads),
            )
        )

    controller = install_index_callback(
        monkeypatch, composition, tmp_path, guard=see_worker
    )
    assert module_name not in sys.modules

    def replaced(actual, value):
        assert actual is controller
        replacement_calls.append(value)
        if method == "lifecycle":
            return 0
        see_worker()
        return None, {}

    def invoke(self, value):
        return globals()["_task52_replaced_controller_method"](self, value)

    original = (
        host.ConsoleChatController._prepare_project_index_scopes
        if method == "callback"
        else host.ConsoleChatController._lifecycle_revision_for
    )
    monkeypatch.setattr(
        host, "_task52_replaced_controller_method", replaced, raising=False
    )
    monkeypatch.setattr(
        original,
        "__code__",
        invoke.__code__.replace(
            co_filename=original.__code__.co_filename,
            co_qualname=original.__code__.co_qualname,
        ),
    )
    await compose(composition, "next-send")

    assert module_name in sys.modules
    if method == "callback":
        assert replacement_calls == [composition.roots]
        assert "project_index_scopes" not in composition.consumed[0]
    else:
        assert replacement_calls == []
        assert composition.consumed[0]["project_index_scopes"]["p"] is composition.scope
    assert worker_reads == [(False, None, None, False, False)] * (
        1 if method == "callback" else 2
    )


@private_profile_test
async def test_owned_key_reuses_original_binding_checks_and_retires_native_connection(
    tmp_path, request
):
    async with _console_index(tmp_path) as (controller, roots, cache):
        app = controller.app
        db = app.workspace_registry_service.db
        db.close()
        (root,) = roots
        service = await asyncio.to_thread(index.get_project_index_service, app)
        args = (root.workspace_id, root.binding_id, root.root)
        expected = await asyncio.to_thread(index.project_index_key, app, *args)

        def derive():
            handles = []
            original = service._reader

            def reader(key):
                result = original(key)
                handles.append(core_cached_connection(db, db._thread_local.conn))
                return result

            service._reader = reader
            try:
                assert index._project_index_key_owned(service, *args) == expected
                with pytest.raises(ValueError, match="index_binding_unavailable"):
                    index._project_index_key_owned(
                        service, root.workspace_id, "missing-binding", root.root
                    )
            finally:
                service._reader = original
            assert len(handles) == 2
            for handle in handles:
                assert isinstance(handle, sqlite3.Connection)
                assert core_cached_connection(db, handle) is None
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    sqlite3.Connection.in_transaction.__get__(handle)

        await asyncio.to_thread(derive)
        assert not cache.exists()
