"""Original app wrappers retain their live patch and callback routes."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.app_module_patches import set_app_global
from tldw_chatbook import app as app_module


def test_startup_wiring_preserves_live_delays_and_callback_order(monkeypatch):
    calls = []
    app = SimpleNamespace(
        _start_staggered_boot_workers=lambda: calls.append("workers"),
        set_timer=lambda delay, callback: calls.append((delay, callback)),
        _schedule_footer_status_updates=object(),
        _start_deferred_audio_service_initialization=object(),
        _deferred_wire_collections_capture_services=object(),
        _deferred_wire_workspace_agent_provisioning=object(),
        _schedule_screen_preimport=object(),
        _deferred_wire_notes_sync_services=object(),
        schedule_media_cleanup=lambda: calls.append("media"),
        _create_deferred_startup_task=lambda work, *, name: calls.append((name, work)),
        _reconcile_interrupted_subscription_work=lambda: "subscriptions",
        _reconcile_citation_artifact_ownership=lambda: "citations",
        _migrate_legacy_citations_idle_unit=lambda: "migration",
        citation_artifact_ownership_coordinator=SimpleNamespace(writes_enabled=True),
        citation_legacy_migration_service=SimpleNamespace(
            ready=True, writes_enabled=True
        ),
        _schedule_launch_wake=lambda: calls.append("wake"),
    )
    monkeypatch.setattr(
        app_module,
        "_install_deferred_notes_sync_facades",
        lambda owner: calls.append(("notes", owner)),
    )
    for delay in (0.123, 0.456):
        monkeypatch.setattr(
            app_module, "DEFERRED_SCREEN_PREIMPORT_DELAY_SECONDS", delay
        )
        calls.clear()
        app_module.TldwCli._schedule_deferred_startup_work(app)
        assert calls == [
            "workers",
            (
                app_module.DEFERRED_DB_SIZE_UPDATE_DELAY_SECONDS,
                app._schedule_footer_status_updates,
            ),
            (
                app_module.DEFERRED_AUDIO_SERVICE_DELAY_SECONDS,
                app._start_deferred_audio_service_initialization,
            ),
            (
                app_module.DEFERRED_COLLECTIONS_CAPTURE_WIRING_DELAY_SECONDS,
                app._deferred_wire_collections_capture_services,
            ),
            (
                app_module.DEFERRED_WORKSPACE_AGENT_PROVISIONING_DELAY_SECONDS,
                app._deferred_wire_workspace_agent_provisioning,
            ),
            (delay, app._schedule_screen_preimport),
            "media",
            ("deferred_subscription_interrupt_reconcile", "subscriptions"),
            ("deferred_citation_artifact_reconciliation", "citations"),
            ("deferred_legacy_citation_migration", "migration"),
            "wake",
            ("notes", app),
            (
                app_module.DEFERRED_NOTES_ORGANIZATION_WIRING_DELAY_SECONDS,
                app._deferred_wire_notes_sync_services,
            ),
        ]


@pytest.mark.parametrize("error_name", ["QueryError", "ScreenStackError"])
def test_footer_lookup_keeps_late_app_types_and_teardown_fallback(
    monkeypatch, error_name
):
    class MissingFooter(Exception):
        pass

    marker = object()
    cached = object()
    selectors = []

    def query(selector):
        selectors.append(selector)
        raise MissingFooter

    app = SimpleNamespace(
        screen=SimpleNamespace(query_one=query), _db_size_status_widget=cached
    )
    monkeypatch.setattr(app_module, error_name, MissingFooter)
    monkeypatch.setattr(app_module, "AppFooterStatus", marker)
    assert app_module.TldwCli._active_footer_status(app) is cached
    assert selectors == [marker]


def test_native_approval_binding_keeps_original_identity():
    def original():
        return None

    app = SimpleNamespace(served_relay_publisher=None)
    assert (
        app_module.TldwCli.served_relay_wrap_pending_approval(app, original) is original
    )


def test_served_run_states_follow_current_controller_and_refuse_failures():
    first, second = object(), object()
    app = SimpleNamespace(console_runtime=None)
    assert app_module.TldwCli._served_relay_run_states(app) is None
    app.console_runtime = SimpleNamespace(
        chat_controller=SimpleNamespace(run_states=first)
    )
    assert app_module.TldwCli._served_relay_run_states(app) is first
    app.console_runtime.chat_controller = SimpleNamespace(run_states=second)
    assert app_module.TldwCli._served_relay_run_states(app) is second

    class Unavailable:
        @property
        def run_states(self):
            raise RuntimeError("unavailable")

    app.console_runtime.chat_controller = Unavailable()
    assert app_module.TldwCli._served_relay_run_states(app) is None


@pytest.mark.parametrize("round_state", ["absent", "unknown", "settled", "active"])
def test_served_decision_requires_current_armed_round(round_state):
    delivered = []
    decisions = {"call": "deny"}
    app = SimpleNamespace(console_runtime=None)
    if round_state != "absent":
        controller = SimpleNamespace(
            _pending_approval_rounds=(
                {}
                if round_state == "unknown"
                else {"round": {"settled": round_state == "settled"}}
            ),
            resolve_pending_approval=lambda values, *, round_id: delivered.append(
                (values, round_id)
            ),
        )
        app.console_runtime = SimpleNamespace(chat_controller=controller)
    assert app_module.TldwCli._served_relay_decide(app, "round", decisions) is (
        round_state == "active"
    )
    if round_state == "active":
        assert delivered == [({"call": "deny"}, "round")]
        assert delivered[0][0] is not decisions
    else:
        assert delivered == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["pending", "succeeded", "failed"])
async def test_served_control_task_retirement_preserves_late_asyncio_and_close_order(
    monkeypatch, mode
):
    calls, warnings, loop_errors = [], [], []
    owned_tasks, attached_sources, projection_requests = [], [], []
    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    entered, release, watching, disconnected = (asyncio.Event() for _ in range(4))
    disconnect_timeouts = []
    identity, source = object(), object()
    task = lifetime = None
    stopped = False

    class WrongCancellation(Exception):
        pass

    def create_task(work, *, name):
        owned = asyncio.create_task(work, name=name)
        owned_tasks.append(owned)
        return owned

    async def start():
        calls.append("start")
        entered.set()
        try:
            await release.wait()
            if mode == "failed":
                raise RuntimeError("offline")
        finally:
            calls.append("retired")

    async def wait_disconnected(*, timeout):
        disconnect_timeouts.append(timeout)
        calls.append("watching")
        watching.set()
        try:
            await disconnected.wait()
        finally:
            calls.append("lifetime_retired")

    def attach(*, projection_source):
        assert app._served_canvas_control_start_task.done()
        calls.append("attach")
        attached_sources.append(projection_source)

    def ensure_remote_projection(host_identity):
        projection_requests.append(host_identity)
        return source

    async def close_publisher():
        assert app._served_control_lifetime_task is None
        assert app._served_canvas_control_start_task is None
        assert all(owned.done() for owned in owned_tasks)
        calls.append("publisher_closed")

    async def close_client():
        assert calls[-1] == "publisher_closed"
        calls.append("client_closed")

    async def join(*tasks, return_exceptions):
        assert return_exceptions is True
        assert len(tasks) == 1
        if tasks[0] is task:
            assert app._served_canvas_control_start_task is None
            assert app._served_control_lifetime_task is None
            calls.append("join_start")
        else:
            assert tasks[0] is lifetime
            assert app._served_control_lifetime_task is None
            assert app._served_canvas_control_start_task is task
            calls.append("join_lifetime")
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    app = SimpleNamespace(
        served_canvas_control=SimpleNamespace(
            start=start,
            wait_disconnected=wait_disconnected,
            host_identity=identity,
            aclose=close_client,
        ),
        served_relay_publisher=SimpleNamespace(attach=attach, aclose=close_publisher),
        console_runtime=SimpleNamespace(
            ensure_remote_projection=ensure_remote_projection,
            _chat_controller=None,
            view=None,
        ),
        _served_canvas_control_start_task=None,
        _served_control_lifetime_task=None,
        _shutting_down=False,
        _close_boot_worker_gate=lambda reason: calls.append(("gate_closed", reason)),
        exit=lambda: calls.append("exit"),
        loguru_logger=SimpleNamespace(warning=lambda *args: warnings.append(args)),
    )
    # A captured early namespace cannot classify the later real cancellation.
    set_app_global(
        monkeypatch,
        "asyncio",
        SimpleNamespace(
            create_task=create_task,
            Task=asyncio.Task,
            CancelledError=WrongCancellation,
        ),
    )
    late_asyncio = SimpleNamespace(
        create_task=create_task,
        Task=asyncio.Task,
        CancelledError=asyncio.CancelledError,
        gather=join,
    )
    try:
        app_module.TldwCli._start_served_canvas_control(app)
        task = app._served_canvas_control_start_task
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == "start_served_canvas_control"
        await entered.wait()
        assert calls == ["start"]
        assert attached_sources == projection_requests == []
        assert app._served_control_lifetime_task is None
        set_app_global(monkeypatch, "asyncio", late_asyncio)
        if mode != "pending":
            release.set()
            start_results = await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            lifetime = app._served_control_lifetime_task
            assert isinstance(lifetime, asyncio.Task)
            if mode == "succeeded":
                assert start_results == [None]
                assert lifetime.get_name() == "served-control-lifetime"
                await watching.wait()
                assert disconnect_timeouts == [60]
                assert len(attached_sources) == 1
                assert callable(attached_sources[0])
                assert projection_requests == []
                assert attached_sources[0]() is source
                assert projection_requests == [identity]
            else:
                assert len(start_results) == 1
                assert isinstance(start_results[0], RuntimeError)
                assert str(start_results[0]) == "offline"
                assert lifetime.get_name() == "served-control-refused"
                await lifetime
                assert attached_sources == projection_requests == []
                assert disconnect_timeouts == []
        await app_module.TldwCli._stop_served_canvas_control(app)
        stopped = True
        await asyncio.sleep(0)
        assert task.done()
        assert task.cancelled() is (mode == "pending")
        assert app._served_canvas_control_start_task is None
        assert app._served_control_lifetime_task is None
        assert app._shutting_down is (mode == "failed")
        if lifetime is not None:
            assert lifetime.done()
            assert lifetime.cancelled() is (mode == "succeeded")
        assert len(owned_tasks) == (1 if mode == "pending" else 2)
        assert (
            calls
            == {
                "pending": [
                    "start",
                    "join_start",
                    "retired",
                    "publisher_closed",
                    "client_closed",
                ],
                "succeeded": [
                    "start",
                    "retired",
                    "attach",
                    "watching",
                    "join_lifetime",
                    "lifetime_retired",
                    "publisher_closed",
                    "client_closed",
                ],
                "failed": [
                    "start",
                    "retired",
                    ("gate_closed", "served control lost"),
                    "exit",
                    "publisher_closed",
                    "client_closed",
                ],
            }[mode]
        )
        assert warnings == (
            [
                (
                    "Served Canvas control unavailable type={} code=connection_failed",
                    "RuntimeError",
                )
            ]
            if mode == "failed"
            else []
        )
    finally:
        set_app_global(monkeypatch, "asyncio", late_asyncio)
        try:
            if not stopped:
                await app_module.TldwCli._stop_served_canvas_control(app)
        finally:
            try:
                await asyncio.sleep(0)
                for owned in owned_tasks:
                    if not owned.done():
                        owned.cancel()
                if owned_tasks:
                    await asyncio.gather(*owned_tasks, return_exceptions=True)
                await asyncio.sleep(0)
            finally:
                loop.set_exception_handler(old_handler)
    assert loop_errors == []
