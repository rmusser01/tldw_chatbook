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
@pytest.mark.parametrize("fails", [False, True])
async def test_served_control_task_retirement_preserves_late_asyncio_and_close_order(
    monkeypatch, fails
):
    calls, warnings, loop_errors = [], [], []
    loop = asyncio.get_running_loop()
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    release = asyncio.Event()

    class WrongCancellation(Exception):
        pass

    async def start():
        calls.append("start")
        try:
            if fails:
                raise RuntimeError("offline")
            await release.wait()
        finally:
            calls.append("retired")

    async def close_publisher():
        assert app._served_canvas_control_start_task is None
        calls.append("publisher_closed")

    async def close_client():
        calls.append("client_closed")

    async def join(*tasks, return_exceptions):
        assert app._served_canvas_control_start_task is None
        calls.append("join")
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)

    source = object()
    app = SimpleNamespace(
        served_canvas_control=SimpleNamespace(start=start, aclose=close_client),
        served_relay_publisher=SimpleNamespace(
            attach=lambda *, run_state_source: calls.append(
                ("attach", run_state_source)
            ),
            aclose=close_publisher,
        ),
        _served_relay_run_states=source,
        _served_canvas_control_start_task=None,
        loguru_logger=SimpleNamespace(warning=lambda *args: warnings.append(args)),
    )
    # A captured early namespace cannot classify the later real cancellation.
    set_app_global(
        monkeypatch,
        "asyncio",
        SimpleNamespace(
            create_task=asyncio.create_task,
            Task=asyncio.Task,
            CancelledError=WrongCancellation,
        ),
    )
    task = None
    try:
        app_module.TldwCli._start_served_canvas_control(app)
        task = app._served_canvas_control_start_task
        assert isinstance(task, asyncio.Task)
        assert task.get_name() == "start_served_canvas_control"
        await asyncio.sleep(0)
        assert calls[:2] == [("attach", source), "start"]
        set_app_global(
            monkeypatch,
            "asyncio",
            SimpleNamespace(
                Task=asyncio.Task, CancelledError=asyncio.CancelledError, gather=join
            ),
        )
        await app_module.TldwCli._stop_served_canvas_control(app)
        await asyncio.sleep(0)
        assert task.done()
        assert calls == (
            [
                ("attach", source),
                "start",
                "retired",
                "publisher_closed",
                "client_closed",
            ]
            if fails
            else [
                ("attach", source),
                "start",
                "join",
                "retired",
                "publisher_closed",
                "client_closed",
            ]
        )
        assert warnings == (
            [
                (
                    "Served Canvas control unavailable type={} code=connection_failed",
                    "RuntimeError",
                )
            ]
            if fails
            else []
        )
        assert loop_errors == []
    finally:
        task = task or app._served_canvas_control_start_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(old_handler)
