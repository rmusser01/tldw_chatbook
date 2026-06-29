"""UI responsiveness diagnostic monitor tests."""

from types import SimpleNamespace

from tldw_chatbook.Utils.ui_responsiveness import UIResponsivenessMonitor


def test_responsiveness_snapshot_records_timers_workers_and_mount_churn():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=100)

    monitor.record_timer_created("footer-token")
    monitor.record_worker_started("console-sync")
    monitor.record_mounts("console-tabs", mounted=3, removed=1)
    monitor.record_worker_finished("console-sync")

    snapshot = monitor.snapshot()

    assert snapshot.active_timers == 1
    assert snapshot.active_workers == 0
    assert snapshot.mounts == 3
    assert snapshot.removes == 1
    assert "timers=1" in snapshot.format_status_line()


def test_responsiveness_snapshot_marks_event_loop_stall():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=50)

    monitor.record_heartbeat_delta(0.075)

    snapshot = monitor.snapshot()

    assert snapshot.max_heartbeat_lag_ms == 75
    assert snapshot.stalled is True


def test_responsiveness_status_line_is_footer_safe():
    monitor = UIResponsivenessMonitor(enabled=True, stall_threshold_ms=50)
    monitor.record_timer_created("ui-heartbeat")
    monitor.record_heartbeat_delta(0.01)

    line = monitor.snapshot().format_status_line()

    assert "\n" not in line
    assert "UI diag: responsive" in line


def test_app_starts_responsiveness_monitor_with_heartbeat_timer(monkeypatch):
    from tldw_chatbook import app as app_module

    app = app_module.TldwCli.__new__(app_module.TldwCli)
    app.ui_responsiveness_monitor = None
    scheduled = []

    def fake_set_interval(seconds, callback):
        scheduled.append((seconds, callback))
        return object()

    monkeypatch.setattr(app, "set_interval", fake_set_interval)
    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        lambda section, key=None, default=None: True
        if (section, key) == ("diagnostics", "ui_responsiveness_enabled")
        else default,
    )

    app_module.TldwCli._start_ui_responsiveness_monitor(app)

    assert app.ui_responsiveness_monitor is not None
    assert app.ui_responsiveness_monitor.snapshot().active_timers == 1
    assert scheduled == [(1.0, app._record_ui_heartbeat)]


def test_app_resets_heartbeat_baseline_when_timer_starts(monkeypatch):
    from tldw_chatbook import app as app_module
    from tldw_chatbook.Utils import ui_responsiveness as responsiveness_module

    current_time = {"value": 0.0}
    monkeypatch.setattr(
        responsiveness_module.time,
        "perf_counter",
        lambda: current_time["value"],
    )

    app = app_module.TldwCli.__new__(app_module.TldwCli)
    app.ui_responsiveness_monitor = UIResponsivenessMonitor(
        enabled=True,
        stall_threshold_ms=250,
        heartbeat_interval_seconds=1.0,
    )
    app._ui_responsiveness_heartbeat_timer = None
    monkeypatch.setattr(app, "set_interval", lambda _seconds, _callback: object())

    current_time["value"] = 10.0
    app_module.TldwCli._start_ui_responsiveness_monitor(app)
    current_time["value"] = 11.0
    app_module.TldwCli._record_ui_heartbeat(app)

    snapshot = app.ui_responsiveness_monitor.snapshot()
    assert snapshot.max_heartbeat_lag_ms == 0
    assert snapshot.stalled is False


def test_app_stops_responsiveness_heartbeat_timer():
    from tldw_chatbook import app as app_module

    stopped = []

    class FakeTimer:
        def stop(self):
            stopped.append(True)

    app = app_module.TldwCli.__new__(app_module.TldwCli)
    app.ui_responsiveness_monitor = UIResponsivenessMonitor(enabled=True)
    app.ui_responsiveness_monitor.record_timer_created("ui-heartbeat")
    app._ui_responsiveness_heartbeat_timer = FakeTimer()

    app_module.TldwCli._stop_ui_responsiveness_monitor(app)

    assert stopped == [True]
    assert app._ui_responsiveness_heartbeat_timer is None
    assert app.ui_responsiveness_monitor.snapshot().active_timers == 0


def test_app_does_not_schedule_heartbeat_when_responsiveness_monitor_is_disabled(monkeypatch):
    from tldw_chatbook import app as app_module

    app = app_module.TldwCli.__new__(app_module.TldwCli)
    app.ui_responsiveness_monitor = None
    scheduled = []

    def fake_set_interval(seconds, callback):
        scheduled.append((seconds, callback))
        return object()

    monkeypatch.setattr(app, "set_interval", fake_set_interval)
    monkeypatch.setattr(
        app_module,
        "get_cli_setting",
        lambda section, key=None, default=None: False
        if (section, key) == ("diagnostics", "ui_responsiveness_enabled")
        else default,
    )

    app_module.TldwCli._start_ui_responsiveness_monitor(app)

    assert app.ui_responsiveness_monitor is not None
    assert app.ui_responsiveness_monitor.enabled is False
    assert app.ui_responsiveness_monitor.snapshot().active_timers == 0
    assert scheduled == []


def test_footer_status_scheduling_records_stable_timer_names():
    from tldw_chatbook import app as app_module

    monitor = UIResponsivenessMonitor(enabled=True)
    scheduled_once = []
    scheduled_periodic = []

    fake_app = SimpleNamespace(
        ui_responsiveness_monitor=monitor,
        query_one=lambda _selector: object(),
        loguru_logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        db_status_manager=SimpleNamespace(
            start_periodic_updates=lambda interval: scheduled_periodic.append(
                ("db", interval)
            )
        ),
        update_db_sizes=lambda: None,
        update_token_count_display=lambda: None,
        call_after_refresh=lambda callback: callback,
        set_timer=lambda delay, callback: scheduled_once.append((delay, callback)),
        set_interval=lambda interval, callback: scheduled_periodic.append(
            ("token", interval, callback)
        ),
        _db_size_status_widget=None,
        _token_count_update_timer=None,
    )

    app_module.TldwCli._schedule_footer_status_updates(fake_app)

    assert monitor.snapshot().active_timers == 2
    assert len(scheduled_once) == 2
    assert len(scheduled_periodic) == 2


def test_app_stops_footer_status_timers_and_diagnostics():
    from tldw_chatbook import app as app_module

    stopped = []

    class FakeTimer:
        def stop(self):
            stopped.append(True)

    app = app_module.TldwCli.__new__(app_module.TldwCli)
    app.ui_responsiveness_monitor = UIResponsivenessMonitor(enabled=True)
    app.ui_responsiveness_monitor.record_timer_created("footer-db-size-periodic")
    app.ui_responsiveness_monitor.record_timer_created("footer-token-periodic")
    app._token_count_update_timer = FakeTimer()

    app_module.TldwCli._stop_footer_status_timers(app)

    assert stopped == [True]
    assert app._token_count_update_timer is None
    assert app.ui_responsiveness_monitor.snapshot().active_timers == 0


def test_console_transcript_sync_timer_updates_responsiveness_monitor():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monitor = UIResponsivenessMonitor(enabled=True)
    stopped = []

    class FakeTimer:
        def stop(self):
            stopped.append(True)

    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(ui_responsiveness_monitor=monitor)
    screen._console_transcript_sync_timer = None
    screen.set_interval = lambda _interval, _callback: FakeTimer()

    ChatScreen._start_console_transcript_sync_timer(screen)

    assert monitor.snapshot().active_timers == 1

    ChatScreen._stop_console_transcript_sync_timer(screen)

    assert monitor.snapshot().active_timers == 0
    assert stopped == [True]


async def test_console_sync_records_worker_lifecycle():
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monitor = UIResponsivenessMonitor(enabled=True)
    screen = ChatScreen.__new__(ChatScreen)
    screen.app_instance = SimpleNamespace(ui_responsiveness_monitor=monitor)
    screen._console_sync_in_progress = False
    screen._console_sync_requested = False
    screen.run_worker = lambda *_args, **_kwargs: None

    def assert_worker_active():
        assert monitor.snapshot().active_workers == 1

    async def async_noop():
        return None

    screen._sync_console_chat_core_state = assert_worker_active
    screen._sync_console_session_draft = lambda: None
    screen._sync_console_control_bar = lambda: None
    screen._sync_console_settings_summary = lambda: None
    screen._sync_console_mode_bar = lambda: None
    screen._sync_console_native_session_tabs = async_noop
    screen._sync_console_workspace_context = lambda: None
    screen._sync_native_console_transcript_to_legacy_surface = async_noop
    screen._sync_console_rail_visibility = lambda _state: None
    screen._current_console_rail_state = lambda: object()

    await ChatScreen._sync_native_console_chat_ui(screen)

    assert monitor.snapshot().active_workers == 0


async def test_console_session_surface_records_tab_mount_churn():
    from tldw_chatbook.Widgets.Console.console_session_surface import ConsoleSessionSurface

    monitor = UIResponsivenessMonitor(enabled=True)

    class FakeChild:
        id = "console-new-chat-tab"

        async def remove(self):
            return None

    class FakeTabStrip:
        def __init__(self):
            self.children = [FakeChild()]

        async def mount(self, widget):
            self.children.append(widget)

    surface = ConsoleSessionSurface.__new__(ConsoleSessionSurface)
    surface.app_instance = SimpleNamespace(ui_responsiveness_monitor=monitor)
    surface._session_sync_lock = __import__("asyncio").Lock()
    tab_strip = FakeTabStrip()
    surface.query_one = lambda _selector, _expect_type=None: tab_strip

    await ConsoleSessionSurface.sync_sessions(
        surface,
        sessions=[SimpleNamespace(id="session-1", title="Console")],
        active_session_id="session-1",
    )

    snapshot = monitor.snapshot()
    assert snapshot.mounts == 3
    assert snapshot.removes == 1
