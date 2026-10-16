"""Original app wrappers retain their live patch and callback routes."""

from types import SimpleNamespace

import pytest

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
        citation_legacy_migration_service=SimpleNamespace(ready=True),
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
