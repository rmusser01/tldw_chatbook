"""Content-free Settings status for physical Console trace maintenance."""

from types import SimpleNamespace

from tldw_chatbook.UI.Screens.settings_privacy_security import (
    build_privacy_posture_rows,
    build_settings_privacy_posture,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def test_settings_reports_completed_compaction_byte_metrics_without_ids() -> None:
    secret_attempt = "attempt-private-identifier"
    posture = build_settings_privacy_posture(
        {},
        environ={},
        trace_maintenance={
            "status": "complete",
            "reason_code": "complete",
            "attempt_id": secret_attempt,
            "allocated_bytes_before": 8 * 1024 * 1024,
            "allocated_bytes_after": 2 * 1024 * 1024,
            "freelist_bytes_before": 6 * 1024 * 1024,
            "freelist_bytes_after": 0,
        },
    )

    rendered = "\n".join(build_privacy_posture_rows(posture))

    assert (
        "Trace physical maintenance: complete; allocated 8.0 MiB → 2.0 MiB, "
        "free 6.0 MiB → 0 B"
    ) in rendered
    assert secret_attempt not in rendered


def test_settings_reports_bounded_retry_and_running_status() -> None:
    pending = build_settings_privacy_posture(
        {},
        environ={},
        trace_maintenance={
            "status": "pending",
            "reason_code": "insufficient_disk",
            "retry_pending": True,
        },
    )
    running = build_settings_privacy_posture(
        {},
        environ={},
        trace_maintenance={
            "status": "running",
            "reason_code": "running",
            "progress_basis_points": 4250,
        },
    )

    assert "Trace physical maintenance: retry pending; insufficient disk" in (
        build_privacy_posture_rows(pending)
    )
    assert "Trace physical maintenance: compacting (42%)" in (
        build_privacy_posture_rows(running)
    )


def test_settings_sanitizes_unrecognized_database_diagnostics() -> None:
    secret = "/private/path secret-token"
    posture = build_settings_privacy_posture(
        {},
        environ={},
        trace_maintenance={
            "status": secret,
            "reason_code": secret,
            "allocated_bytes_before": "1234",
        },
    )

    rendered = "\n".join(build_privacy_posture_rows(posture))

    assert "Trace physical maintenance: unavailable" in rendered
    assert secret not in rendered


def test_settings_screen_reads_status_without_exposing_database_identity() -> None:
    class FakeDB:
        def get_console_trace_compaction_status(self) -> dict[str, object]:
            return {
                "status": "pending",
                "reason_code": "connections_busy",
                "retry_pending": True,
                "last_gc_request_id": "private-gc-id",
            }

    screen = SettingsScreen(
        SimpleNamespace(app_config={}, chachanotes_db=FakeDB())
    )

    rendered = "\n".join(screen._privacy_posture_rows())

    assert "Trace physical maintenance: retry pending; connections busy" in rendered
    assert "private-gc-id" not in rendered
