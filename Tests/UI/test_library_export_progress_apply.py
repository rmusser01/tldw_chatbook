from types import SimpleNamespace
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _fake(run_id=7, running=True):
    calls = []
    fake = SimpleNamespace(
        _library_export_run_id=run_id,
        _library_export_running=running,
        _library_export_status="",
        _refresh_library_export_status_line=lambda: calls.append("refresh"),
    )
    return fake, calls


def test_progress_apply_ignores_stale_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 3, "media", 5, 10)  # 3 != 7
    assert fake._library_export_status == ""
    assert calls == []


def test_progress_apply_ignores_when_not_running():
    fake, calls = _fake(run_id=7, running=False)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._library_export_status == ""
    assert calls == []


def test_progress_apply_updates_current_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._library_export_status == "Collecting media…  5/10"
    assert calls == ["refresh"]
