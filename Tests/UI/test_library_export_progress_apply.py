from types import SimpleNamespace
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


def _fake(run_id=7, running=True):
    calls = []
    fake = SimpleNamespace(
        # Task 4 cleanup: the screen's flat `_library_export_<field>` shim
        # is gone -- `_apply_library_export_progress`'s body now reads
        # `self._export_state.<field>`, so this fake nests its export
        # fields under `_export_state` (recipe §11's "unbound fake-self"
        # retarget precedent).
        _export_state=SimpleNamespace(
            run_id=run_id,
            running=running,
            status="",
        ),
        _refresh_library_export_status_line=lambda: calls.append("refresh"),
    )
    return fake, calls


def test_progress_apply_ignores_stale_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 3, "media", 5, 10)  # 3 != 7
    assert fake._export_state.status == ""
    assert calls == []


def test_progress_apply_ignores_when_not_running():
    fake, calls = _fake(run_id=7, running=False)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._export_state.status == ""
    assert calls == []


def test_progress_apply_updates_current_run():
    fake, calls = _fake(run_id=7)
    LibraryScreen._apply_library_export_progress(fake, 7, "media", 5, 10)
    assert fake._export_state.status == "Collecting media…  5/10"
    assert calls == ["refresh"]
