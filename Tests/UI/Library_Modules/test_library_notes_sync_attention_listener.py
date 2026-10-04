"""The runtime-status listener that keeps Notes truthful (TASK-34000.2, fix round 1).

Review Important #2: a hold a WATCHER pass produced while the user was idle
reached the runtime but no Notes surface until the next interaction. The
bridge in ``library_notes_sync_attention`` must marshal onto the app thread,
coalesce a burst of publications into one refresh, and never raise into the
runtime. Pure unit tests over small fakes; the real-stack behaviour is pinned
by ``Tests/UI/test_library_notes_sync_attention_listener.py``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention as attention

#: ``bootstrap_profile`` for the same reason as ``test_library_quit_guard.py``:
#: a Tests/UI file run under the per-test profile redirect trips
#: ``RecoveryRequired: raw_source_selection_changed`` at setup (lessons-testing-evidence).
pytestmark = [pytest.mark.unit, pytest.mark.bootstrap_profile]


class _App:
    def __init__(self, *, same_thread: bool = True) -> None:
        self._thread_id = threading.get_ident() if same_thread else -1
        self.timers: list[tuple[float, object]] = []
        self.marshalled: list[tuple[object, tuple[object, ...]]] = []

    def set_timer(self, delay: float, callback) -> None:
        self.timers.append((delay, callback))

    def call_from_thread(self, callback, *args) -> None:
        self.marshalled.append((callback, args))


class _Host:
    def __init__(self, app: _App | None) -> None:
        self.app_instance = app
        self.scheduled: list[dict[str, object]] = []

    def run_worker(self, work, **kwargs) -> None:
        self.scheduled.append(kwargs)


def test_a_burst_of_publications_becomes_one_debounced_refresh() -> None:
    app = _App()
    host = _Host(app)
    listener = attention.LibraryNotesSyncAttentionListener(host)

    for _ in range(5):
        listener(SimpleNamespace(root_id="root-1", status="needs_attention"))

    assert len(app.timers) == 1, "one debounce per burst"
    assert app.timers[0][0] == attention.STATUS_LISTENER_DEBOUNCE_SECONDS
    assert host.scheduled == []
    app.timers[0][1]()
    assert len(host.scheduled) == 1
    assert host.scheduled[0]["exclusive"] is True
    assert host.scheduled[0]["group"] == "library_notes_sync_attention"

    # The next burst arms again.
    listener(SimpleNamespace(root_id="root-1", status="up_to_date"))
    assert len(app.timers) == 2


def test_a_publication_from_another_thread_is_marshalled_to_the_app() -> None:
    app = _App(same_thread=False)
    host = _Host(app)
    listener = attention.LibraryNotesSyncAttentionListener(host)

    listener(SimpleNamespace(root_id="root-1", status="failed"))

    assert app.timers == [], "nothing runs on the publishing thread"
    assert len(app.marshalled) == 1
    callback, args = app.marshalled[0]
    callback(*args)
    assert len(app.timers) == 1


def test_the_listener_never_raises_into_the_runtime() -> None:
    class _ExplodingApp(_App):
        def set_timer(self, delay: float, callback) -> None:
            raise RuntimeError("timer unavailable")

    attention.LibraryNotesSyncAttentionListener(_Host(None))(object())
    attention.LibraryNotesSyncAttentionListener(_Host(_ExplodingApp()))(object())
    attention.LibraryNotesSyncAttentionListener(SimpleNamespace())(object())


def test_ensure_registers_once_and_release_unregisters() -> None:
    added: list[object] = []
    removed: list[object] = []
    runtime = SimpleNamespace(
        add_status_listener=added.append, remove_status_listener=removed.append
    )
    host = _Host(_App())
    host.app_instance.notes_sync_runtime_owner = runtime

    attention.ensure_library_notes_sync_attention_listener(host)
    attention.ensure_library_notes_sync_attention_listener(host)
    assert len(added) == 2 and added[0] is added[1], (
        "the same listener object each time"
    )

    attention.release_library_notes_sync_attention_listener(host)
    assert removed == [added[0]]
    # Releasing twice, or a host that never registered, is a no-op.
    attention.release_library_notes_sync_attention_listener(host)
    attention.release_library_notes_sync_attention_listener(_Host(None))
    assert removed == [added[0]]


# --- TASK-32633 slice (N-03): the Notes side's signal, and Manage rows following


class _SignalRuntime:
    def __init__(self, *, hinted=("root-1",), raise_error: bool = False) -> None:
        self.hinted = hinted
        self.raise_error = raise_error
        self.asked: list[str] = []

    async def note_changed(self, note_id: str):
        self.asked.append(note_id)
        if self.raise_error:
            raise RuntimeError("runtime closed")
        return self.hinted


@pytest.mark.asyncio
async def test_signal_library_note_lasting_sync_hints_the_runtime_and_never_raises() -> (
    None
):
    """One seam for the Notes side's writes that bypass the editor port."""

    runtime = _SignalRuntime()
    host = _Host(_App())
    host.app_instance.notes_sync_runtime_owner = runtime
    assert await attention.signal_library_note_lasting_sync(host, "note-1") == (
        "root-1",
    )
    assert runtime.asked == ["note-1"]

    # No runtime yet (boot-deferred), no note id, or a refusing runtime: the
    # write that just succeeded is never failed by its signal.
    assert (
        await attention.signal_library_note_lasting_sync(_Host(_App()), "note-1") == ()
    )
    assert await attention.signal_library_note_lasting_sync(host, "") == ()
    assert runtime.asked == ["note-1"]
    host.app_instance.notes_sync_runtime_owner = _SignalRuntime(raise_error=True)
    assert await attention.signal_library_note_lasting_sync(host, "note-2") == ()


class _SyncController:
    """The Manage sync folders projection, as the attention refresh sees it."""

    def __init__(self, rows) -> None:
        self._rows = iter(rows)
        self.snapshot = SimpleNamespace(roots=next(self._rows))
        self.calls: list[bool] = []

    def refresh_roots(self, *, publish: bool = True) -> None:
        self.calls.append(publish)
        if not publish:
            self.snapshot = SimpleNamespace(roots=next(self._rows, self.snapshot.roots))


def test_manage_sync_folders_rows_follow_a_publication_and_publish_only_a_change() -> (
    None
):
    """The rows re-project through the same listener seam; a no-change refresh
    publishes nothing, so the publication's own attention refresh cannot loop."""

    changed = _SyncController([("row-old",), ("row-new",)])
    host = SimpleNamespace(
        _library_notes_view="lasting_roots", _library_notes_sync_controller=changed
    )
    assert attention.refresh_manage_sync_folders_rows(host) is True
    assert changed.calls == [False, True]

    unchanged = _SyncController([("row-same",), ("row-same",)])
    host = SimpleNamespace(
        _library_notes_view="lasting_roots", _library_notes_sync_controller=unchanged
    )
    assert attention.refresh_manage_sync_folders_rows(host) is False
    assert unchanged.calls == [False]

    # Not on Manage sync folders: nothing is re-projected.
    elsewhere = _SyncController([("row-old",), ("row-new",)])
    host = SimpleNamespace(
        _library_notes_view="list", _library_notes_sync_controller=elsewhere
    )
    assert attention.refresh_manage_sync_folders_rows(host) is False
    assert elsewhere.calls == []
    assert attention.refresh_manage_sync_folders_rows(SimpleNamespace()) is False
