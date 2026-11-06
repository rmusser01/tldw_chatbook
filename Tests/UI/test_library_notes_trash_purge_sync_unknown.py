"""Delete forever asks lasting sync through the async resolver, never a peek.

PR #3423 review, I-1. Dev's ``e017592b44``/``ab43a0303b`` turned
``library_notes_sync_attention._runtime`` into a non-constructing peek: it
returns the app's ``_notes_sync_runtime_owner`` slot, which ``app.py`` always
creates as ``None``. The purge worker and the Trash page's kept-file read
still called it. So when the lasting-sync runtime FAILED to build (a checked
config or path read raised: ``_notes_sync_runtime_build_failure`` set, the
slot left ``None``), the purge was told "no lasting sync in this app" and
deleted for good a note a sync folder still held -- exactly what
``PURGE_REFUSED_SYNC_BOUND`` exists to prevent. The fix resolves the runtime
with ``_prepare_runtime``, the async readers' own resolver, whose
"could not say" answer refuses with ``PURGE_REFUSED_SYNC_UNKNOWN``.

Two boots of the real ``LibraryScreen`` with the production stylesheet over a
real ChaChaNotes database (the 8a seed: ``_NotesProfile`` plus 25
tombstones), with lasting sync's real device store holding the bindings:

- The runtime's build raises, through the app's real safe reader, while the
  newest tombstone has a live binding: Delete forever deletes nothing, the
  row is still soft-deleted, and the sync-unknown sentence is painted. Once
  the runtime builds (the setter clears the failure), the same press is the
  sync-bound refusal -- the seed's binding is the live one the purge would
  have violated.
- The happy path: a runtime built on first ask, with no live binding on the
  note (only a disconnected folder's kept file), still purges, and the
  confirm keeps its "file stays on disk" clause.

Base (``bd01c0e46b``): the first boot purges the bound note ("Deleted forever
· Deleted 01") and the first wait times out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.library_quit_guard_support import _record_toasts
from Tests.UI.test_library_notes_trash_pages_and_purges import (
    WIDE,
    _focused_id,
    _painted,
    _settled_on,
    _until,
)
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _DIGEST,
    _confirm_open,
    _delete_forever,
    _open_trash,
    _painted_in,
    _painted_toasts,
    _PurgeSeed,
    _title,
)
from Tests.UI.test_library_shell import LibraryProductionCSSHarness, _seed_conversations
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootRecord,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_scope_service import (
    PURGE_REFUSED_SYNC_BOUND,
    PURGE_REFUSED_SYNC_UNKNOWN,
)
from tldw_chatbook.Notes.notes_sync_models import NotesSyncSerializationProfile
from tldw_chatbook.Notes.notes_sync_runtime import build_notes_sync_runtime_owner
from tldw_chatbook.RAG_Search import ingestion_indexing
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library.library_notes_canvas import PURGE_KEPT_FILE_CLAUSE

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The newest tombstone: the Trash's first row, focused when it opens.
NEWEST = "Deleted 01"


def _bind(seed: _PurgeSeed, note_id: str, state: NotesSyncBindingState) -> None:
    """Record one binding for ``note_id``, as lasting sync's own store does."""
    store = NotesDeviceStateStore(seed.state_path)
    try:
        store.initialize()
        store.create_root(
            NotesSyncRootRecord(
                root_id="root-1",
                note_scope_id="local_note",
                logical_folder_id="folder-sync-1",
                canonical_path=str(seed.vault.resolve()),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.ACTIVE,
            )
        )
        store.create_binding(
            NotesSyncBindingRecord(
                binding_id="binding-1",
                root_id="root-1",
                note_scope_id="local_note",
                note_id=note_id,
                normalized_relative_path="deleted-01.md",
                stable_identity_digest=_DIGEST,
                state=state,
                serialization=NotesSyncSerializationProfile(
                    utf8_bom=False, newline="lf", final_newline=True, mode=0o644
                ),
                content_digest=_DIGEST,
                note_version=2,
            )
        )
    finally:
        store.close()


def _runtime_over(seed: _PurgeSeed):
    """The production runtime over the seed's private device store."""
    profile = seed.trash.profile
    return build_notes_sync_runtime_owner(
        notes_scope_service=profile.scope_service,
        cutover_admitted=True,
        profile_process_is_sole=True,
        database_path=seed.state_path,
        migrate_legacy=lambda: None,
        local_user_id="default_user",
        recovery_capacity_bytes=1024 * 1024,
    )


def _mounted(tmp_path: Path, monkeypatch):
    app = _build_test_app()
    _seed_conversations(app, [])
    seed = _PurgeSeed(tmp_path, special=False)
    profile = seed.trash.profile
    app.chachanotes_db = profile.db
    app.notes_scope_service = profile.scope_service
    app.notes_service = profile.interop
    return app, seed


def _host(app, monkeypatch):
    host = LibraryProductionCSSHarness(app, screen=LibraryScreen(app))
    # The Library notifies through ``app_instance``; route its toasts to the
    # running host so they paint.
    app.notify = host.notify
    return host, _record_toasts(monkeypatch, app)


@pytest.fixture
def index_removals(monkeypatch) -> list[str]:
    """The semantic-index removals a purge runs (no RAG store in a test)."""
    queued: list[str] = []
    monkeypatch.setattr(
        ingestion_indexing,
        "remove_purged_note_projection",
        lambda note_id: queued.append(note_id) or 1,
    )
    return queued


async def test_delete_forever_refuses_when_lasting_sync_failed_to_build(
    tmp_path, monkeypatch, index_removals
) -> None:
    app, seed = _mounted(tmp_path, monkeypatch)
    bound_id = seed.ids[NEWEST]
    _bind(seed, bound_id, NotesSyncBindingState.ACTIVE)

    def failed_build():
        raise OSError("synthetic: a checked lasting-sync path read failed")

    monkeypatch.setattr(app, "_construct_notes_sync_runtime_owner", failed_build)
    # The boot read on a profile with a sync folder (``on_mount``): the build
    # raises inside the real safe reader, which records it and leaves the
    # slot empty -- the TASK-21249 / PR #3084 state.
    assert app._read_notes_sync_runtime_owner_safe() is None
    assert app._notes_sync_runtime_build_failure is not None
    assert app.__dict__["_notes_sync_runtime_owner"] is None
    host, toasts = _host(app, monkeypatch)
    runtime = None
    try:
        # Toasts painted (``run_test`` turns them off by default).
        async with host.run_test(size=WIDE, notifications=True) as pilot:
            screen = await _open_trash(host, pilot, count=25)
            assert _title(screen) == NEWEST

            await _delete_forever(host, pilot)
            await _until(
                pilot,
                lambda: any(
                    PURGE_REFUSED_SYNC_UNKNOWN in toast
                    for toast in _painted_toasts(host)
                ),
                lambda: f"the sync-unknown refusal never painted: "
                f"{_painted_toasts(host)!r} (notified: {toasts!r})",
            )
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: host.screen is screen and _title(screen) == NEWEST,
                lambda: f"back on the refused row (focus: {_focused_id(screen)!r}, "
                f"toasts: {toasts!r}): {_painted(host)}",
            )
            assert toasts == [PURGE_REFUSED_SYNC_UNKNOWN], toasts
            assert NEWEST in seed.deleted()
            assert "notes.id" in seed.traces(bound_id)
            assert index_removals == []
            # Still the failed state: nothing rebuilt it behind the refusal.
            assert app._notes_sync_runtime_build_failure is not None
            assert app.__dict__["_notes_sync_runtime_owner"] is None

            # The seed's control: once lasting sync builds (the setter clears
            # the recorded failure), the same press is the sync-bound refusal.
            runtime = _runtime_over(seed)
            app.notes_sync_runtime_owner = runtime
            await _delete_forever(host, pilot)
            await _until(
                pilot,
                lambda: PURGE_REFUSED_SYNC_BOUND in toasts,
                lambda: f"the sync-bound refusal (toasts: {toasts!r})",
            )
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: host.screen is screen and _title(screen) == NEWEST,
                lambda: f"back on the bound row (focus: {_focused_id(screen)!r})",
            )
            assert toasts == [PURGE_REFUSED_SYNC_UNKNOWN, PURGE_REFUSED_SYNC_BOUND]
            assert NEWEST in seed.deleted()
            assert index_removals == []
    finally:
        if runtime is not None:
            await runtime.shutdown()


async def test_delete_forever_still_purges_under_a_ready_runtime_with_no_binding(
    tmp_path, monkeypatch, index_removals
) -> None:
    app, seed = _mounted(tmp_path, monkeypatch)
    note_id = seed.ids[NEWEST]
    # No live binding: a disconnected folder only keeps the note's file.
    _bind(seed, note_id, NotesSyncBindingState.DISCONNECTED)
    runtime = _runtime_over(seed)
    # Built on first ask, through the app's real lazy property.
    monkeypatch.setattr(app, "_construct_notes_sync_runtime_owner", lambda: runtime)
    host, toasts = _host(app, monkeypatch)
    try:
        async with host.run_test(size=WIDE, notifications=True) as pilot:
            screen = await _open_trash(host, pilot, count=25)
            assert _title(screen) == NEWEST

            await pilot.press("x")
            await _confirm_open(host, pilot)
            assert PURGE_KEPT_FILE_CLAUSE in _painted_in(host, ".dialog-message"), (
                _painted_in(host, ".dialog-message")
            )
            await pilot.press("tab")
            await _until(
                pilot,
                lambda: getattr(host.screen.focused, "id", "") == "confirm-button",
                "Delete forever to take focus",
            )
            await pilot.press("enter")
            await _settled_on(
                pilot,
                host,
                screen,
                lambda: f"Deleted forever · {NEWEST}" in toasts
                and _title(screen) == "Deleted 02",
                lambda: f"the purge landed (focus: {_focused_id(screen)!r}, "
                f"toasts: {toasts!r}): {_painted(host)}",
            )
            assert toasts == [f"Deleted forever · {NEWEST}"], toasts
            assert NEWEST not in seed.deleted()
            assert seed.traces(note_id) == {}
            assert index_removals == [note_id]
            assert app.__dict__["_notes_sync_runtime_owner"] is runtime
    finally:
        await runtime.shutdown()
