"""B10: the identity fallback hashes each discovered file once per pass.

The rename fallback in ``observe_root`` used to re-run
``stable_identity_digest`` over every discovered file once per path-missed
binding -- O(missed bindings x files) SHA-256 computations per pass, on the
event loop. An index built once on the first miss must compute each file's
digest exactly once while preserving the old scan's semantics exactly:
only a UNIQUE digest match resolves; zero matches, or two or more files
sharing one identity (e.g. hardlinks), resolve to nothing.

Every world here is real: a real temp sync root, a real ``CharactersRAGDB``,
the real device-state store, and the real Posix filesystem adapter.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

import tldw_chatbook.Notes.notes_sync_runtime as runtime_module
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import plan_reconciliation
from tldw_chatbook.Notes.notes_sync_runtime import _ProductionRuntimeAdapter

pytestmark = pytest.mark.unit

_USER = "test-user"
_FILES = 50
_BOUND = 10
_RENAMED = 3


def _content(index: int) -> str:
    return f"note body {index}\n"


def _rel(index: int) -> str:
    return f"note-{index:04d}.md"


def _moved(index: int) -> str:
    return f"moved-{index}.md"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _DbBackedLocalNotes:
    """The production local-notes call surface over a real DB."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def get_note_by_id(self, _user_id: str, note_id: str):
        return self._db.get_note_by_id(note_id)

    def get_note_version_states(self, _user_id: str, note_ids):
        return self._db.get_note_version_states(note_ids)

    def update_note(self, _user_id: str, note_id: str, data, expected_version: int):
        return self._db.update_note(note_id, data, expected_version)

    def soft_delete_note(self, _user_id: str, note_id: str, version: int):
        return self._db.soft_delete_note(note_id, version)

    def add_note(self, _user_id: str, title: str, content: str, note_id=None):
        return self._db.add_note(title, content, note_id=note_id)


@dataclass
class _World:
    root_dir: Path
    scope_service: NotesScopeService
    store: NotesDeviceStateStore
    root: NotesSyncRootRecord
    adapter: _ProductionRuntimeAdapter
    db: CharactersRAGDB

    def fresh_adapter(self) -> _ProductionRuntimeAdapter:
        """A cold adapter over the same durable state: the ground truth."""

        return _ProductionRuntimeAdapter(
            self.store,
            self.scope_service,
            local_user_id=_USER,
            recovery_capacity_bytes=64 * 1024 * 1024,
        )


@pytest.fixture()
def world(tmp_path: Path) -> _World:
    root_dir = (tmp_path / "sync-root").resolve()
    root_dir.mkdir(mode=0o700)
    db = CharactersRAGDB(tmp_path / "chachanotes.sqlite3", client_id=_USER)
    scope_service = NotesScopeService(
        _DbBackedLocalNotes(db),
        None,
        folder_repository=LocalNoteFolderRepository(db),
    )
    store = NotesDeviceStateStore(tmp_path / "device-state.sqlite3")
    store.initialize()
    root = NotesSyncRootRecord(
        root_id="root-1",
        note_scope_id="local_note",
        logical_folder_id="folder-1",
        canonical_path=str(root_dir),
        direction=NotesSyncDirection.BIDIRECTIONAL,
        state=NotesSyncRootState.ACTIVE,
    )
    store.create_root(root)

    fs = PosixNotesSyncFilesystem(root_dir)
    with fs:
        for index in range(_FILES):
            (root_dir / _rel(index)).write_text(_content(index), encoding="utf-8")
        for index in range(_BOUND):
            snapshot = fs.observe(_rel(index))
            db.add_note(f"Note {index}", _content(index), note_id=f"note-{index:04d}")
            store.create_binding(
                NotesSyncBindingRecord(
                    binding_id=f"binding-{index:04d}",
                    root_id="root-1",
                    note_scope_id="local_note",
                    note_id=f"note-{index:04d}",
                    normalized_relative_path=_rel(index),
                    stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                        snapshot
                    ),
                    state=NotesSyncBindingState.ACTIVE,
                    serialization=snapshot.observation.serialization,
                    content_digest=_digest(_content(index)),
                    note_version=1,
                )
            )

    # Three bound files are renamed on disk. Same inodes -> the identity
    # fallback must still bind them; the path lookup misses.
    for index in range(_RENAMED):
        (root_dir / _rel(index)).rename(root_dir / _moved(index))

    adapter = _ProductionRuntimeAdapter(
        store,
        scope_service,
        local_user_id=_USER,
        recovery_capacity_bytes=64 * 1024 * 1024,
    )
    built = _World(
        root_dir=root_dir,
        scope_service=scope_service,
        store=store,
        root=root,
        adapter=adapter,
        db=db,
    )
    yield built
    adapter.close()
    db.close_connection()


async def _observe_and_release(adapter: _ProductionRuntimeAdapter, root):
    request = await adapter.observe_root(root)
    adapter.release_observation(plan_reconciliation(request).observation_token)
    return request


async def test_rename_fallback_hashes_each_file_once_per_pass(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass over 50 files with 3 path misses: each file's digest is
    computed exactly once. Before the index, every missed binding re-hashed
    all 50 discovered files (3x each) and the matched file was hashed again
    downstream."""
    calls: list[str] = []
    real = NotesSyncExecutor.stable_identity_digest

    def counting_digest(item):
        calls.append(item.observation.relative_path)
        return real(item)

    monkeypatch.setattr(
        NotesSyncExecutor,
        "stable_identity_digest",
        staticmethod(counting_digest),
    )

    request = await world.adapter.observe_root(world.root)
    world.adapter.release_observation(
        plan_reconciliation(request).observation_token
    )

    per_file = Counter(calls)
    recomputed = {path: count for path, count in per_file.items() if count > 1}
    assert not recomputed, f"digests recomputed within one pass: {recomputed}"
    # Evidence bound: per-pass digest computations <= bindings + files.
    assert len(calls) <= _BOUND + _FILES, f"digest computations: {len(calls)}"


async def test_rename_fallback_still_resolves_unique_identities(
    world: _World,
) -> None:
    request = await _observe_and_release(world.adapter, world.root)

    by_binding = {item.binding_id: item for item in request.bindings}
    for index in range(_RENAMED):
        observed = by_binding[f"binding-{index:04d}"]
        assert observed.relative_path == _moved(index)
        assert observed.file_digest == _digest(_content(index))
        # The reused matched digest equals the baseline identity it matched.
        assert (
            observed.file_identity_digest == observed.baseline_identity_digest
        )
    # The untouched bindings keep their path binding.
    for index in range(_RENAMED, _BOUND):
        assert by_binding[f"binding-{index:04d}"].relative_path == _rel(index)


async def test_ambiguous_identity_still_resolves_to_nothing(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two discovered files claiming one identity digest must refuse to
    match, exactly as the old ``len(identity_matches) == 1`` guard did.

    The real Posix filesystem refuses multi-link files at discovery
    (``multiple_links``), so the ambiguity is staged at the index seam --
    which is also where the guard lives after B10.
    """
    index_of = 0  # binding-0000's disk file only matches by identity
    real_index_builder = runtime_module.identity_digest_index

    def ambiguous_index_builder(discovered):
        built = real_index_builder(discovered)
        bindings = {
            binding.normalized_relative_path: binding
            for binding in world.store.list_bindings(world.root.root_id)
        }
        digest = bindings[_rel(index_of)].stable_identity_digest
        decoy = next(
            items[0] for key, items in built.items() if key != digest
        )
        built[digest] = [built[digest][0], decoy]
        return built

    monkeypatch.setattr(
        runtime_module, "identity_digest_index", ambiguous_index_builder
    )

    request = await world.adapter.observe_root(world.root)
    world.adapter.release_observation(
        plan_reconciliation(request).observation_token
    )

    by_binding = {item.binding_id: item for item in request.bindings}
    observed = by_binding[f"binding-{index_of:04d}"]
    assert observed.file_digest is None
    assert observed.relative_path == _rel(index_of)  # baseline path, unresolved
    assert observed.file_identity_digest is None


async def test_cold_adapter_reads_the_same_world(world: _World) -> None:
    """The fixture's pass equals a cold adapter's pass over the same state."""

    warm = await _observe_and_release(world.adapter, world.root)
    cold = world.fresh_adapter()
    cold_request = await _observe_and_release(cold, world.root)
    cold.close()
    assert warm == cold_request


def test_identity_digest_index_groups_ambiguous_identities(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    (root / "a.md").write_text("same bytes\n", encoding="utf-8")
    (root / "c.md").write_text("other bytes\n", encoding="utf-8")
    with PosixNotesSyncFilesystem(root) as filesystem:
        first = filesystem.observe("a.md")
        third = filesystem.observe("c.md")
    # A second discovered file carrying a.md's identity (device+inode) under
    # another path -- the ambiguity a hardlink pair would present. (The real
    # filesystem refuses multi-link files, so this is built by copying the
    # frozen observation.)
    second = replace(
        first, observation=replace(first.observation, relative_path="b.md")
    )
    assert NotesSyncExecutor.stable_identity_digest(first) == (
        NotesSyncExecutor.stable_identity_digest(second)
    )

    index = runtime_module.identity_digest_index(
        {"a.md": first, "b.md": second, "c.md": third}
    )
    shared = NotesSyncExecutor.stable_identity_digest(first)
    assert index[shared] == [first, second]
    # A unique identity still yields a single-entry list.
    assert index[NotesSyncExecutor.stable_identity_digest(third)] == [third]
