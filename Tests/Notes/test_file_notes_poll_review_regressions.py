"""File Notes polling must retain unvisited files and retry uncertain reads."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Notes import file_notes_service as service_module
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "private-vault-name"
    root.mkdir()
    replica = FileNotesReplica(":memory:")
    service = FileNotesService(root, replica)
    yield root, replica, service
    service.close()
    replica.close()


def test_depth_truncation_preserves_previously_opened_file(vault):
    root, replica, service = vault
    parent = root.joinpath(*["d" for _ in range(service_module.WALK_MAX_DEPTH + 1)])
    parent.mkdir(parents=True)
    target = parent / "note.md"
    target.write_text("retained", encoding="utf-8")
    relative = target.relative_to(root).as_posix()
    service.open_file(relative)
    assert [
        row.relative_path for row in replica.list_active_files(service.root_key)
    ] == [relative]

    for _ in range(2):
        result = service.reconcile()
        assert result.deleted == ()
        assert result.vault_unchanged is False
        assert [
            row.relative_path for row in replica.list_active_files(service.root_key)
        ] == [relative]
        assert replica.list_deleted(service.root_key) == []
    assert target.read_text(encoding="utf-8") == "retained"


def test_unreadable_file_recovers_without_size_or_mtime_change(vault):
    root, replica, service = vault
    target = root / "note.md"
    target.write_text("retained", encoding="utf-8")
    before = target.stat()
    target.chmod(0)
    try:
        first = service.reconcile()
        assert first.entries[0].read_only_reason == "unreadable"
        assert first.replica_warning is None
        target.chmod(0o600)
        assert target.stat().st_size == before.st_size
        assert target.stat().st_mtime_ns == before.st_mtime_ns

        recovered = service.reconcile()
        assert recovered.vault_unchanged is False
        assert recovered.entries[0].read_only_reason is None
        assert recovered.created == ("note.md",)
        assert len(replica.list_active_files(service.root_key)) == 1
        assert service.reconcile().vault_unchanged is True
    finally:
        target.chmod(0o600)


def test_uncertain_new_file_cannot_replay_previous_clean_projection(vault, monkeypatch):
    root, replica, service = vault
    (root / "existing.md").write_text("existing", encoding="utf-8")
    assert service.reconcile().created == ("existing.md",)
    assert service.reconcile().vault_unchanged is True
    target = root / "new.md"
    target.write_text("new", encoding="utf-8")
    real_lstat = service_module.os.lstat

    def inaccessible_file(path, *args, **kwargs):
        if path == target:
            raise PermissionError("temporarily unreadable")
        return real_lstat(path, *args, **kwargs)

    with monkeypatch.context() as current:
        current.setattr(service_module.os, "lstat", inaccessible_file)
        uncertain = service.reconcile()
    assert uncertain.vault_unchanged is False
    assert [
        (entry.relative_path, entry.read_only_reason) for entry in uncertain.entries
    ] == [
        ("existing.md", None),
        ("new.md", "unreadable"),
    ]
    recovered = service.reconcile()
    assert recovered.created == ("new.md",)
    assert len(replica.list_active_files(service.root_key)) == 2
    assert service.reconcile().vault_unchanged is True


@pytest.mark.parametrize("limit", ["depth", "files", "entries"])
def test_walk_limit_diagnostics_do_not_disclose_vault_path(vault, monkeypatch, limit):
    root, _replica, service = vault
    if limit == "depth":
        monkeypatch.setattr(service_module, "WALK_MAX_DEPTH", 1)
        nested = root / "private-child" / "deep"
        nested.mkdir(parents=True)
        (nested / "note.md").write_text("retained", encoding="utf-8")
    else:
        for index in range(3):
            (root / f"note-{index}.md").write_text("retained", encoding="utf-8")
        monkeypatch.setattr(
            service_module,
            "WALK_MAX_FILES" if limit == "files" else "WALK_MAX_ENTRIES",
            1,
        )
    diagnostics = []
    monkeypatch.setattr(
        service_module,
        "logger",
        SimpleNamespace(
            warning=lambda message, *args: diagnostics.append(message.format(*args))
        ),
    )
    service.reconcile()
    service.reconcile()
    assert len(diagnostics) == 1
    assert str(root) not in diagnostics[0]
    assert "private-vault-name" not in diagnostics[0]
    assert "private-child" not in diagnostics[0]
