from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import os
from pathlib import Path

import pytest

from Tests.LLM_Management.snapshot_fixtures import commit_test_snapshot
from Tests.LLM_Management.snapshot_fixtures import test_evidence as evidence
from tldw_chatbook.LLM_Management.snapshot_models import (
    COMPATIBILITY_STATE_KEYS,
    SlotReceipt,
    SnapshotError,
)


@pytest.fixture
def store(tmp_path):
    assert importlib.util.find_spec("tldw_chatbook.LLM_Management.snapshot_store"), (
        "private snapshot store is not implemented"
    )
    from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore

    return SnapshotStore(tmp_path / "snapshots")


def _save(store, *, keep=10, payload=b"original", slot=0, model="Test model"):
    working = store.reserve_save("test-launch-a", slot)
    working.path.write_bytes(payload)
    receipt = SlotReceipt(
        slot_id=slot, filename=working.path.name, tokens=7, bytes=len(payload)
    )
    return store.commit_save(working, receipt, evidence(), model, keep)


def test_fixture_uses_complete_effective_evidence():
    actual = evidence()
    assert {key for key, _ in actual.state_settings} == COMPATIBILITY_STATE_KEYS
    assert dict(actual.state_settings)["effective-slot-contexts"] == "0:4096"


@pytest.mark.parametrize("raises", [False, True])
def test_publication_predicate_rejects_without_publishing_or_pruning(store, raises):
    old = _save(store).record
    working = store.reserve_save("test-launch-a", 0)
    working.path.write_bytes(b"new data")
    receipt = SlotReceipt(slot_id=0, filename=working.path.name, tokens=7, bytes=8)

    def reject():
        if raises:
            raise RuntimeError("private-validation-canary")
        return False

    with pytest.raises(SnapshotError, match="publication_invalidated") as raised:
        store.commit_save(
            working, receipt, evidence(), "Test model", 1, validate_publication=reject
        )
    assert "private-validation-canary" not in str(raised.value)
    assert store.list_records().records == (old,)
    assert working.path.read_bytes() == b"new data"


def test_reservation_is_private_and_only_commit_makes_it_visible(store):
    working = store.reserve_save("launch-a", 2)
    assert working.path.stat().st_mode & 0o777 == 0o600
    assert working.path.parent.stat().st_mode & 0o777 == 0o700
    assert store.list_records().records == ()
    record = commit_test_snapshot(store, payload=b"original", slot_id=2)
    assert store.list_records().records == (record,)
    assert record.sha256 == hashlib.sha256(b"original").hexdigest()
    assert store.list_records().stored_bytes == 8


def test_corrupt_snapshot_never_produces_restore_staging(store, tmp_path):
    record = commit_test_snapshot(store, payload=b"original", slot_id=0)
    binary = tmp_path / "snapshots" / "catalog" / record.filename
    binary.write_bytes(b"modified")
    with pytest.raises(SnapshotError, match="integrity_mismatch"):
        store.stage_restore(record.snapshot_id, "test-launch-b")
    assert binary.read_bytes() == b"modified"
    assert not list(
        (tmp_path / "snapshots" / "working" / "test-launch-b").glob("*.bin")
    )


def test_newest_count_crosses_models_and_ignores_clock_rollback(store, monkeypatch):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    monkeypatch.setattr(module, "_utc_now", lambda: "2026-09-04T12:00:00Z")
    first = _save(store, model="A").record
    second = _save(store, model="B").record
    monkeypatch.setattr(module, "_utc_now", lambda: "2020-01-01T00:00:00Z")
    result = _save(store, keep=2, model="C")
    assert result.removed_ids == (first.snapshot_id,)
    assert store.list_records().records == (result.record, second)
    assert len({first.filename, second.filename, result.record.filename}) == 3
    assert result.record.publication_sequence == 3
    last = _save(store, keep=1)
    assert store.list_records().records == (last.record,)


@pytest.mark.parametrize(
    "change",
    [
        {"tokens": 0},
        {"bytes": 0},
        {"bytes": 99},
        {"slot_id": 1},
        {"filename": "wrong.bin"},
    ],
)
def test_invalid_receipt_never_prunes(store, change):
    old = _save(store).record
    working = store.reserve_save("launch-a", 0)
    working.path.write_bytes(b"new")
    values = {"slot_id": 0, "filename": working.path.name, "tokens": 1, "bytes": 3}
    values.update(change)
    with pytest.raises(SnapshotError):
        store.commit_save(working, SlotReceipt(**values), evidence(), "Model", 1)
    assert store.list_records().records == (old,)


@pytest.mark.parametrize("failure", ["flush", "metadata", "durability"])
def test_failed_publication_never_prunes_last_snapshot(store, monkeypatch, failure):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    working = store.reserve_save("launch-a", 0)
    working.path.write_bytes(b"new")
    receipt = SlotReceipt(slot_id=0, filename=working.path.name, tokens=1, bytes=3)

    def fail(*args, **kwargs):
        raise OSError("injected disk failure")

    if failure == "flush":
        monkeypatch.setattr(module, "_flush_binary", fail)
    elif failure == "metadata":
        original = module.atomic_private_write_text

        def write(path, *args, **kwargs):
            if Path(path).parent.name == "catalog" and Path(path).suffix == ".json":
                fail()
            return original(path, *args, **kwargs)

        monkeypatch.setattr(module, "atomic_private_write_text", write)
    else:
        monkeypatch.setattr(module, "_sync_directory", fail)
    with pytest.raises(SnapshotError):
        store.commit_save(working, receipt, evidence(), "Model", 1)
    assert old in store.list_records().records


def test_failed_prune_reports_committed_success_and_reconciles_tombstone(
    store, monkeypatch, tmp_path
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    original = module._unlink_verified

    def fail(path, *args, **kwargs):
        if Path(path).name == old.filename:
            raise OSError("injected unlink failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_unlink_verified", fail)
        result = _save(store, keep=1)
    assert result.cleanup_failed_ids == (old.snapshot_id,)
    assert store.list_records().records == (result.record,)
    assert store.list_records().residual_bytes >= old.bytes
    assert store.reconcile(frozenset()) == ()
    assert not (tmp_path / "snapshots" / "catalog" / old.filename).exists()


@pytest.mark.parametrize(
    "kind", ["malformed", "oversize", "foreign", "symlink", "hardlink"]
)
def test_foreign_catalog_entries_are_not_adopted_or_removed(store, tmp_path, kind):
    catalog = tmp_path / "snapshots" / "catalog"
    outside = tmp_path / "outside"
    outside.write_bytes(b"foreign bytes")
    entry = catalog / ("f" * 32 + ".json")
    if kind == "symlink":
        entry.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, entry)
    elif kind == "oversize":
        entry.write_bytes(b" " * (65536 + 1))
    else:
        entry.write_text("{}" if kind == "malformed" else "foreign")
    before = entry.lstat()
    _save(store, keep=1)
    _save(store, keep=1)
    store.reconcile(frozenset())
    assert entry.exists()
    assert entry.lstat().st_mode == before.st_mode
    assert outside.read_bytes() == b"foreign bytes"


def test_queued_working_directory_swap_never_traverses_external_files(
    store, tmp_path, monkeypatch
):
    working = store.reserve_save("launch-a", 0)
    external = tmp_path / "external"
    external.mkdir()
    (external / "private.bin").write_bytes(b"external payload")
    original = store._entries

    def swap(directory):
        entries = original(directory)
        if directory == store.working:
            working.path.parent.rename(tmp_path / "moved-launch")
            working.path.parent.symlink_to(external, target_is_directory=True)
        return entries

    monkeypatch.setattr(store, "_entries", swap)
    page = store.list_records()
    assert page.residual_bytes != len(b"external payload")
    assert (external / "private.bin").read_bytes() == b"external payload"


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "replacement"])
def test_changed_reserved_identity_is_never_published(store, tmp_path, kind):
    working = store.reserve_save("launch-a", 0)
    outside = tmp_path / "outside"
    outside.write_bytes(b"original")
    working.path.unlink()
    if kind == "symlink":
        working.path.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, working.path)
    else:
        working.path.write_bytes(b"original")
    with pytest.raises(SnapshotError):
        store.commit_save(
            working,
            SlotReceipt(slot_id=0, filename=working.path.name, tokens=1, bytes=8),
            evidence(),
            "Model",
            1,
        )
    assert store.list_records().records == ()
    assert outside.read_bytes() == b"original"


def test_pagination_prunes_the_complete_catalog_not_just_first_page(store):
    for _ in range(54):
        _save(store, keep=100)
    page = store.list_records()
    assert len(page.records) == 50
    assert page.next_offset == 50
    assert len(store.list_records(offset=50).records) == 4
    result = _save(store, keep=1)
    assert len(result.removed_ids) == 54
    assert store.list_records().records == (result.record,)


def test_incomplete_scan_keeps_old_records_and_reports_unknown_totals(
    store, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    monkeypatch.setattr(module, "MAX_SCAN_ENTRIES", 1)
    result = _save(store, keep=1)
    assert "cleanup_incomplete" in result.cleanup_failed_ids
    page = store.list_records()
    assert not page.scan_complete
    assert page.stored_bytes is None and page.residual_bytes is None
    monkeypatch.undo()
    assert old in store.list_records().records


@pytest.mark.parametrize("counter", ["missing", "corrupt", "nested"])
def test_incomplete_scan_cannot_recover_sequence_counter(
    store, tmp_path, monkeypatch, counter
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    counter_path = tmp_path / "snapshots" / "publication.json"
    if counter == "missing":
        counter_path.unlink()
    elif counter == "nested":
        counter_path.write_text("[" * 10000 + "0" + "]" * 10000)
    else:
        counter_path.write_text("{}")
    monkeypatch.setattr(module, "MAX_SCAN_ENTRIES", 1)
    with pytest.raises(SnapshotError, match="ordering_unavailable"):
        _save(store, keep=1)
    monkeypatch.undo()
    assert store.list_records().records == (old,)
    assert _save(store).record.publication_sequence == 2


def test_deep_counter_recovers_only_from_complete_catalog(store):
    old = _save(store).record
    (store.root / "publication.json").write_text("[" * 10000 + "0" + "]" * 10000)
    result = _save(store)
    assert result.record.publication_sequence == 2
    assert old in store.list_records().records


@pytest.mark.parametrize("state", ["reserved", "acknowledged", "terminal", "unknown"])
def test_cleanup_obeys_operation_state_and_reports_residuals(store, state):
    working = store.reserve_save("launch-a", 0)
    working.path.write_bytes(b"residual")
    if state != "reserved":
        store.set_operation_state(working, state)
    failures = store.cleanup(working)
    assert working.path.exists() == (state == "unknown")
    assert failures == (("operation_unknown",) if state == "unknown" else ())
    if state == "unknown":
        assert store.list_records().residual_bytes >= 8
        assert store.reconcile(frozenset()) == ()
        assert working.path.exists()
        assert store.reconcile(frozenset({"launch-a"})) == ()
        assert not working.path.exists()


def test_empty_acknowledged_save_releases_only_on_owner_cleanup(store):
    working = store.reserve_save("launch-a", 0)
    store.set_operation_state(working, "acknowledged")
    with pytest.raises(SnapshotError):
        store.commit_save(
            working,
            SlotReceipt(slot_id=0, filename=working.path.name, tokens=0, bytes=0),
            evidence(),
            "Model",
            1,
        )
    store.reconcile(frozenset())
    assert working.path.exists()
    assert store.cleanup(working) == ()
    assert not working.path.exists()


def test_repeated_acknowledged_restore_cycles_leave_no_staged_binaries(store, tmp_path):
    record = _save(store).record
    for _ in range(3):
        working = store.stage_restore(record.snapshot_id, "launch-b")
        assert working.path.read_bytes() == b"original"
        source = tmp_path / "snapshots" / "catalog" / record.filename
        assert working.path.stat().st_ino != source.stat().st_ino
        store.set_operation_state(working, "unknown")
        store.set_operation_state(working, "acknowledged")
        assert store.cleanup(working) == ()
    assert not list((tmp_path / "snapshots" / "working" / "launch-b").iterdir())


@pytest.mark.parametrize("failure", ["short", "full", "changed"])
def test_restore_staging_failure_cleans_owned_partial_and_retains_source(
    store, tmp_path, monkeypatch, failure
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    record = _save(store).record
    original = module._write_chunk

    def write(stream, chunk):
        if failure == "short":
            return 0
        if failure == "full":
            raise OSError("disk full")
        result = original(stream, chunk)
        member = next((tmp_path / "snapshots" / "working" / "launch-b").glob("*.bin"))
        replacement = member.with_suffix(".replacement")
        replacement.write_bytes(b"unowned")
        replacement.replace(member)
        return result

    monkeypatch.setattr(module, "_write_chunk", write)
    with pytest.raises(SnapshotError):
        store.stage_restore(record.snapshot_id, "launch-b")
    assert store.list_records().records == (record,)
    remaining = list((tmp_path / "snapshots" / "working" / "launch-b").glob("*.bin"))
    if failure == "changed":
        assert remaining[0].read_bytes() == b"unowned"
    else:
        assert remaining == []


def test_cleanup_failure_is_visible_and_terminal_reconciliation_retries(
    store, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    working = store.reserve_save("launch-a", 0)
    working.path.write_bytes(b"residual")
    store.set_operation_state(working, "terminal")

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_unlink_verified", fail)
        assert store.cleanup(working) == ("cleanup_failed",)
    assert store.list_records().residual_bytes >= 8
    assert store.reconcile(frozenset()) == ()
    assert not working.path.exists()


def _publisher(root, entered, release):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    store = module.SnapshotStore(root)
    original = module._flush_binary

    def flush(stream):
        entered.set()
        if not release.wait(15):
            raise RuntimeError("test gate expired")
        return original(stream)

    module._flush_binary = flush
    _save(store, keep=1)


def _delete_and_reconcile(root, snapshot_id, started, finished):
    from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore

    store = SnapshotStore(root)
    started.set()
    assert store.reconcile(frozenset()) == ()
    assert store.delete(snapshot_id) == ()
    finished.set()


def test_two_process_publication_blocks_deletion_and_acknowledged_reaping(
    store, tmp_path
):
    old = _save(store).record
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    entered, release, started, finished = [manager.Event() for _ in range(4)]
    root = tmp_path / "snapshots"
    publisher = ctx.Process(target=_publisher, args=(root, entered, release))
    deleter = ctx.Process(
        target=_delete_and_reconcile, args=(root, old.snapshot_id, started, finished)
    )
    publisher.start()
    try:
        assert entered.wait(15)
        deleter.start()
        assert started.wait(15)
        assert not finished.wait(0.2)
        release.set()
        publisher.join(15)
        deleter.join(15)
        assert publisher.exitcode == 0 and deleter.exitcode == 0
        assert len(store.list_records().records) == 1
        assert store.list_records().records[0].snapshot_id != old.snapshot_id
    finally:
        release.set()
        for process in (publisher, deleter):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(5)
        manager.shutdown()


def test_staged_bytes_are_verified_even_if_write_returns_correct_count(
    store, tmp_path, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    record = _save(store).record

    def corrupt(stream, chunk):
        return os.write(stream.fileno(), b"x" * len(chunk))

    monkeypatch.setattr(module, "_write_chunk", corrupt)
    with pytest.raises(SnapshotError, match="integrity_mismatch"):
        store.stage_restore(record.snapshot_id, "launch-b")
    assert not list((tmp_path / "snapshots" / "working" / "launch-b").glob("*.bin"))


def test_failed_sidecar_owned_orphan_can_be_cleaned_without_adopting_foreign_files(
    store, monkeypatch, tmp_path
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    working = store.reserve_save("launch-a", 0)
    working.path.write_bytes(b"new")
    original = module.atomic_private_write_text

    def fail(path, *args, **kwargs):
        if Path(path).parent.name == "catalog":
            raise OSError("sidecar failed")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_private_write_text", fail)
        with pytest.raises(SnapshotError):
            store.commit_save(
                working,
                SlotReceipt(slot_id=0, filename=working.path.name, tokens=1, bytes=3),
                evidence(),
                "Model",
                1,
            )
    orphan = tmp_path / "snapshots" / "catalog" / working.path.name
    assert orphan.exists()
    assert store.cleanup(working) == ()
    assert not orphan.exists()
    assert store.list_records().records == (old,)


def test_prepare_launch_directory_is_private_and_confined_without_reserving_save(
    store, tmp_path
):
    prepared = store.prepare_launch_directory("launch-new")
    assert prepared == tmp_path / "snapshots" / "working" / "launch-new"
    assert prepared.stat().st_mode & 0o777 == 0o700
    assert list(prepared.iterdir()) == []
    with pytest.raises(SnapshotError):
        store.prepare_launch_directory("../outside")


def test_unreadable_catalog_member_disables_pruning_and_totals(
    store, monkeypatch, tmp_path
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    old = _save(store).record
    original = module._read_json

    def fail(path):
        if Path(path).name == f"{old.snapshot_id}.json":
            raise PermissionError("cannot obtain complete scan")
        return original(path)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_read_json", fail)
        result = _save(store, keep=1)
        assert result.cleanup_failed_ids == ("cleanup_incomplete",)
        assert store.list_records().stored_bytes is None
    assert old in store.list_records().records


def test_reconcile_failed_state_persistence_reports_cleanup_failure(store, monkeypatch):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    working = store.reserve_save("launch-a", 0)
    store.set_operation_state(working, "unknown")

    def fail(*args, **kwargs):
        raise OSError("cannot settle manifest")

    monkeypatch.setattr(module, "atomic_private_write_text", fail)
    assert store.reconcile(frozenset({"launch-a"})) == ("cleanup_failed",)
    assert working.path.exists()


def test_metadata_schema_and_receipt_limits_are_enforced_without_pruning(
    store, tmp_path
):
    import json

    record = _save(store).record
    metadata = tmp_path / "snapshots" / "catalog" / f"{record.snapshot_id}.json"
    contents = json.loads(metadata.read_text())
    contents["schema_version"] = 2
    metadata.write_text(json.dumps(contents))
    assert store.list_records().records == ()
    _save(store, keep=1)
    assert metadata.exists()
    assert (metadata.parent / record.filename).exists()


def test_truncated_record_remains_browsable_and_deletable_but_cannot_restore(
    store, tmp_path
):
    record = _save(store).record
    binary = tmp_path / "snapshots" / "catalog" / record.filename
    binary.write_bytes(b"short")
    assert store.list_records().records == (record,)
    with pytest.raises(SnapshotError, match="integrity_mismatch"):
        store.stage_restore(record.snapshot_id, "launch-b")
    assert binary.read_bytes() == b"short"
    assert store.delete(record.snapshot_id) == ()
    assert not binary.exists()


def test_unsupported_platform_fails_before_creating_snapshot_storage(
    tmp_path, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    monkeypatch.setattr(module, "_supported_platform", lambda: False, raising=False)
    root = tmp_path / "not-created"
    with pytest.raises(SnapshotError, match="unsupported_platform"):
        module.SnapshotStore(root)
    assert not root.exists()


def test_directory_parent_swap_after_validation_never_opens_external_tree(
    store, tmp_path, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    external = tmp_path / "external"
    (external / "catalog").mkdir(parents=True)
    (external / "catalog" / "sentinel").write_bytes(b"outside")
    original = module.secure_private_directory

    def swap(path, **kwargs):
        result = original(path, **kwargs)
        if path == store.catalog:
            store.root.rename(tmp_path / "moved-root")
            store.root.symlink_to(external, target_is_directory=True)
        return result

    monkeypatch.setattr(module, "secure_private_directory", swap)
    with pytest.raises((SnapshotError, OSError)), module._directory(store.catalog):
        pytest.fail("opened a catalog through a substituted ancestor symlink")


def test_cleanup_is_idempotent_after_successful_owner_release(store):
    working = store.reserve_save("launch-a", 0)
    assert store.cleanup(working) == ()
    assert store.cleanup(working) == ()


def _pending_tombstone(store):
    """Represent a real record after its deletion intent replaced its commit marker."""
    import json

    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    record = _save(store).record
    binary = store.catalog / record.filename
    info = binary.stat()
    envelope = {
        "record": record.model_dump(mode="json"),
        "device": info.st_dev,
        "inode": info.st_ino,
    }
    tombstone = store.catalog / f"{record.snapshot_id}.deleting"
    atomic_private_write_text(tombstone, json.dumps(envelope))
    (store.catalog / f"{record.snapshot_id}.json").unlink()
    return binary, tombstone, envelope


@pytest.mark.parametrize(
    "payload",
    ["null", "1", '[{"x":1}]', "true", '"text"', "[]", "[" * 10000 + "0" + "]" * 10000],
    ids=["null", "number", "array", "bool", "text", "empty", "deep"],
)
def test_nonobject_tombstone_is_preserved_and_reconciliation_continues(
    store, monkeypatch, payload
):
    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    malformed = store.catalog / ("0" * 32 + ".deleting")
    atomic_private_write_text(malformed, payload)
    binary, valid_tombstone, _ = _pending_tombstone(store)
    original = store._entries

    def malformed_first(directory):
        entries, complete = original(directory)
        return sorted(entries, key=lambda entry: entry != malformed), complete

    monkeypatch.setattr(store, "_entries", malformed_first)

    store.reconcile(frozenset())

    assert malformed.read_text() == payload
    assert not binary.exists()
    assert not valid_tombstone.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"tokens": 0},
        {"bytes": 0},
        {"publication_sequence": 0},
        {"source_slot": 99},
        {"created_utc": "invalid timestamp"},
        {"snapshot_id": "not-the-owned-id"},
        {"incomplete_compatibility": True},
    ],
)
def test_malformed_tombstone_record_never_authorizes_deletion(store, change):
    import json

    from tldw_chatbook.Utils.private_paths import atomic_private_write_text

    binary, tombstone, envelope = _pending_tombstone(store)
    if "incomplete_compatibility" in change:
        envelope["record"]["compatibility"]["state_settings"] = [["ctx-size", "4096"]]
    else:
        envelope["record"].update(change)
    payload = json.dumps(envelope)
    atomic_private_write_text(tombstone, payload)
    original_bytes = binary.read_bytes()
    original_inode = binary.stat().st_ino

    store.reconcile(frozenset())

    assert binary.read_bytes() == original_bytes
    assert binary.stat().st_ino == original_inode
    assert tombstone.read_text() == payload
