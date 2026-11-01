"""R142 local refusal mechanisms, never mocked positive source authority.

The parent-groups fixture mocks the upstream committed journal/manifest prefix.
These tests exercise new guard boundaries with real temporary journals, native
maintenance, descriptor custody and files. Genuine positives are separate.
"""

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_safety_parent_groups import local_members
from tldw_chatbook.Backup_Recovery import (
    journal,
    later_rollback,
    mobile_canonical_copy,
    publication,
)
from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation as preservation
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


def invoke(case):
    return preservation._mobile_safety_members(
        case.plan,
        case.journal,
        case.parent,
        case.prepared,
        case.receipt,
        case.session,
        case.archive,
        case.original,
        _TraceReadBudget(),
        Event(),
    )


@pytest.mark.parametrize(
    "fault",
    [
        "family",
        "terminal",
        "evidence_digest",
        "cipher",
        "receipt_archive",
        "receipt_manifest",
        "acquired_archive",
        "manifest_bytes",
        "source_locator",
        "self_journal",
        "directory_identity",
        "physical_lock",
    ],
)
def test_local_source_refusal_before_downstream_archive(tmp_path, monkeypatch, fault):
    native_matches = journal._matches
    holds = len(storage._failed_capture_holds)
    with local_members(tmp_path, monkeypatch) as case:
        reader = later_rollback._accounted_snapshot_evidence
        proof, digest, terminal = reader()
        reads = []

        def evidence(*args, **kwargs):
            reads.append(args[0])
            assert fault not in {
                "source_locator",
                "self_journal",
                "directory_identity",
                "physical_lock",
            }, "identity refusal must precede source SH reader"
            return (
                object() if fault == "family" else proof,
                "wrong" if fault == "evidence_digest" else digest,
                "mobile_rolled_back" if fault == "terminal" else terminal,
            )

        def unexpected(*args, **kwargs):
            pytest.fail("invalid source reached downstream archive verification")

        monkeypatch.setattr(later_rollback, "_accounted_snapshot_evidence", evidence)
        monkeypatch.setattr(
            mobile_canonical_copy, "_verify_canonical_archive", unexpected
        )
        if fault == "cipher":
            Path(proof.ciphertext.path).write_bytes(b"changed private cipher")
            monkeypatch.setattr(journal, "_matches", native_matches)
        elif fault == "receipt_archive":
            case.receipt.archive_digest = "wrong"
        elif fault == "receipt_manifest":
            case.receipt.manifest_digest = "wrong"
        elif fault == "acquired_archive":
            case.archive = replace(case.archive, digest="wrong")
        elif fault == "manifest_bytes":
            case.archive = replace(case.archive, manifest_bytes=b"changed manifest")
        elif fault == "source_locator":
            case.plan.local_snapshot = replace(
                case.plan.local_snapshot, operation_id="wrong"
            )
        elif fault == "self_journal":
            case.original = case.journal
            case.plan.local_snapshot = LocalSnapshotSource(
                case.journal.root.parent, case.journal.operation_id, digest
            )
        elif fault == "directory_identity":
            retained = case.original.root.with_name("retained-original-journal")
            case.original.root.rename(retained)
            case.original.root.mkdir(mode=0o700)
            lock = case.original.root / "journal.lock"
            lock.write_bytes((retained / "journal.lock").read_bytes())
            lock.chmod(0o600)
        elif fault == "physical_lock":
            old_lock = case.original.root / "journal.lock"
            old_lock.rename(case.original.root / "retained-original.lock")
            os.link(case.journal.root / "journal.lock", old_lock)
        message = (
            "mobile_safety_source_journal_changed"
            if fault
            in {"source_locator", "self_journal", "directory_identity", "physical_lock"}
            else "mobile_safety_source_changed"
        )
        with pytest.raises(ValueError, match=message):
            invoke(case)
        assert not case.session._scopes
        assert len(reads) == (0 if "journal_changed" in message else 1)
    assert not case.session._active and len(storage._failed_capture_holds) == holds


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "unheld",
        "unpublished",
        "foreign",
        "current_entry",
        "pending",
        "proposed",
    ],
)
def test_local_namespace_snapshot_refusals(tmp_path, monkeypatch, fault):
    with local_members(tmp_path, monkeypatch) as case:
        source_members = preservation._mobile_source_members

        def changed(*args, **kwargs):
            members, records, profiles, registry, budget = source_members(
                *args, **kwargs
            )
            registry = copy.deepcopy(registry)  # Negative snapshot fault only.
            entry = registry["unit.current-source"]
            if fault == "missing":
                del registry["unit.current-source"]
            elif fault == "unheld":
                registry["foreign.unheld"] = registry.pop("unit.current-source")
                case.prepared.publication.namespaces = ("foreign.unheld",)
            elif fault == "unpublished":
                case.prepared.publication.namespaces = ()
            elif fault == "foreign":
                foreign = tmp_path / "foreign"
                foreign.mkdir(mode=0o700)
                entry["roots"] = [str(foreign)]
            elif fault == "current_entry":
                entry["historical"] = ["changed-native-entry"]
            elif fault == "pending":
                entry["pending"] = "other-operation"
            elif fault == "proposed":
                entry["proposed"] = [str(tmp_path / "proposed")]
            return members, records, profiles, registry, budget

        monkeypatch.setattr(preservation, "_mobile_source_members", changed)
        with pytest.raises(ValueError, match="mobile_safety_source_binding_unverified"):
            invoke(case)
        assert not case.session._scopes
    assert not case.session._active


@pytest.mark.parametrize(
    "fault",
    ["inode", "type", "mode", "mtime", "write_overlap", "not_preserved", "stale_cover"],
)
def test_local_member_and_covering_root_refusals(tmp_path, monkeypatch, fault):
    from contextlib import contextmanager

    with local_members(tmp_path, monkeypatch) as case:
        leaf = case.files[0]
        if fault == "write_overlap":
            case.plan.restore = (("write", leaf),)
        elif fault == "not_preserved":
            case.plan.preserve = tuple(
                row for row in case.plan.preserve if row[0] != "file2"
            )
        elif fault in {"inode", "type", "mode", "mtime"}:
            if fault in {"inode", "type"}:
                leaf.rename(case.group / "retained-leaf")
                if fault == "inode":
                    leaf.write_bytes(b"\x01" * 2)
                    leaf.chmod(0o600)
                else:
                    leaf.mkdir(mode=0o700)
            elif fault == "mode":
                leaf.chmod(0o400)
            else:
                info = leaf.stat()
                os.utime(leaf, ns=(info.st_atime_ns, info.st_mtime_ns + 1000))
            if fault == "type":
                # Negative fixture: match the new directory inode so this case
                # reaches the native file-type predicate instead of inode first.
                observed = next(
                    row
                    for row in case.prepared.safety_sources
                    if row.logical_id == "file2"
                )
                info = leaf.stat()
                observed.source = observed.source.model_copy(
                    update={"device": info.st_dev, "inode": info.st_ino}
                )
            # Keep the unit parent snapshot current to isolate the stale LEAF
            # predicate. This mocked prefix is never committed authority.
            info = case.group.stat()
            meta = replace(case.members["group"].metadata, mtime_ns=info.st_mtime_ns)
            case.members["group"] = replace(case.members["group"], metadata=meta)
            case.records["group"].metadata = meta
            next(
                row for row in case.prepared.safety_sources if row.logical_id == "group"
            ).source = journal._Object.model_validate(
                publication._observe_safety_source(case.group)
            )
        else:
            native_pin = storage._CaptureFileDescriptors.pinned_directory
            pins = []

            @contextmanager
            def pin(custody, path):
                if path == case.root:
                    pins.append(path)
                    # Actual fixture has covering pin, package pin, group-parent
                    # pin/reopen, package reopen, then final covering reopen.
                    if len(pins) == 6:
                        case.root.rename(tmp_path / "retained-cover")
                        case.root.mkdir(mode=0o700)
                        (case.root / "foreign").write_bytes(b"retain")
                with native_pin(custody, path) as fd:
                    yield fd

            monkeypatch.setattr(
                storage._CaptureFileDescriptors, "pinned_directory", pin
            )
        expected = (
            "mobile_safety_source_not_preserved"
            if fault in {"write_overlap", "not_preserved"}
            else "mobile_safety_source_changed"
        )
        if fault == "stale_cover":
            expected = "mobile_safety_source_binding_unverified"
        with pytest.raises(ValueError, match=expected):
            invoke(case)
        assert not case.session._scopes
        if fault == "stale_cover":
            assert len(pins) == 6
            assert (case.root / "foreign").read_bytes() == b"retain"
            assert (
                tmp_path / "retained-cover" / "expressions" / leaf.name
            ).read_bytes() == b"\x01" * 2
    assert not case.session._active


def test_native_nonpackage_outside_original_profile_still_refuses(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import replacement

    def unexpected(*args, **kwargs):
        pytest.fail("nonpackage source reached alternate builtin proof")

    with replacement_case(tmp_path, monkeypatch) as (_, plan, current, session, _, _):
        # Negative containment seam only; actual native prepared/journal/session
        # checks run. No fabricated record or destination can be accepted.
        monkeypatch.setattr(storage, "_contains_owned_path", lambda *a, **k: False)
        monkeypatch.setattr(preservation, "_mobile_safety_members", unexpected)
        budget = _TraceReadBudget()
        with pytest.raises(ValueError, match="rollback_source_binding_unverified"):
            replacement._checked_originals(
                plan, current, session, _source_read_budget=budget
            )
        assert budget._remaining == 16 * 1024**2


@pytest.mark.parametrize(
    "fault", ["owner", "extra_member", "historical", "parent_closure"]
)
def test_pure_mapping_rejects_missing_r142_member_shapes(fault):
    from Tests.Backup_Recovery.test_archive_reader import manifest
    from tldw_chatbook.Backup_Recovery.archive_models import ArchiveManifest
    from tldw_chatbook.Backup_Recovery.models import (
        FileMetadata,
        Inventory,
        StorageItem,
    )
    from tldw_chatbook.Persona_Visual.recovery import recovery_adapters

    owner = "persona.visual_identity_builtin"
    package = next(
        item for item in recovery_adapters() if item.owner_id == owner
    )._root({})
    root = "profile:profile:" + owner
    doc = manifest()
    doc["owners"][0]["owner_id"] = owner
    doc["directories"][0].update(logical_id=root, root_id=root, synthetic=False)
    doc["files"][0].update(
        root_id=root,
        parent_id=root,
        owner_id=owner,
        metadata={"version": 1, "mtime_ns": 0, "mode": 0o600},
    )
    doc["producer_inventory"] = [
        {
            "logical_id": root,
            "owner_id": owner,
            "status": "included_directory",
            "dependencies": [],
            "shared_group": None,
        },
        {
            "logical_id": "file",
            "owner_id": owner,
            "status": "included",
            "dependencies": [],
            "shared_group": None,
        },
    ]
    document = ArchiveManifest.model_validate_json(json.dumps(doc))
    root_item = StorageItem(
        owner,
        root,
        package,
        "included_directory",
        (),
        metadata=FileMetadata(1, root, "", None, "directory", 0o700, 0, "external"),
    )
    leaf = StorageItem(
        owner,
        "file",
        package / "note.txt",
        "included",
        (),
        metadata=FileMetadata(1, root, "note.txt", root, "file", 0o600, 0, "external"),
    )
    items = (root_item, leaf)
    if fault == "owner":
        items = (root_item, replace(leaf, owner="foreign.owner"))
    elif fault == "extra_member":
        items += (
            replace(
                leaf,
                logical_id="unarchived",
                path=package / "other",
                metadata=replace(leaf.metadata, relative_path="other"),
            ),
        )
    elif fault == "historical":
        document = document.model_copy(
            update={
                "directories": (
                    document.directories[0].model_copy(update={"synthetic": True}),
                )
            }
        )
    else:
        items = (
            root_item,
            replace(leaf, metadata=replace(leaf.metadata, parent_id="missing")),
        )
        document = document.model_copy(
            update={
                "files": (
                    document.files[0].model_copy(update={"parent_id": "missing"}),
                )
            }
        )
    expected = (
        "mobile_canonical_builtin_history_unsupported"
        if fault == "historical"
        else "mobile_canonical_builtin_members_changed"
    )
    # Pure shape checks only, no package reads, discovery, capture or authority.
    with pytest.raises(ValueError, match=expected):
        preservation._ordinary_builtin_members(
            document, Inventory(items, True, "unit-shape", ()), _TraceReadBudget()
        )


def test_native_prepared_descriptor_tamper_refuses_before_alternative(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import replacement

    def unexpected(*args, **kwargs):
        pytest.fail("changed native descriptor reached alternate source proof")

    with replacement_case(tmp_path, monkeypatch) as (_, plan, current, session, _, _):
        with current._locked(exclusive=False) as parent:
            records = current._records(parent)
        receipt = replacement._CandidateReceipt.model_validate(records[0].evidence)
        descriptor = Path(receipt.descriptor.path)
        retained = descriptor.with_name("retained-descriptor-before-negative")
        retained.write_bytes(descriptor.read_bytes())
        retained.chmod(0o600)
        descriptor.write_bytes(retained.read_bytes() + b" ")
        monkeypatch.setattr(preservation, "_mobile_safety_members", unexpected)
        budget = _TraceReadBudget()
        with pytest.raises(ValueError, match="candidate_receipt_changed"):
            replacement._checked_originals(
                plan, current, session, _source_read_budget=budget
            )
        assert descriptor.read_bytes() == retained.read_bytes() + b" "
        assert budget._remaining == 16 * 1024**2


def test_fixed_source_prefix_rejects_unregistered_failed_wrong_custody(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case

    with replacement_case(tmp_path, monkeypatch) as (_, plan, current, session, _, _):
        # Only malformed input refusal: this uncommitted source never reaches a
        # reader or issues a grant, and the session itself is actual native.
        plan = replace(
            plan,
            local_snapshot=LocalSnapshotSource(
                current.root.parent, current.operation_id, "unit"
            ),
        )
        proof = mobile_canonical_copy._CanonicalCopy(
            SimpleNamespace(manifest_digest="unit"), "unit", False, None
        )
        failed = storage._CaptureFileDescriptors(None)
        failed.close_failed = True
        unregistered = storage._CaptureFileDescriptors(None)
        budget = _TraceReadBudget()
        for custody in (object(), failed, unregistered):
            with pytest.raises(ValueError, match="mobile_canonical_custody_changed"):
                preservation._mobile_source_members(
                    plan,
                    proof,
                    None,
                    "unit",
                    session,
                    _read_budget=budget,
                    _custody=custody,
                )
        assert budget._remaining == 16 * 1024**2
        assert not failed.fds and not unregistered.fds
