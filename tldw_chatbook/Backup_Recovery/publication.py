"""Locally prepared, journaled native publication; never releases admission."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from threading import Event

from . import archive_reader as reader
from .admission import Admission, fcntl
from .bootstrap import _key, _overlap, _read, _records, _registry
from .journal import (
    _CandidateReceipt,
    _matches,
    _Prepared,
    _PublicationContext,
    _Rollback,
    _states,
    observe_artifact,
)
from .limits import ArchiveLimits
from .native_files import (
    _rename_new,
    create_private_directory,
    flush_directory,
    pinned_directory,
    publish_new,
)
from .qualification import _qualified_identity, native_identity
from .restore_plan import RestorePlan, recheck_targets


def _plan_digest(plan):
    if type(plan) is not RestorePlan:
        raise ValueError("publication_plan_invalid")
    value = {
        "archive_digest": plan.archive_digest,
        "target_fingerprint": plan.target_fingerprint,
        "mode": plan.mode,
        "paths": {
            name: [(key, str(path)) for key, path in getattr(plan, name)]
            for name in ("restore", "retire", "preserve", "destinations", "selectors")
        },
        "containers": [
            (key, str(path)) for key, path in getattr(plan, "containers", ())
        ],
        "profile_names": plan.profile_names,
        "metadata": [
            (key, desired.model_dump() if desired else None, applied.model_dump())
            for key, desired, applied in plan.metadata
        ],
        "issues": plan.issues,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _pending(journal, context, *, targets=(), durable=False):
    pending, _ = _records(Path(context.bootstrap_root))
    expected = {
        "version": 1,
        "operation_id": journal.operation_id,
        "namespaces": context.namespaces,
        "selectors": context.selectors,
        "control_root": str(journal.root.parent),
    }
    if [row for row in pending if row["operation_id"] == journal.operation_id] != [
        expected
    ]:
        raise ValueError("publication_pending_mismatch")
    registry = _registry(Path(context.bootstrap_root))
    if registry is not None:
        if any(name not in registry for name in context.namespaces):
            raise ValueError("publication_scope_uncovered")
        fenced = [Path(path) for path in context.selectors] + [
            Path(path)
            for name in context.namespaces
            for path in registry[name]["roots"]
        ]
        if any(
            not any(_overlap(target, root) for root in fenced) for target in targets
        ):
            raise ValueError("publication_scope_uncovered")
        # A directory publication can encompass several independently enrolled
        # stores. Every affected store must be fenced, not merely one child.
        for name, entry in registry.items():
            roots = [Path(path) for path in entry["roots"]]
            if (
                any(_overlap(target, root) for target in targets for root in roots)
                and name not in context.namespaces
                and not any(_overlap(root, fence) for root in roots for fence in fenced)
            ):
                raise ValueError("publication_scope_uncovered")
    if durable:
        root = Path(context.bootstrap_root)
        with pinned_directory(root) as parent:
            journal._flush_record(
                parent, f"pending-{_key(journal.operation_id)}.json", expected
            )
            flush_directory(parent)
        # Any of these local directories may have been created by registration.
        # The filesystem root itself has no containing directory entry to flush.
        for ancestor in root.parents:
            if ancestor == Path("/"):
                break
            with pinned_directory(ancestor) as parent:
                flush_directory(parent)


def _descriptor(candidate, plan):
    with pinned_directory(candidate) as fd:
        if os.fstat(fd).st_uid != os.geteuid() or os.fstat(fd).st_mode & 0o077:
            raise ValueError("private_candidate_required")
        document = _read(fd, "candidate.json")
    if (
        document.get("archive_digest") != plan.archive_digest
        or document.get("target_fingerprint") != plan.target_fingerprint
    ):
        raise ValueError("candidate_plan_mismatch")
    if document.get("profile_names") != dict(plan.profile_names) or document.get(
        "issues"
    ) != list(plan.issues):
        raise ValueError("candidate_plan_mismatch")
    roots = [Path(value) for value in document["private_roots"]]
    for root in roots:
        with pinned_directory(root) as fd:
            if os.fstat(fd).st_uid != os.geteuid() or os.fstat(fd).st_mode & 0o077:
                raise ValueError("private_candidate_required")
    rows = document["artifacts"]
    if len(rows) != len(plan.restore) or {
        row["logical_id"]: Path(row["destination"]) for row in rows
    } != dict(plan.restore):
        raise ValueError("candidate_mapping_mismatch")
    for row in [*document.get("containers", []), *rows]:
        path = Path(row["candidate"])
        if not any(root in path.parents for root in roots):
            raise ValueError("candidate_root_mismatch")
        observed = observe_artifact(path)
        info = path.lstat()
        if (
            row["identity"]
            != [info.st_dev, info.st_ino, info.st_mode, info.st_mtime_ns]
            or observed["kind"] != row["kind"]
        ):
            raise ValueError("candidate_identity_changed")
        if row["kind"] == "file" and (
            row["size"] != info.st_size or row["sha256"] != reader._hash(path, Event())
        ):
            raise ValueError("candidate_content_changed")
    containers = document.get("containers", [])
    if {row["logical_id"]: Path(row["destination"]) for row in containers} != dict(
        getattr(plan, "containers", ())
    ):
        raise ValueError("candidate_container_mismatch")
    return document


def _flush_original(fd, device):
    """Flush existing owned bytes without normalizing their supported metadata."""
    info = os.fstat(fd)
    if info.st_dev != device or info.st_uid != os.geteuid():
        raise ValueError("retirement_identity_changed")
    if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
        os.fsync(fd)
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    elif stat.S_ISDIR(info.st_mode):
        for name in os.listdir(fd):
            child = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
            )
            try:
                _flush_original(child, device)
            finally:
                os.close(child)
        flush_directory(fd)
    else:
        raise ValueError("retirement_object_unverified")


def _retire(item):
    source, destination = Path(item.target), Path(item.retained)
    with (
        pinned_directory(source.parent) as source_parent,
        pinned_directory(destination.parent) as target_parent,
    ):
        parents = {row.path: (row.device, row.inode) for row in item.parents}
        for path, fd in (
            (source.parent, source_parent),
            (destination.parent, target_parent),
        ):
            info = os.fstat(fd)
            if parents.get(str(path)) != (info.st_dev, info.st_ino):
                raise ValueError("publication_parent_changed")
        identity = native_identity(target_parent)
        for operation in (
            "publish_new",
            "publish_directory"
            if item.previous.kind == "directory"
            else "publish_file",
        ):
            allowed, reason = _qualified_identity(operation, identity)
            if not allowed:
                raise ValueError(reason)
        fd = os.open(
            source.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=source_parent,
        )
        try:
            if os.fstat(fd).st_dev != os.fstat(target_parent).st_dev:
                raise ValueError("cross_volume_retirement_unqualified")
            _flush_original(fd, os.fstat(fd).st_dev)
            if not _matches(item.previous_metadata, item.target, metadata=True):
                raise ValueError("retirement_identity_changed")
            if os.stat(
                source.name, dir_fd=source_parent, follow_symlinks=False
            ) != os.fstat(fd):
                raise ValueError("retirement_identity_changed")
            _rename_new(source_parent, source.name, target_parent, destination.name)
            flush_directory(target_parent)
            flush_directory(source_parent)
        finally:
            os.close(fd)
    if not _matches(item.previous_metadata, item.retained, metadata=True):
        raise ValueError("retirement_identity_changed")


def _parent_evidence(path, containers):
    if str(path) in containers:
        candidate = Path(containers[str(path)])
    else:
        candidate = path
    with pinned_directory(candidate) as fd:
        info = os.fstat(fd)
        return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def _check_parents(item):
    expected = {str(Path(item.target).parent)}
    if item.retained:
        expected.add(str(Path(item.retained).parent))
    if item.candidate:
        expected.add(str(Path(item.candidate.path).parent))
    if {row.path for row in item.parents} != expected:
        raise ValueError("publication_parent_unverified")
    try:
        for row in item.parents:
            with pinned_directory(Path(row.path)) as fd:
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino) != (row.device, row.inode):
                    raise ValueError("publication_parent_changed")
    except (OSError, RuntimeError):
        raise ValueError("publication_parent_changed") from None


def _prepare(
    journal, candidate, plan, bootstrap_root, namespaces, selectors, generation
):
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if len(records) != 1 or records[0].event != "candidate_staged":
            raise ValueError("candidate_receipt_required")
        receipt = _CandidateReceipt.model_validate(records[0].evidence)
        if (
            receipt.plan_digest != _plan_digest(plan)
            or receipt.archive_digest != plan.archive_digest
        ):
            raise ValueError("candidate_plan_mismatch")
        if (
            receipt.stage.path != str(candidate)
            or not _matches(receipt.stage, str(candidate))
            or not _matches(receipt.descriptor, str(candidate / "candidate.json"))
        ):
            raise ValueError("candidate_receipt_changed")
        context = _PublicationContext(
            bootstrap_root=str(bootstrap_root),
            namespaces=list(Admission._names(namespaces)),
            selectors=[str(path) for path in selectors],
            archive_digest=plan.archive_digest,
            plan_digest=_plan_digest(plan),
            descriptor=receipt.descriptor,
        )
        if not {str(path) for _, path in plan.selectors} <= set(context.selectors):
            raise ValueError("publication_selector_mismatch")
        _pending(
            journal,
            context,
            targets=[path for _, path in (*plan.restore, *plan.retire)],
        )
        recheck_targets(plan)
        document = _descriptor(candidate, plan)
        containers = {
            row["destination"]: row["candidate"]
            for row in document.get("containers", [])
        }
        from .owner_registry import install_adapters

        owners = {owner.owner_id: owner for owner in install_adapters()}
        target_items = (
            {item.path: item for item in plan.target.items} if plan.target else {}
        )
        roots = [Path(path) for path in document["private_roots"]]
        retained_roots = {}
        artifacts = []
        rows = [dict(row, action="container") for row in document.get("containers", [])]
        # Retire only maximal reviewed objects; their exact subtrees are verified.
        for key, path in sorted(plan.retire, key=lambda row: len(row[1].parts)):
            if any(
                Path(row["destination"]) in path.parents
                for row in rows
                if row["action"] == "retire"
            ):
                continue
            rows.append(
                {
                    "logical_id": key,
                    "destination": str(path),
                    "candidate": None,
                    "action": "retire",
                }
            )
        rows.extend(
            dict(row, action="publish")
            for row in document["artifacts"]
            if row["publication_unit"]
        )
        for row in rows:
            target = Path(row["destination"])
            if any(
                target == path or target in path.parents for _, path in plan.preserve
            ):
                raise ValueError("publication_preserved_overlap")
            previous = metadata = retained = None
            requires_owner = False
            try:
                target.lstat()
            except FileNotFoundError:
                if row["action"] == "retire":
                    raise ValueError("retirement_target_missing") from None
            else:
                if plan.mode != "replace" or row["action"] == "container":
                    raise ValueError("publication_target_occupied")
                previous = observe_artifact(target)
                metadata = observe_artifact(target, metadata=True)
                device = previous["device"]
                if device not in retained_roots:
                    selected = next(
                        (root for root in roots if root.stat().st_dev == device), None
                    )
                    if selected is None:
                        raise ValueError("retirement_volume_unavailable")
                    retained_root = selected / (
                        "retained-"
                        + hashlib.sha256(journal.operation_id.encode()).hexdigest()
                    )
                    create_private_directory(retained_root)
                    retained_roots[device] = retained_root
                retained = str(
                    retained_roots[device]
                    / hashlib.sha256(row["logical_id"].encode()).hexdigest()
                )
                source = target_items.get(target)
                owner = owners.get(source.owner) if source else None
                if owner is None:
                    raise ValueError("rollback_owner_unavailable")
                for local_path, local_item in target_items.items():
                    if local_path == target or target in local_path.parents:
                        local_owner = owners.get(local_item.owner)
                        if local_owner is None:
                            raise ValueError("rollback_owner_unavailable")
                        policy = local_owner.schema_policy()
                        requires_owner |= policy is not None and bool(policy.schema_sql)
            artifacts.append(
                {
                    "logical_id": row["logical_id"],
                    "action": row["action"],
                    "candidate": observe_artifact(Path(row["candidate"]))
                    if row["candidate"]
                    else None,
                    "target": str(target),
                    "previous": previous,
                    "retained": retained,
                    "previous_metadata": metadata,
                    "rollback_requires_owner": requires_owner,
                    "parents": [
                        _parent_evidence(path, containers)
                        for path in sorted(
                            {target.parent}
                            | ({Path(retained).parent} if retained else set())
                            | (
                                {Path(row["candidate"]).parent}
                                if row["candidate"]
                                else set()
                            )
                        )
                    ],
                }
            )
        journal._append(
            parent,
            "prepared",
            {
                "generation": generation,
                "mode": plan.mode,
                "artifacts": artifacts,
                "publication": context.model_dump(),
            },
        )


def _archive_object(document, logical_id, *, metadata=False):
    """Compute exactly the raw file/subtree represented by a manifest record."""
    entries = {row.logical_id: row for row in (*document.directories, *document.files)}
    selected = entries.get(logical_id)
    if selected is None:
        raise ValueError("rollback_coverage_missing")
    directory_ids = {row.logical_id for row in document.directories}
    base = Path(selected.relative_path)
    chosen = [
        row
        for row in entries.values()
        if row.logical_id == logical_id
        or logical_id in directory_ids
        and row.root_id == selected.root_id
        and base in Path(row.relative_path).parents
    ]
    rows, total = [], 0
    for row in sorted(chosen, key=lambda row: Path(row.relative_path).parts):
        relative = (
            ""
            if row.logical_id == logical_id
            else "/" + str(Path(row.relative_path).relative_to(base))
        )
        if row.logical_id in directory_ids:
            value = (relative, "directory")
        else:
            value = (relative, "file", row.size, row.sha256)
            total += row.size
        if metadata:
            if row.metadata is None:
                raise ValueError("rollback_metadata_unverified")
            value += (row.metadata.mode, row.metadata.mtime_ns)
        rows.append(value)
    # Filesystem traversal is depth-first; lexicographic path order matches it.
    return total, hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()


def _verify_rollback(journal, path, password, work_root, cancel, coverage):
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if not records or records[-1].event != "prepared":
            raise ValueError("rollback_preparation_required")
        prepared = _Prepared.model_validate(records[-1].evidence)
        if prepared.publication is None:
            raise ValueError("publication_context_unverified")
        _pending(
            journal,
            prepared.publication,
            targets=[Path(item.target) for item in prepared.artifacts],
        )
        previous = {
            item.logical_id: item
            for item in prepared.artifacts
            if item.previous is not None
        }
        if set(coverage) != set(previous):
            raise ValueError("rollback_coverage_mismatch")
        if any(item.rollback_requires_owner for item in previous.values()):
            raise ValueError("rollback_sqlite_owner_receipt_required")
        before = observe_artifact(path)
        with reader._regular(path) as source:
            if not source.read(20).startswith(b"age-encryption.org/"):
                raise ValueError("encrypted_rollback_required")
        archive = reader.acquire(path, work_root, ArchiveLimits(), password, cancel)
        document = reader.verify_sealed(archive, cancel)
        if (
            document.credential_policy != "rollback"
            or document.consistency != "coherent"
        ):
            raise ValueError("rollback_policy_required")
        if observe_artifact(path) != before:
            raise ValueError("rollback_ciphertext_changed")
        for key, item in previous.items():
            if not _matches(item.previous_metadata, item.target, metadata=True):
                raise ValueError("rollback_source_changed")
            size, digest = _archive_object(document, coverage[key], metadata=True)
            if (size, digest) != (
                item.previous_metadata.size,
                item.previous_metadata.sha256,
            ):
                raise ValueError("rollback_coverage_mismatch")
        journal._append(
            parent,
            "rollback_verified",
            {
                "ciphertext": before,
                "sealed_digest": archive.digest,
                "manifest_digest": hashlib.sha256(archive.manifest_bytes).hexdigest(),
                "coverage": dict(coverage),
            },
        )


def publish_candidate(
    candidate: Path, plan: RestorePlan, journal, rollback_archive: Path | None
) -> None:
    """Publish only fully verified local context; incomplete evidence stays fenced."""
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        if not records or records[0].event != "candidate_staged":
            raise ValueError("candidate_receipt_required")
        receipt = _CandidateReceipt.model_validate(records[0].evidence)
        record = next((row for row in records if row.event == "prepared"), None)
        if record is None:
            raise ValueError("publication_preparation_required")
        prepared = _Prepared.model_validate(record.evidence)
        context = prepared.publication
        if (
            context is None
            or receipt.plan_digest != _plan_digest(plan)
            or receipt.descriptor != context.descriptor
            or receipt.stage.path != str(candidate)
            or context.plan_digest != _plan_digest(plan)
            or context.archive_digest != plan.archive_digest
        ):
            raise ValueError("publication_context_unverified")
        _pending(journal, context)
        if context.descriptor.path != str(candidate / "candidate.json") or not _matches(
            context.descriptor, str(candidate / "candidate.json")
        ):
            raise ValueError("candidate_receipt_changed")
        if prepared.mode == "replace":
            proof = next(
                (row for row in records if row.event == "rollback_verified"), None
            )
            if proof is None or rollback_archive is None:
                raise ValueError("rollback_required")
            verified = _Rollback.model_validate(proof.evidence)
            if verified.ciphertext.path != str(rollback_archive) or not _matches(
                verified.ciphertext, str(rollback_archive)
            ):
                raise ValueError("rollback_ciphertext_changed")
        elif rollback_archive is not None:
            raise ValueError("unexpected_rollback_archive")
        started = any(row.event == "publication_started" for row in records)
        if not started:
            recheck_targets(plan)
            _descriptor(candidate, plan)
            for item in prepared.artifacts:
                if item.candidate and not _matches(item.candidate, item.candidate.path):
                    raise ValueError("publication_objects_changed")
                if item.previous_metadata and not _matches(
                    item.previous_metadata, item.target, metadata=True
                ):
                    raise ValueError("publication_objects_changed")
            journal._append(parent, "publication_started", {})
        _pending(
            journal,
            context,
            targets=[Path(item.target) for item in prepared.artifacts],
            durable=True,
        )
        journal._flush_records(parent)
        for item in prepared.artifacts:
            _check_parents(item)
            state = _states(prepared, logical_id=item.logical_id)[item.logical_id]
            if state == "uncertain":
                raise ValueError("publication_objects_changed")
            if state == "staged" and item.previous is not None:
                _retire(item)
                journal._append(
                    parent,
                    "artifact_retired",
                    {
                        "logical_id": item.logical_id,
                        "observed": observe_artifact(
                            Path(item.retained), metadata=True
                        ),
                    },
                )
                state = "retired"
            if item.candidate is not None and state in {"staged", "retired"}:
                if not _matches(item.candidate, item.candidate.path):
                    raise ValueError("candidate_content_changed")
                parents = {row.path: (row.device, row.inode) for row in item.parents}
                publish_new(
                    Path(item.candidate.path),
                    Path(item.target),
                    parent_identities=(
                        parents[str(Path(item.candidate.path).parent)],
                        parents[str(Path(item.target).parent)],
                    ),
                )
                if not _matches(item.candidate, item.target):
                    raise ValueError("publication_objects_changed")
                journal._append(
                    parent,
                    "artifact_published",
                    {
                        "logical_id": item.logical_id,
                        "observed": observe_artifact(Path(item.target)),
                    },
                )
