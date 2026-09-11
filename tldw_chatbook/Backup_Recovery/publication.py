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
from .capture import _item_validator
from .journal import (
    _CandidateReceipt,
    _DirectoryState,
    _evidence_digest,
    _IsolatedProfile,
    _matches,
    _Object,
    _Prepared,
    _PublicationContext,
    _ReplacementProfile,
    _RetainedCredentials,
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
    if plan.safety_scope:
        value["safety_scope"] = plan.safety_scope
    if plan.acknowledged_credential_issues:
        value["acknowledged_credential_issues"] = plan.acknowledged_credential_issues
    if plan.local_snapshot is not None:
        value["local_snapshot"] = plan.local_snapshot.record()
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _finalization_session(session, context, prepared):
    from .control_records import UNBOUND_NAMESPACE
    from .storage_admission import MaintenanceSession, _contains_owned_path

    if type(session) is not MaintenanceSession:
        raise ValueError("maintenance_session_required")
    MaintenanceSession._check(session)
    if UNBOUND_NAMESPACE not in session._names:
        raise ValueError("finalization_unbound_guard_required")
    if session._control != Path(context.bootstrap_root) / "admission":
        raise ValueError("conflicting_admission_authority")
    with pinned_directory(session._control) as fd:
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) != session._control_identity:
            raise ValueError("finalization_authority_changed")
    registry = _registry(Path(context.bootstrap_root))
    if (
        registry is None
        or not set(context.namespaces) <= set(session._names)
        or any(name not in registry for name in context.namespaces)
        or any(
            not any(_contains_owned_path(root, path) for root in session._roots)
            for path in (
                *map(Path, context.selectors),
                *_publication_targets(prepared),
                *(
                    Path(path)
                    for name in context.namespaces
                    for path in registry[name]["roots"]
                ),
            )
        )
    ):
        raise ValueError("finalization_scope_uncovered")


def _activation_proof(journal, context, generation, selectors, owners):
    """Read and flush the exact pair plus immutable requirements, never approvals."""
    from .activation import ActivationStore, _private
    from .bootstrap import _binding, _control_records

    root = Path(context.bootstrap_root)
    _, profiles, associations = _control_records(root)
    registry = _registry(root)
    store = ActivationStore(journal.root.parent / "activation")
    paths = []
    for selector in selectors:
        profile = next((p for p in profiles if p["selector"] == selector), None)
        association = next((a for a in associations if a["selector"] == selector), None)
        if (
            profile is None
            or association is None
            or _binding(Path(selector), [profile], registry) is None
        ):
            raise ValueError("finalization_activation_unverified")
        expected = {
            "operation_id": journal.operation_id,
            "generation": generation,
            "owners": owners,
            "namespaces": profile["namespaces"],
            "store_root": str(store.root),
        }
        if (
            profile.get("activation") != expected
            or association["activation"] != expected
            or not set(profile["namespaces"]) <= set(context.namespaces)
        ):
            raise ValueError("finalization_activation_unverified")
        with _private(root) as parent:
            for prefix, value in (("profile-", profile), ("activation-", association)):
                name = prefix + _key(selector) + ".json"
                journal._flush_record(parent, name, value)
                paths.append(root / name)
            flush_directory(parent)
    directory = store._generation(generation)
    with _private(directory) as parent:
        requirement = store._required(parent, generation)
        if requirement.owners != owners:
            raise ValueError("finalization_activation_unverified")
        journal._flush_record(parent, "required.json", requirement.model_dump())
        flush_directory(parent)
    paths.append(directory / "required.json")
    return [observe_artifact(path) for path in sorted(paths)]


def finalize_candidate(candidate: Path, plan: RestorePlan, journal, *, session) -> str:
    """Commit current installed proof and retire only its pending native fence.

    Internal executor composition: the caller retains the actual session through
    publication and completion, including bootstrap.unbound so new unbound storage
    users remain blocked after pointer retirement. That guard is not an affected
    profile namespace and does not enter the pending record or activation binding.
    Config-owned restored items are selectors; managed
    relocation directories are not. A restored config requires every installed
    activation-required capability, including omitted/preserved dependent stores.
    Existing per-owner reviews survive identical retries. No service is started.

    An absent pointer is accepted only with this operation's durable typed commit
    and freshly rechecked installed objects and paired activation identities. An
    unlink/barrier failure may be retried under fresh native recovery maintenance;
    this reestablishes the bootstrap barrier without recreating a pointer.
    """
    from .activation import bind_activation
    from .owner_registry import install_adapters
    from .staging import _items

    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        row = next((row for row in records if row.event == "prepared"), None)
        if row is None:
            raise ValueError("publication_incomplete")
        prepared = _Prepared.model_validate(row.evidence)
        context = prepared.publication
        if context is None or context.plan_digest != _plan_digest(plan):
            raise ValueError("publication_context_unverified")
        _finalization_session(session, context, prepared)
        if any(
            not _matches(row.source, row.source.path, metadata=True)
            for row in prepared.safety_sources
        ):
            raise ValueError("safety_source_changed")
        committed = records[-1] if records[-1].event == "committed" else None
        _pending(
            journal,
            context,
            targets=_publication_targets(prepared),
            committed=committed,
        )
    root = Path(context.bootstrap_root)
    name = "pending-" + _key(journal.operation_id) + ".json"
    with pinned_directory(root) as parent:
        root_identity = (os.fstat(parent).st_dev, os.fstat(parent).st_ino)
        before = observe_artifact(root / name) if os.path.lexists(root / name) else None
    installed = _validate_installed(journal, candidate, plan)
    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        _finalization_session(session, context, prepared)
        with reader._regular(journal.root / "verified-manifest.json") as stream:
            manifest = stream.read(ArchiveLimits().manifest_bytes + 1)
        if hashlib.sha256(manifest).hexdigest() != installed["manifest_digest"]:
            raise ValueError("verified_manifest_changed")
        doc = reader._manifest(manifest, ArchiveLimits(), True)
        selectors = sorted(
            {
                str(item.path)
                for item in _items(doc, plan).values()
                if item.owner == "config" and item.status == "included"
            }
        )
        if not selectors or not set(selectors) <= set(context.selectors):
            raise ValueError("finalization_config_selector_required")
        owners = sorted(
            owner.owner_id for owner in install_adapters() if owner.activation_required
        )
        prior = next(
            (row for row in records if row.event == "activation_recorded"), None
        )
        if prior is None:
            for selector in selectors:
                bind_activation(
                    root,
                    journal.operation_id,
                    Path(selector),
                    prepared.generation,
                    tuple(owners),
                    session=session,
                )
        proof = {
            "generation": prepared.generation,
            "plan_digest": context.plan_digest,
            "installed_digest": _evidence_digest(installed),
            "selectors": selectors,
            "owners": owners,
            "records": _activation_proof(
                journal, context, prepared.generation, selectors, owners
            ),
        }

        def check_installed():
            if any(
                not _matches(_Object.model_validate(item), item["path"], metadata=True)
                for item in installed["artifacts"]
            ):
                raise ValueError("installed_evidence_changed")

        check_installed()
        if prior is None:
            journal._append(parent, "activation_recorded", proof)
        elif prior.evidence != proof:
            raise ValueError("finalization_activation_changed")
        check_installed()
        catalog_proof = None
        if prepared.isolated_profiles:
            catalog = next(
                (row for row in records if row.event == "catalog_registered"), None
            )
            catalog_proof = {
                "generation": prepared.generation,
                "activation_digest": _evidence_digest(proof),
                "records": _catalog_proof(journal, prepared, register=catalog is None),
            }
            if catalog is None:
                journal._append(parent, "catalog_registered", catalog_proof)
            elif catalog.evidence != catalog_proof:
                raise ValueError("finalization_catalog_changed")
        credential = next(
            (row for row in records if row.event == "credentials_completed"), None
        )
        if prepared.credential_scopes:
            from .replacement import _verify_applied_credentials

            _verify_applied_credentials(
                candidate, journal, session=session, records=records
            )
        if prepared.incoming_credentials and not _matches(
            prepared.incoming_credentials.ciphertext,
            prepared.incoming_credentials.ciphertext.path,
        ):
            raise ValueError("retained_ciphertext_changed")
        if committed is None:
            journal._append(
                parent,
                "committed",
                {
                    "generation": prepared.generation,
                    "activation_digest": _evidence_digest(proof),
                    "credential_digest": _evidence_digest(credential.evidence)
                    if credential
                    else None,
                    "catalog_digest": _evidence_digest(catalog_proof)
                    if catalog_proof
                    else None,
                },
            )
        journal._flush_records(parent)
        _finalization_session(session, context, prepared)
        check_installed()
        if (
            catalog_proof
            and _catalog_proof(journal, prepared) != catalog_proof["records"]
        ):
            raise ValueError("finalization_catalog_changed")
        if (
            _activation_proof(journal, context, prepared.generation, selectors, owners)
            != proof["records"]
        ):
            raise ValueError("finalization_activation_changed")
        if any(
            not _matches(row.source, row.source.path, metadata=True)
            for row in prepared.safety_sources
        ):
            raise ValueError("safety_source_changed")
        if prepared.incoming_credentials and not _matches(
            prepared.incoming_credentials.ciphertext,
            prepared.incoming_credentials.ciphertext.path,
        ):
            raise ValueError("retained_ciphertext_changed")
        if prepared.credential_scopes:
            _verify_applied_credentials(
                candidate, journal, session=session, records=journal._records(parent)
            )
        committed = journal._records(parent)[-1]
        _pending(
            journal,
            context,
            targets=_publication_targets(prepared),
            durable=True,
            committed=committed,
        )
        with pinned_directory(root) as bootstrap:
            info = os.fstat(bootstrap)
            if (info.st_dev, info.st_ino) != root_identity:
                raise ValueError("finalization_authority_changed")
            after = (
                observe_artifact(root / name) if os.path.lexists(root / name) else None
            )
            if after != before:
                raise ValueError("finalization_pending_changed")
            if before is not None:
                os.unlink(name, dir_fd=bootstrap)
            flush_directory(bootstrap)
    return prepared.generation


def _catalog_proof(journal, prepared, *, register=False):
    """Register/recheck real catalog locators and retained encrypted material."""
    from .profile_catalog import ProfileCatalog, _name

    if prepared.retained_credentials is not None:
        item = prepared.retained_credentials.ciphertext
        if not _matches(item, item.path):
            raise ValueError("retained_ciphertext_changed")
    catalog = ProfileCatalog(journal.root.parent)
    records = []
    for entry in prepared.isolated_profiles:
        config, data = Path(entry.config), Path(entry.data)
        if register:
            catalog.register(entry.profile_id, config, data)
        if catalog.resolve(entry.profile_id) != (config, data):
            raise ValueError("catalog_mapping_changed")
        records.append(observe_artifact(catalog.root / _name(entry.profile_id)))
    return sorted(records, key=lambda row: row["path"])


def _pending(journal, context, *, targets=(), durable=False, committed=None):
    pending, _ = _records(Path(context.bootstrap_root))
    expected = {
        "version": 1,
        "operation_id": journal.operation_id,
        "namespaces": context.namespaces,
        "selectors": context.selectors,
        "control_root": str(journal.root.parent),
    }
    matching = [row for row in pending if row["operation_id"] == journal.operation_id]
    absent_commit = (
        not matching and committed is not None and committed.event == "committed"
    )
    if matching != [expected] and not absent_commit:
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
            if not absent_commit:
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
    if document.get("local_snapshot") != (
        plan.local_snapshot.record() if plan.local_snapshot else None
    ):
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


def _directory_state(path):
    with pinned_directory(Path(path)) as fd:
        info = os.fstat(fd)
        if info.st_uid != os.geteuid():
            raise ValueError("directory_metadata_changed")
        return _DirectoryState(
            path=str(path),
            device=info.st_dev,
            inode=info.st_ino,
            mode=stat.S_IMODE(info.st_mode),
            mtime_ns=info.st_mtime_ns,
            ctime_ns=info.st_ctime_ns,
        )


def _directory_expected(prepared, records):
    expected = {
        item.previous.path: item.previous for item in prepared.directory_metadata
    }
    for record in records:
        if record.event in {"artifact_retired", "artifact_published", "move_observed"}:
            for value in record.evidence.get("directories", []):
                state = _DirectoryState.model_validate(value)
                expected[state.path] = state
        elif record.event == "directory_metadata_applied":
            state = _DirectoryState.model_validate(record.evidence["observed"])
            expected[state.path] = state
    return expected


def _directory_transition_states(before, mode, mtime_ns):
    # The native sequence is fchmod, then utime; the fourth Cartesian pair
    # (old mode, new time) is not an interrupted operation when both differ.
    return {(before.mode, before.mtime_ns), (mode, before.mtime_ns), (mode, mtime_ns)}


def _check_directory_states(prepared, records):
    expected = _directory_expected(prepared, records)
    for item in prepared.directory_metadata:
        with pinned_directory(Path(item.parent.path)) as fd:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != (item.parent.device, item.parent.inode):
                raise ValueError("publication_parent_changed")
        state = _directory_state(item.previous.path)
        intent = next(
            (
                row
                for row in records
                if row.event == "directory_metadata_started"
                and row.evidence["logical_id"] == item.logical_id
            ),
            None,
        )
        applied = any(
            row.event == "directory_metadata_applied"
            and row.evidence["logical_id"] == item.logical_id
            for row in records
        )
        if intent is not None and not applied:
            before = _DirectoryState.model_validate(intent.evidence["before"])
            if (state.device, state.inode) != (before.device, before.inode) or (
                state.mode,
                state.mtime_ns,
            ) not in _directory_transition_states(
                before, item.applied.mode, item.applied.mtime_ns
            ):
                raise ValueError("directory_metadata_changed")
        elif state != expected[item.previous.path]:
            raise ValueError("directory_metadata_unproven")


def _directory_observations(prepared, item):
    observations = []
    for row in prepared.directory_metadata:
        if row.previous.path != str(Path(item.target).parent):
            continue
        state = _directory_state(row.previous.path)
        if (state.device, state.inode, state.mode) != (
            row.previous.device,
            row.previous.inode,
            row.previous.mode,
        ):
            raise ValueError("directory_metadata_changed")
        observations.append(state.model_dump())
    return observations


def _publication_targets(prepared):
    return (
        [Path(item.source.path) for item in prepared.safety_sources]
        + [Path(item.target) for item in prepared.artifacts]
        + [Path(item.previous.path) for item in prepared.directory_metadata]
    )


def _sidecar_main(item, items, owners):
    """Resolve only the exact locally classified transient main/WAL/SHM relation."""
    if item.owner != "sqlite.transient":
        return item
    main = next(
        (
            row
            for row in items
            if len(item.dependencies) == 1 and row.logical_id == item.dependencies[0]
        ),
        None,
    )
    adapter = owners.get(main.owner) if main else None
    policy = (
        _item_validator(adapter, main).schema_policy()
        if adapter and main.path
        else None
    )
    if (
        main is None
        or main.path is None
        or policy is None
        or not policy.schema_sql
        or item.status != "intentionally_excluded"
        or item.path
        not in {Path(str(main.path) + suffix) for suffix in ("-wal", "-shm")}
    ):
        raise ValueError("rollback_sidecar_unclassified")
    return main


def _rollback_sources(plan, artifacts, owners):
    """Bind installed SQLite ownership and physical groups before safety capture."""
    if plan.target is None:
        return []
    groups = []
    for item in plan.target.items:
        if item.path is None or item.owner not in owners:
            continue
        policy = _item_validator(owners[item.owner], item).schema_policy()
        if item.status != "included" or policy is None or not policy.schema_sql:
            continue
        sidecars = {}
        for row in plan.target.items:
            if row.owner != "sqlite.transient":
                continue
            main = _sidecar_main(row, plan.target.items, owners)
            # Semantic aliases at one declared path share the same live WAL/SHM.
            # A hardlink or another pathname never borrows that path's sidecars.
            if row.path.exists() and (
                main.logical_id == item.logical_id
                or item.shared_group
                and main.shared_group == item.shared_group
                and main.path == item.path
            ):
                sidecars[row.logical_id] = row
        paths = [item.path, *(row.path for row in sidecars.values())]
        affected = sorted(
            row["logical_id"]
            for row in artifacts
            if row["previous"] is not None
            and any(
                path == Path(row["target"]) or Path(row["target"]) in path.parents
                for path in paths
            )
        )
        if not affected:
            continue
        if not any(
            row["previous"] is not None
            and (
                item.path == Path(row["target"])
                or Path(row["target"]) in item.path.parents
            )
            for row in artifacts
        ):
            raise ValueError("rollback_sidecar_main_unaffected")
        actual_sidecars = {
            Path(str(item.path) + suffix)
            for suffix in ("-wal", "-shm")
            if os.path.lexists(str(item.path) + suffix)
        }
        if actual_sidecars != {row.path for row in sidecars.values()}:
            raise ValueError("rollback_sidecar_unclassified")
        groups.append(
            {
                "logical_id": item.logical_id,
                "owner_id": item.owner,
                "source": observe_artifact(item.path, metadata=True),
                "sidecars": {
                    key: observe_artifact(row.path, metadata=True)
                    for key, row in sidecars.items()
                },
                "artifacts": affected,
            }
        )
    return sorted(groups, key=lambda row: row["logical_id"])


def _prepare(
    journal,
    candidate,
    plan,
    bootstrap_root,
    namespaces,
    selectors,
    generation,
    *,
    replacement_profiles=(),
    incoming_credentials=None,
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
        from .staging import _items

        with reader._regular(journal.root / "verified-manifest.json") as stream:
            manifest = stream.read(ArchiveLimits().manifest_bytes + 1)
        if hashlib.sha256(manifest).hexdigest() != receipt.manifest_digest:
            raise ValueError("verified_manifest_changed")
        doc = reader._manifest(manifest, ArchiveLimits(), True)
        # Managed DB/data relocation keys stay bound by the plan, not fingerprinted
        # as config selectors. Data-only primitives may retain a local selector.
        restored_selectors = {
            str(item.path)
            for item in _items(doc, plan).values()
            if item.owner == "config" and item.status == "included"
        }
        if not restored_selectors <= set(context.selectors):
            raise ValueError("publication_selector_mismatch")
        _pending(
            journal,
            context,
            targets=[path for _, path in (*plan.restore, *plan.retire)],
        )
        recheck_targets(plan)
        document = _descriptor(candidate, plan)
        isolated_profiles = [
            _IsolatedProfile.model_validate(row)
            for row in document.get("isolated_profiles", [])
        ]
        retained_credentials = document.get("retained_credentials")
        replacement_profiles = [
            _ReplacementProfile.model_validate(row) for row in replacement_profiles
        ]
        if replacement_profiles and (
            plan.mode != "replace"
            or isolated_profiles
            or {row.config for row in replacement_profiles} != restored_selectors
            or len(replacement_profiles) != len(restored_selectors)
            or len({row.installation_id for row in replacement_profiles})
            != len(replacement_profiles)
        ):
            raise ValueError("replacement_identity_mapping_invalid")

        if isolated_profiles:
            configs = {
                row.logical_id.split(":")[1]: str(dict(plan.restore)[row.logical_id])
                for row in doc.files
                if row.owner_id == "config" and row.logical_id in dict(plan.restore)
            }
            if (
                plan.mode != "isolated"
                or len({row.profile_id for row in isolated_profiles})
                != len(isolated_profiles)
                or len({row.installation_id for row in isolated_profiles})
                != len(isolated_profiles)
                or {row.source_profile: row.config for row in isolated_profiles}
                != configs
                or len(isolated_profiles) != len(configs)
                or any(
                    row.data
                    != str(
                        dict(plan.selectors).get(
                            f"profile:{row.source_profile}:paths.data_dir"
                        )
                    )
                    for row in isolated_profiles
                )
            ):
                raise ValueError("isolated_catalog_intent_invalid")
            if doc.credential_policy != "exclude" and retained_credentials is None:
                raise ValueError("encrypted_retention_required")
        if replacement_profiles:
            if doc.credential_policy != "exclude" and incoming_credentials is None:
                raise ValueError("encrypted_retention_required")
            if incoming_credentials is not None:
                retained = _RetainedCredentials.model_validate(incoming_credentials)
                if (
                    Path(retained.ciphertext.path)
                    != journal.root.parent
                    / ("replacement-" + journal.operation_id)
                    / "credentials.age"
                    or retained.plaintext_digest != receipt.archive_digest
                    or retained.manifest_digest != receipt.manifest_digest
                    or not _matches(retained.ciphertext, retained.ciphertext.path)
                ):
                    raise ValueError("encrypted_retention_unverified")
        if retained_credentials is not None:
            retained = _RetainedCredentials.model_validate(retained_credentials)
            if (
                not isolated_profiles
                or Path(retained.ciphertext.path)
                != journal.root.parent
                / ("isolated-" + journal.operation_id)
                / "credentials.age"
                or retained.plaintext_digest != receipt.archive_digest
                or retained.manifest_digest != receipt.manifest_digest
                or not _matches(retained.ciphertext, retained.ciphertext.path)
            ):
                raise ValueError("encrypted_retention_unverified")
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
        # A restored projection replaces its complete reviewed root as one unit.
        projection_units = {
            item.path
            for item in (plan.target.items if plan.target else ())
            if item.owner == "rag.projections"
            and item.metadata
            and item.logical_id == item.metadata.root_id
            and (item.logical_id, item.path) in plan.retire
        }
        restored_paths = {path for _, path in plan.restore}
        projection_replacements = projection_units & restored_paths
        # Retire only maximal reviewed objects; their exact subtrees are verified.
        for key, path in sorted(plan.retire, key=lambda row: len(row[1].parts)):
            if any(
                path == root or root in path.parents for root in projection_replacements
            ):
                continue
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
            if (
                row["publication_unit"]
                or Path(row["destination"]) in projection_replacements
            )
            and not any(
                root in Path(row["destination"]).parents
                for root in projection_replacements
            )
        )
        for row in rows:
            target = Path(row["destination"])
            if any(
                target == path or target in path.parents for _, path in plan.preserve
            ):
                raise ValueError("publication_preserved_overlap")
            previous = metadata = retained = None
            requires_owner = False
            projection_roots = []
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
                source = (
                    _sidecar_main(source, plan.target.items, owners) if source else None
                )
                owner = owners.get(source.owner) if source else None
                if owner is None:
                    raise ValueError("rollback_owner_unavailable")
                from .projection_publication import projection_groups

                projection_items = [
                    item
                    for item in plan.target.items
                    if item.path is not None
                    and (item.path == target or target in item.path.parents)
                    and item.owner == "rag.projections"
                ]
                for key, group in projection_groups(projection_items).items():
                    projection_root = next(
                        item.path for item in group if item.logical_id == key
                    )
                    evidence = observe_artifact(projection_root, metadata=True)
                    if evidence not in projection_roots:
                        projection_roots.append(evidence)
                for local_path, local_item in target_items.items():
                    if local_path == target or target in local_path.parents:
                        main = _sidecar_main(local_item, plan.target.items, owners)
                        local_owner = owners.get(main.owner)
                        if local_owner is None:
                            raise ValueError("rollback_owner_unavailable")
                        policy = _item_validator(local_owner, main).schema_policy()
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
                    "rollback_projection_roots": projection_roots,
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
        installed_paths = []
        directory_metadata = []
        for row in [*document["artifacts"], *document.get("containers", [])]:
            target = Path(row["destination"])
            moved = any(
                item["candidate"] is not None
                and (
                    target == Path(item["target"])
                    or Path(item["target"]) in target.parents
                )
                and item["action"] != "container"
                for item in artifacts
            ) or row in document.get("containers", [])
            source = Path(row["candidate"]) if moved else target
            if row["kind"] == "directory" and not moved:
                local = target_items.get(target)
                if local is None or local.owner not in owners:
                    raise ValueError("directory_metadata_owner_required")
                directory_metadata.append(
                    {
                        "logical_id": row["logical_id"],
                        "owner_id": local.owner,
                        "previous": _directory_state(target).model_dump(),
                        "parent": _parent_evidence(target.parent, {}),
                        "applied": row["applied_metadata"],
                    }
                )
            info = source.lstat()
            installed_paths.append(
                {"path": str(target), "device": info.st_dev, "inode": info.st_ino}
            )
        if plan.mode == "replace":
            credential_issues = (
                ["credential_isolated_retention_required"]
                if any(
                    json.loads(value)["action"] == "retain"
                    for value in document.get("credential_scopes", {}).values()
                )
                else []
            )
            if document.get("credential_issues", []) != credential_issues or not set(
                credential_issues
            ) <= set(plan.acknowledged_credential_issues):
                raise ValueError("credential_omission_acknowledgement_required")
        credential_material = (
            document.get("credential_material") if plan.mode == "replace" else None
        )
        if document.get("credential_scopes") and plan.mode == "replace":
            if credential_material is None:
                raise ValueError("credential_material_unverified")
            material = _Object.model_validate(credential_material)
            if material.path != str(
                candidate / "credential-recovery.json"
            ) or not _matches(material, material.path):
                raise ValueError("credential_material_unverified")
        safety_sources = []
        target_by_id = (
            {item.logical_id: item for item in plan.target.items} if plan.target else {}
        )
        for key in plan.safety_scope:
            item = target_by_id[key]
            adapter = owners.get(item.owner)
            policy = _item_validator(adapter, item).schema_policy() if adapter else None
            if (
                item.status != "included"
                or adapter is None
                or policy is None
                or policy.schema_sql
                or (item.logical_id, item.path) not in plan.preserve
                or item.shared_group
                and any(
                    other.logical_id not in plan.safety_scope
                    for other in plan.target.items
                    if other.shared_group == item.shared_group
                )
            ):
                raise ValueError("safety_scope_capability_unavailable")
            safety_sources.append(
                {
                    "logical_id": key,
                    "owner_id": item.owner,
                    "source": observe_artifact(item.path, metadata=True),
                }
            )
        journal._append(
            parent,
            "prepared",
            {
                "generation": generation,
                "mode": plan.mode,
                "isolated_profiles": [row.model_dump() for row in isolated_profiles],
                "retained_credentials": retained_credentials,
                "replacement_profiles": [
                    row.model_dump() for row in replacement_profiles
                ],
                "safety_sources": safety_sources,
                "incoming_credentials": incoming_credentials,
                "credential_scopes": document.get("credential_scopes", {})
                if plan.mode == "replace"
                else {},
                "credential_material": credential_material,
                "artifacts": artifacts,
                "publication": context.model_dump(),
                "installed_paths": installed_paths,
                "rollback_sources": _rollback_sources(plan, artifacts, owners),
                "directory_metadata": directory_metadata,
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
            targets=_publication_targets(prepared),
        )
        previous = {
            item.logical_id: item
            for item in prepared.artifacts
            if item.previous is not None
        }
        directory_metadata = {
            item.logical_id: item for item in prepared.directory_metadata
        }
        if prepared.safety_sources:
            raise ValueError("rollback_safety_held_capture_required")
        if set(coverage) != set(previous) | set(directory_metadata):
            raise ValueError("rollback_coverage_mismatch")
        if any(item.rollback_projection_roots for item in previous.values()):
            raise ValueError("rollback_projection_held_capture_required")
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
        producers = {item.logical_id: item for item in document.producer_inventory}
        directories = {item.logical_id: item for item in document.directories}
        for key, item in directory_metadata.items():
            original = item.previous
            if _directory_state(original.path) != original:
                raise ValueError("rollback_source_changed")
            archived = directories.get(coverage[key])
            producer = producers.get(coverage[key])
            if (
                archived is None
                or archived.synthetic
                or producer is None
                or producer.owner_id != item.owner_id
                or producer.status != "included_directory"
                or (archived.metadata.mode, archived.metadata.mtime_ns)
                != (original.mode, original.mtime_ns)
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
    candidate: Path,
    plan: RestorePlan,
    journal,
    rollback_archive: Path | None,
    *,
    session=None,
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
        if any(
            not _matches(row.source, row.source.path, metadata=True)
            for row in prepared.safety_sources
        ):
            raise ValueError("safety_source_changed")
        if prepared.incoming_credentials and not _matches(
            prepared.incoming_credentials.ciphertext,
            prepared.incoming_credentials.ciphertext.path,
        ):
            raise ValueError("retained_ciphertext_changed")
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
        if prepared.credential_scopes:
            from .replacement import _verify_applied_credentials

            _verify_applied_credentials(
                candidate, journal, session=session, records=records
            )
        if any(row.event == "publication_started" for row in records):
            _reconcile_moves(journal, parent, prepared, finish=True)
            records = journal._records(parent)
        started = any(row.event == "publication_started" for row in records)
        if not started:
            _check_directory_states(prepared, records)
            recheck_targets(plan)
            _descriptor(candidate, plan)
            for item in prepared.artifacts:
                if item.candidate and not _matches(item.candidate, item.candidate.path):
                    raise ValueError("publication_objects_changed")
                if item.previous_metadata and not _matches(
                    item.previous_metadata, item.target, metadata=True
                ):
                    raise ValueError("publication_objects_changed")
            # Only extant services have caches. Importing RAG here would try to
            # start a new participant while the native publication fence is held.
            import sys

            generation = sys.modules.get("tldw_chatbook.RAG_Search.generation")
            if generation is not None:
                generation._clear_caches()
            journal._append(parent, "publication_started", {})
        _pending(
            journal,
            context,
            targets=_publication_targets(prepared),
            durable=True,
        )
        journal._flush_records(parent)
        for item in prepared.artifacts:
            _check_parents(item)
            state = _states(prepared, logical_id=item.logical_id)[item.logical_id]
            if state == "uncertain":
                raise ValueError("publication_objects_changed")
            if state == "staged" and item.previous is not None:
                _check_directory_states(prepared, journal._records(parent))
                intent = _begin_move(journal, parent, item, "retire")
                _retire(item)
                _complete_move(journal, parent, prepared, intent, moved=True)
                journal._append(
                    parent,
                    "artifact_retired",
                    {
                        "logical_id": item.logical_id,
                        "observed": observe_artifact(
                            Path(item.retained), metadata=True
                        ),
                        "directories": _directory_observations(prepared, item),
                    },
                )
                state = "retired"
            if item.candidate is not None and state in {"staged", "retired"}:
                _check_directory_states(prepared, journal._records(parent))
                if not _matches(item.candidate, item.candidate.path):
                    raise ValueError("candidate_content_changed")
                parents = {row.path: (row.device, row.inode) for row in item.parents}
                intent = _begin_move(journal, parent, item, "publish")
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
                _complete_move(journal, parent, prepared, intent, moved=True)
                journal._append(
                    parent,
                    "artifact_published",
                    {
                        "logical_id": item.logical_id,
                        "observed": observe_artifact(Path(item.target)),
                        "directories": _directory_observations(prepared, item),
                    },
                )


def _installed_file_digest(path, size):
    with reader._regular(path) as stream:
        before = os.fstat(stream.fileno())
        if (
            before.st_size != size
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
        ):
            raise ValueError("installed_content_changed")
        remaining = size
        digest = hashlib.sha256()
        while remaining:
            block = stream.read(min(remaining, 64 * 1024))
            if not block:
                raise ValueError("installed_content_changed")
            digest.update(block)
            remaining -= len(block)
        if stream.read(1) or reader._identity(before) != reader._identity(
            os.fstat(stream.fileno())
        ):
            raise ValueError("installed_content_changed")
        return digest.hexdigest()


def _installed_metadata(
    path, expected, metadata, *, previous=None, parent_identity=None
):
    """Apply supported metadata only through the checked object's native handle."""
    with pinned_directory(path.parent) as parent:
        parent_info = os.fstat(parent)
        if (
            parent_identity is not None
            and (parent_info.st_dev, parent_info.st_ino) != parent_identity
        ):
            raise ValueError("publication_parent_changed")
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            if previous is not None and (
                (stat.S_IMODE(before.st_mode), before.st_mtime_ns)
                not in _directory_transition_states(
                    previous, metadata["mode"], metadata["mtime_ns"]
                )
            ):
                raise ValueError("directory_metadata_changed")
            if (
                before.st_dev,
                before.st_ino,
            ) != expected or before.st_uid != os.geteuid():
                raise ValueError("installed_identity_changed")
            if not (stat.S_ISREG(before.st_mode) or stat.S_ISDIR(before.st_mode)):
                raise ValueError("installed_kind_changed")
            os.fchmod(fd, metadata["mode"])
            os.utime(fd, ns=(before.st_atime_ns, metadata["mtime_ns"]))
            os.fsync(fd)
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (named.st_dev, named.st_ino) != expected:
                raise ValueError("installed_identity_changed")
            flush_directory(parent)
        finally:
            os.close(fd)


def _validate_installed(journal, candidate, plan):
    """Recheck published objects, validate disposable copies, then persist proof."""
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from .owner_registry import install_adapters
    from .staging import _items

    with journal._locked(exclusive=True) as parent:
        records = journal._records(parent)
        committed = (
            records[-1] if records and records[-1].event == "committed" else None
        )
        receipt_row = next(
            (row for row in records if row.event == "candidate_staged"), None
        )
        prepared_row = next((row for row in records if row.event == "prepared"), None)
        if (
            receipt_row is None
            or prepared_row is None
            or not any(row.event == "publication_started" for row in records)
        ):
            raise ValueError("publication_incomplete")
        receipt = _CandidateReceipt.model_validate(receipt_row.evidence)
        prepared = _Prepared.model_validate(prepared_row.evidence)
        context = prepared.publication
        if (
            context is None
            or receipt.plan_digest != _plan_digest(plan)
            or context.plan_digest != receipt.plan_digest
            or receipt.stage.path != str(candidate)
            or context.descriptor != receipt.descriptor
        ):
            raise ValueError("publication_context_unverified")
        if not _matches(receipt.descriptor, str(candidate / "candidate.json")):
            raise ValueError("candidate_receipt_changed")
        with pinned_directory(candidate) as fd:
            descriptor = _read(fd, "candidate.json")
        with reader._regular(journal.root / "verified-manifest.json") as stream:
            info = os.fstat(stream.fileno())
            if (
                info.st_uid != os.geteuid()
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                raise ValueError("verified_manifest_changed")
            manifest = stream.read(ArchiveLimits().manifest_bytes + 1)
            if reader._identity(info) != reader._identity(os.fstat(stream.fileno())):
                raise ValueError("verified_manifest_changed")
        if (
            len(manifest) > ArchiveLimits().manifest_bytes
            or hashlib.sha256(manifest).hexdigest() != receipt.manifest_digest
        ):
            raise ValueError("verified_manifest_changed")
        doc = reader._manifest(manifest, ArchiveLimits(), True)
        raw_configs = {}
        if plan.local_snapshot is not None:
            from .later_rollback import verify_snapshot_source

            _, source = verify_snapshot_source(plan)
            if source.manifest_digest != receipt.manifest_digest:
                raise ValueError("local_snapshot_source_changed")
            raw_configs = {
                row.logical_id: row for row in doc.files if row.owner_id == "config"
            }
        rows = [*descriptor["artifacts"], *descriptor.get("containers", [])]
        expected = {
            item.path: (item.device, item.inode) for item in prepared.installed_paths
        }
        if set(expected) != {row["destination"] for row in rows}:
            raise ValueError("installed_identity_required")

        owners = {owner.owner_id: owner for owner in install_adapters()}
        items = _items(doc, plan)
        sqlite_paths = []
        for key, item in items.items():
            owner = owners[item.owner]
            role_check = getattr(owner, "restore_role", None)
            policy = owner.schema_policy()
            role = (
                role_check(item)
                if callable(role_check)
                else "sqlite"
                if policy is not None and policy.schema_sql
                else "file"
            )
            if item.metadata.kind == "file" and role == "sqlite":
                sqlite_paths.append(dict(plan.restore)[key])

        def verify():
            _check_directory_states(prepared, journal._records(parent))
            for path in sqlite_paths:
                if any(
                    os.path.lexists(str(path) + suffix)
                    for suffix in ("-wal", "-shm", "-journal")
                ):
                    raise ValueError("installed_sqlite_sidecar_present")
            _pending(
                journal,
                context,
                targets=_publication_targets(prepared),
                committed=committed,
            )
            states = _states(prepared)
            for item in prepared.artifacts:
                _check_parents(item)
                required = "retired" if item.action == "retire" else "published"
                if states[item.logical_id] != required:
                    raise ValueError("installed_objects_changed")
            for row in rows:
                path = Path(row["destination"])
                info = path.lstat()
                if (info.st_dev, info.st_ino) != expected[str(path)]:
                    raise ValueError("installed_identity_changed")
                if row["kind"] == "file" and (
                    info.st_size != row["size"]
                    or _installed_file_digest(path, row["size"]) != row["sha256"]
                ):
                    raise ValueError("installed_content_changed")

        verify()
        metadata_paths = {item.previous.path for item in prepared.directory_metadata}
        if any(
            row["kind"] == "directory"
            and expected[row["destination"]] != tuple(row["identity"][:2])
            and row["destination"] not in metadata_paths
            for row in rows
        ):
            raise ValueError("installed_directory_rollback_required")
        if metadata_paths:
            rollback = next(
                (row for row in records if row.event == "rollback_verified"), None
            )
            if rollback is None or not {
                item.logical_id for item in prepared.directory_metadata
            } <= set(rollback.evidence["coverage"]):
                raise ValueError("installed_directory_rollback_required")
            _check_directory_states(prepared, records)
        prior_validation = next(
            (row for row in reversed(records) if row.event == "installed_validated"),
            None,
        )
        if prior_validation is not None:
            from .journal import _Object

            for evidence in prior_validation.evidence["artifacts"]:
                if not _matches(
                    _Object.model_validate(evidence), evidence["path"], metadata=True
                ):
                    raise ValueError("installed_metadata_changed")
        _pending(
            journal,
            context,
            targets=_publication_targets(prepared),
            durable=True,
            committed=committed,
        )
        journal._flush_records(parent)
        # The existing SQLite seam accepts disposable candidates, never live WAL
        # readers. Copy exact installed bytes, then check installed evidence again.
        work = Path(tempfile.mkdtemp(prefix="installed-check-", dir=candidate))
        try:
            candidates = {}
            physical_candidates = {}
            restored = dict(plan.restore)
            topology = {}
            directories = {row.logical_id for row in doc.directories}
            for record in (*doc.directories, *doc.files):
                kind = "directory" if record.logical_id in directories else "file"
                topology[record.logical_id] = (
                    record.root_id,
                    record.parent_id,
                    record.relative_path,
                    kind,
                )
                if record.logical_id not in restored:
                    continue
                source = restored[record.logical_id]
                physical_key = (str(source), expected[str(source)])
                if physical_key in physical_candidates:
                    candidates[record.logical_id] = physical_candidates[physical_key]
                    continue
                destination = work / record.root_id / record.relative_path
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if kind == "directory":
                    destination.mkdir(mode=0o700, exist_ok=True)
                else:
                    row = next(
                        row for row in rows if row["logical_id"] == record.logical_id
                    )
                    remaining = row["size"]
                    digest = hashlib.sha256()
                    with (
                        reader._regular(source) as stream,
                        destination.open("xb") as output,
                    ):
                        while remaining:
                            block = stream.read(min(remaining, 64 * 1024))
                            if not block:
                                raise ValueError("installed_content_changed")
                            output.write(block)
                            digest.update(block)
                            remaining -= len(block)
                        if stream.read(1) or digest.hexdigest() != row["sha256"]:
                            raise ValueError("installed_content_changed")
                    destination.chmod(0o600)
                candidates[record.logical_id] = destination
                physical_candidates[physical_key] = destination
            verify()
            synthetic = {row.logical_id for row in doc.directories if row.synthetic}
            # A fresh worker owns only private read probes. Executor shutdown
            # waits even if result() is interrupted, keeping native maintenance
            # and these private files alive until every callback has retired.
            private_items = {
                key: replace(item, path=candidates[key]) for key, item in items.items()
            }
            with ThreadPoolExecutor(max_workers=1) as worker:
                worker.submit(
                    _validate_installed_copies,
                    private_items,
                    candidates,
                    topology,
                    synthetic,
                    owners,
                    raw_configs,
                ).result()
            verify()
            for row in sorted(
                rows,
                key=lambda row: (
                    row["kind"] == "directory",
                    -len(Path(row["destination"]).parts),
                ),
            ):
                directory = next(
                    (
                        item
                        for item in prepared.directory_metadata
                        if item.previous.path == row["destination"]
                    ),
                    None,
                )
                if directory is not None:
                    _apply_directory_metadata(journal, parent, prepared, directory)
                else:
                    _installed_metadata(
                        Path(row["destination"]),
                        expected[row["destination"]],
                        row["applied_metadata"],
                    )
            verify()
            for row in rows:
                info = Path(row["destination"]).lstat()
                applied = row["applied_metadata"]
                if (
                    stat.S_IMODE(info.st_mode) != applied["mode"]
                    or info.st_mtime_ns != applied["mtime_ns"]
                ):
                    raise ValueError("installed_metadata_changed")
            evidence = [
                observe_artifact(Path(path), metadata=True) for path in sorted(expected)
            ]
            proof = {
                "plan_digest": receipt.plan_digest,
                "descriptor_digest": receipt.descriptor.sha256,
                "manifest_digest": receipt.manifest_digest,
                "artifacts": evidence,
            }
            if prior_validation is None:
                journal._append(parent, "installed_validated", proof)
            elif prior_validation.evidence != proof:
                raise ValueError("installed_evidence_changed")
            return proof
        finally:
            shutil.rmtree(work)


def _apply_directory_metadata(journal, parent, prepared, item):
    records = journal._records(parent)
    _check_directory_states(prepared, records)
    if any(
        row.event == "directory_metadata_applied"
        and row.evidence["logical_id"] == item.logical_id
        for row in records
    ):
        return
    if not any(
        row.event == "directory_metadata_started"
        and row.evidence["logical_id"] == item.logical_id
        for row in records
    ):
        journal._append(
            parent,
            "directory_metadata_started",
            {
                "logical_id": item.logical_id,
                "before": _directory_state(item.previous.path).model_dump(),
                "applied": item.applied.model_dump(),
            },
        )
    journal._flush_records(parent)
    intent = next(
        row
        for row in journal._records(parent)
        if row.event == "directory_metadata_started"
        and row.evidence["logical_id"] == item.logical_id
    )
    _installed_metadata(
        Path(item.previous.path),
        (item.previous.device, item.previous.inode),
        item.applied.model_dump(),
        previous=_DirectoryState.model_validate(intent.evidence["before"]),
        parent_identity=(item.parent.device, item.parent.inode),
    )
    state = _directory_state(item.previous.path)
    if (state.mode, state.mtime_ns) != (item.applied.mode, item.applied.mtime_ns):
        raise ValueError("directory_metadata_changed")
    journal._append(
        parent,
        "directory_metadata_applied",
        {"logical_id": item.logical_id, "observed": state.model_dump()},
    )


def _validate_installed_copies(
    items, candidates, topology, synthetic, owners, raw_configs=None
):
    """Validate operation-private copies without receiving live native authority."""
    from types import MappingProxyType

    from .sqlite_validation import validate_candidate
    from .storage_admission import _preview_reads

    with _preview_reads():
        for key, item in items.items():
            owner = owners[item.owner]
            if item.metadata.kind == "file":
                role_check = getattr(owner, "restore_role", None)
                policy = owner.schema_policy()
                role = (
                    role_check(item)
                    if callable(role_check)
                    else "sqlite"
                    if policy is not None and policy.schema_sql
                    else "file"
                )
                if role == "sqlite":
                    issues = validate_candidate(
                        owner, candidates[key], Event(), migrate=False
                    )
                elif item.owner == "config" and key in (raw_configs or {}):
                    from .later_rollback import validate_snapshot_config

                    issues = validate_snapshot_config(raw_configs[key], candidates[key])
                else:
                    validator = getattr(owner, "validate_restore", None)
                    issues = (
                        validator(item, candidates[key])
                        if callable(validator)
                        else owner.validate(candidates[key])
                    )
                if issues:
                    raise ValueError(issues[0])
            if key in synthetic:
                continue
            validator = getattr(owner, "validate_restore_dependencies", None)
            legacy = getattr(owner, "validate_dependencies", None)
            if callable(validator):
                issues = validator(
                    item,
                    candidates[key],
                    MappingProxyType(candidates),
                    topology=MappingProxyType(topology),
                )
            elif callable(legacy):
                issues = legacy(item, candidates[key], MappingProxyType(candidates))
            else:
                issues = ()
            if issues:
                raise ValueError(issues[0])


def _move_child(parent, path):
    """Observe an immediate sibling without reading its payload or following links."""
    from .journal import _MoveChild

    info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    return _MoveChild(
        path=str(path),
        device=info.st_dev,
        inode=info.st_ino,
        mode=info.st_mode,
        owner=info.st_uid,
        size=info.st_size,
        links=info.st_nlink,
        mtime_ns=info.st_mtime_ns,
        ctime_ns=info.st_ctime_ns,
    )


def _move_topology(source, destination):
    """Pin both parents and record unaffected immediate entries, without capture."""
    from .journal import MAX_EVENTS

    parents = []
    for path in sorted({source.parent, destination.parent}):
        excluded = {p.name for p in (source, destination) if p.parent == path}
        state = _directory_state(path)
        with pinned_directory(path) as fd:
            names = sorted(os.listdir(fd))
            if len(names) > MAX_EVENTS:
                raise ValueError("move_topology_limit")
            children = {
                name: _move_child(fd, path / name).model_dump()
                for name in names
                if name not in excluded
            }
            if _directory_state(path) != state or names != sorted(os.listdir(fd)):
                raise ValueError("move_parent_changed")
            if any(
                _move_child(fd, path / name).model_dump() != value
                for name, value in children.items()
            ):
                raise ValueError("move_parent_changed")
        parents.append({"state": state.model_dump(), "children": children})
    return parents


def _move_position(intent):
    """Only the exact before/after name transition can reconcile an interrupted move."""
    from .journal import _absent

    source, target = Path(intent.source.path), Path(intent.destination)
    before = _matches(intent.source, str(source)) and _absent(str(target))
    after = _absent(str(source)) and _matches(intent.source, str(target))
    if before == after:
        raise ValueError("move_state_uncertain")
    for entry in intent.parents:
        path = Path(entry.state.path)
        current = _directory_state(path)
        if (current.device, current.inode, current.mode) != (
            entry.state.device,
            entry.state.inode,
            entry.state.mode,
        ):
            raise ValueError("move_parent_changed")
        if before and current != entry.state:
            raise ValueError("move_parent_changed")
        excluded = {p.name for p in (source, target) if p.parent == path}
        with pinned_directory(path) as fd:
            names = set(os.listdir(fd))
            if names - excluded != set(entry.children):
                raise ValueError("move_topology_changed")
            for name, child in entry.children.items():
                if _move_child(fd, path / name) != child:
                    raise ValueError("move_topology_changed")
    return "after" if after else "before"


def _complete_move(journal, parent, prepared, intent, *, moved):
    from .journal import _evidence_digest

    if _move_position(intent) != ("after" if moved else "before"):
        raise ValueError("move_state_uncertain")
    directories = [
        _directory_state(item.previous.path).model_dump()
        for item in prepared.directory_metadata
    ]
    journal._append(
        parent,
        "move_observed",
        {
            "intent_digest": _evidence_digest(intent.model_dump()),
            "moved": moved,
            "directories": directories,
        },
    )
    journal._flush_records(parent)


def _begin_move(journal, parent, item, step):
    from .journal import _MoveIntent

    if step == "retire":
        source, destination = Path(item.target), Path(item.retained)
    elif step == "publish":
        source, destination = Path(item.candidate.path), Path(item.target)
    elif step == "unpublish":
        source, destination = Path(item.target), Path(item.candidate.path)
    else:
        source, destination = Path(item.retained), Path(item.target)
    proof = {
        "logical_id": item.logical_id,
        "step": step,
        "source": observe_artifact(source),
        "destination": str(destination),
        "parents": _move_topology(source, destination),
    }
    intent = _MoveIntent.model_validate(proof)
    if _move_position(intent) != "before":
        raise ValueError("move_state_uncertain")
    journal._append(parent, "move_intended", proof)
    journal._flush_records(parent)
    return intent


def _reverse_native_move(intent):
    """Preserve checked original bytes and metadata through native no-replace."""
    source, destination = Path(intent.source.path), Path(intent.destination)
    if _move_position(intent) != "before":
        raise ValueError("move_state_uncertain")
    with (
        pinned_directory(source.parent) as left,
        pinned_directory(destination.parent) as right,
    ):
        parents = {
            entry.state.path: (entry.state.device, entry.state.inode)
            for entry in intent.parents
        }
        for path, fd in ((source.parent, left), (destination.parent, right)):
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != parents[str(path)]:
                raise ValueError("move_parent_changed")
        info = os.fstat(left)
        if info.st_dev != os.fstat(right).st_dev:
            raise ValueError("cross_volume_retirement_unqualified")
        for operation in (
            "publish_new",
            "publish_directory"
            if intent.source.kind == "directory"
            else "publish_file",
        ):
            allowed, reason = _qualified_identity(operation, native_identity(right))
            if not allowed:
                raise ValueError(reason)
        fd = os.open(
            source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=left
        )
        try:
            _flush_original(fd, info.st_dev)
            if _move_position(intent) != "before" or os.stat(
                source.name, dir_fd=left, follow_symlinks=False
            ) != os.fstat(fd):
                raise ValueError("move_state_uncertain")
            _rename_new(left, source.name, right, destination.name)
            flush_directory(right)
            flush_directory(left)
        finally:
            os.close(fd)
    if _move_position(intent) != "after":
        raise ValueError("move_state_uncertain")


def _reconcile_moves(journal, parent, prepared, *, finish):
    """Reconcile actual intent boundaries, never infer a missing move from labels."""
    from .journal import _MoveIntent

    records = journal._records(parent)
    if records[-1].event == "move_intended":
        intent = _MoveIntent.model_validate(records[-1].evidence)
        position = _move_position(intent)
        if position == "before" and finish:
            _reverse_native_move(intent)
            position = "after"
        _complete_move(journal, parent, prepared, intent, moved=position == "after")
    records = journal._records(parent)
    if any(row.event == "rollback_started" for row in records):
        return
    for index, record in enumerate(records):
        if record.event != "move_observed" or not record.evidence["moved"]:
            continue
        intent = _MoveIntent.model_validate(records[index - 1].evidence)
        event = {"retire": "artifact_retired", "publish": "artifact_published"}.get(
            intent.step
        )
        if event is None or any(
            row.event == event and row.evidence["logical_id"] == intent.logical_id
            for row in records
        ):
            continue
        item = next(
            item for item in prepared.artifacts if item.logical_id == intent.logical_id
        )
        journal._append(
            parent,
            event,
            {
                "logical_id": item.logical_id,
                "observed": observe_artifact(
                    Path(intent.destination), metadata=event == "artifact_retired"
                ),
                "directories": [
                    row
                    for row in record.evidence["directories"]
                    if row["path"] == str(Path(item.target).parent)
                ],
            },
        )
        journal._flush_records(parent)
