"""Fixed rollback restart authentication over original native archive members."""

import hashlib
import zipfile
from pathlib import Path
from uuid import uuid4

from tldw_chatbook.DB.private_sqlite import open_recovery_validation
from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, native_files, sqlite_validation
from .limits import ArchiveLimits
from .mobile_captured_sqlite import _retire_staging
from .mobile_native_operation import _require_readers_retired
from .mobile_rollback_io import observation
from .storage_admission import _CaptureFileDescriptors, _local


def _authenticate_stages(session, frame, password, cancel):
    """Decrypt now; compare exact fixed members and validate disposable owners."""
    frame.verify()
    budget = frame.budget
    budget.container(len(frame.records) * 4)
    stages = [
        row for row in frame.records if row.event == "mobile_rollback_stage_intended"
    ]
    canonical = next(
        row
        for row in frame.records
        if row.event == "mobile_capture_verified"
        and row.evidence["purpose"] == "canonical"
    )
    artifact = canonical.evidence["archive"]
    budget.value(artifact, text_copies=8)
    source = Path(artifact["path"])
    root = source.parent / (".mobile-rollback-read-" + uuid4().hex)
    root_identity = None
    with session._discovery_reads():
        scope = _local.discovery_scope  # The actual installed native issuer.
        custody = _CaptureFileDescriptors(scope)
        try:
            if observation(source, budget, custody, cancel) != artifact:
                raise ValueError("mobile_capture_changed")
            native_files.create_private_directory(root, _custody=custody)
            info = root.lstat()
            root_identity = info.st_dev, info.st_ino
            sealed, document = archive_reader._acquire_with_manifest(
                source,
                root,
                ArchiveLimits(),
                password,
                cancel,
                _custody=custody,
                _read_budget=budget,
            )
            if (
                sealed.encrypted_source is None
                or hashlib.sha256(sealed.manifest_bytes).hexdigest()
                != canonical.evidence["manifest_digest"]
            ):
                raise ValueError("mobile_capture_changed")
            # This is the actual document just parsed and authenticated by the
            # installed reader in this call, not a cached/caller-supplied model.
            if (
                document.consistency != "coherent"
                or document.credential_policy != "rollback"
            ):
                raise ValueError("mobile_rollback_capture_incomplete")
            budget.container(
                len(document.files)
                + len(document.owners)
                + len(document.producer_inventory)
            )
            with archive_reader._regular(sealed.path, _custody=custody) as stream:
                central = archive_reader._central_preflight(
                    stream, ArchiveLimits(), _read_budget=budget
                )
                with zipfile.ZipFile(stream) as archive:
                    offsets = archive_reader._local_ranges(
                        stream, archive.infolist(), central
                    )
                    for stage in stages:
                        scope.check()
                        value = stage.evidence
                        matches = [
                            row
                            for row in document.files
                            if row.logical_id == value["logical_id"]
                            and row.owner_id == value["owner_id"]
                        ]
                        declarations = [
                            row
                            for row in document.producer_inventory
                            if row.logical_id == value["logical_id"]
                            and row.owner_id == value["owner_id"]
                        ]
                        owners = [
                            row
                            for row in document.owners
                            if row.owner_id == value["owner_id"]
                        ]
                        if (
                            len(matches) != 1
                            or len(declarations) != 1
                            or declarations[0].status != "included"
                            or len(owners) != 1
                        ):
                            raise ValueError("mobile_rollback_member_changed")
                        payload = matches[0]
                        if (
                            payload.payload,
                            payload.size,
                            payload.sha256,
                            payload.metadata.model_dump() if payload.metadata else None,
                        ) != (
                            value["member_path"],
                            value["payload_size"],
                            value["payload_sha256"],
                            value["metadata"],
                        ):
                            raise ValueError("mobile_rollback_member_changed")
                        budget.reserve(2 * 65536 + 16384)
                        budget.value(value["owner_id"], text_copies=8)
                        candidate = root / (value["owner_id"] + ".db")
                        digest, size = hashlib.sha256(), 0
                        with native_files.create_private_file(
                            candidate, _custody=custody
                        ) as fd:
                            member = archive.getinfo(payload.payload)
                            for chunk in archive_reader._member_chunks(
                                stream, member, offsets[member.filename], cancel
                            ):
                                size += len(chunk)
                                if size > payload.size:
                                    raise ValueError("mobile_rollback_member_changed")
                                digest.update(chunk)
                                view = memoryview(chunk)
                                while view:
                                    count = os.write(fd, view)
                                    if count <= 0:
                                        raise OSError(
                                            "mobile_rollback_snapshot_write_failed"
                                        )
                                    view = view[count:]
                        if size != payload.size or digest.hexdigest() != payload.sha256:
                            raise ValueError("mobile_rollback_member_changed")
                        with open_recovery_validation(
                            value["owner_id"],
                            candidate,
                            writable=False,
                            with_restrictions=True,
                            cancel=cancel,
                            _custody=custody,
                        ) as (connection, restrictions):
                            owner = sqlite_validation._installed_owner(
                                value["owner_id"]
                            )
                            issues, version = sqlite_validation._check(
                                connection, owner, owner.schema_policy(), restrictions
                            )
                            if issues or version != owners[0].schema_version:
                                raise ValueError("mobile_rollback_schema_changed")
                        if (
                            archive_reader._hash(candidate, cancel, _custody=custody)
                            != payload.sha256
                        ):
                            raise ValueError("mobile_rollback_member_changed")
            if (
                archive_reader._hash(sealed.path, cancel, _custody=custody)
                != sealed.digest
                or observation(source, budget, custody, cancel) != artifact
            ):
                raise ValueError("mobile_capture_changed")
            frame.verify()
        finally:
            # Crypto output pins and descriptor-backed SQLite leases own names
            # inside this tree. Retire the actual native resources in reverse
            # ownership order before removing their witnessed staging paths.
            budget.container(len(scope.resources))
            for resource in reversed(tuple(scope.resources)):
                resource.retire()
            # The installed native remover owns only this witnessed disposable
            # tree, after descriptor/SQLite physical retirement has completed.
            if root_identity is not None:
                _retire_staging(scope, root, root_identity, _read_budget=budget)
    _require_readers_retired(session)


def _resume_rollback(session, journal, password, cancel):
    """Resume the actual rollback direction; an import still uses its own entry."""
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from . import bootstrap
    from .control_records import _Pending
    from .inventory import read_profile_config
    from .journal import _mobile_state
    from .mobile_consolidation import _source_claim_locations
    from .mobile_consolidation_apply import _SourceClaim
    from .mobile_journal_frame import _locked_mobile_rollback
    from .mobile_rollback import (
        _append_capacity,
        _complete_rollback,
        _file_observation_capacity,
        _move_capacity,
        _reference,
        _reference_cost,
    )
    from .mobile_rollback_io import (
        metadata,
        move_original,
        move_position,
        parent_census,
        sync_moved,
    )
    from .profile_paths import legacy_mobile_dir
    from .storage_admission import (
        _mobile_admission,
        _mobile_check_witnesses,
        _mobile_coordinator_budget,
    )

    admission = _mobile_admission(session)
    if admission[2] is None:
        admission[2] = _TraceReadBudget()
    budget = _mobile_coordinator_budget(session)
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    claim = None
    try:
        # Existing pending/selector data derives only the claim locator. The
        # native locked frame below supplies all current history authority.
        pending, _ = bootstrap._records(
            bootstrap.default_bootstrap_root(), _budget=budget, _custody=custody
        )
        selected = [
            row for row in pending if row["operation_id"] == journal.operation_id
        ]
        if len(selected) != 1:
            raise ValueError("mobile_pending_scope_changed")
        budget.value(selected[0], text_copies=4)
        association = _Pending.model_validate(selected[0])
        if Path(association.control_root) != journal.root.parent:
            raise ValueError("mobile_pending_scope_changed")
        config = read_profile_config(
            Path(association.selectors[0]), _read_budget=budget, _custody=custody
        )
        budget.value(config, text_copies=8)
        budget.reserve(16384)
        source = legacy_mobile_dir(config)
        pairs = _source_claim_locations(source)
        found = [pair for pair in pairs if pair[0].exists() or pair[1].exists()]
        if len(found) != 1:
            raise ValueError("mobile_source_claim_ambiguous")
        pair = found[0]
        path = pair[1]
        claim = _SourceClaim._issue(session, journal, path, pair[0])
        with claim.custody.pinned_directory(path.parent) as parent:
            claim._lock_existing(parent)
        from .mobile_lineage import _resolve_owned_claim

        _resolve_owned_claim(claim, journal, source)
        if claim.path != claim.original_path:
            from .mobile_lineage import _read_lineage
            from .mobile_lineage_ancestors import _issue_ancestors

            leaf = _read_lineage(claim.original_path, source, budget, claim.custody)
            if leaf is None or leaf[0].root != journal.root:
                raise ValueError("mobile_source_claim_changed")
            from .mobile_journal_frame import _bind_rollback

            _bind_rollback(journal, session, claim, leaf[1], budget)
            _issue_ancestors(claim, journal, source)
        with _locked_mobile_rollback(journal, session, claim, cancel) as frame:
            if not any(row.event == "mobile_rollback_started" for row in frame.records):
                return False
            if any(row.event == "mobile_committed" for row in frame.records):
                raise ValueError("mobile_rollback_unavailable")
            session._mobile_attempted_journal = journal
            prepared, _ = _mobile_state(frame.records)
            _require_readers_retired(session)
            _authenticate_stages(session, frame, password, cancel)
            budget.container(len(frame.records) * 4)
            stages = []
            for row in frame.records:
                if row.event == "mobile_rollback_stage_intended":
                    reference = _reference(frame, row)
                    observed = next(
                        (
                            item
                            for item in frame.records
                            if item.event == "mobile_rollback_stage_observed"
                            and item.evidence["intended"] == reference
                        ),
                        None,
                    )
                    if observed is None:
                        raise ValueError("mobile_rollback_stage_unassociated")
                    stages.append((row, observed))
            required = {
                row.owner_id
                for row in prepared.targets
                if row.witnesses[-1].inode is not None
            }
            if {row.evidence["owner_id"] for row, _ in stages} != required:
                raise ValueError("mobile_rollback_stage_missing")
            last = frame.records[-1]
            if last.event == "mobile_rollback_move_intended":
                intent = last.evidence
                position = move_position(intent, budget, custody, cancel)
                frame.verify()
                if position == "before":
                    move_original(
                        intent,
                        budget,
                        custody,
                        cancel,
                        _before_effect=lambda cost: _move_capacity(
                            frame, intent, last, cost
                        ),
                    )
                else:
                    sync_moved(intent, budget, custody, cancel)
                if move_position(intent, budget, custody, cancel) != "after":
                    raise ValueError("mobile_rollback_move_uncertain")
                frame.append(
                    "mobile_rollback_move_observed",
                    {
                        "version": 1,
                        "intended": _reference(frame, last),
                        "destination": observation(
                            Path(intent["destination"]), budget, custody, cancel
                        ),
                        "parents": [
                            parent_census(
                                Path(row["state"]["path"]), budget, custody, cancel
                            )["state"]
                            for row in intent["parents"]
                        ],
                    },
                )
            elif last.event == "mobile_rollback_metadata_intended":
                intent = last.evidence
                target = Path(intent["object"]["path"])
                if observation(target, budget, custody, cancel) != intent[
                    "object"
                ] or metadata(target, intent["object"], budget, custody) not in (
                    intent["previous"],
                    intent["applied"],
                ):
                    raise ValueError("mobile_rollback_metadata_changed")
                frame.verify()
                frame.require_capacity(
                    8192
                    + 16 * len(str(target))
                    + _file_observation_capacity(target, custody)
                    + _append_capacity(frame, intent)
                    + 8192
                    + _reference_cost(frame, last)
                )
                if (
                    metadata(
                        target,
                        intent["object"],
                        budget,
                        custody,
                        apply=intent["applied"],
                    )
                    != intent["applied"]
                ):
                    raise ValueError("mobile_rollback_metadata_changed")
                frame.append(
                    "mobile_rollback_metadata_observed",
                    {
                        "version": 1,
                        "intended": _reference(frame, last),
                        "object": observation(target, budget, custody, cancel),
                        "metadata": intent["applied"],
                    },
                )
            prepared, witnesses = _mobile_state(frame.records)
            _mobile_check_witnesses(frame.records, witnesses)
            for stage, observed in stages:
                restored = next(
                    (
                        row
                        for row in frame.records
                        if row.event == "mobile_rollback_move_observed"
                        and frame.records[
                            row.evidence["intended"]["sequence"]
                        ].evidence["step"]
                        == "restore_snapshot"
                        and frame.records[
                            row.evidence["intended"]["sequence"]
                        ].evidence["subject"]["owner_id"]
                        == stage.evidence["owner_id"]
                    ),
                    None,
                )
                expected = (
                    restored.evidence["destination"]
                    if restored
                    else observed.evidence["staged"]
                )
                if (
                    observation(Path(expected["path"]), budget, custody, cancel)
                    != expected
                ):
                    raise ValueError("mobile_rollback_stage_changed")
            for row in frame.records:
                if (
                    row.event == "mobile_capture_verified"
                    and observation(
                        Path(row.evidence["archive"]["path"]), budget, custody, cancel
                    )
                    != row.evidence["archive"]
                ):
                    raise ValueError("mobile_capture_changed")
                if (
                    row.event == "mobile_rollback_move_observed"
                    and frame.records[row.evidence["intended"]["sequence"]].evidence[
                        "step"
                    ]
                    != "restore_snapshot"
                    and observation(
                        Path(row.evidence["destination"]["path"]),
                        budget,
                        custody,
                        cancel,
                    )
                    != row.evidence["destination"]
                ):
                    raise ValueError("mobile_rollback_retained_changed")
            started = next(
                row for row in frame.records if row.event == "mobile_rollback_started"
            )
            _complete_rollback(
                session, journal, claim, frame, custody, started, stages, cancel
            )
        return True
    finally:
        try:
            custody.retire()
        finally:
            if claim is not None:
                claim.retire()
