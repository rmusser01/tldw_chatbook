"""Explicit pre-cutover restoration of authenticated native mobile snapshots."""

import sys
from contextlib import ExitStack
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from . import bootstrap, control_records
from .archive_reader import _check
from .journal import _mobile_ref, _mobile_state
from .mobile_consolidation_apply import _SourceClaim
from .mobile_native_operation import _require_readers_retired
from .mobile_rollback_io import (
    copy_snapshot,
    metadata,
    move_original,
    move_position,
    observation,
    parent_census,
)
from .mobile_rollback_records import _retained_name, _stage_name
from .native_files import flush_directory
from .storage_admission import (
    _CaptureFileDescriptors,
    _mobile_check_witnesses,
    _mobile_coordinator_budget,
)


def _directory(census):
    return {key: census["state"][key] for key in ("path", "device", "inode")}


def _reference_cost(frame, record):
    frame.check()
    frame.budget.reserve(256)
    return 512 + 32 * frame.files[f"{record.sequence:06d}.json"][0][5]


def _reference(frame, record):
    frame.budget.reserve(_reference_cost(frame, record))
    return _mobile_ref(record)


def _append_capacity(frame, evidence):
    """Bound the fixed native observed-event allocation and full append check."""
    budget = frame.budget

    def costs(value):
        budget.reserve(256)  # Cost arithmetic/tuple before each retained pair.
        if isinstance(value, str):
            return 64 + 32 * len(value), 2 + 12 * len(value)
        if isinstance(value, dict):
            amount, encoded = 64 + 32 * len(value), 2 + 2 * len(value)
            for key, item in value.items():
                left, right = costs(key)
                amount, encoded = amount + left, encoded + right
                left, right = costs(item)
                amount, encoded = amount + left, encoded + right
            return amount, encoded
        if isinstance(value, list):
            amount, encoded = 64 + 32 * len(value), 2 + len(value)
            for item in value:
                left, right = costs(item)
                amount, encoded = amount + left, encoded + right
            return amount, encoded
        return 64, 32

    value_cost, encoded_bound = costs(evidence)
    encoded_bound += 4096  # Fixed event envelope, operation ID and digest.
    return (
        frame.verification_cost()
        + value_cost
        + 8192
        + 2048
        + 256 * ((encoded_bound + 4095) // 4096)
        + 256
    )


def _file_observation_capacity(path, custody, *, expected_size=None):
    from .journal import _artifact_component_limit

    with custody.pinned_directory(path.parent) as parent:
        component = _artifact_component_limit(parent)
    # Exact one-file observation reservations, including native content buffers.
    size = path.stat().st_size if expected_size is None else expected_size
    return 159872 + 12 * component + 192 * len(str(path)) + 512 * (size // 65536 + 3)


def _move_capacity(frame, intent, intended, validation_cost):
    """Price the two post-rename observations and durable native append."""
    budget = frame.budget
    budget.value(intent["source"], text_copies=4)
    budget.container(len(intent["parents"]) * 16)
    for row in intent["parents"]:
        budget.value(row["state"], text_copies=4)
    evidence = {
        "version": 1,
        "intended": _reference(frame, intended),
        "destination": {**intent["source"], "path": intent["destination"]},
        "parents": [row["state"] for row in intent["parents"]],
    }

    delta = abs(len(intent["source"]["path"]) - len(intent["destination"]))
    # Each post-move observation sees the identical object/tree and siblings,
    # with only the two fixed path spellings exchanged. Timestamp scalars have
    # fixed representation reservations. Actual post-effect charges remain.
    observation_cost = (
        2 * validation_cost + 4096 + 1024 * (delta + 1) * len(intent["parents"])
    )
    append_cost = _append_capacity(frame, evidence)
    frame.require_capacity(
        observation_cost + append_cost + _reference_cost(frame, intended)
    )


def _restore_captured(captured, password, cancel):
    """Keep the actual capture session and original source flock until terminal."""
    session, journal, claim = captured.session, captured.journal, captured.claim
    budget = _mobile_coordinator_budget(session)
    custody = _CaptureFileDescriptors(None)
    session._scopes.append(custody)
    locks = ExitStack()
    frame = None

    def records():
        _check(cancel)
        _SourceClaim._check_binding(claim)
        if frame is not None:
            frame.verify()
            current = frame.records
        else:
            with journal._locked(exclusive=False, _custody=custody) as parent:
                current = journal._records(parent, _budget=budget, _custody=custody)
        prepared, _ = _mobile_state(current)
        control_records._mobile_pending(session, journal, prepared)
        return current

    def append(event, evidence):
        _check(cancel)
        _SourceClaim._check_binding(claim)
        if frame is None:
            raise ValueError("mobile_native_journal_required")
        return frame.append(event, evidence)

    try:
        from .mobile_journal_frame import _locked_mobile

        # Capture/claim checks below share this actual native lock frame. They
        # still verify current raw journal evidence before reading its records.
        frame = locks.enter_context(
            _locked_mobile(journal, session, claim.custody, cancel)
        )
        current = records()
        prepared, witnesses = _mobile_state(current)
        if any(
            row.event
            in {"mobile_committed", "mobile_rollback_started", "mobile_rolled_back"}
            for row in current
        ):
            raise ValueError("mobile_rollback_recovery_required")
        _mobile_check_witnesses(current, witnesses)
        claim_record = next(
            row for row in current if row.event == "mobile_claim_observed"
        )
        canonical = next(
            row
            for row in current
            if row.event == "mobile_capture_verified"
            and row.evidence["purpose"] == "canonical"
        )
        stages = []
        # The native reader validates the original cipher, manifest, producer
        # declarations and every fixed SQLite owner before a target is displaced.
        with captured.sqlite_reads(password=password, cancel=cancel) as reads:
            document = reads._canonical_archive[1]
            members = []
            for target, family in zip(
                prepared.targets, ("chat", "workspaces", "runs"), strict=True
            ):
                value = reads._connections.get(("canonical", family))
                if target.witnesses[-1].inode is None:
                    if value is not None:
                        raise ValueError("mobile_rollback_absence_changed")
                    continue
                if value is None:
                    raise ValueError("mobile_rollback_snapshot_missing")
                candidate, digest = value[3:]
                payloads = [
                    row
                    for row in document.files
                    if row.owner_id == target.owner_id and row.sha256 == digest
                ]
                if len(payloads) != 1 or payloads[0].metadata is None:
                    raise ValueError("mobile_rollback_snapshot_ambiguous")
                payload = payloads[0]
                path = Path(target.witnesses[-1].path)
                temporary = path.parent / _stage_name(
                    journal.operation_id, target.owner_id
                )
                if temporary.exists():
                    raise ValueError("mobile_rollback_stage_unassociated")
                parent = parent_census(path.parent, budget, custody, cancel)
                members.append((target, candidate, payload, temporary, parent))
            budget.container(len(members) * 16)
            reads.check()
            started = append(
                "mobile_rollback_started",
                {
                    "version": 1,
                    "prepared": _reference(frame, current[0]),
                    "claim": _reference(frame, claim_record),
                    "canonical_capture": _reference(frame, canonical),
                    "source_fingerprint": prepared.source_fingerprint,
                    "destination_identity": captured.preview.destination.identity,
                },
            )
            for target, candidate, payload, temporary, parent in members:
                reads.check()
                intended = append(
                    "mobile_rollback_stage_intended",
                    {
                        "version": 1,
                        "started": _reference(frame, started),
                        "owner_id": target.owner_id,
                        "logical_id": payload.logical_id,
                        "member_path": payload.payload,
                        "payload_sha256": payload.sha256,
                        "payload_size": payload.size,
                        "target": target.witnesses[-1].path,
                        "temporary": str(temporary),
                        "parent": _directory(parent),
                        "metadata": payload.metadata.model_dump(),
                    },
                )
                frame.verify()
                # The intent contains every variable field of its observation
                # plus additional path/member/metadata fields; it bounds that
                # smaller append without inventing a future physical identity.
                copy_cost = (
                    2 * 65536
                    + 16384
                    + 128
                    + 16 * (len(str(candidate)) + len(str(temporary)))
                )
                frame.require_capacity(
                    copy_cost
                    + _file_observation_capacity(
                        temporary, custody, expected_size=payload.size
                    )
                    + _append_capacity(frame, intended.evidence)
                    + 8192
                    + _reference_cost(frame, intended)
                )
                copy_snapshot(
                    candidate,
                    temporary,
                    payload.size,
                    payload.sha256,
                    budget,
                    custody,
                    cancel,
                )
                actual = observation(temporary, budget, custody, cancel)
                observed = append(
                    "mobile_rollback_stage_observed",
                    {
                        "version": 1,
                        "intended": _reference(frame, intended),
                        "staged": actual,
                    },
                )
                stages.append((intended, observed))
        _require_readers_retired(session)

        _complete_rollback(
            session, journal, claim, frame, custody, started, stages, cancel
        )
    finally:
        try:
            # Propagate the actual body failure through the held context; a
            # cleanup-time success verification must not mask its native cause.
            locks.__exit__(*sys.exc_info())
        finally:
            custody.retire()


def _complete_rollback(
    session, journal, claim, frame, custody, started, stages, cancel
):
    """Continue only exact durable rollback steps in the actual native frame."""
    budget = frame.budget

    def records():
        _check(cancel)
        _SourceClaim._check_binding(claim)
        frame.verify()
        prepared, _ = _mobile_state(frame.records)
        control_records._mobile_pending(session, journal, prepared)
        return frame.records

    def append(event, evidence):
        _check(cancel)
        _SourceClaim._check_binding(claim)
        return frame.append(event, evidence)

    current = records()
    prepared, _ = _mobile_state(current)
    if current[-1].event == "mobile_rolled_back":
        # The restart reader authenticated the original archive and every
        # staged/restored/retained object before reaching this terminal branch.
        for intended, _ in stages:
            path = Path(intended.evidence["target"])
            actual = observation(path, budget, custody, cancel)
            if metadata(path, actual, budget, custody) != intended.evidence["metadata"]:
                raise ValueError("mobile_rollback_metadata_changed")
        _retire_rollback_pending(
            session,
            journal,
            claim,
            frame,
            custody,
            prepared,
            current[-1].evidence,
            cancel,
        )
        return
    budget.container(len(current) * 4)
    ordinal = sum(row.event == "mobile_rollback_move_intended" for row in current)
    retained = [
        row
        for row in current
        if row.event == "mobile_rollback_move_observed"
        and current[row.evidence["intended"]["sequence"]].evidence["step"]
        != "restore_snapshot"
    ]

    def completed_move(subject, step):
        for row in frame.records:
            if row.event == "mobile_rollback_move_observed":
                intent = frame.records[row.evidence["intended"]["sequence"]].evidence
                if intent["subject"] == subject and intent["step"] == step:
                    return row
        return None

    def move(subject, source, destination, step):
        nonlocal ordinal
        _require_readers_retired(session)
        completed = completed_move(subject, step)
        if completed is not None:
            if (
                observation(
                    Path(completed.evidence["destination"]["path"]),
                    budget,
                    custody,
                    cancel,
                )
                != completed.evidence["destination"]
            ):
                raise ValueError("mobile_rollback_retained_changed")
            return completed
        source = Path(source)
        destination = Path(destination)
        parents = [
            parent_census(path, budget, custody, cancel)
            for path in sorted({source.parent, destination.parent})
        ]
        if destination.exists():
            raise ValueError("mobile_rollback_destination_occupied")
        evidence = {
            "version": 1,
            "started": _reference(frame, started),
            "ordinal": ordinal,
            "subject": subject,
            "step": step,
            "source": observation(source, budget, custody, cancel),
            "destination": str(destination),
            "parents": parents,
        }
        intended = append("mobile_rollback_move_intended", evidence)
        frame.verify()
        move_original(
            evidence,
            budget,
            custody,
            cancel,
            _before_effect=lambda cost: _move_capacity(frame, evidence, intended, cost),
        )
        if move_position(evidence, budget, custody, cancel) != "after":
            raise ValueError("mobile_rollback_move_uncertain")
        observed = append(
            "mobile_rollback_move_observed",
            {
                "version": 1,
                "intended": _reference(frame, intended),
                "destination": observation(destination, budget, custody, cancel),
                "parents": [
                    parent_census(Path(row["state"]["path"]), budget, custody, cancel)[
                        "state"
                    ]
                    for row in parents
                ],
            },
        )
        ordinal += 1
        if step != "restore_snapshot":
            retained.append(observed)
        return observed

    for target in prepared.targets:
        path = Path(target.witnesses[-1].path)
        for component, suffix in (
            ("wal", "-wal"),
            ("shm", "-shm"),
            ("journal", "-journal"),
            ("main", ""),
        ):
            selected = Path(str(path) + suffix)
            subject = {
                "owner_id": target.owner_id,
                "component": component,
                "path": str(selected),
            }
            if completed_move(subject, "retain_current") is not None:
                continue
            if selected.exists():
                move(
                    subject,
                    selected,
                    selected.parent / _retained_name(journal.operation_id, ordinal),
                    "retain_current",
                )
        stage = next(
            (
                item
                for item in stages
                if item[0].evidence["owner_id"] == target.owner_id
            ),
            None,
        )
        if stage is None:
            continue
        intended, _observed = stage
        subject = {"owner_id": target.owner_id, "component": "main", "path": str(path)}
        restored = move(
            subject, Path(intended.evidence["temporary"]), path, "restore_snapshot"
        )
        actual = restored.evidence["destination"]
        previous = metadata(path, actual, budget, custody)
        applied = intended.evidence["metadata"]
        completed_metadata = next(
            (
                row
                for row in frame.records
                if row.event == "mobile_rollback_metadata_observed"
                and frame.records[row.evidence["intended"]["sequence"]].evidence[
                    "subject"
                ]
                == subject
            ),
            None,
        )
        if completed_metadata is not None:
            if (
                previous != applied
                or observation(path, budget, custody, cancel)
                != completed_metadata.evidence["object"]
            ):
                raise ValueError("mobile_rollback_metadata_changed")
            continue
        intended_metadata = append(
            "mobile_rollback_metadata_intended",
            {
                "version": 1,
                "started": _reference(frame, started),
                "subject": subject,
                "object": actual,
                "previous": previous,
                "applied": applied,
                "parent": _directory(
                    parent_census(path.parent, budget, custody, cancel)
                ),
            },
        )
        frame.verify()
        frame.require_capacity(
            8192
            + 16 * len(str(path))
            + _file_observation_capacity(path, custody)
            + _append_capacity(frame, intended_metadata.evidence)
            + 8192
            + _reference_cost(frame, intended_metadata)
        )
        if metadata(path, actual, budget, custody, apply=applied) != applied:
            raise ValueError("mobile_rollback_metadata_changed")
        append(
            "mobile_rollback_metadata_observed",
            {
                "version": 1,
                "intended": _reference(frame, intended_metadata),
                "object": observation(path, budget, custody, cancel),
                "metadata": applied,
            },
        )

    history = next(
        (row for row in current if row.event == "mobile_history_verified"), None
    )
    if history is not None:
        path = Path(
            next(
                row.evidence["publication"]
                for row in current
                if row.event == "mobile_history_intended"
            )
        )
        move(
            {"owner_id": None, "component": "history", "path": str(path)},
            path,
            path.parent / _retained_name(journal.operation_id, ordinal),
            "retain_current",
        )
    # Created parents are removed only through separately witnessed retention;
    # absence is not represented by deleting an imported tree.
    created = {
        Path(row.path)
        for target in prepared.targets
        for row in target.witnesses[:-1]
        if row.inode is None
    }
    for path in sorted(created, key=lambda row: len(row.parts), reverse=True):
        if path.exists():
            census = parent_census(path, budget, custody, cancel)
            if census["children"]:
                raise ValueError("mobile_rollback_created_parent_not_empty")
            move(
                {"owner_id": None, "component": "created_parent", "path": str(path)},
                path,
                path.parent / _retained_name(journal.operation_id, ordinal),
                "retain_created",
            )

    current = records()
    prepared, witnesses = _mobile_state(current)
    _mobile_check_witnesses(current, witnesses)
    from .storage_admission import _mobile_witness

    targets, restored = [], []
    for target in prepared.targets:
        path = target.witnesses[-1].path
        sidecars = [
            _mobile_witness(Path(path + suffix))
            for suffix in ("-wal", "-shm", "-journal")
        ]
        targets.append(
            {
                "owner_id": target.owner_id,
                "witnesses": [witnesses[row.path] for row in target.witnesses],
                "sidecars": sidecars,
            }
        )
        stage = next(
            (
                item
                for item in stages
                if item[0].evidence["owner_id"] == target.owner_id
            ),
            None,
        )
        restored.append(
            {
                "owner_id": target.owner_id,
                "stage": _reference(frame, stage[1]) if stage else None,
                "current": witnesses[path],
                "snapshot_sha256": stage[0].evidence["payload_sha256"]
                if stage
                else None,
            }
        )
    root = Path(prepared.bootstrap_root)
    name = "pending-" + bootstrap._key(journal.operation_id) + ".json"
    with custody.pinned_directory(root) as parent:
        info = os.fstat(parent)
        pending = observation(root / name, budget, custody, cancel)
        terminal = append(
            "mobile_rolled_back",
            {
                "version": 1,
                "started": _reference(frame, started),
                "source_fingerprint": prepared.source_fingerprint,
                "destination_identity": started.evidence["destination_identity"],
                "targets": targets,
                "restored": restored,
                "retained": [_reference(frame, row) for row in retained],
                "pending": pending,
                "bootstrap": {
                    "path": str(root),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                },
            },
        )
    _retire_rollback_pending(
        session, journal, claim, frame, custody, prepared, terminal.evidence, cancel
    )


def _retire_rollback_pending(
    session, journal, claim, frame, custody, prepared, terminal, cancel
):
    """Retire only this durable rollback terminal's exact witnessed pointer."""
    budget = frame.budget
    root = Path(prepared.bootstrap_root)
    name = "pending-" + bootstrap._key(journal.operation_id) + ".json"
    with custody.pinned_directory(root) as parent:
        info = os.fstat(parent)
        if terminal["bootstrap"] != {
            "path": str(root),
            "device": info.st_dev,
            "inode": info.st_ino,
        }:
            raise ValueError("mobile_terminal_scope_changed")
        _require_readers_retired(session)
        _SourceClaim._check_binding(claim)
        control_records._mobile_pending(session, journal, prepared)
        if observation(root / name, budget, custody, cancel) != terminal["pending"]:
            raise ValueError("mobile_pending_scope_changed")
        frame.verify()
        frame.require_capacity(frame.verification_cost())
        bootstrap.advance_admission_epoch()
        os.unlink(name, dir_fd=parent)
        flush_directory(parent)
        bootstrap.advance_admission_epoch()


def _rollback_import(service, operation_id, password, cancel, *, _budgets):
    """Enter the installed pending capture path for an explicit rollback request."""
    from .mobile_consolidation_apply import _resume_captured_mobile
    from .mobile_consolidation_recovery import _recovery_pending_session
    from .mobile_rollback_resume import _resume_rollback
    from .owner_registry import install_adapters
    from .recovery_copies import _journal
    from .replacement import require_rollback_password
    from .service_storage import ensure_storage

    require_rollback_password(password)
    install_adapters()
    with _recovery_pending_session(service, operation_id, _budgets, cancel) as session:
        journal = _journal(service.control_root, operation_id)
        if _resume_rollback(session, journal, password, cancel):
            return
        destination = ensure_storage(service.control_root)
        with _resume_captured_mobile(
            session, journal, destination=destination, password=password, cancel=cancel
        ) as captured:
            _restore_captured(captured, password, cancel)
