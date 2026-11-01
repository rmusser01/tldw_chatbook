"""Finite rollback-only mobile successors under the original source flock."""

from pathlib import Path

from . import bootstrap
from .journal import (
    _ArtifactObservationLimits,
    _mobile_ref,
    _mobile_referenced,
    _mobile_state,
    _MobileSourceClaim,
    _MobileSuccessor,
    observe_artifact,
)
from .recovery_copies import _journal

CLAIM_NAME = "mobile-source-claim.json"


def _claim_path(control, operation):
    import hashlib

    return (
        Path(control)
        / ("operation-" + hashlib.sha256(operation.encode()).hexdigest())
        / CLAIM_NAME
    )


def _validate_successor(evidence, prior):
    value = _MobileSuccessor.model_validate(evidence)
    if not prior or prior[-1].event != "mobile_rolled_back":
        raise ValueError("mobile_successor_requires_rollback")
    _mobile_referenced(value.rolled_back, prior, "mobile_rolled_back")
    if (
        value.rolled_back.model_dump() != _mobile_ref(prior[-1])
        or value.operation_id == prior[0].operation_id
        or value.prepared.sequence != 0
        or Path(value.claim_path) != _claim_path(value.control_root, value.operation_id)
    ):
        raise ValueError("mobile_successor_scope_changed")
    return value.model_dump()


def _observe_claim(path, budget, custody):
    size = path.lstat().st_size
    if not 0 <= size <= bootstrap.MAX_RECORD:
        raise ValueError("mobile_source_claim_changed")
    if budget is not None:
        budget.reserve(147456 + 256 * len(str(path)))
    return observe_artifact(
        path,
        _custody=custody,
        _limits=_ArtifactObservationLimits(1, size, 0, 0, 0, 255),
    )


def _validate_local_claim(journal, parent, records, present, budget, custody):
    """One exact artifact, authenticated locally without another journal lock."""
    if not present:
        if (
            records
            and records[0].evidence.get("predecessor") is not None
            and any(row.event == "mobile_claim_observed" for row in records)
        ):
            raise ValueError("mobile_source_claim_changed")
        return
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .storage_admission import _CaptureFileDescriptors, _local, _mobile_admission

    session = getattr(_local, "maintenance_session", None)
    if budget is None and session is not None:
        budget = _mobile_admission(session)[4]
    if budget is None:
        budget = _TraceReadBudget()  # Ordinary fixed-artifact observation only.
    own = custody is None
    if own:
        budget.reserve(512)
        custody = _CaptureFileDescriptors(None)
        if session is not None:
            session._scopes.append(custody)
    try:
        _check_local_claim(journal, parent, records, present, budget, custody)
    finally:
        if own:
            custody.retire()


def _check_local_claim(journal, parent, records, present, budget, custody):
    successor = bool(
        records
        and records[0].event == "mobile_prepared"
        and records[0].evidence.get("predecessor") is not None
    )
    if not present:
        if successor and any(row.event == "mobile_claim_observed" for row in records):
            raise ValueError("mobile_source_claim_changed")
        return
    if not successor:
        raise ValueError("journal_sequence_invalid")
    path = journal.root / CLAIM_NAME
    raw = bootstrap._read(parent, CLAIM_NAME, _budget=budget, _custody=custody)
    if budget is not None:
        budget.value(raw, text_copies=4)
    claim = _MobileSourceClaim.model_validate(raw)
    intended = [row.evidence for row in records if row.event == "mobile_claim_intended"]
    if (
        claim.operation_id != journal.operation_id
        or claim.control_root != str(journal.root.parent)
        or claim.claim_path != str(path)
        or intended != [claim.model_dump()]
    ):
        raise ValueError("mobile_source_claim_changed")
    observed = [row.evidence for row in records if row.event == "mobile_claim_observed"]
    if observed and observed != [_observe_claim(path, budget, custody)]:
        raise ValueError("mobile_source_claim_changed")


def _read_lineage(original, source, budget, custody):
    """Follow exact reciprocal links; callers retain the original native flock."""
    with custody.pinned_directory(original.parent) as parent:
        try:
            raw = bootstrap._read(
                parent, original.name, _budget=budget, _custody=custody
            )
        except FileNotFoundError:
            return None
    if budget is not None:
        budget.value(raw, text_copies=8)
    first = _MobileSourceClaim.model_validate(raw)
    if first.claim_path != str(original) or first.source_dir != str(source):
        raise ValueError("mobile_source_claim_changed")
    path, operation, control = original, first.operation_id, Path(first.control_root)
    seen, predecessor, reservation = set(), None, None
    for _ in range(bootstrap.MAX_RECORDS):
        if budget is not None:
            budget.reserve(8192 + 32 * (len(operation) + len(str(path))))
        if operation in seen:
            raise ValueError("mobile_lineage_cycle")
        seen.add(operation)
        from .mobile_journal_frame import _local as frame_local

        frame = getattr(frame_local, "frame", None)
        if frame is not None and not (
            frame.journal.operation_id == operation
            and frame.journal.root.parent == control
        ):
            from .mobile_lineage_ancestors import _ancestor_records

            journal, records = _ancestor_records(control, operation)
        else:
            if frame is not None:
                frame.check()
                journal = frame.journal
            else:
                journal = _journal(control, operation)
            with journal._locked(exclusive=False, _custody=custody) as parent:
                records = journal._records(parent, _budget=budget, _custody=custody)
        if not records or records[0].event != "mobile_prepared":
            raise ValueError("mobile_source_claim_changed")
        if budget is not None:
            budget.value(records[0].evidence, text_copies=4)
        prepared, _ = _mobile_state(records)
        if (
            prepared.source_fingerprint != first.source_fingerprint
            or (prepared.predecessor.model_dump() if prepared.predecessor else None)
            != predecessor
            or reservation is not None
            and reservation.prepared.model_dump() != _mobile_ref(records[0])
        ):
            raise ValueError("mobile_lineage_changed")
        intents = [
            row.evidence for row in records if row.event == "mobile_claim_intended"
        ]
        observed = [
            row.evidence for row in records if row.event == "mobile_claim_observed"
        ]
        claim = None
        if path.exists():
            with custody.pinned_directory(path.parent) as parent:
                raw = bootstrap._read(
                    parent, path.name, _budget=budget, _custody=custody
                )
            if budget is not None:
                budget.value(raw, text_copies=8)
            claim = _MobileSourceClaim.model_validate(raw)
            if (
                claim.operation_id != operation
                or Path(claim.control_root) != control
                or claim.claim_path != str(path)
                or claim.source_dir != str(source)
                or claim.source_fingerprint != first.source_fingerprint
                or claim.source_witnesses != first.source_witnesses
                or claim.destination_identity != first.destination_identity
                or intents != [claim.model_dump()]
            ):
                raise ValueError("mobile_lineage_changed")
            if observed and observed != [_observe_claim(path, budget, custody)]:
                raise ValueError("mobile_source_claim_changed")
        elif observed or predecessor is None:
            raise ValueError("mobile_source_claim_changed")
        links = [row for row in records if row.event == "mobile_successor_intended"]
        if not links:
            return journal, records, claim, path
        if (
            len(links) != 1
            or links[0] is not records[-1]
            or claim is None
            or len(observed) != 1
        ):
            raise ValueError("mobile_lineage_changed")
        reservation = _MobileSuccessor.model_validate(links[0].evidence)
        rolled = _mobile_referenced(
            reservation.rolled_back, records, "mobile_rolled_back"
        )
        if Path(reservation.control_root) != control:
            raise ValueError("mobile_lineage_changed")
        predecessor = {
            "operation_id": operation,
            "control_root": str(control),
            "claim_path": str(path),
            "claim": observed[0],
            "rolled_back": _mobile_ref(rolled),
        }
        operation, path = reservation.operation_id, Path(reservation.claim_path)
    raise ValueError("mobile_lineage_limit")


def _selection(claim, journal, preview):
    from .mobile_consolidation import MobileConsolidationRefused
    from .mobile_consolidation_recovery import _terminal_bindings
    from .mobile_rollback_records import _RolledBack
    from .storage_admission import (
        _mobile_admission,
        _mobile_check_witnesses,
        _mobile_coordinator_budget,
    )

    if _mobile_admission(claim.session)[4] is None:
        # Preserve the installed private first-operation reader. Successors are
        # coordinated only by the public worker; never spend its new work from
        # the original private retained-input/proof allowance.
        from .mobile_consolidation_apply import _existing_claim

        existing = _existing_claim(claim.original_path, preview.source_dir)
        if existing is not None and (
            existing.operation_id != journal.operation_id
            or existing.control_root != str(journal.root.parent)
            or existing.destination_identity != preview.destination.identity
        ):
            raise MobileConsolidationRefused(
                "mobile_source_already_claimed",
                operation_id=existing.operation_id,
                control_root=Path(existing.control_root),
            )
        if claim.path != claim.original_path:
            raise ValueError("mobile_public_successor_required")
        return existing, None
    budget = _mobile_coordinator_budget(claim.session)
    leaf = _read_lineage(claim.original_path, preview.source_dir, budget, claim.custody)
    if leaf is None:
        return None, None
    previous, records, source_claim, path = leaf
    budget.value(records[0].evidence, text_copies=4)
    prepared, witnesses = _mobile_state(records)
    if previous.operation_id == journal.operation_id:
        if previous.root != journal.root:
            raise ValueError("mobile_lineage_changed")
        claim.path = path
        return (
            source_claim,
            prepared.predecessor.model_dump() if prepared.predecessor else None,
        )
    if records[-1].event != "mobile_rolled_back":
        raise MobileConsolidationRefused(
            "mobile_source_already_claimed",
            operation_id=previous.operation_id,
            control_root=previous.root.parent,
        )
    if source_claim is None or previous.root.parent != journal.root.parent:
        raise ValueError("mobile_lineage_changed")
    budget.value(records[-1].evidence, text_copies=4)
    terminal = _RolledBack.model_validate(records[-1].evidence)
    if (
        source_claim.source_fingerprint != preview.source_fingerprint
        or source_claim.destination_identity != preview.destination.identity
        or tuple(
            (row.path, row.device, row.inode, row.mode)
            for row in source_claim.source_witnesses
        )
        != preview.source_witnesses
    ):
        raise ValueError("mobile_source_changed_review_required")
    _terminal_bindings(
        claim.session, previous, prepared, terminal, budget, claim.custody
    )
    _mobile_check_witnesses(records, witnesses)
    expected = tuple(
        (
            row.witnesses[-1].path,
            tuple(
                (item.path, item.device, item.inode, item.mode)
                for item in row.witnesses
            ),
        )
        for row in terminal.targets
    )
    sidecars = tuple(
        (
            row.witnesses[-1].path,
            tuple(
                (item.path, item.device, item.inode, item.mode) for item in row.sidecars
            ),
        )
        for row in terminal.targets
    )
    if (
        preview.destination_witnesses != expected
        or preview.destination_sidecars != sidecars
    ):
        raise ValueError("mobile_successor_target_changed")
    claim.path = journal.root / CLAIM_NAME
    observed = next(
        row.evidence for row in records if row.event == "mobile_claim_observed"
    )
    return None, {
        "operation_id": previous.operation_id,
        "control_root": str(previous.root.parent),
        "claim_path": str(path),
        "claim": observed,
        "rolled_back": _mobile_ref(records[-1]),
    }


def _prepared_predecessor(session, journal, preview):
    """Derive from the actual active source-lock owner, never caller records."""
    from .mobile_consolidation_apply import _SourceClaim

    claims = [
        row
        for row in session._scopes
        if type(row) is _SourceClaim
        and row.operation_id == journal.operation_id
        and hasattr(row, "original_path")
        and hasattr(row, "descriptor")
        and row.descriptor in row.custody.fds
    ]
    if not claims:
        return None  # Existing private first-operation preparation remains valid.
    if len(claims) != 1:
        raise ValueError("mobile_source_claim_ambiguous")
    _SourceClaim._check_binding(claims[0])
    return _selection(claims[0], journal, preview)[1]


def _reserve_successor(claim, journal, preview):
    """Publish reciprocal reservation between separate native journal locks."""
    from .storage_admission import _mobile_admission, _mobile_coordinator_budget

    if _mobile_admission(claim.session)[4] is None:
        if claim.path != claim.original_path:
            raise ValueError("mobile_public_successor_required")
        return
    budget = _mobile_coordinator_budget(claim.session)
    with journal._locked(exclusive=False, _custody=claim.custody) as fd:
        child = journal._records(fd, _budget=budget, _custody=claim.custody)
    budget.value(child[0].evidence, text_copies=4)
    prepared, _ = _mobile_state(child)
    predecessor = prepared.predecessor
    if predecessor is None:
        return
    from . import control_records

    control_records._mobile_pending(
        claim.session, journal, prepared, _budget=budget, _custody=claim.custody
    )
    leaf = _read_lineage(claim.original_path, preview.source_dir, budget, claim.custody)
    if leaf is None or leaf[0].operation_id not in {
        journal.operation_id,
        predecessor.operation_id,
    }:
        raise ValueError("mobile_lineage_changed")
    parent = _journal(Path(predecessor.control_root), predecessor.operation_id)
    value = {
        "version": 1,
        "rolled_back": predecessor.rolled_back.model_dump(),
        "operation_id": journal.operation_id,
        "control_root": str(journal.root.parent),
        "claim_path": str(journal.root / CLAIM_NAME),
        "prepared": _mobile_ref(child[0]),
    }
    with parent._locked(exclusive=True, _custody=claim.custody) as fd:
        rows = parent._records(fd, _budget=budget, _custody=claim.custody)
        rolled = _mobile_referenced(predecessor.rolled_back, rows, "mobile_rolled_back")
        observed = [
            row.evidence for row in rows if row.event == "mobile_claim_observed"
        ]
        intended = [
            row.evidence for row in rows if row.event == "mobile_claim_intended"
        ]
        if (
            observed != [predecessor.claim.model_dump()]
            or len(intended) != 1
            or intended[0]["claim_path"] != predecessor.claim_path
            or rolled.evidence["source_fingerprint"] != prepared.source_fingerprint
            or rolled.evidence["destination_identity"] != preview.destination.identity
            or _observe_claim(Path(predecessor.claim_path), budget, claim.custody)
            != predecessor.claim.model_dump()
        ):
            raise ValueError("mobile_lineage_changed")
        links = [
            row.evidence for row in rows if row.event == "mobile_successor_intended"
        ]
        if links:
            if links != [value]:
                raise ValueError("mobile_lineage_changed")
        else:
            parent._append(fd, "mobile_successor_intended", value)
    # Reauthenticate both immutable associations after publication, before claim.
    leaf = _read_lineage(claim.original_path, preview.source_dir, budget, claim.custody)
    if leaf is None or leaf[0].root != journal.root:
        raise ValueError("mobile_lineage_changed")


def _inventory_lineage(path, source):
    """Validate controls within actual live claim custody or a fresh read flock."""
    from contextlib import nullcontext

    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .mobile_consolidation import _source_claim_locations
    from .mobile_consolidation_apply import _SourceClaim
    from .mobile_cutover_observer import _claim_read_lock
    from .mobile_preview_budget import read_options
    from .storage_admission import (
        _CaptureFileDescriptors,
        _local,
        _mobile_coordinator_budget,
    )

    budget, custody = read_options()
    session = getattr(_local, "maintenance_session", None)
    if session is not None:
        from .storage_admission import _mobile_admission

        private = _mobile_admission(session)[4] is None
        active = [
            row
            for row in session._scopes
            if type(row) is _SourceClaim
            and row.path == row.original_path == path
            and session._mobile_journal is not None
            and getattr(row, "journal", None) is session._mobile_journal
            and row.operation_id == session._mobile_journal.operation_id
            and getattr(row, "receipt", None) is not None
            and getattr(row, "descriptor", None) in row.custody.fds
        ]
        if private and len(active) == 1:
            # The issuer published and observed this receipt before yielding the
            # same original-lock scope. No internal successor path is accepted.
            _SourceClaim._check_binding(active[0])
            if active[0].receipt["path"] != str(path):
                raise ValueError("mobile_source_claim_changed")
            return
    if budget is None and session is not None:
        budget = _mobile_coordinator_budget(session)
    own = custody is None
    if own:
        custody = _CaptureFileDescriptors(None)
        if session is not None:
            session._scopes.append(custody)
    if budget is None:
        budget = _TraceReadBudget()
    try:
        pairs = [
            pair
            for pair in _source_claim_locations(source)
            if pair[0].exists() or pair[1].exists()
        ]
        if len(pairs) != 1 or pairs[0][1] != path:
            raise ValueError("mobile_source_claim_ambiguous")
        claims = (
            []
            if session is None
            else [
                row
                for row in session._scopes
                if type(row) is _SourceClaim
                and row.original_path == path
                and hasattr(row, "descriptor")
                and row.descriptor in row.custody.fds
            ]
        )
        if len(claims) > 1:
            raise ValueError("mobile_source_claim_ambiguous")
        if claims:
            _SourceClaim._check_binding(claims[0])
        with (
            nullcontext() if claims else _claim_read_lock(pairs[0][0], budget, custody)
        ):
            if _read_lineage(path, source, budget, custody) is None:
                raise ValueError("mobile_source_claim_changed")
    finally:
        if own:
            custody.retire()


def _resolve_owned_claim(claim, journal, source):
    """Resolve a recovery locator only after acquiring the original real flock."""
    from .mobile_consolidation_apply import _SourceClaim
    from .storage_admission import _mobile_coordinator_budget

    _SourceClaim._check_binding(claim)
    leaf = _read_lineage(
        claim.original_path,
        source,
        _mobile_coordinator_budget(claim.session),
        claim.custody,
    )
    if leaf is None or leaf[0].root != journal.root or leaf[2] is None:
        raise ValueError("mobile_source_claim_changed")
    claim.path = leaf[3]
