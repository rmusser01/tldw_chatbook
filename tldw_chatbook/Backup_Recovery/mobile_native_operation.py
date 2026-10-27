"""Same-operation native input and physical retirement; no import activation."""

from . import archive_reader, control_records
from .mobile_content_graph import _ContentBlocker


class _NativeOperationBlocked(ValueError):
    """Exact body-free outstanding work, never an import-completion receipt."""

    def __init__(self, blockers):
        super().__init__("mobile_native_operation_blocked")
        self.blockers = blockers


def _require_readers_retired(session):
    """Check actual native custody, including scopes whose local marker is gone."""
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_consolidation_apply import (
        _CapturedMobile,
        _MobileCapturePartition,
        _SourceClaim,
    )
    from .mobile_history_files import _HistoryFiles
    from .storage_admission import (
        _CaptureFileDescriptors,
        _CaptureScope,
        _DiscoveryScope,
        _MobileScope,
    )

    session._check()
    for scope in session._scopes:
        if type(scope) not in (
            _CapturedMobile,
            _MobileCapturePartition,
            _SourceClaim,
            _CapturedSQLite,
            _HistoryFiles,
            _CaptureScope,
            _DiscoveryScope,
            _MobileScope,
            _CaptureFileDescriptors,
        ):
            raise ValueError("mobile_native_scope_unqualified")
        if type(scope) is _SourceClaim and scope.custody.close_failed:
            raise ValueError("mobile_native_readers_not_retired")
        if (
            getattr(scope, "close_failed", False)
            or getattr(scope, "resource_close_failed", False)
            or getattr(scope, "resources", ())
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if (
            type(scope)
            in (
                _CapturedSQLite,
                _HistoryFiles,
                _CaptureScope,
                _DiscoveryScope,
                _MobileScope,
            )
            and scope.active
        ):
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CaptureScope and scope.sqlite_directory is not None:
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CapturedSQLite and scope.roots:
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _CaptureFileDescriptors and scope.fds:
            raise ValueError("mobile_native_readers_not_retired")
        if type(scope) is _DiscoveryScope and scope.sqlite_copies is not None:
            copies = scope.sqlite_copies
            if copies.active or copies.resources or copies.directory is not None:
                raise ValueError("mobile_native_readers_not_retired")


def _native_capture_check(captured, cancel):
    """Recheck the original captured cohort before native owner effects."""
    from .journal import _mobile_state
    from .mobile_consolidation import _path_witnesses
    from .storage_admission import _mobile_check_witnesses

    captured.check()
    archive_reader._check(cancel)
    if captured.claim.custody.close_failed:
        raise ValueError("mobile_native_claim_retirement_uncertain")
    if (
        _path_witnesses((captured.preview.source_dir,))[0][1]
        != captured.preview.source_witnesses
    ):
        raise ValueError("mobile_source_changed")
    with captured.journal._locked(exclusive=False) as fd:
        records = captured.journal._records(fd)
    prepared, witnesses = _mobile_state(records)
    control_records._mobile_pending(captured.session, captured.journal, prepared)
    if prepared.source_fingerprint != captured.preview.source_fingerprint:
        raise ValueError("mobile_native_source_changed")
    _mobile_check_witnesses(records, witnesses, recovering_mode=True)
    archive_reader._check(cancel)


def _native_preopen(captured, cancel):
    """Refuse retained resources before even native path or journal creation."""
    _native_capture_check(captured, cancel)
    _require_readers_retired(captured.session)


def _required_handlers(plan, budget):
    """Report actual pending dispositions; none of their names proves a handler."""
    budget.container(len(plan.blockers) + len(plan.pending_duties) + 3)
    blockers = list(plan.blockers)
    for duty in plan.pending_duties:
        budget.container(5)
        blockers.append(
            _ContentBlocker(
                duty.table,
                duty.original,
                "native_consumer_required",
                duty.owner,
                duty.requirement,
            )
        )
    for member in plan.history:
        budget.container(1)
        budget.container(5)
        budget.reserve(32)
        blockers.append(
            _ContentBlocker(
                "agents.history",
                (member.logical_id,),
                "native_runlog_receipt_membership_required",
            )
        )
    for row in plan.rows:
        for column in row.pending_columns:
            budget.container(5)
            budget.reserve(32)
            blockers.append(
                _ContentBlocker(
                    row.table,
                    row.original,
                    "native_field_owner_required",
                    "column",
                    column,
                )
            )
    # Fixed owning transactions are the next serial increment, even for empty input.
    for owner in ("db.chachanotes.primary", "db.workspaces", "db.agent_runs"):
        budget.container(5)
        blockers.append(
            _ContentBlocker(
                "mobile_consolidation_batches",
                (),
                "native_owner_writer_required",
                "owner-write",
                owner,
            )
        )
    budget.container(len(blockers))
    return tuple(blockers)


def _consolidate(captured, *, password, cancel):
    """Derive complete input in this call; retain no reusable readiness object."""
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .mobile_runs_preflight import _plan_runs

    captured.check()
    archive_reader._check(cancel)
    if captured.session._mobile_attempted_journal is not None:
        raise ValueError("mobile_native_operation_already_attempted")
    # Denial-only session state survives reminting this operation's lifecycle.
    captured.session._mobile_attempted_journal = captured.journal
    budget = _TraceReadBudget()
    budget.container(12)
    budget.value(captured.preview.source_witnesses)
    budget.value(captured.preview.destination_witnesses)
    _native_preopen(captured, cancel)
    budget.container(len(captured.session._scopes))
    issued_scopes = tuple(captured.session._scopes)
    cohort = captured.binding
    source_fingerprint = captured.preview.source_fingerprint
    operation_id = captured.journal.operation_id
    with captured.sqlite_reads(password=password, cancel=cancel) as reads:
        budget.container(6)
        native_input = (
            reads.binding,
            reads.versions,
            reads._source_archive,
            reads._canonical_archive,
        )
        plan = _plan_runs(reads, _read_budget=budget)
        blockers = _required_handlers(plan, budget)
        reads.check()
    # The same frame retains the actual immutable proofs while their readers close.
    if (
        captured.binding is not cohort
        or reads not in captured.session._scopes
        or len(captured.session._scopes) < len(issued_scopes)
        or any(
            captured.session._scopes[index] is not scope
            for index, scope in enumerate(issued_scopes)
        )
        or captured.preview.source_fingerprint != source_fingerprint
        or captured.journal.operation_id != operation_id
        or reads.binding is not native_input[0]
        or reads.versions is not native_input[1]
        or reads._source_archive is not native_input[2]
        or reads._canonical_archive is not native_input[3]
    ):
        raise ValueError("mobile_native_input_cohort_changed")
    budget.value(captured.preview.source_witnesses)
    budget.value(captured.preview.destination_witnesses)
    _native_preopen(captured, cancel)
    budget.container(3)
    raise _NativeOperationBlocked(blockers)
