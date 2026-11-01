"""Public mobile operation composition over installed native maintenance."""

from contextlib import contextmanager
from pathlib import Path

from . import bootstrap


@contextmanager
def _open_completion_read(path):
    """Own a fixed ordinary read connection, never a preview-copy capability."""
    from tldw_chatbook.DB.private_sqlite import connect_private_sqlite

    from .mobile_preview_budget import _current_completion, path_cost

    scope = _current_completion()
    scope.read_budget.reserve(1024 + path_cost(path))
    connection = connect_private_sqlite(
        "recovery.mobile.completion.read",
        path,
        read_only=True,
        must_exist=True,
        immutable=False,
        isolation_level=None,
        cached_statements=0,
    )
    try:
        scope.bind(connection, path)
        yield connection
    finally:
        try:
            connection.close()
        except BaseException:
            scope.uncertain(connection)
            raise
        else:
            scope.retired(connection)


def _terminal_bindings(session, journal, prepared, terminal, budget, custody):
    """Check terminal admission without changing ongoing _mobile_pending rules."""
    from .capture_service import _selectors
    from .control_records import _Pending
    from .mobile_consolidation import _selector_observation

    session._check()
    budget.reserve(4096)
    budget.value(prepared.selector_review, text_copies=2)
    budget.value(prepared.namespaces, text_copies=4)
    if (
        prepared.bootstrap_root != str(session._control.parent)
        or session._control.parent != bootstrap.default_bootstrap_root()
        or not {*prepared.namespaces, "bootstrap.unbound"} <= set(session._names)
    ):
        raise bootstrap.RecoveryRequired("mobile_session_scope_changed")
    selectors = _selectors(
        (Path(prepared.source_selector), Path(prepared.target_selector)),
        include_known_profiles=True,
        _budget=budget,
        _custody=custody,
    )
    if tuple(str(path) for path in selectors) != tuple(
        row.selector for row in prepared.selector_review
    ):
        raise bootstrap.RecoveryRequired("mobile_selector_review_changed")
    for row in prepared.selector_review:
        if _selector_observation(
            Path(row.selector), _budget=budget, _custody=custody
        ) != (
            row.fingerprint,
            tuple(
                (item.path, item.device, item.inode, item.mode)
                for item in row.witnesses
            ),
        ):
            raise bootstrap.RecoveryRequired("mobile_selector_review_changed")
    pending, profiles = bootstrap._records(
        session._control.parent, _budget=budget, _custody=custody
    )
    expected = _Pending(
        operation_id=journal.operation_id,
        control_root=str(journal.root.parent),
        namespaces=prepared.namespaces,
        selectors=list(
            dict.fromkeys((prepared.source_selector, prepared.target_selector))
        ),
    )
    if pending and pending != [expected.model_dump()]:
        raise bootstrap.RecoveryRequired("mobile_pending_scope_changed")
    registry = bootstrap._registry(
        session._control.parent, _budget=budget, _custody=custody
    )
    for selector, fingerprint in zip(
        (prepared.source_selector, prepared.target_selector),
        prepared.selector_fingerprints,
        strict=True,
    ):
        binding = bootstrap._binding(
            Path(selector), profiles, registry, _budget=budget, _custody=custody
        )
        if (
            binding is None
            or binding["fingerprint"] != fingerprint
            or not set(binding["namespaces"]) <= set(session._names)
        ):
            raise bootstrap.RecoveryRequired(
                "mobile_close_old_instance_enrollment_required"
            )
    if terminal.bootstrap.path != prepared.bootstrap_root:
        raise bootstrap.RecoveryRequired("mobile_terminal_scope_changed")


def _verify_terminal_owners(
    session, journal, terminal, budget, cancel, *, _custody=None
):
    """Read completion records under ordinary or maintenance-owned snapshots."""
    import hashlib
    from contextlib import closing

    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt

    from .mobile_consolidation import _open_preview_read
    from .mobile_native_operation import _require_readers_retired
    from .mobile_preview_budget import _completion_observation
    from .sqlite_validation import _restrict_connection, _version

    for target, family in zip(
        terminal.targets, ("chat", "workspaces", "runs"), strict=True
    ):
        budget.container(20)
        with (
            _completion_observation(budget, _custody)
            if session is None
            else session._discovery_reads(private_sqlite=True, _read_budget=budget),
            (
                _open_completion_read(Path(target.witnesses[-1].path))
                if session is None
                else closing(_open_preview_read(Path(target.witnesses[-1].path)))
            ) as connection,
        ):
            # Ordinary reads hold the live native read-only transaction;
            # maintenance reads retain their strict native private copy.
            # Every imported/schema query follows restriction installation;
            # the unchanged authorizer grants no transaction statements.
            connection.execute("BEGIN")
            restrictions = _restrict_connection(connection, cancel)
            if _version(connection, target.owner_id) != target.schema_version:
                raise ValueError("mobile_terminal_schema_changed")
            result = _read_owner_receipt(
                connection,
                journal.operation_id,
                terminal.source_fingerprint,
                family,
                budget,
            )
            if terminal.kind == "no_data":
                if result is not None:
                    raise ValueError("mobile_terminal_owner_changed")
            else:
                expected = next(
                    row for row in terminal.owners if row.owner_id == target.owner_id
                )
                if result is None:
                    raise ValueError("mobile_terminal_owner_changed")
                master, _children = result
                if any(
                    master[key] != getattr(expected, key)
                    for key in (
                        "entry_count",
                        "entries_sha256",
                        "command_history_count",
                    )
                ):
                    raise ValueError("mobile_terminal_owner_changed")
                size = connection.execute(
                    "SELECT length(CAST(mapping_json AS BLOB)) FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                    (journal.operation_id, family),
                ).fetchone()[0]
                budget.reserve(128 + size * 5)
                raw = connection.execute(
                    "SELECT mapping_json FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                    (journal.operation_id, family),
                ).fetchone()[0]
                if (
                    hashlib.sha256(raw.encode("utf-8")).hexdigest()
                    != expected.master_sha256
                ):
                    raise ValueError("mobile_terminal_owner_changed")
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
            # Physical native close retires the read transaction.
        if session is not None:
            _require_readers_retired(session)


def _require_mobile_cutover(source_dir, stores):
    """Observe one committed source claim while ordinary native admission is held."""
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_cutover_budget
    from tldw_chatbook.Utils.platform_files import os

    from .journal import _mobile_ref, _mobile_state, _MobileCommitted, observe_artifact
    from .mobile_consolidation import CanonicalConsoleStores, _source_claim_locations
    from .mobile_cutover_observer import (
        _claim_read_lock,
        _observe_committed_selectors,
        _retained_cutover_evidence,
    )
    from .profile_paths import lexical_path
    from .storage_admission import (
        _CaptureFileDescriptors,
        _mobile_check_witnesses,
        acquire_storage,
    )

    if type(stores) is not CanonicalConsoleStores:
        raise ValueError("mobile_canonical_stores_required")
    source_dir = lexical_path(source_dir)
    budget = _mobile_cutover_budget()
    custody = _CaptureFileDescriptors(None)
    try:
        budget.value(str(source_dir))
        locations = _source_claim_locations(source_dir)
        budget.container(len(locations) * 2)
        found = [
            (lock, path) for lock, path in locations if path.exists() or lock.exists()
        ]
        if len(found) != 1:
            raise ValueError("mobile_cutover_required")
        lock, path = found[0]
        with _claim_read_lock(lock, budget, custody):
            from .mobile_lineage import _read_lineage

            leaf = _read_lineage(path, source_dir, budget, custody)
            if leaf is None or leaf[2] is None:
                raise ValueError("mobile_cutover_required")
            journal, records, claim, path = leaf
            if claim.destination_identity != stores.identity:
                raise ValueError("mobile_cutover_target_changed")
            if not records or records[-1].event != "mobile_committed":
                raise ValueError("mobile_cutover_required")
            budget.value(records[-1].evidence, text_copies=4)
            terminal = _MobileCommitted.model_validate(records[-1].evidence)
            reference = _mobile_ref(records[-1])
            prepared, witnesses = _mobile_state(records)
            if (
                prepared.bootstrap_root != str(bootstrap.default_bootstrap_root())
                or terminal.destination_identity != stores.identity
                or terminal.source_fingerprint != claim.source_fingerprint
                or tuple(row.witnesses[-1].path for row in terminal.targets)
                != tuple(str(path) for path in stores.paths)
                or next(
                    row.evidence
                    for row in records
                    if row.event == "mobile_claim_intended"
                )
                != claim.model_dump()
            ):
                raise ValueError("mobile_cutover_target_changed")
            # This is the ordinary installed storage lease, not completion evidence
            # converted into owner permission. The gate creates no DB or pointer.
            with acquire_storage(stores.chat_path, related_paths=stores.paths[1:]):
                configuration = _observe_committed_selectors(
                    prepared, claim, terminal, budget, custody
                )
                with custody.pinned_directory(Path(prepared.bootstrap_root)) as parent:
                    info = os.fstat(parent)
                    if (info.st_dev, info.st_ino) != (
                        terminal.bootstrap.device,
                        terminal.bootstrap.inode,
                    ):
                        raise ValueError("mobile_cutover_profile_changed")
                budget.reserve(4096 + path.stat().st_size * 8)
                if (
                    observe_artifact(path, _custody=custody)
                    != records[terminal.claim.sequence].evidence
                ):
                    raise ValueError("mobile_source_claim_changed")
                from .mobile_restored_observer import _direct_restoration

                with _direct_restoration(
                    journal,
                    records,
                    prepared,
                    witnesses,
                    claim,
                    terminal,
                    configuration,
                    budget,
                    custody,
                ) as restored:
                    _retained_cutover_evidence(records, terminal, budget, custody)
                    if not restored:
                        _verify_terminal_owners(
                            None, journal, terminal, budget, None, _custody=custody
                        )
                    with journal._locked(exclusive=False, _custody=custody) as parent:
                        current = journal._records(
                            parent, _budget=budget, _custody=custody
                        )
                    if (
                        _mobile_ref(current[-1]) != reference
                        or _observe_committed_selectors(
                            prepared, claim, terminal, budget, custody
                        )
                        != configuration
                    ):
                        raise ValueError("mobile_cutover_changed")
                    if not restored:
                        _mobile_check_witnesses(current, witnesses)
    finally:
        custody.retire()


def _finish_terminal(service, operation_id, cancel, *, _budgets):
    """Reacquire actual maintenance and the original flock; never replay import."""
    from tldw_chatbook.Chat.console_trace_read_budget import _require_mobile_budget_pair
    from tldw_chatbook.Utils.platform_files import os

    from .control_records import _existing_admission_authority
    from .journal import (
        _mobile_ref,
        _mobile_state,
        _MobileCommitted,
        _MobileSourceClaim,
        observe_artifact,
    )
    from .mobile_consolidation import _source_claim_locations
    from .mobile_consolidation_apply import _SourceClaim
    from .mobile_native_operation import _require_readers_retired
    from .native_files import flush_directory
    from .recovery_copies import _journal
    from .storage_admission import (
        _attach_mobile_budgets,
        _CaptureFileDescriptors,
        _mobile_check_witnesses,
    )

    budget = _require_mobile_budget_pair(_budgets)
    budget.reserve(4096 + 16 * len(operation_id))
    journal = _journal(service.control_root, operation_id)
    custody = _CaptureFileDescriptors(None)
    try:
        with journal._locked(exclusive=False, _custody=custody) as parent:
            initial = journal._records(parent, _budget=budget, _custody=custody)
        if not initial or initial[-1].event != "mobile_committed":
            raise ValueError("mobile_terminal_required")
        prepared, _ = _mobile_state(initial)
        if prepared.bootstrap_root != str(bootstrap.default_bootstrap_root()):
            raise ValueError("mobile_terminal_scope_changed")
        budget.value(initial[-1].evidence, text_copies=4)
        reference = _mobile_ref(initial[-1])
        authority = _existing_admission_authority(Path(prepared.bootstrap_root))
        with authority.maintenance(
            tuple(sorted({*prepared.namespaces, "bootstrap.unbound"})),
            30,
            cancel=cancel,
        ) as session:
            session._scopes.append(custody)
            _attach_mobile_budgets(session, _budgets)
            with journal._locked(exclusive=False, _custody=custody) as parent:
                records = journal._records(parent, _budget=budget, _custody=custody)
            if _mobile_ref(records[-1]) != reference:
                raise ValueError("mobile_terminal_changed")
            terminal = _MobileCommitted.model_validate(records[-1].evidence)
            prepared, witnesses = _mobile_state(records)
            _terminal_bindings(session, journal, prepared, terminal, budget, custody)
            intended = next(
                row.evidence for row in records if row.event == "mobile_claim_intended"
            )
            claim_path = Path(intended["claim_path"])
            locations = _source_claim_locations(Path(intended["source_dir"]))
            found = [pair for pair in locations if pair[0].exists() or pair[1].exists()]
            if len(found) != 1:
                raise ValueError("mobile_source_claim_ambiguous")
            pair = found[0]
            claim = _SourceClaim._issue(session, journal, pair[1], pair[0])
            try:
                with claim.custody.pinned_directory(pair[0].parent) as parent:
                    claim._lock_existing(parent)
                from .mobile_lineage import _resolve_owned_claim

                _resolve_owned_claim(claim, journal, Path(intended["source_dir"]))
                if claim.path != claim_path:
                    raise ValueError("mobile_source_claim_changed")
                with claim.custody.pinned_directory(claim_path.parent) as parent:
                    actual = bootstrap._read(
                        parent, claim_path.name, _budget=budget, _custody=claim.custody
                    )
                    budget.value(actual, text_copies=2)
                    if (
                        _MobileSourceClaim.model_validate(actual).model_dump()
                        != intended
                    ):
                        raise ValueError("mobile_source_claim_changed")
                    observed = records[terminal.claim.sequence].evidence
                    budget.reserve(
                        4096
                        + os.stat(
                            claim_path.name, dir_fd=parent, follow_symlinks=False
                        ).st_size
                        * 8
                    )
                    if observe_artifact(claim_path, _custody=claim.custody) != observed:
                        raise ValueError("mobile_source_claim_changed")
                    session._mobile_journal = journal
                    claim.journal, claim.receipt = journal, observed
                    from .mobile_lineage_ancestors import _issue_ancestors

                    _issue_ancestors(claim, journal, Path(intended["source_dir"]))
                    _SourceClaim._check_binding(claim)
                    _mobile_check_witnesses(records, witnesses)
                    _verify_terminal_owners(session, journal, terminal, budget, cancel)
                    with journal._locked(
                        exclusive=True, _custody=claim.custody
                    ) as journal_parent:
                        current = journal._records(
                            journal_parent, _budget=budget, _custody=claim.custody
                        )
                        if _mobile_ref(current[-1]) != reference:
                            raise ValueError("mobile_terminal_changed")
                        _terminal_bindings(
                            session, journal, prepared, terminal, budget, custody
                        )
                        _mobile_check_witnesses(current, witnesses)
                        _SourceClaim._check_binding(claim)
                        _require_readers_retired(session)
                        with claim.custody.pinned_directory(
                            Path(prepared.bootstrap_root)
                        ) as pending_parent:
                            info = os.fstat(pending_parent)
                            if (info.st_dev, info.st_ino) != (
                                terminal.bootstrap.device,
                                terminal.bootstrap.inode,
                            ):
                                raise ValueError("mobile_terminal_scope_changed")
                            name = Path(terminal.pending.path).name
                            try:
                                info = os.stat(
                                    name, dir_fd=pending_parent, follow_symlinks=False
                                )
                            except FileNotFoundError:
                                pass
                            else:
                                budget.reserve(4096 + info.st_size * 8)
                                if (
                                    observe_artifact(
                                        Path(terminal.pending.path),
                                        _custody=claim.custody,
                                    )
                                    != terminal.pending.model_dump()
                                ):
                                    raise ValueError("mobile_pending_scope_changed")
                                bootstrap.advance_admission_epoch()
                                os.unlink(name, dir_fd=pending_parent)
                            flush_directory(pending_parent)
                            bootstrap.advance_admission_epoch()
            finally:
                claim.retire()
    finally:
        custody.retire()


@contextmanager
def _recovery_pending_session(service, operation_id, pair, cancel):
    """Bound pending data supplies only locators for actual native maintenance."""
    from tldw_chatbook.Chat.console_trace_read_budget import _require_mobile_budget_pair

    from .control_records import _existing_admission_authority, _Pending
    from .storage_admission import _attach_mobile_budgets, _CaptureFileDescriptors

    budget = _require_mobile_budget_pair(pair)
    budget.reserve(4096 + 16 * len(operation_id))
    custody = _CaptureFileDescriptors(None)
    root = bootstrap.default_bootstrap_root()
    try:
        pending, _ = bootstrap._records(root, _budget=budget, _custody=custody)
        budget.container(len(pending))
        selected = [row for row in pending if row["operation_id"] == operation_id]
        if len(selected) != 1:
            raise ValueError("mobile_pending_scope_changed")
        budget.value(selected[0], text_copies=4)
        association = _Pending.model_validate(selected[0])
        if Path(association.control_root) != service.control_root:
            raise ValueError("recovery_control_mismatch")
    finally:
        custody.retire()
    authority = _existing_admission_authority(root)
    budget.container(4 * (len(association.namespaces) + 1))
    with authority.maintenance(
        tuple(sorted({*association.namespaces, "bootstrap.unbound"})), 30, cancel=cancel
    ) as session:
        _attach_mobile_budgets(session, pair)
        yield session


def _finish_import(service, operation_id, password, cancel, *, _budgets):
    """Resume installed capture/import in one newly acquired native session."""
    from .mobile_consolidation_apply import _resume_captured_mobile
    from .owner_registry import install_adapters
    from .recovery_copies import _journal
    from .replacement import require_rollback_password
    from .service_storage import ensure_storage

    require_rollback_password(password)
    # A fresh recovery process has not necessarily performed a public preview.
    # Install the same inert declarations before any native inventory read.
    install_adapters()
    with _recovery_pending_session(service, operation_id, _budgets, cancel) as session:
        journal = _journal(service.control_root, operation_id)
        destination = ensure_storage(service.control_root)
        with _resume_captured_mobile(
            session, journal, destination=destination, password=password, cancel=cancel
        ) as captured:
            captured.consolidate(password=password, cancel=cancel)


def _mobile_names(preview, control_root, budget, custody):
    """Resolve existing profile associations; a preview cannot enroll a writer."""
    budget.reserve(4096)
    root = bootstrap.default_bootstrap_root()
    pending, profiles = bootstrap._records(root, _budget=budget, _custody=custody)
    # Native _Profile has no control_root field. The operation/control binding
    # belongs to _Pending; profile authority is its checked registry association.
    if any(Path(row["control_root"]) != control_root for row in pending):
        raise bootstrap.RecoveryRequired("recovery_control_mismatch")
    registry = bootstrap._registry(root, _budget=budget, _custody=custody)
    budget.container(4 * (len(registry) + 1))
    names = {"bootstrap.unbound"}
    for selector in (preview.source_selector, preview.target_selector):
        binding = bootstrap._binding(
            selector, profiles, registry, _budget=budget, _custody=custody
        )
        if binding is None:
            raise bootstrap.RecoveryRequired(
                "mobile_close_old_instance_enrollment_required"
            )
        names.update(binding["namespaces"])
    return tuple(sorted(names))


def _start_mobile(service, operation, cancel, preview, password):
    from .journal import Journal
    from .mobile_consolidation_apply import _captured_mobile
    from .mobile_enrollment import _start_session
    from .owner_registry import install_adapters

    install_adapters()
    with _start_session(preview, service.control_root, cancel) as (
        session,
        inventory,
        destination,
    ):
        # First enrollment does not replace the installed profile/control check.
        from .storage_admission import (
            _CaptureFileDescriptors,
            _mobile_coordinator_budget,
        )

        budget = _mobile_coordinator_budget(session)
        budget.reserve(4096)
        custody = _CaptureFileDescriptors(None)
        session._scopes.append(custody)
        try:
            if not set(
                _mobile_names(preview, service.control_root, budget, custody)
            ) <= set(session._names):
                raise ValueError("mobile_session_scope_changed")
        finally:
            custody.retire()
        control = service.control_root
        journal = Journal(control, operation)
        service._update(
            operation,
            phase="preparing",
            result={"journal_operation_id": operation, "control_root": str(control)},
        )
        try:
            with _captured_mobile(
                session,
                journal,
                preview,
                inventory.scope_digest,
                destination=destination,
                password=password,
                cancel=cancel,
            ) as captured:
                service._update(operation, phase="recovery_required")
                captured.consolidate(password=password, cancel=cancel)
        except BaseException:
            if any(
                row["operation_id"] == operation
                and Path(row["control_root"]) == control
                for row in bootstrap._records(session._control.parent)[0]
            ):
                service._update(operation, phase="recovery_required")
            raise
    service._update(
        operation,
        phase="mobile_committed",
        result={
            "journal_operation_id": operation,
            "control_root": str(control),
            "mobile_consolidated": True,
        },
    )
