"""The public worker releases ordinary storage only after native completion."""

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed


def test_public_service_completes_same_native_target(tmp_path, monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import bootstrap, recovery_service
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    errors = []
    native_issue = recovery_service.issue_code

    def observe_issue(error, **kwargs):
        import json
        import traceback

        errors.append(
            (type(error).__name__, str(error), traceback.format_tb(error.__traceback__))
        )
        (tmp_path / "copy-preview-errors.json").write_text(json.dumps(errors))
        return native_issue(error, **kwargs)

    monkeypatch.setattr(recovery_service, "issue_code", observe_issue)
    import os
    import stat

    inherited_umask = os.umask(0o077)
    try:
        with _capture_fixture(tmp_path, monkeypatch) as values:
            _, journal, config, stores, preview, _, authority = values
    finally:
        os.umask(inherited_umask)
    for directory in (preview.source_dir.parent.parent, preview.source_dir.parent):
        info = directory.stat()
        assert info.st_uid == os.geteuid()
        assert stat.S_IMODE(info.st_mode) == 0o700
    import json

    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(preview.target_selector))
    source_before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    budgets = []
    native_init = _TraceReadBudget.__init__

    def counted(actual):
        native_init(actual)
        budgets.append(actual)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted)
    try:
        service = RecoveryService(journal.root.parent)
        try:
            preview = service.preview_mobile_consolidation(config, config)
            assert (
                mobile_consolidation_fixture.fingerprints(preview.source_dir)
                == source_before
            )
            operation = service.start_mobile_consolidation(
                preview,
                destination_confirmed=True,
                rollback_password=b"test-private-mobile-rollback",
            )
            state = service.wait(operation)
            assert state["state"] == "succeeded", (dict(state), errors)
            assert state["result"]["mobile_consolidated"] is True
            assert service.pending_operations() == ()
            assert bootstrap.startup_permission(config, authority.control_root.parent)[
                0
            ]
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

            chat = CharactersRAGDB(stores.chat_path, "cold-public-mobile-reader")
            try:
                assert len(chat.get_all_conversation_ids()) == 2
            finally:
                chat.close_connection()
            from Tests.Backup_Recovery.test_mobile_cutover_budget import (
                assert_original_observer_accounting,
            )

            assert_original_observer_accounting(tmp_path, monkeypatch, preview, stores)
        finally:
            service.close()
        from tldw_chatbook.Backup_Recovery.recovery_copies import list_recovery_copies

        copies = [
            row
            for row in list_recovery_copies(journal.root.parent)
            if row.operation_id == operation
        ]
        assert len(copies) == 1
        assert copies[0].status == "verified" and not copies[0].pending_operation
        from dataclasses import replace
        from threading import Event

        from tldw_chatbook.Backup_Recovery import archive_reader
        from tldw_chatbook.Backup_Recovery.capture_service import preview_capture
        from tldw_chatbook.Backup_Recovery.later_rollback import _verify_snapshot_source
        from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
        from tldw_chatbook.Backup_Recovery.mobile_canonical_copy import _canonical_copy
        from tldw_chatbook.Backup_Recovery.recovery_copies import (
            _journal,
            delete_recovery_copy,
        )
        from tldw_chatbook.Backup_Recovery.restore_plan import LocalSnapshotSource

        proof = _canonical_copy(_journal(journal.root.parent, operation))
        source = LocalSnapshotSource(
            journal.root.parent, operation, proof.evidence_digest
        )
        acquired = archive_reader.acquire(
            copies[0].path,
            tmp_path / "canonical-acquisition",
            ArchiveLimits(),
            b"test-private-mobile-rollback",
            Event(),
        )
        assert (
            _verify_snapshot_source(source, "replace", acquired.digest, acquired)[1]
            == proof
        )
        # No-archive callers authenticate source evidence, not an asserted plaintext.
        assert _verify_snapshot_source(source, "replace", "f" * 64)[1] == proof
        altered = replace(
            acquired,
            encrypted_source=replace(acquired.encrypted_source, digest="0" * 64),
        )
        with pytest.raises(ValueError, match="local_snapshot_source_changed"):
            _verify_snapshot_source(source, "replace", altered.digest, altered)
        with pytest.raises(ValueError, match="mobile_canonical_copy_retained"):
            delete_recovery_copy(journal.root.parent, operation, user_selected=True)
        edited = CharactersRAGDB(stores.chat_path, "ordinary-later-note-writer")
        try:
            note_id = edited.add_note(
                "Later canonical edit",
                "Preserve this later edit in the new safety copy",
            )
            assert note_id is not None
        finally:
            edited.close_connection()
        current = preview_capture(
            tuple(dict.fromkeys((preview.source_selector, preview.target_selector))),
            options={},
        )
        assert current.complete
        later = RecoveryService(journal.root.parent)
        try:
            plan = later.preview_rollback(
                operation, old_password=b"test-private-mobile-rollback", target=current
            )
            assert plan.local_snapshot == source
            assert plan.archive_digest == acquired.digest
            replacement = later.start_rollback(
                operation,
                plan,
                old_password=b"test-private-mobile-rollback",
                new_password=b"new-test-only-safety-password",
            )
            replaced = later.wait(replacement)
            assert replaced["state"] == "succeeded", (dict(replaced), errors)
            replacement_id = replaced["result"]["journal_operation_id"]
            assert replacement_id != operation
            assert later.pending_operations() == ()
        finally:
            later.close()
        restored = CharactersRAGDB(stores.chat_path, "ordinary-later-restored-reader")
        try:
            assert restored.get_all_conversation_ids() == ["shared-chat"]
            assert restored.get_note_by_id(note_id) is None
        finally:
            restored.close_connection()
        assert bootstrap.startup_permission(config, authority.control_root.parent)[0]
        safety = next(
            row
            for row in list_recovery_copies(journal.root.parent)
            if row.operation_id == replacement_id
        )
        assert safety.status == "verified" and not safety.pending_operation
        saved = archive_reader.acquire(
            safety.path,
            tmp_path / "new-safety-acquisition",
            ArchiveLimits(),
            b"new-test-only-safety-password",
            Event(),
        )
        document = archive_reader.verify_sealed(saved)
        core = next(
            row for row in document.files if row.owner_id == "db.chachanotes.primary"
        )
        import sqlite3
        import zipfile
        from contextlib import closing

        snapshot = tmp_path / "new-safety-core.db"
        with zipfile.ZipFile(saved.path) as packed:
            snapshot.write_bytes(packed.read(core.payload))
        snapshot.chmod(0o600)
        with closing(
            sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
        ) as connection:
            assert (
                connection.execute(
                    "SELECT count(*) FROM conversations WHERE deleted=0"
                ).fetchone()[0]
                == 2
            )
            assert connection.execute(
                "SELECT content FROM notes WHERE id=?", (note_id,)
            ).fetchone() == ("Preserve this later edit in the new safety copy",)
        assert (
            _verify_snapshot_source(source, "replace", acquired.digest, acquired)[
                1
            ].evidence_digest
            == proof.evidence_digest
        )
        try:
            from Tests.Backup_Recovery.test_mobile_restored_observer import (
                assert_actual_restored_gate_controls,
            )

            assert_actual_restored_gate_controls(
                tmp_path,
                monkeypatch,
                preview,
                stores,
                _journal(journal.root.parent, operation),
                _journal(journal.root.parent, replacement_id),
                budgets,
            )
        except Exception as error:
            import traceback

            (tmp_path / "restored-cutover-gate.json").write_text(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "code": str(error),
                        "stack": traceback.format_tb(error.__traceback__),
                    }
                )
            )
            raise
    finally:
        source_after = mobile_consolidation_fixture.fingerprints(preview.source_dir)
        profile_violations, network_violations = take_violations(), blocked_attempts()
        pairs = [
            {
                "coupled": _mobile_budget_pair(row._parent, row),
                "aggregate_remaining": row._parent._remaining,
                "native_remaining": row._remaining,
            }
            for row in budgets
            if row._parent is not None
        ]
        (tmp_path / "copy-final-evidence.json").write_text(
            json.dumps(
                {
                    "source_before": source_before,
                    "source_after": source_after,
                    "profile_violations": profile_violations,
                    "network_violations": network_violations,
                    "pairs_after_services_close": pairs,
                    "errors": errors,
                }
            )
        )
    assert source_before == source_after
    assert not profile_violations and not network_violations
    assert len(pairs) == 1 and pairs[0]["coupled"]


@pytest.mark.parametrize("position", ["before_unlink", "after_unlink"])
def test_fresh_service_finishes_only_witnessed_terminal_pointer(
    tmp_path, monkeypatch, position
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import bootstrap, recovery_service
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Utils.platform_files import os

    errors = []
    native_issue = recovery_service.issue_code

    def observe_issue(error, **kwargs):
        import traceback

        errors.append(
            (type(error).__name__, str(error), traceback.format_tb(error.__traceback__))
        )
        import json

        (tmp_path / "terminal-errors.json").write_text(json.dumps(errors))
        return native_issue(error, **kwargs)

    monkeypatch.setattr(recovery_service, "issue_code", observe_issue)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        _, journal, config, _, preview, _, authority = values
    service = RecoveryService(journal.root.parent)
    native_unlink = os.unlink
    injected = []

    def interrupted(path, *args, **kwargs):
        if str(path).startswith("pending-") and not injected:
            injected.append(position)
            if position == "after_unlink":
                native_unlink(path, *args, **kwargs)
            raise OSError("injected terminal pointer interruption")
        return native_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as fault:
            fault.setattr(os, "unlink", interrupted)
            operation = service.start_mobile_consolidation(
                preview,
                destination_confirmed=True,
                rollback_password=b"test-private-mobile-rollback",
            )
            assert service.wait(operation)["state"] == "recovery_required"
        assert injected == [position]
    finally:
        service.close()
    fresh = RecoveryService(journal.root.parent)
    try:
        assert fresh.status(operation)["actions"] == ("finish",)
        resume = fresh.start_recovery(operation, action="finish")
        assert fresh.wait(resume)["state"] == "succeeded", errors
        assert fresh.pending_operations() == ()
        assert bootstrap.startup_permission(config, authority.control_root.parent)[0]
    finally:
        fresh.close()


@pytest.mark.parametrize("action", ["finish", "rollback"])
def test_fresh_service_resumes_import_before_terminal_publication(
    tmp_path, tmp_path_factory, monkeypatch, action
):
    import json
    import traceback

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import recovery_service
    from tldw_chatbook.Backup_Recovery.mobile_journal_frame import _Frame
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    errors = []
    issue = recovery_service.issue_code

    def observed(error, **kwargs):
        errors.append(
            (type(error).__name__, str(error), traceback.format_tb(error.__traceback__))
        )
        (tmp_path / "resume-errors.json").write_text(json.dumps(errors))
        chain, seen, current = [], set(), error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(
                (
                    type(current).__name__,
                    str(current),
                    traceback.format_tb(current.__traceback__),
                )
            )
            current = current.__cause__ or current.__context__
        (tmp_path / "resume-error-chain.json").write_text(json.dumps(chain))
        return issue(error, **kwargs)

    monkeypatch.setattr(recovery_service, "issue_code", observed)
    from tldw_chatbook.Backup_Recovery.storage_admission import _CaptureFileDescriptors

    close_descriptor = _CaptureFileDescriptors.close_descriptor
    close_failures = []

    def observed_close(custody, descriptor):
        try:
            return close_descriptor(custody, descriptor)
        except BaseException as error:
            close_failures.append(
                {
                    "descriptor": descriptor,
                    "fds": list(custody.fds),
                    "iterators": len(custody.directory_iterators),
                    "close_failed": custody.close_failed,
                    "exception": repr(error),
                    "context": repr(error.__context__),
                    "stack": traceback.format_stack(),
                }
            )
            (tmp_path / "resume-close-errors.json").write_text(
                json.dumps(close_failures)
            )
            raise

    monkeypatch.setattr(_CaptureFileDescriptors, "close_descriptor", observed_close)
    case = tmp_path_factory.mktemp("r") if action == "rollback" else tmp_path
    with _capture_fixture(case, monkeypatch) as values:
        _, journal, _, stores, preview, _, _ = values
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _TraceReadBudget,
    )

    source_before = mobile_consolidation_fixture.fingerprints(preview.source_dir)
    issued = []
    native_budget_init = _TraceReadBudget.__init__

    def counted_budget(actual):
        native_budget_init(actual)
        issued.append(actual)

    monkeypatch.setattr(_TraceReadBudget, "__init__", counted_budget)
    try:
        service = RecoveryService(journal.root.parent)
        append = _Frame.append
        reached = []

        def interrupted(actual, event, evidence, **kwargs):
            if event == "mobile_committed":
                reached.append(actual.journal.operation_id)
                raise OSError("injected before mobile terminal publication")
            return append(actual, event, evidence, **kwargs)

        try:
            with monkeypatch.context() as fault:
                fault.setattr(_Frame, "append", interrupted)
                operation = service.start_mobile_consolidation(
                    preview,
                    destination_confirmed=True,
                    rollback_password=b"test-private-mobile-rollback",
                )
                assert service.wait(operation)["state"] == "recovery_required"
            assert reached == [operation]
        finally:
            service.close()
        fresh = RecoveryService(journal.root.parent)
        try:
            assert action in fresh.status(operation)["actions"]
            resumed = fresh.start_recovery(
                operation,
                action=action,
                rollback_password=b"test-private-mobile-rollback",
            )
            final = fresh.wait(resumed)
            assert final["state"] == "succeeded", errors
            assert fresh.pending_operations() == ()
            if action == "rollback":
                assert final["result"]["rolled_back"]
        finally:
            fresh.close()
        from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
            require_mobile_cutover,
        )

        if action == "finish":
            assert require_mobile_cutover(preview.source_dir, stores) is None
        else:
            with pytest.raises(ValueError, match="mobile_cutover_required"):
                require_mobile_cutover(preview.source_dir, stores)
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

            original = CharactersRAGDB(stores.chat_path, "ordinary-rolled-back-reader")
            try:
                assert original.get_all_conversation_ids() == ["shared-chat"]
            finally:
                original.close_connection()

            successor = RecoveryService(journal.root.parent)
            try:
                reviewed = successor.preview_mobile_consolidation(
                    preview.source_selector, preview.target_selector
                )
                assert reviewed.source_fingerprint == preview.source_fingerprint
                new_operation = successor.start_mobile_consolidation(
                    reviewed,
                    destination_confirmed=True,
                    rollback_password=b"test-private-mobile-rollback",
                )
                assert new_operation != operation
                result = successor.wait(new_operation)
                assert result["state"] == "succeeded", (dict(result), errors)
                assert successor.pending_operations() == ()
            finally:
                successor.close()
            assert require_mobile_cutover(preview.source_dir, stores) is None
    finally:
        # Record even a failed assertion after each service.close; quarantine may remain on failure.
        source_after = mobile_consolidation_fixture.fingerprints(preview.source_dir)
        profile_violations = take_violations()
        network_violations = blocked_attempts()
        native_children = [row for row in issued if row._parent is not None]
        pair_facts = [
            {
                "coupled": _mobile_budget_pair(row._parent, row),
                "aggregate_remaining": row._parent._remaining,
                "native_remaining": row._remaining,
            }
            for row in native_children
        ]
        (tmp_path / "resume-final-evidence.json").write_text(
            json.dumps(
                {
                    "source_before": source_before,
                    "source_after": source_after,
                    "profile_violations": profile_violations,
                    "network_violations": network_violations,
                    "pairs_after_service_close": pair_facts,
                    "action": action,
                    "errors": errors,
                }
            )
        )
    assert source_after == source_before
    assert not profile_violations and not network_violations
    assert len(native_children) == (3 if action == "rollback" else 2)
    assert all(row["coupled"] for row in pair_facts)


@pytest.mark.parametrize("source_state", ["absent", "empty"])
def test_no_data_refuses_uncapturable_required_target_before_preparation(
    tmp_path, monkeypatch, source_state
):
    import json

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import recovery_service
    from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    def seed(stores, **kwargs):
        value = _small_native_seed(stores, **kwargs)
        # This fixture chooses an already existing private canonical parent;
        # all three canonical database leaves remain genuinely absent.
        (tmp_path / "profile" / "new" / "nested").mkdir(
            mode=0o700, parents=True, exist_ok=True
        )
        return value

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    with _capture_fixture(
        tmp_path,
        monkeypatch,
        fresh=True,
        source_state=source_state,
        require_complete_inventory=False,
    ) as values:
        _, old_journal, _, stores, preview, inventory, _ = values
        assert not preview.has_data
        assert not any(path.exists() for path in stores.paths)
        (tmp_path / "no-data-inventory.json").write_text(
            json.dumps(
                [
                    (row.owner, row.logical_id, row.status)
                    for row in inventory.items
                    if row.status in {"unsupported", "missing_required", "unavailable"}
                ]
            )
        )
    errors = []
    native_issue = recovery_service.issue_code

    def observe_issue(error, **kwargs):
        import traceback

        errors.append(
            (type(error).__name__, str(error), traceback.format_tb(error.__traceback__))
        )
        (tmp_path / "no-data-errors.json").write_text(json.dumps(errors))
        return native_issue(error, **kwargs)

    monkeypatch.setattr(recovery_service, "issue_code", observe_issue)
    original_init = CharactersRAGDB.__init__
    opened = []

    def observed_init(database, path, *args, **kwargs):
        from pathlib import Path

        opened.append(Path(path))
        assert preview.source_dir not in Path(path).parents
        return original_init(database, path, *args, **kwargs)

    monkeypatch.setattr(CharactersRAGDB, "__init__", observed_init)
    service = RecoveryService(old_journal.root.parent)
    try:
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"test-private-mobile-rollback",
        )
        state = service.wait(operation)
        cached = service._states[operation]
        (tmp_path / "no-data-worker.json").write_text(
            json.dumps(
                {
                    key: cached[key]
                    for key in ("state", "phase", "issues", "review_issues")
                }
            )
        )
        assert state["state"] == "failed", (dict(state), errors)
        assert "missing_required" in state["review_issues"]
        assert service.pending_operations() == ()
    finally:
        service.close()
    assert preview.source_dir.exists() is (source_state == "empty")
    assert not tuple(preview.source_dir.glob("*.db"))
    assert not any(path.exists() for path in stores.paths)
    assert opened == []
    with pytest.raises(FileNotFoundError):
        _journal(old_journal.root.parent, operation)
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _source_claim_locations,
    )

    assert not any(
        path.exists()
        for pair in _source_claim_locations(preview.source_dir)
        for path in pair
    )
