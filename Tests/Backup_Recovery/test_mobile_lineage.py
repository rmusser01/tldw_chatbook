"""Local successor grammar never turns a filename into source authority."""

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_admission import prepare, prepared


def test_old_prepared_absence_remains_exact_native_grammar(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery.journal import _validate

    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        with journal._locked(exclusive=False) as parent:
            rows = journal._records(parent)
        assert rows[0].evidence["predecessor"] is None
        historical = dict(rows[0].evidence)
        historical.pop("predecessor")
        assert _validate("mobile_prepared", historical, []) == historical


def test_fixed_claim_filename_is_refused_in_a_first_operation(tmp_path, monkeypatch):
    with prepared(tmp_path, monkeypatch) as (
        session,
        journal,
        config,
        _,
        preview,
        inventory,
        _,
    ):
        prepare(session, journal, config, preview, inventory)
        path = journal.root / "mobile-source-claim.json"
        path.write_text("{}")
        path.chmod(0o600)
        with (
            journal._locked(exclusive=False) as parent,
            pytest.raises(ValueError, match="journal_sequence_invalid"),
        ):
            journal._records(parent)


def test_lineage_inventory_keeps_actual_locked_journal_and_worker_pair(
    tmp_path, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _claim_source
    from tldw_chatbook.Backup_Recovery.mobile_journal_frame import _locked_mobile
    from tldw_chatbook.Backup_Recovery.mobile_lineage import _inventory_lineage
    from tldw_chatbook.Backup_Recovery.storage_admission import _attach_mobile_budgets
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_operation_budgets,
        _TraceReadBudget,
    )

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        issued = []
        original = _TraceReadBudget.__init__

        def counted(value):
            original(value)
            issued.append(value)

        monkeypatch.setattr(_TraceReadBudget, "__init__", counted)
        pair = _mobile_operation_budgets()
        _attach_mobile_budgets(session, pair)
        with _claim_source(
            session, journal, preview, approved_scope=inventory.scope_digest
        ) as claim:
            with _locked_mobile(journal, session, claim.custody, Event()) as frame:
                before = tuple(frame.records)
                _inventory_lineage(claim.original_path, preview.source_dir)
                frame.verify()
                assert tuple(frame.records) == before
                assert frame.journal is journal
            assert issued == list(pair)


def test_verification_requires_exact_current_native_frame(tmp_path, monkeypatch):
    from threading import Event

    from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
        _capture_fixture,
    )
    from tldw_chatbook.Backup_Recovery import mobile_journal_frame as frames
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _claim_source
    from tldw_chatbook.Backup_Recovery.storage_admission import _attach_mobile_budgets
    from tldw_chatbook.Chat.console_trace_read_budget import _mobile_operation_budgets

    with _capture_fixture(tmp_path, monkeypatch) as (
        session,
        journal,
        _,
        _,
        preview,
        inventory,
        _,
    ):
        pair = _mobile_operation_budgets()
        _attach_mobile_budgets(session, pair)
        with _claim_source(
            session, journal, preview, approved_scope=inventory.scope_digest
        ) as claim:
            cancel = Event()
            with (
                pytest.raises(ValueError, match="mobile_current_frame_required"),
                frames._verify_current_mobile(journal, session, claim.custody, cancel),
            ):
                pytest.fail("unlocked verification admitted")
            with frames._locked_mobile(
                journal, session, claim.custody, cancel
            ) as frame:
                with frames._verify_current_mobile(
                    journal, session, claim.custody, cancel
                ) as actual:
                    assert actual is frame
                    assert actual.budget is pair[0]
                for supplied in (
                    (object(), session, claim.custody, cancel),
                    (journal, object(), claim.custody, cancel),
                    (journal, session, object(), cancel),
                    (journal, session, claim.custody, Event()),
                ):
                    with (
                        pytest.raises(ValueError, match="mobile_.*frame"),
                        frames._verify_current_mobile(*supplied),
                    ):
                        pytest.fail("foreign association admitted")
                with (
                    pytest.raises(
                        ValueError, match="mobile_nested_journal_write_forbidden"
                    ),
                    journal._locked(exclusive=True),
                ):
                    pytest.fail("nested publication admitted")
                with (
                    pytest.raises(ValueError, match="mobile_native_journal_required"),
                    frames._locked_mobile(journal, session, claim.custody, cancel),
                ):
                    pytest.fail("nested issuance admitted")
            with (
                pytest.raises(ValueError, match="mobile_current_frame_required"),
                frames._verify_current_mobile(journal, session, claim.custody, cancel),
            ):
                pytest.fail("retired frame admitted")
