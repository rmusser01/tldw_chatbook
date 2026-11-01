"""Early terminal pricing is a real native pre-effect boundary."""

import hashlib
import sys
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed


@pytest.mark.parametrize("source_state", [None, "absent"], ids=["populated", "no-data"])
def test_terminal_cost_refusal_precedes_first_native_target_owner(
    tmp_path, monkeypatch, source_state
):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import _CapturedMobile
    from tldw_chatbook.Backup_Recovery.recovery_copies import _journal
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with _capture_fixture(tmp_path, monkeypatch, source_state=source_state) as values:
        _, journal, _, stores, preview, _, _ = values
    assert preview.has_data is (source_state is None)
    source_before = (
        mobile_consolidation_fixture.fingerprints(preview.source_dir)
        if preview.source_dir.exists()
        else None
    )
    targets_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stores.paths
    }
    original_reserve = _TraceReadBudget.reserve
    original_owner = _CapturedMobile.owner
    priced = []
    owner_entries = []

    def exhaust_at_early_price(budget, amount):
        if sys._getframe(1).f_code.co_name == "_preprice_terminal" and not priced:
            priced.append(budget)
            # Spend the SAME real allowance at the exact tested boundary. This
            # does not alter the production issuer, defaults, or native pair.
            original_reserve(budget, budget._remaining)
        return original_reserve(budget, amount)

    @contextmanager
    def guarded_owner(captured, owner_id):
        owner_entries.append(owner_id)
        assert priced, "first target owner reached before terminal prepricing"
        with original_owner(captured, owner_id) as owner:
            yield owner

    monkeypatch.setattr(_TraceReadBudget, "reserve", exhaust_at_early_price)
    monkeypatch.setattr(_CapturedMobile, "owner", guarded_owner)
    service = RecoveryService(journal.root.parent)
    try:
        operation = service.start_mobile_consolidation(
            preview,
            destination_confirmed=True,
            rollback_password=b"test-private-mobile-rollback",
        )
        state = service.wait(operation)
        assert state["state"] == "recovery_required", dict(state)
        assert priced, "no terminal prepricing observed"
        assert owner_entries == []
        actual = _journal(journal.root.parent, operation)
        with actual._locked(exclusive=False) as parent:
            records = actual._records(parent)
        assert not any(
            row.event.startswith(("mobile_owner_", "mobile_path_", "mobile_history_"))
            for row in records
        )
        assert service.pending_operations()
    finally:
        service.close()
    assert source_before == (
        mobile_consolidation_fixture.fingerprints(preview.source_dir)
        if preview.source_dir.exists()
        else None
    )
    assert targets_before == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in stores.paths
    }
    assert take_violations() == []
    assert not blocked_attempts()


def test_private_original_allowance_preprices_before_native_receipts(
    tmp_path, monkeypatch
):
    import json
    from threading import Event

    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation
    from Tests.Backup_Recovery.test_mobile_owning_transactions import (
        _native_terminal_interruption,
    )
    from Tests.network_guard import blocked_attempts
    from Tests.real_profile_guard import take_violations
    from tldw_chatbook.Backup_Recovery import mobile_terminal_pricing, storage_admission

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    native = mobile_terminal_pricing._preprice_terminal
    priced = []
    budgets = []
    report = {"scope": "private original16 preterminal component"}

    def observed(captured, **kwargs):
        admission = storage_admission._mobile_admission(captured.session)
        assert admission[4] is None and admission[2]._parent is None
        budget = admission[2]
        budgets.append(budget)
        before = budget._remaining
        result = native(captured, **kwargs)
        priced.append((before, budget._remaining))
        return result

    monkeypatch.setattr(mobile_terminal_pricing, "_preprice_terminal", observed)
    try:
        with captured_operation(tmp_path, monkeypatch) as captured:
            source = captured.preview.source_dir
            before = mobile_consolidation_fixture.fingerprints(source)
            with _native_terminal_interruption(monkeypatch):
                captured.consolidate(
                    password=b"test-private-mobile-rollback", cancel=Event()
                )
            assert len(priced) == 1 and priced[0][1] < priced[0][0] <= 16 * 1024**2
            for owner_id, family in (
                ("db.chachanotes.primary", "chat"),
                ("db.workspaces", "workspaces"),
                ("db.agent_runs", "runs"),
            ):
                with (
                    captured.owner(owner_id) as owner,
                    owner.transaction() as connection,
                ):
                    assert (
                        connection.execute(
                            "SELECT count(*) FROM mobile_consolidation_batches WHERE operation_id=? AND batch_key=?",
                            (captured.journal.operation_id, family),
                        ).fetchone()[0]
                        == 1
                    )
            report["three_native_receipts"] = True
        report["source_unchanged"] = (
            before == mobile_consolidation_fixture.fingerprints(source)
        )
        assert report["source_unchanged"]
    finally:
        report.update(
            preprice=priced,
            native_remaining=[b._remaining for b in budgets],
            profile_guard=take_violations(),
            network_guard=blocked_attempts(),
        )
        (tmp_path / "private-terminal-pricing.json").write_text(
            json.dumps(report, indent=2)
        )
    assert not report["profile_guard"] and not report["network_guard"]
