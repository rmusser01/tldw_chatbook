"""Public coordinator review keeps its aggregate independently of private proof."""

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import (
    _capture_fixture,
    _prepare_review,
)
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed


def test_real_public_resume_review_uses_original_aggregate(tmp_path, monkeypatch):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import mobile_consolidation
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _review_mobile_operation,
    )
    from tldw_chatbook.Backup_Recovery.storage_admission import (
        _attach_mobile_budgets,
        _local,
        _mobile_admission,
    )
    from tldw_chatbook.Chat.console_trace_read_budget import (
        _mobile_budget_pair,
        _mobile_operation_budgets,
    )

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", _small_native_seed)
    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, config, _, preview, inventory, _ = values
        aggregate, native = _mobile_operation_budgets()
        _attach_mobile_budgets(session, (aggregate, native))
        _prepare_review(session, journal, config, preview, inventory)
        original = mobile_consolidation._preview_selected_mobile
        observed = []

        def checked(*args, **kwargs):
            scope = _local.discovery_scope
            assert kwargs["_read_budget"] is aggregate
            assert scope.read_budget is aggregate
            assert _mobile_admission(session)[2] is native
            assert _mobile_admission(session)[4] is aggregate
            observed.append(scope)
            return original(*args, **kwargs)

        monkeypatch.setattr(mobile_consolidation, "_preview_selected_mobile", checked)
        before = aggregate._remaining
        resumed = _review_mobile_operation(session, journal)
        assert resumed.source_fingerprint == preview.source_fingerprint
        assert observed and aggregate._remaining < before
        assert native._remaining == 16 * 1024**2
        assert _mobile_budget_pair(aggregate, native)
        assert all(not scope.active for scope in observed)
