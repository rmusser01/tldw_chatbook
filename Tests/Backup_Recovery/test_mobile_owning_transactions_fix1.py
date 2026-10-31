"""Native attached -> populated -> detached replay under original relationships."""

from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_native_operation import captured_operation
from Tests.Backup_Recovery.test_mobile_owning_transactions import _small_native_seed


@pytest.mark.parametrize(
    "source_case",
    ["detached_original", "detached_current", "detached_twice", "empty_detached"],
)
def test_actual_proved_original_detached_replay(tmp_path, monkeypatch, source_case):
    from Tests.Backup_Recovery import mobile_consolidation_fixture
    from tldw_chatbook.Backup_Recovery import mobile_native_rows
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
    from tldw_chatbook.DB.mobile_consolidation_receipts import _read_owner_receipt

    observed = {}
    prepare = mobile_native_rows._prepare_rows

    def seed(stores, **kwargs):
        return _small_native_seed(stores, **kwargs, source_case=source_case)

    def prepared(plan, budget):
        observed["rows"] = tuple(
            row for row in plan.rows if row.table.startswith("console_trace_")
        )
        return prepare(plan, budget)

    monkeypatch.setattr(mobile_consolidation_fixture, "_seed", seed)
    monkeypatch.setattr(mobile_native_rows, "_prepare_rows", prepared)
    with captured_operation(tmp_path, monkeypatch) as captured:
        captured.consolidate(password=b"test-private-mobile-rollback", cancel=Event())
        with (
            captured.owner("db.chachanotes.primary") as owner,
            owner.transaction() as connection,
        ):
            assert (
                _read_owner_receipt(
                    connection.connection,
                    captured.journal.operation_id,
                    captured.preview.source_fingerprint,
                    "chat",
                    _TraceReadBudget(),
                )
                is not None
            )
            for row in observed["rows"]:
                mobile_native_rows._verify_row(connection, row, dict(row.values))
            assert (
                connection.execute(
                    "SELECT count(*) FROM mobile_consolidation_batches WHERE batch_key='chat' AND operation_id=?",
                    (captured.journal.operation_id,),
                ).fetchone()[0]
                == 1
            )
            assert connection.execute(
                "SELECT count(*) FROM console_trace_owners WHERE attached=0"
            ).fetchone()[0] == (2 if source_case == "detached_twice" else 1)
