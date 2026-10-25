"""Real owned SQLite claims and retention protect at-most-once dispatch."""

from dataclasses import replace

import pytest

from Tests.agent_runs_owner import owned_agent_runs_db
from tldw_chatbook.Chat.console_remote_models import ConsoleHostIdentity
from tldw_chatbook.DB.console_command_receipts import (
    ConsoleCommandReceipt,
    ConsoleCommandReceiptRepository,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def repository(tmp_path):
    database = owned_agent_runs_db(tmp_path / "agent_runs.db")
    yield ConsoleCommandReceiptRepository(database)
    database.close()


def receipt(command_id="command-a", **changes):
    return replace(
        ConsoleCommandReceipt(
            command_id=command_id,
            server_epoch="server",
            console_id="console",
            action="send",
            target=ConsoleHostIdentity(server_epoch="server", console_id="console"),
            payload_fingerprint="a" * 64,
            state="pending",
            issued_ns=1,
            expires_ns=301,
        ),
        **changes,
    )


def test_only_atomic_inserter_owns_pending_dispatch(repository):
    first = repository.claim(receipt())
    assert first.inserted
    duplicate = repository.claim(receipt())
    assert not duplicate.inserted
    assert duplicate.receipt == first.receipt
    with pytest.raises(ValueError, match="mismatch"):
        repository.claim(receipt(payload_fingerprint="b" * 64))


def test_prune_retains_valid_and_uncertain_evidence(repository):
    for command_id, state in [
        ("valid", "applied"),
        ("pending", "pending"),
        ("unknown", "outcome_unknown"),
        ("expired", "refused"),
    ]:
        repository.claim(
            receipt(
                command_id,
                state=state,
                expires_ns=1000 if command_id == "valid" else 301,
            )
        )
    assert repository.prune_invalid(receipt().owner_epoch, 302) == 1
    assert repository.get("expired") is None
    assert all(repository.get(key) for key in ("valid", "pending", "unknown"))


def test_transition_is_compare_and_swap_and_unresolved_pages_are_bounded(repository):
    for number in range(4):
        repository.claim(receipt(f"c{number}"))
    changed = repository.transition(
        "c0", expected_state="pending", state="accepted", native_id="turn"
    )
    assert changed.native_id == "turn"
    assert (
        repository.transition("c0", expected_state="pending", state="refused")
        == changed
    )
    page = repository.list_unresolved(limit=2)
    assert len(page.items) == 2 and page.next_cursor
    following = repository.list_unresolved(cursor=page.next_cursor, limit=2)
    assert len(following.items) == 1 and following.next_cursor is None
    with pytest.raises(ValueError):
        repository.list_unresolved(limit=True)


def test_read_only_handle_queries_without_migration_or_mutation(repository):
    from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionUnavailable
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    repository.claim(receipt())
    reader = AgentRunsDB.open_read_only(repository.runs_db.db_path)
    try:
        assert reader.command_receipts.get("command-a").state == "pending"
        with pytest.raises(ConsoleExecutionUnavailable):
            reader.command_receipts.claim(receipt("other"))
    finally:
        reader.close()


def test_peer_connections_only_one_claims(repository):
    from concurrent.futures import ThreadPoolExecutor

    def claim():
        try:
            return repository.claim(receipt()).inserted
        finally:
            repository.runs_db.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        assert sorted(workers.map(lambda _index: claim(), range(2))) == [False, True]


@pytest.mark.parametrize(
    "changes",
    [
        {"state": "invented"},
        {"action": "shell"},
        {"expires_ns": 0},
        {"issued_ns": True},
        {"payload_fingerprint": "body"},
        {"console_id": "different"},
    ],
)
def test_repository_receipt_rejects_invalid_boundary_values(changes):
    with pytest.raises(ValueError):
        receipt(**changes)


def test_expired_known_acceptance_can_prune_but_never_valid_evidence(repository):
    for index, state in enumerate(("accepted", "queued", "applied", "refused")):
        repository.claim(receipt(f"known-{index}", state=state))
    assert repository.prune_invalid(receipt().owner_epoch, 300) == 0
    assert repository.prune_invalid(receipt().owner_epoch, 301) == 4


@pytest.mark.parametrize("protected", [False, True])
def test_full_prior_owner_known_receipts_reclaim_only_one_under_pressure(
    repository, protected
):
    from tldw_chatbook.DB.console_command_receipts import MAX_COMMAND_RECEIPTS

    current = ConsoleHostIdentity(server_epoch="successor", console_id="console")
    protected_receipts = (
        [
            receipt("old-pending", state="pending"),
            receipt("old-unknown", state="outcome_unknown"),
            receipt(
                "current-valid",
                server_epoch=current.server_epoch,
                target=current,
                state="accepted",
                expires_ns=999999,
            ),
        ]
        if protected
        else []
    )
    known_count = MAX_COMMAND_RECEIPTS - len(protected_receipts)
    for index in range(known_count):
        repository.claim(
            receipt(
                f"known-{index:04}",
                state=("accepted", "queued", "applied", "refused")[index % 4],
                updated_at="2026-10-25T00:00:00+00:00",
            )
        )
    for item in protected_receipts:
        repository.claim(item)
    owner_epoch = receipt(
        "new", server_epoch=current.server_epoch, target=current
    ).owner_epoch
    repository.invalidate_prior_owner(owner_epoch)
    # Restart alone retains all known history for review, without interpreting old clocks.
    with repository.runs_db.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_command_receipts"
            ).fetchone()[0]
            == MAX_COMMAND_RECEIPTS
        )
    assert repository.get("known-0000") is not None
    new = receipt("new", server_epoch=current.server_epoch, target=current)
    assert repository.claim(new).inserted
    assert repository.get("known-0000") is None
    assert repository.get("known-0001") is not None
    with repository.runs_db.connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM console_command_receipts"
            ).fetchone()[0]
            == MAX_COMMAND_RECEIPTS
        )
    assert not repository.claim(new).inserted
    assert repository.get("known-0001") is not None  # Matching retries reclaim nothing.
    with pytest.raises(ValueError, match="mismatch"):
        repository.claim(replace(new, payload_fingerprint="b" * 64))
    assert repository.get("known-0001") is not None
    if protected:
        assert repository.get("old-pending").state == "outcome_unknown"
        assert repository.get("old-unknown").state == "outcome_unknown"
        assert repository.get("current-valid").state == "accepted"


def test_pressure_does_not_discard_uncertain_or_current_owner_receipts(
    repository, monkeypatch
):
    from tldw_chatbook.DB import console_command_receipts as module

    monkeypatch.setattr(module, "MAX_COMMAND_RECEIPTS", 3)
    current = ConsoleHostIdentity(server_epoch="successor", console_id="console")
    for item in (
        receipt("old-pending"),
        receipt("old-unknown", state="outcome_unknown"),
        receipt(
            "current-valid",
            server_epoch=current.server_epoch,
            target=current,
            state="accepted",
        ),
    ):
        repository.claim(item)
    with pytest.raises(module.ConsoleReceiptCapacity):
        repository.claim(
            receipt("new", server_epoch=current.server_epoch, target=current)
        )
    assert all(
        repository.get(key) is not None
        for key in ("old-pending", "old-unknown", "current-valid")
    )
