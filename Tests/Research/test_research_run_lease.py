"""Run leases (task-18060): exactly one executor may hold a run."""

from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService


def _service() -> LocalResearchService:
    return LocalResearchService(":memory:")


def test_first_claim_succeeds_and_returns_a_lease_id():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    lease_id = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert isinstance(lease_id, str) and lease_id


def test_second_claim_is_refused_while_the_lease_is_live():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60) is None
