from unittest.mock import Mock

import pytest

from tldw_chatbook.Evaluations_Interop.server_evaluations_service import ServerEvaluationsService
from tldw_chatbook.runtime_policy.types import PolicyDecision, PolicyDeniedError


class FakeEvaluationsClient:
    def __init__(self):
        self.calls = []

    async def list_evaluations(self, **kwargs):
        self.calls.append(("list_evaluations", kwargs))
        return {"data": []}

    async def create_evaluation(self, request_data):
        self.calls.append(("create_evaluation", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "eval_1", "name": "Eval"}

    async def list_evaluation_datasets(self, **kwargs):
        self.calls.append(("list_evaluation_datasets", kwargs))
        return {"data": []}

    async def create_evaluation_dataset(self, request_data):
        self.calls.append(("create_evaluation_dataset", request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "dataset_1", "name": "Dataset", "sample_count": 1}

    async def delete_evaluation_dataset(self, dataset_id):
        self.calls.append(("delete_evaluation_dataset", dataset_id))
        return None

    async def create_evaluation_run(self, eval_id, request_data):
        self.calls.append(("create_evaluation_run", eval_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"id": "run_1", "eval_id": eval_id, "status": "queued"}

    async def cancel_evaluation_run(self, run_id):
        self.calls.append(("cancel_evaluation_run", run_id))
        return {"id": run_id, "status": "cancellation_requested"}


@pytest.mark.asyncio
async def test_server_evaluations_service_enforces_policy_actions():
    client = FakeEvaluationsClient()
    policy = Mock()
    service = ServerEvaluationsService(client=client, policy_enforcer=policy)

    await service.list_evaluations(limit=25)
    await service.create_evaluation(
        name="Eval",
        eval_type="classification",
        eval_spec={"metric": "accuracy"},
    )
    await service.list_datasets(limit=10)
    await service.create_dataset(name="Dataset", samples=[{"input": "Q", "expected": "A"}])
    await service.delete_dataset("dataset_1")
    await service.create_run("eval_1", target_model="openai:gpt-4.1-mini")
    await service.cancel_run("run_1")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "evaluations.dataset.list.server",
        "evaluations.dataset.create.server",
        "evaluations.dataset.list.server",
        "evaluations.dataset.create.server",
        "evaluations.dataset.delete.server",
        "evaluations.run.launch.server",
        "evaluations.run.update.server",
    ]


@pytest.mark.asyncio
async def test_server_evaluations_service_hard_stops_denied_ui_policy_decision():
    policy = Mock()
    policy.require_allowed = None
    policy.require_ui_action_allowed = Mock(
        return_value=PolicyDecision(
            allowed=False,
            reason_code="server_unreachable",
            user_message="Blocked.",
            effective_source="server",
            authority_owner="server",
        )
    )
    client = FakeEvaluationsClient()
    service = ServerEvaluationsService(client=client, policy_enforcer=policy)

    with pytest.raises(PolicyDeniedError) as exc:
        await service.list_evaluations()

    assert exc.value.reason_code == "server_unreachable"
    assert client.calls == []
