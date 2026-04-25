import pytest

from tldw_chatbook.Evaluations_Interop.evaluation_scope_service import EvaluationScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeLocalEvaluationService:
    def __init__(self):
        self.calls = []

    def list_evaluations(self, *, limit=100, offset=0, eval_type=None):
        self.calls.append(("list_evaluations", limit, offset, eval_type))
        return [
            {
                "id": "task_123",
                "name": "local_eval",
                "description": "Local evaluation",
                "task_type": "question_answer",
                "config_format": "custom",
                "config_data": {"metrics": ["accuracy"]},
                "dataset_id": "dataset_123",
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:02:00Z",
                "version": 2,
                "client_id": "local_client",
            }
        ]

    def get_evaluation(self, eval_id):
        self.calls.append(("get_evaluation", eval_id))
        return {
            "id": eval_id,
            "name": "local_eval",
            "description": "Local evaluation",
            "task_type": "question_answer",
            "config_format": "custom",
            "config_data": {"metrics": ["accuracy"]},
            "dataset_id": "dataset_123",
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:02:00Z",
            "version": 2,
            "client_id": "local_client",
        }

    def create_evaluation(self, **kwargs):
        self.calls.append(("create_evaluation", kwargs))
        return "task_123"

    def update_evaluation(self, eval_id, **kwargs):
        self.calls.append(("update_evaluation", eval_id, kwargs))
        return True

    def list_targets(self, *, provider=None, limit=100, offset=0):
        self.calls.append(("list_targets", provider, limit, offset))
        return [
            {
                "id": "model_123",
                "name": "Preferred Local",
                "provider": "openai",
                "model_id": "gpt-4.1-mini",
                "config": {"temperature": 0.2},
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:02:00Z",
                "client_id": "local_client",
            }
        ]

    def create_run(self, eval_id, *, target_id=None, target_model=None, config=None, run_name=None, dataset_override=None, webhook_url=None):
        self.calls.append(("create_run", eval_id, target_id, target_model, config, run_name, dataset_override, webhook_url))
        return "run_new"

    def get_run_artifacts(self, run_id):
        self.calls.append(("get_run_artifacts", run_id))
        return {
            "run": {
                "id": run_id,
                "task_id": "task_123",
                "name": "local_run",
                "status": "completed",
                "model_id": "model_123",
                "model_name": "gpt-4.1-mini",
                "created_at": "2026-04-20T00:00:00Z",
                "metrics_summary": {"accuracy": 0.95},
                "config_overrides": {"temperature": 0.2},
            },
            "metrics": {"accuracy": 0.95},
            "results": [{"sample_id": "sample_1", "metrics": {"accuracy": 1.0}}],
            "detail_available": True,
        }

    def list_runs(self, *, eval_id=None, status=None, limit=100, offset=0):
        self.calls.append(("list_runs", eval_id, status, limit, offset))
        return [
            {
                "id": "run_123",
                "task_id": eval_id or "task_123",
                "name": "local_run",
                "status": "completed",
                "model_id": "model_123",
                "model_name": "gpt-4.1-mini",
                "created_at": "2026-04-20T00:00:00Z",
                "start_time": "2026-04-20T00:01:00Z",
                "end_time": "2026-04-20T00:02:00Z",
                "metrics_summary": {"accuracy": 0.9},
                "config_overrides": {"temperature": 0.2},
            }
        ]

    def cancel_run(self, run_id):
        self.calls.append(("cancel_run", run_id))
        return {"status": "cancelled", "id": run_id}

    def create_dataset(self, *, name, format="custom", source_path=None, description=None, metadata=None):
        self.calls.append(("create_dataset", name, format, source_path, description, metadata))
        return {
            "id": "dataset_local_new",
            "name": name,
            "description": description,
            "format": format,
            "source_path": source_path,
            "metadata": metadata or {},
        }

    def delete_dataset(self, dataset_id):
        self.calls.append(("delete_dataset", dataset_id))
        return None


class FakeServerEvaluationService:
    def __init__(self):
        self.calls = []

    async def list_evaluations(self, *, limit=100, after=None, eval_type=None):
        self.calls.append(("list_evaluations", limit, after, eval_type))
        return [
            {
                "id": "eval_123",
                "object": "evaluation",
                "name": "server_eval",
                "description": "Server evaluation",
                "eval_type": "classification",
                "eval_spec": {"metrics": ["f1"]},
                "dataset_id": "dataset_999",
                "created": 1713571200,
                "updated": 1713571260,
                "created_by": "user_1",
                "metadata": {"project": "server"},
            }
        ]

    async def list_runs(self, *, eval_id, limit=100, after=None, status=None):
        self.calls.append(("list_runs", eval_id, limit, after, status))
        return [
            {
                "id": "run_999",
                "object": "run",
                "eval_id": eval_id,
                "status": "running",
                "target_model": "gpt-4.1",
                "created": 1713571200,
                "progress": {
                    "completed_samples": 3,
                    "total_samples": 10,
                    "percent_complete": 30.0,
                },
            }
        ]

    async def create_run(self, eval_id, *, target_model=None, dataset_override=None, config=None, webhook_url=None, run_name=None, target_id=None):
        self.calls.append(("create_run", eval_id, target_model, dataset_override, config, webhook_url, run_name, target_id))
        return {
            "id": "run_srv",
            "object": "run",
            "eval_id": eval_id,
            "status": "pending",
            "target_model": target_model or "openai:gpt-4.1-mini",
            "created": 1713571200,
            "results": {"accuracy": 0.91},
            "config": config or {},
        }

    async def get_run_artifacts(self, run_id):
        self.calls.append(("get_run_artifacts", run_id))
        return {
            "run": {
                "id": run_id,
                "object": "run",
                "eval_id": "eval_123",
                "status": "completed",
                "target_model": "openai:gpt-4.1-mini",
                "created": 1713571200,
                "results": {"accuracy": 0.91},
            },
            "metrics": {"accuracy": 0.91},
            "results": None,
            "detail_available": False,
        }

    async def cancel_run(self, run_id):
        self.calls.append(("cancel_run", run_id))
        return {"status": "cancellation_requested", "id": run_id}

    async def create_dataset(self, *, name, samples, description=None, metadata=None):
        self.calls.append(("create_dataset", name, samples, description, metadata))
        return {
            "id": "dataset_server_new",
            "name": name,
            "description": description,
            "sample_count": len(samples),
            "metadata": metadata or {},
        }

    async def delete_dataset(self, dataset_id):
        self.calls.append(("delete_dataset", dataset_id))
        return None


class FakePolicyEnforcer:
    def __init__(self, denied_reason: str | None = None):
        self.denied_reason = denied_reason
        self.calls = []

    @classmethod
    def deny(cls, reason_code: str) -> "FakePolicyEnforcer":
        return cls(denied_reason=reason_code)

    def require_allowed(self, *, action_id: str) -> None:
        self.calls.append(action_id)
        if self.denied_reason is None:
            return
        raise PolicyDeniedError(
            action_id=action_id,
            reason_code=self.denied_reason,
            user_message=f"{action_id} denied",
            effective_source="local",
            authority_owner="server",
        )


@pytest.mark.asyncio
async def test_scope_service_routes_evaluation_list_by_backend():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_records = await scope.list_evaluations(mode="local")
    server_records = await scope.list_evaluations(mode="server")

    assert local_records[0]["record_id"] == "local:evaluation:task_123"
    assert local_records[0]["eval_type"] == "question_answer"
    assert server_records[0]["record_id"] == "server:evaluation:eval_123"
    assert server_records[0]["metadata"]["project"] == "server"


@pytest.mark.asyncio
async def test_scope_service_routes_runs_using_eval_id_for_each_backend():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_runs = await scope.list_runs(mode="local", eval_id="task_123")
    server_runs = await scope.list_runs(mode="server", eval_id="eval_123")

    assert local_runs[0]["evaluation_id"] == "task_123"
    assert local_runs[0]["target_model"] == "gpt-4.1-mini"
    assert server_runs[0]["evaluation_id"] == "eval_123"
    assert server_runs[0]["progress"]["percent_complete"] == 30.0


@pytest.mark.asyncio
async def test_scope_service_lists_local_targets_and_routes_run_creation_by_backend():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_targets = await scope.list_targets(mode="local")
    local_run = await scope.create_run(
        mode="local",
        eval_id="task_123",
        target_id="model_123",
        run_name="local_run",
        config={"temperature": 0.2},
    )
    server_run = await scope.create_run(
        mode="server",
        eval_id="eval_123",
        target_model="openai:gpt-4.1-mini",
        config={"max_workers": 2},
    )

    assert local_targets[0]["record_id"] == "local:evaluation_target:model_123"
    assert local_targets[0]["target_model"] == "openai:gpt-4.1-mini"
    assert local_run["record_id"] == "local:evaluation_run:run_new"
    assert local_run["evaluation_id"] == "task_123"
    assert server_run["record_id"] == "server:evaluation_run:run_srv"
    assert server_run["target_model"] == "openai:gpt-4.1-mini"


@pytest.mark.asyncio
async def test_evaluation_scope_service_denies_server_run_creation_in_local_mode():
    policy_enforcer = FakePolicyEnforcer.deny("wrong_source")
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=FakeServerEvaluationService(),
        policy_enforcer=policy_enforcer,
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await scope.create_run(
            mode="server",
            eval_id="eval_123",
            target_model="openai:gpt-4.1",
        )

    assert exc.value.reason_code == "wrong_source"
    assert policy_enforcer.calls == ["evaluations.run.launch.server"]


@pytest.mark.asyncio
async def test_evaluation_scope_service_keeps_normalizing_server_runs_after_policy_passes():
    policy_enforcer = FakePolicyEnforcer()
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=FakeServerEvaluationService(),
        policy_enforcer=policy_enforcer,
    )

    runs = await scope.list_runs(mode="server", eval_id="eval_123")

    assert runs[0]["record_id"] == "server:evaluation_run:run_999"
    assert policy_enforcer.calls == ["evaluations.run.list.server"]


@pytest.mark.asyncio
async def test_scope_service_create_and_update_local_evaluation_resolves_raw_local_responses():
    local = FakeLocalEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=FakeServerEvaluationService())

    created = await scope.create_evaluation(
        mode="local",
        name="local_eval",
        eval_type="question_answer",
        eval_spec={"metrics": ["accuracy"]},
    )
    updated = await scope.update_evaluation(
        "task_123",
        mode="local",
        description="Updated evaluation",
    )

    assert created["record_id"] == "local:evaluation:task_123"
    assert updated["description"] == "Local evaluation"
    assert ("create_evaluation", {"name": "local_eval", "description": None, "eval_type": "question_answer", "eval_spec": {"metrics": ["accuracy"]}, "dataset_id": None, "dataset": None, "metadata": None}) in local.calls
    assert ("get_evaluation", "task_123") in local.calls


@pytest.mark.asyncio
async def test_scope_service_get_run_artifacts_reflects_backend_detail_limitations():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_artifacts = await scope.get_run_artifacts(mode="local", run_id="run_123")
    server_artifacts = await scope.get_run_artifacts(mode="server", run_id="run_999")

    assert local_artifacts["detail_available"] is True
    assert local_artifacts["metrics"]["accuracy"] == 0.95
    assert local_artifacts["results"][0]["sample_id"] == "sample_1"
    assert local_artifacts["run"]["record_id"] == "local:evaluation_run:run_123"
    assert server_artifacts["detail_available"] is False
    assert server_artifacts["metrics"]["accuracy"] == 0.91
    assert server_artifacts["results"] is None
    assert server_artifacts["run"]["record_id"] == "server:evaluation_run:run_999"


@pytest.mark.asyncio
async def test_scope_service_cancel_run_returns_backend_response():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_payload = await scope.cancel_run(mode="local", run_id="run_123")
    server_payload = await scope.cancel_run(mode="server", run_id="run_999")

    assert local_payload["status"] == "cancelled"
    assert server_payload["status"] == "cancellation_requested"


@pytest.mark.asyncio
async def test_scope_service_routes_dataset_create_delete_by_backend_with_policy():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    policy_enforcer = FakePolicyEnforcer()
    scope = EvaluationScopeService(
        local_service=local,
        server_service=server,
        policy_enforcer=policy_enforcer,
    )

    local_dataset = await scope.create_dataset(
        mode="local",
        name="Offline Dataset",
        format="json",
        source_path="/tmp/offline.json",
        metadata={"project": "offline"},
    )
    server_dataset = await scope.create_dataset(
        mode="server",
        name="Server Dataset",
        samples=[{"input": "Q", "expected": "A"}],
        metadata={"project": "server"},
    )
    await scope.delete_dataset(mode="local", dataset_id="dataset_local_new")
    await scope.delete_dataset(mode="server", dataset_id="dataset_server_new")

    assert local_dataset["record_id"] == "local:evaluation_dataset:dataset_local_new"
    assert server_dataset["record_id"] == "server:evaluation_dataset:dataset_server_new"
    assert local.calls[-2:] == [
        ("create_dataset", "Offline Dataset", "json", "/tmp/offline.json", None, {"project": "offline"}),
        ("delete_dataset", "dataset_local_new"),
    ]
    assert server.calls[-2:] == [
        ("create_dataset", "Server Dataset", [{"input": "Q", "expected": "A"}], None, {"project": "server"}),
        ("delete_dataset", "dataset_server_new"),
    ]
    assert policy_enforcer.calls[-4:] == [
        "evaluations.dataset.create.local",
        "evaluations.dataset.create.server",
        "evaluations.dataset.delete.local",
        "evaluations.dataset.delete.server",
    ]
