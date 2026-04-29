import pytest

from tldw_chatbook.Prompt_Studio_Interop.server_prompt_studio_service import ServerPromptStudioService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakePromptStudioClient:
    def __init__(self):
        self.calls = []

    async def create_prompt_studio_project(self, request_data, idempotency_key=None):
        self.calls.append(("create_prompt_studio_project", request_data, idempotency_key))
        return {"success": True, "data": {"id": 1, "name": request_data.name, "status": request_data.status}}

    async def list_prompt_studio_projects(self, **kwargs):
        self.calls.append(("list_prompt_studio_projects", kwargs))
        return {"success": True, "data": [{"id": 1, "name": "Smoke", "status": "draft"}], "metadata": {"total": 1}}

    async def get_prompt_studio_project(self, project_id):
        self.calls.append(("get_prompt_studio_project", project_id))
        return {"success": True, "data": {"id": project_id, "name": "Smoke", "status": "draft"}}

    async def update_prompt_studio_project(self, project_id, request_data):
        self.calls.append(("update_prompt_studio_project", project_id, request_data))
        return {"success": True, "data": {"id": project_id, "name": request_data.name, "status": "active"}}

    async def delete_prompt_studio_project(self, project_id, permanent=False):
        self.calls.append(("delete_prompt_studio_project", project_id, permanent))
        return {"message": "deleted"}

    async def archive_prompt_studio_project(self, project_id):
        self.calls.append(("archive_prompt_studio_project", project_id))
        return {"success": True, "data": {"id": project_id, "status": "archived"}}

    async def unarchive_prompt_studio_project(self, project_id):
        self.calls.append(("unarchive_prompt_studio_project", project_id))
        return {"success": True, "data": {"id": project_id, "status": "draft"}}

    async def get_prompt_studio_project_stats(self, project_id):
        self.calls.append(("get_prompt_studio_project_stats", project_id))
        return {"success": True, "data": {"project_id": project_id, "prompt_count": 2}}

    async def create_prompt_studio_prompt(self, request_data, idempotency_key=None):
        self.calls.append(("create_prompt_studio_prompt", request_data, idempotency_key))
        return {"success": True, "data": {"id": 11, "project_id": request_data.project_id, "name": request_data.name}}

    async def list_prompt_studio_prompts(self, project_id, **kwargs):
        self.calls.append(("list_prompt_studio_prompts", project_id, kwargs))
        return {"success": True, "data": [{"id": 11, "project_id": project_id, "name": "Prompt"}]}

    async def get_prompt_studio_prompt(self, prompt_id):
        self.calls.append(("get_prompt_studio_prompt", prompt_id))
        return {"success": True, "data": {"id": prompt_id, "project_id": 1, "name": "Prompt"}}

    async def update_prompt_studio_prompt(self, prompt_id, request_data):
        self.calls.append(("update_prompt_studio_prompt", prompt_id, request_data))
        return {"success": True, "data": {"id": prompt_id, "project_id": 1, "name": request_data.name}}

    async def get_prompt_studio_prompt_history(self, prompt_id):
        self.calls.append(("get_prompt_studio_prompt_history", prompt_id))
        return {"success": True, "data": [{"version": 1, "prompt_id": prompt_id}]}

    async def revert_prompt_studio_prompt(self, prompt_id, version):
        self.calls.append(("revert_prompt_studio_prompt", prompt_id, version))
        return {"success": True, "data": {"id": prompt_id, "version": version}}

    async def preview_prompt_studio_prompt(self, request_data):
        self.calls.append(("preview_prompt_studio_prompt", request_data))
        return {"success": True, "data": {"project_id": request_data.project_id, "rendered": "Hello Ada"}}

    async def convert_prompt_studio_prompt(self, request_data):
        self.calls.append(("convert_prompt_studio_prompt", request_data))
        return {"success": True, "data": {"project_id": request_data.project_id, "prompt_format": "structured"}}

    async def execute_prompt_studio_prompt(self, request_data):
        self.calls.append(("execute_prompt_studio_prompt", request_data))
        return {"output": "ok", "tokens_used": 4, "execution_time": 0.1}

    async def create_prompt_studio_test_case(self, request_data):
        self.calls.append(("create_prompt_studio_test_case", request_data))
        return {"success": True, "data": {"id": 21, "project_id": request_data.project_id, "name": request_data.name}}

    async def create_prompt_studio_test_cases_bulk(self, request_data):
        self.calls.append(("create_prompt_studio_test_cases_bulk", request_data))
        return {"success": True, "data": [{"id": 22, "project_id": request_data.project_id, "name": "Bulk"}]}

    async def list_prompt_studio_test_cases(self, project_id, **kwargs):
        self.calls.append(("list_prompt_studio_test_cases", project_id, kwargs))
        return {"success": True, "data": [{"id": 21, "project_id": project_id, "name": "Case"}]}

    async def get_prompt_studio_test_case(self, test_case_id):
        self.calls.append(("get_prompt_studio_test_case", test_case_id))
        return {"success": True, "data": {"id": test_case_id, "project_id": 1, "name": "Case"}}

    async def update_prompt_studio_test_case(self, test_case_id, request_data):
        self.calls.append(("update_prompt_studio_test_case", test_case_id, request_data))
        return {"success": True, "data": {"id": test_case_id, "project_id": 1, "name": request_data.name}}

    async def delete_prompt_studio_test_case(self, test_case_id, permanent=False):
        self.calls.append(("delete_prompt_studio_test_case", test_case_id, permanent))
        return {"message": "deleted"}

    async def import_prompt_studio_test_cases(self, request_data):
        self.calls.append(("import_prompt_studio_test_cases", request_data))
        return {"success": True, "data": {"project_id": request_data.project_id, "imported": 3}}

    async def import_prompt_studio_test_cases_csv_upload(
        self,
        project_id,
        csv_content,
        filename="prompt_studio_test_cases.csv",
        signature_id=None,
        auto_generate_names=True,
    ):
        self.calls.append(
            (
                "import_prompt_studio_test_cases_csv_upload",
                project_id,
                csv_content,
                filename,
                signature_id,
                auto_generate_names,
            )
        )
        return {"success": True, "data": {"project_id": project_id, "imported": 2}}

    async def get_prompt_studio_test_cases_csv_template(self, signature_id=None):
        self.calls.append(("get_prompt_studio_test_cases_csv_template", signature_id))
        return {"content": b"name,inputs\n", "filename": "prompt_studio_test_cases_template.csv", "content_type": "text/csv"}

    async def export_prompt_studio_test_cases(self, project_id, request_data):
        self.calls.append(("export_prompt_studio_test_cases", project_id, request_data))
        return {"success": True, "data": "[]"}

    async def generate_prompt_studio_test_cases(self, **kwargs):
        self.calls.append(("generate_prompt_studio_test_cases", kwargs))
        return {"success": True, "data": [{"id": 23, "project_id": kwargs["project_id"], "name": "Generated"}]}

    async def run_prompt_studio_test_cases(self, request_data):
        self.calls.append(("run_prompt_studio_test_cases", request_data))
        return {"results": [{"test_case_id": 21, "passed": True}]}

    async def create_prompt_studio_evaluation(self, request_data):
        self.calls.append(("create_prompt_studio_evaluation", request_data))
        return {"id": 31, "project_id": request_data.project_id, "prompt_id": request_data.prompt_id, "status": "pending"}

    async def list_prompt_studio_evaluations(self, **kwargs):
        self.calls.append(("list_prompt_studio_evaluations", kwargs))
        return {
            "evaluations": [{"id": 31, "project_id": 1, "prompt_id": 11, "status": "pending"}],
            "total": 1,
            "limit": kwargs.get("limit", 100),
            "offset": kwargs.get("offset", 0),
        }

    async def get_prompt_studio_evaluation(self, evaluation_id):
        self.calls.append(("get_prompt_studio_evaluation", evaluation_id))
        return {"id": evaluation_id, "project_id": 1, "prompt_id": 11, "status": "completed"}

    async def delete_prompt_studio_evaluation(self, evaluation_id):
        self.calls.append(("delete_prompt_studio_evaluation", evaluation_id))
        return {"message": "deleted"}

    async def create_prompt_studio_optimization(self, request_data, idempotency_key=None):
        self.calls.append(("create_prompt_studio_optimization", request_data, idempotency_key))
        return {"success": True, "data": {"id": 41, "project_id": request_data.project_id, "status": "pending"}}

    async def create_prompt_studio_optimization_simple(self, request_data):
        self.calls.append(("create_prompt_studio_optimization_simple", request_data))
        return {"id": "job-1", "status": "queued"}

    async def list_prompt_studio_optimizations(self, project_id, **kwargs):
        self.calls.append(("list_prompt_studio_optimizations", project_id, kwargs))
        return {"success": True, "data": [{"id": 41, "project_id": project_id, "status": "pending"}]}

    async def get_prompt_studio_optimization(self, optimization_id):
        self.calls.append(("get_prompt_studio_optimization", optimization_id))
        return {"success": True, "data": {"id": optimization_id, "project_id": 1, "status": "pending"}}

    async def get_prompt_studio_optimization_job_status(self, job_id):
        self.calls.append(("get_prompt_studio_optimization_job_status", job_id))
        return {"id": job_id, "status": "queued"}

    async def cancel_prompt_studio_optimization(self, optimization_id, reason=None):
        self.calls.append(("cancel_prompt_studio_optimization", optimization_id, reason))
        return {"success": True, "data": {"id": optimization_id, "status": "cancelled", "reason": reason}}

    async def get_prompt_studio_optimization_strategies(self):
        self.calls.append(("get_prompt_studio_optimization_strategies",))
        return {"success": True, "data": ["cot", "few_shot"]}

    async def get_prompt_studio_optimization_history(self, optimization_id):
        self.calls.append(("get_prompt_studio_optimization_history", optimization_id))
        return {"success": True, "data": [{"optimization_id": optimization_id, "iteration_number": 1}]}

    async def add_prompt_studio_optimization_iteration(self, optimization_id, request_data):
        self.calls.append(("add_prompt_studio_optimization_iteration", optimization_id, request_data))
        return {"success": True, "data": {"id": 51, "optimization_id": optimization_id, "iteration_number": request_data.iteration_number}}

    async def list_prompt_studio_optimization_iterations(self, optimization_id, **kwargs):
        self.calls.append(("list_prompt_studio_optimization_iterations", optimization_id, kwargs))
        return {"success": True, "data": [{"id": 51, "optimization_id": optimization_id, "iteration_number": 1}]}

    async def compare_prompt_studio_optimization_strategies(self, request_data):
        self.calls.append(("compare_prompt_studio_optimization_strategies", request_data))
        return {"success": True, "data": {"prompt_id": request_data.prompt_id, "winner": "cot"}}

    async def get_prompt_studio_status(self, warn_seconds=30):
        self.calls.append(("get_prompt_studio_status", warn_seconds))
        return {"success": True, "data": {"queue_depth": 0, "warn_seconds": warn_seconds}}

    async def stream_prompt_studio_events(self, client_id=None, project_id=None):
        self.calls.append(("stream_prompt_studio_events", client_id, project_id))
        yield {"event": "optimization.started", "project_id": project_id}
        yield {"event": "stream.complete", "project_id": project_id}


class FakeClientProvider:
    def __init__(self, client):
        self.client = client
        self.build_calls = 0

    def build_client(self):
        self.build_calls += 1
        return self.client


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_server_prompt_studio_service_from_config_can_use_provider_backed_client(monkeypatch):
    def fail_build_client(_app_config):
        raise AssertionError("legacy config builder should not run")

    monkeypatch.setattr(
        "tldw_chatbook.Prompt_Studio_Interop.server_prompt_studio_service.build_runtime_api_client_from_config",
        fail_build_client,
    )

    provider = FakeClientProvider(FakePromptStudioClient())
    service = ServerPromptStudioService.from_config(
        {"tldw_api": {"base_url": "https://example.com"}},
        client_provider=provider,
    )

    result = await service.list_projects(search="provider")

    assert service.client is None
    assert service.client_provider is provider
    assert provider.build_calls == 1
    assert result["data"][0]["record_id"] == "server:prompt_studio_project:1"


@pytest.mark.asyncio
async def test_server_prompt_studio_service_routes_and_normalizes_full_rest_and_sse_surface():
    client = FakePromptStudioClient()
    policy = FakePolicyEnforcer()
    service = ServerPromptStudioService(client, policy_enforcer=policy)

    project = await service.create_project({"name": "Smoke"}, idempotency_key="create-project-1")
    projects = await service.list_projects(status="draft", search="Smoke")
    loaded_project = await service.get_project(1)
    updated_project = await service.update_project(1, {"name": "Active", "status": "active"})
    deleted_project = await service.delete_project(1)
    archived_project = await service.archive_project(1)
    unarchived_project = await service.unarchive_project(1)
    project_stats = await service.get_project_stats(1)
    prompt = await service.create_prompt({"project_id": 1, "name": "Prompt"}, idempotency_key="create-prompt-1")
    prompts = await service.list_prompts(1, include_deleted=True)
    loaded_prompt = await service.get_prompt(11)
    updated_prompt = await service.update_prompt(11, {"name": "Updated", "change_description": "rename"})
    prompt_history = await service.get_prompt_history(11)
    reverted_prompt = await service.revert_prompt(11, version=1)
    preview = await service.preview_prompt({"project_id": 1, "variables": {"name": "Ada"}})
    converted = await service.convert_prompt({"project_id": 1, "system_prompt": "sys", "user_prompt": "user"})
    execution = await service.execute_prompt({"prompt_id": 11, "inputs": {"name": "Ada"}})
    test_case = await service.create_test_case({"project_id": 1, "name": "Case", "inputs": {"name": "Ada"}})
    bulk_cases = await service.create_test_cases_bulk(
        {"project_id": 1, "test_cases": [{"name": "Bulk", "inputs": {"name": "Grace"}}]}
    )
    test_cases = await service.list_test_cases(1, is_golden=True)
    loaded_case = await service.get_test_case(21)
    updated_case = await service.update_test_case(21, {"name": "Case 2"})
    deleted_case = await service.delete_test_case(21)
    imported_cases = await service.import_test_cases({"project_id": 1, "format": "json", "data": "[]"})
    uploaded_cases = await service.import_test_cases_csv_upload(1, b"name,inputs\n", filename="cases.csv", signature_id=7)
    csv_template = await service.get_test_cases_csv_template(signature_id=7)
    exported_cases = await service.export_test_cases(1, {"format": "json"})
    generated_cases = await service.generate_test_cases(project_id=1, prompt_id=11, num_cases=2)
    run_cases = await service.run_test_cases({"prompt_id": 11, "test_case_ids": [21], "project_id": 1})
    evaluation = await service.create_evaluation({"project_id": 1, "prompt_id": 11})
    evaluations = await service.list_evaluations(project_id=1, prompt_id=11)
    loaded_evaluation = await service.get_evaluation(31)
    deleted_evaluation = await service.delete_evaluation(31)
    optimization = await service.create_optimization(
        {
            "project_id": 1,
            "initial_prompt_id": 11,
            "optimization_config": {"optimizer_type": "grid", "target_metric": "accuracy"},
        },
        idempotency_key="create-opt-1",
    )
    simple_optimization = await service.create_optimization_simple({"prompt_id": 11, "project_id": 1})
    optimizations = await service.list_optimizations(1, status="pending")
    loaded_optimization = await service.get_optimization(41)
    optimization_job = await service.get_optimization_job_status("job-1")
    cancelled = await service.cancel_optimization(41, reason="stop")
    strategies = await service.get_optimization_strategies()
    optimization_history = await service.get_optimization_history(41)
    iteration = await service.add_optimization_iteration(41, {"iteration_number": 1})
    iterations = await service.list_optimization_iterations(41)
    comparison = await service.compare_optimization_strategies(
        {"prompt_id": 11, "test_case_ids": [21], "strategies": ["cot"]}
    )
    status = await service.get_status(warn_seconds=60)
    events = [event async for event in service.stream_events(client_id="chatbook-1", project_id=1)]

    assert project["record_id"] == "server:prompt_studio_project:1"
    assert projects["data"][0]["record_id"] == "server:prompt_studio_project:1"
    assert loaded_project["record_id"] == "server:prompt_studio_project:1"
    assert updated_project["name"] == "Active"
    assert deleted_project["backend"] == "server"
    assert archived_project["record_id"] == "server:prompt_studio_project:1"
    assert unarchived_project["status"] == "draft"
    assert project_stats["record_id"] == "server:prompt_studio_project_stats:1"
    assert prompt["record_id"] == "server:prompt_studio_prompt:11"
    assert prompts["data"][0]["record_id"] == "server:prompt_studio_prompt:11"
    assert loaded_prompt["record_id"] == "server:prompt_studio_prompt:11"
    assert updated_prompt["name"] == "Updated"
    assert prompt_history["data"][0]["record_id"] == "server:prompt_studio_prompt_version:11:1"
    assert reverted_prompt["record_id"] == "server:prompt_studio_prompt:11"
    assert preview["record_id"] == "server:prompt_studio_prompt_preview:1"
    assert converted["record_id"] == "server:prompt_studio_prompt_conversion:1"
    assert execution["record_id"] == "server:prompt_studio_prompt_execution:11"
    assert test_case["record_id"] == "server:prompt_studio_test_case:21"
    assert bulk_cases["data"][0]["record_id"] == "server:prompt_studio_test_case:22"
    assert test_cases["data"][0]["record_id"] == "server:prompt_studio_test_case:21"
    assert loaded_case["record_id"] == "server:prompt_studio_test_case:21"
    assert updated_case["name"] == "Case 2"
    assert deleted_case["backend"] == "server"
    assert imported_cases["record_id"] == "server:prompt_studio_test_case_import:1"
    assert uploaded_cases["record_id"] == "server:prompt_studio_test_case_import:1"
    assert csv_template["record_id"] == "server:prompt_studio_test_case_template:7"
    assert exported_cases["record_id"] == "server:prompt_studio_test_case_export:1"
    assert generated_cases["data"][0]["record_id"] == "server:prompt_studio_test_case:23"
    assert run_cases["record_id"] == "server:prompt_studio_test_case_run:11"
    assert evaluation["record_id"] == "server:prompt_studio_evaluation:31"
    assert evaluations["evaluations"][0]["record_id"] == "server:prompt_studio_evaluation:31"
    assert loaded_evaluation["record_id"] == "server:prompt_studio_evaluation:31"
    assert deleted_evaluation["backend"] == "server"
    assert optimization["record_id"] == "server:prompt_studio_optimization:41"
    assert simple_optimization["record_id"] == "server:prompt_studio_optimization_job:job-1"
    assert optimizations["data"][0]["record_id"] == "server:prompt_studio_optimization:41"
    assert loaded_optimization["record_id"] == "server:prompt_studio_optimization:41"
    assert optimization_job["record_id"] == "server:prompt_studio_optimization_job:job-1"
    assert cancelled["status"] == "cancelled"
    assert strategies["record_id"] == "server:prompt_studio_optimization_strategies:catalog"
    assert optimization_history["data"][0]["record_id"] == "server:prompt_studio_optimization_iteration:41:1"
    assert iteration["record_id"] == "server:prompt_studio_optimization_iteration:41:51"
    assert iterations["data"][0]["record_id"] == "server:prompt_studio_optimization_iteration:41:51"
    assert comparison["record_id"] == "server:prompt_studio_optimization_comparison:11"
    assert status["record_id"] == "server:prompt_studio_status:queue"
    assert events[-1]["backend"] == "server"
    assert policy.calls == [
        "prompt_studio.projects.create.server",
        "prompt_studio.projects.list.server",
        "prompt_studio.projects.detail.server",
        "prompt_studio.projects.update.server",
        "prompt_studio.projects.delete.server",
        "prompt_studio.projects.archive.server",
        "prompt_studio.projects.restore.server",
        "prompt_studio.project_stats.detail.server",
        "prompt_studio.prompts.create.server",
        "prompt_studio.prompts.list.server",
        "prompt_studio.prompts.detail.server",
        "prompt_studio.prompts.update.server",
        "prompt_studio.prompt_versions.list.server",
        "prompt_studio.prompts.restore.server",
        "prompt_studio.prompts.preview.server",
        "prompt_studio.prompts.process.server",
        "prompt_studio.prompts.launch.server",
        "prompt_studio.test_cases.create.server",
        "prompt_studio.test_cases.create.server",
        "prompt_studio.test_cases.list.server",
        "prompt_studio.test_cases.detail.server",
        "prompt_studio.test_cases.update.server",
        "prompt_studio.test_cases.delete.server",
        "prompt_studio.test_cases.import.server",
        "prompt_studio.test_cases.import.server",
        "prompt_studio.test_cases.export.server",
        "prompt_studio.test_cases.export.server",
        "prompt_studio.test_cases.launch.server",
        "prompt_studio.test_cases.launch.server",
        "prompt_studio.evaluations.create.server",
        "prompt_studio.evaluations.list.server",
        "prompt_studio.evaluations.detail.server",
        "prompt_studio.evaluations.delete.server",
        "prompt_studio.optimizations.create.server",
        "prompt_studio.optimizations.launch.server",
        "prompt_studio.optimizations.list.server",
        "prompt_studio.optimizations.detail.server",
        "prompt_studio.optimizations.detail.server",
        "prompt_studio.optimizations.cancel.server",
        "prompt_studio.optimization_strategies.list.server",
        "prompt_studio.optimization_iterations.list.server",
        "prompt_studio.optimization_iterations.create.server",
        "prompt_studio.optimization_iterations.list.server",
        "prompt_studio.optimization_strategies.launch.server",
        "prompt_studio.status.detail.server",
        "prompt_studio.events.observe.server",
    ]
    assert client.calls[0][2] == "create-project-1"
    assert client.calls[8][2] == "create-prompt-1"


@pytest.mark.asyncio
async def test_server_prompt_studio_service_denies_before_dispatch():
    client = FakePromptStudioClient()
    service = ServerPromptStudioService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.list_projects()

    assert client.calls == []
