from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    PromptStudioCompareStrategiesRequest,
    PromptStudioEvaluationCreate,
    PromptStudioEvaluationListResponse,
    PromptStudioEvaluationResponse,
    PromptStudioOptimizationConfig,
    PromptStudioOptimizationCreate,
    PromptStudioOptimizationIterationCreate,
    PromptStudioOptimizationSimpleCreateRequest,
    PromptStudioPromptConvertRequest,
    PromptStudioPromptCreate,
    PromptStudioPromptExecuteRequest,
    PromptStudioPromptPreviewRequest,
    PromptStudioPromptUpdate,
    PromptStudioProjectCreate,
    PromptStudioProjectUpdate,
    PromptStudioRunTestCasesRequest,
    PromptStudioStandardResponse,
    PromptStudioStatusResponse,
    PromptStudioTestCaseBase,
    PromptStudioTestCaseBulkCreate,
    PromptStudioTestCaseCreate,
    PromptStudioTestCaseExportRequest,
    PromptStudioTestCaseImportRequest,
    PromptStudioTestCaseUpdate,
    TLDWAPIClient,
)


@pytest.mark.asyncio
async def test_prompt_studio_projects_prompts_and_test_cases_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"success": True, "data": {"id": 1, "name": "Eval Lab", "status": "active"}},
            {
                "success": True,
                "data": [{"id": 1, "name": "Eval Lab", "status": "active"}],
                "metadata": {"page": 2, "per_page": 10, "total": 1, "total_pages": 1},
            },
            {"success": True, "data": {"id": 1, "name": "Eval Lab"}},
            {"success": True, "data": {"id": 1, "name": "Eval Lab v2", "status": "active"}},
            {"success": True, "data": {"message": "Project soft deleted"}},
            {"success": True, "data": {"id": 1, "status": "archived"}},
            {"success": True, "data": {"id": 1, "status": "active"}},
            {"success": True, "data": {"prompt_count": 2, "test_case_count": 3}},
            {"success": True, "data": {"id": 11, "project_id": 1, "name": "Summarizer", "version_number": 1}},
            {
                "success": True,
                "data": [{"id": 11, "project_id": 1, "name": "Summarizer", "version_number": 1}],
                "metadata": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
            },
            {"success": True, "data": {"id": 11, "project_id": 1, "name": "Summarizer", "version_number": 1}},
            {"success": True, "data": {"id": 12, "project_id": 1, "name": "Summarizer", "version_number": 2}},
            {"success": True, "data": [{"id": 11, "version_number": 1}, {"id": 12, "version_number": 2}]},
            {"success": True, "data": {"id": 13, "project_id": 1, "name": "Summarizer", "version_number": 3}},
            {"success": True, "data": {"assembled_messages": [{"role": "user", "content": "Hi"}]}},
            {"success": True, "data": {"prompt_format": "structured", "prompt_schema_version": 1}},
            {"output": "ok", "tokens_used": 7, "execution_time": 0.2},
            {"success": True, "data": {"id": 21, "project_id": 1, "name": "Smoke", "inputs": {"text": "Hi"}}},
            {
                "success": True,
                "data": [{"id": 21, "project_id": 1, "name": "Smoke", "inputs": {"text": "Hi"}}],
                "metadata": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
            },
            {"success": True, "data": {"id": 21, "project_id": 1, "name": "Smoke", "inputs": {"text": "Hi"}}},
            {"success": True, "data": {"id": 21, "project_id": 1, "name": "Smoke v2", "inputs": {"text": "Hi"}}},
            {"success": True, "data": {"message": "Test case soft deleted"}},
            {"success": True, "data": [{"id": 22}, {"id": 23}]},
            {"success": True, "data": {"imported": 2, "errors": [], "total_test_cases": 3}},
            {"success": True, "data": {"format": "json", "data": "[{}]", "content_type": "application/json"}},
            {"success": True, "data": [{"id": 24, "is_generated": True}]},
            {"results": [{"test_case_id": 21, "passed": True}]},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created_project = await client.create_prompt_studio_project(
        PromptStudioProjectCreate(name="Eval Lab", description="Prompt experiments", status="active"),
        idempotency_key="project-key",
    )
    projects = await client.list_prompt_studio_projects(
        page=2,
        per_page=10,
        status="active",
        include_deleted=True,
        search="Eval",
    )
    project = await client.get_prompt_studio_project(1)
    updated_project = await client.update_prompt_studio_project(
        1,
        PromptStudioProjectUpdate(name="Eval Lab v2", status="active"),
    )
    deleted_project = await client.delete_prompt_studio_project(1, permanent=False)
    archived_project = await client.archive_prompt_studio_project(1)
    unarchived_project = await client.unarchive_prompt_studio_project(1)
    project_stats = await client.get_prompt_studio_project_stats(1)
    created_prompt = await client.create_prompt_studio_prompt(
        PromptStudioPromptCreate(
            project_id=1,
            name="Summarizer",
            system_prompt="Summarize clearly.",
            user_prompt="{{text}}",
        ),
        idempotency_key="prompt-key",
    )
    prompts = await client.list_prompt_studio_prompts(1, include_deleted=True)
    prompt = await client.get_prompt_studio_prompt(11)
    updated_prompt = await client.update_prompt_studio_prompt(
        11,
        PromptStudioPromptUpdate(system_prompt="Summarize tersely.", change_description="tighten"),
    )
    prompt_history = await client.get_prompt_studio_prompt_history(11)
    reverted_prompt = await client.revert_prompt_studio_prompt(11, 1)
    preview = await client.preview_prompt_studio_prompt(
        PromptStudioPromptPreviewRequest(project_id=1, user_prompt="{{text}}", variables={"text": "Hi"})
    )
    converted = await client.convert_prompt_studio_prompt(
        PromptStudioPromptConvertRequest(project_id=1, system_prompt="S", user_prompt="{{text}}")
    )
    execution = await client.execute_prompt_studio_prompt(
        PromptStudioPromptExecuteRequest(prompt_id=11, inputs={"text": "Hi"}, provider="openai", model="gpt-4o-mini")
    )
    created_case = await client.create_prompt_studio_test_case(
        PromptStudioTestCaseCreate(project_id=1, name="Smoke", inputs={"text": "Hi"})
    )
    cases = await client.list_prompt_studio_test_cases(1, is_golden=True, tags=["smoke", "gold"], search="Smoke")
    case = await client.get_prompt_studio_test_case(21)
    updated_case = await client.update_prompt_studio_test_case(
        21,
        PromptStudioTestCaseUpdate(name="Smoke v2", is_golden=True),
    )
    deleted_case = await client.delete_prompt_studio_test_case(21)
    bulk_cases = await client.create_prompt_studio_test_cases_bulk(
        PromptStudioTestCaseBulkCreate(project_id=1, test_cases=[PromptStudioTestCaseBase(name="A", inputs={})])
    )
    imported_cases = await client.import_prompt_studio_test_cases(
        PromptStudioTestCaseImportRequest(project_id=1, format="json", data="[]")
    )
    exported_cases = await client.export_prompt_studio_test_cases(
        1,
        PromptStudioTestCaseExportRequest(format="json", include_golden_only=False, tag_filter=["smoke"]),
    )
    generated_cases = await client.generate_prompt_studio_test_cases(project_id=1, prompt_id=11, num_cases=2)
    run_cases = await client.run_prompt_studio_test_cases(
        PromptStudioRunTestCasesRequest(project_id=1, prompt_id=11, test_case_ids=[21], model="gpt-4o-mini")
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/prompt-studio/projects/")
    assert mocked.await_args_list[0].kwargs["headers"] == {"Idempotency-Key": "project-key"}
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/prompt-studio/projects/")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "page": 2,
        "per_page": 10,
        "status": "active",
        "include_deleted": "true",
        "search": "Eval",
    }
    assert mocked.await_args_list[3].args[:2] == ("PUT", "/api/v1/prompt-studio/projects/update/1")
    assert mocked.await_args_list[3].kwargs["json_data"] == {"name": "Eval Lab v2", "status": "active"}
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/prompt-studio/projects/delete/1")
    assert mocked.await_args_list[4].kwargs["params"] == {"permanent": "false"}
    assert mocked.await_args_list[8].args[:2] == ("POST", "/api/v1/prompt-studio/prompts/create")
    assert mocked.await_args_list[8].kwargs["headers"] == {"Idempotency-Key": "prompt-key"}
    assert mocked.await_args_list[9].args[:2] == ("GET", "/api/v1/prompt-studio/prompts/list/1")
    assert mocked.await_args_list[9].kwargs["params"] == {"page": 1, "per_page": 20, "include_deleted": "true"}
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/prompt-studio/prompts/preview")
    assert mocked.await_args_list[16].args[:2] == ("POST", "/api/v1/prompt-studio/prompts/execute")
    assert mocked.await_args_list[18].args[:2] == ("GET", "/api/v1/prompt-studio/test-cases/list/1")
    assert mocked.await_args_list[18].kwargs["params"]["tags"] == "smoke,gold"
    assert mocked.await_args_list[22].args[:2] == ("POST", "/api/v1/prompt-studio/test-cases/bulk")
    assert mocked.await_args_list[25].args[:2] == ("POST", "/api/v1/prompt-studio/test-cases/generate")
    assert mocked.await_args_list[26].args[:2] == ("POST", "/api/v1/prompt-studio/test-cases/run")
    assert isinstance(created_project, PromptStudioStandardResponse)
    assert projects.metadata["page"] == 2
    assert project.data["name"] == "Eval Lab"
    assert updated_project.data["name"] == "Eval Lab v2"
    assert deleted_project.data["message"] == "Project soft deleted"
    assert archived_project.data["status"] == "archived"
    assert unarchived_project.data["status"] == "active"
    assert project_stats.data["prompt_count"] == 2
    assert created_prompt.data["name"] == "Summarizer"
    assert prompts.data[0]["name"] == "Summarizer"
    assert prompt.data["id"] == 11
    assert updated_prompt.data["version_number"] == 2
    assert prompt_history.data[1]["version_number"] == 2
    assert reverted_prompt.data["version_number"] == 3
    assert preview.data["assembled_messages"][0]["content"] == "Hi"
    assert converted.data["prompt_format"] == "structured"
    assert execution.output == "ok"
    assert created_case.data["name"] == "Smoke"
    assert cases.data[0]["id"] == 21
    assert case.data["inputs"]["text"] == "Hi"
    assert updated_case.data["name"] == "Smoke v2"
    assert deleted_case.data["message"] == "Test case soft deleted"
    assert bulk_cases.data[1]["id"] == 23
    assert imported_cases.data["imported"] == 2
    assert exported_cases.data["content_type"] == "application/json"
    assert generated_cases.data[0]["is_generated"] is True
    assert run_cases.results[0]["passed"] is True


@pytest.mark.asyncio
async def test_prompt_studio_evaluations_optimizations_and_status_routes(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"id": 31, "project_id": 1, "prompt_id": 11, "status": "running", "metrics": {}, "config": {}, "tags": []},
            {
                "evaluations": [
                    {"id": 31, "project_id": 1, "prompt_id": 11, "status": "running", "metrics": {}, "config": {}, "tags": []}
                ],
                "total": 1,
                "limit": 50,
                "offset": 0,
            },
            {"id": 31, "project_id": 1, "prompt_id": 11, "status": "completed", "metrics": {}, "config": {}, "tags": []},
            {"message": "Evaluation 31 deleted successfully"},
            {"success": True, "data": {"optimization": {"id": 41, "status": "pending"}, "job_id": "job-1"}},
            {"id": "job-2", "status": "pending"},
            {
                "success": True,
                "data": [{"id": 41, "project_id": 1, "status": "pending"}],
                "metadata": {"page": 1, "per_page": 20, "total": 1, "total_pages": 1},
            },
            {"success": True, "data": {"id": 41, "project_id": 1, "status": "running"}},
            {"id": "job-1", "status": "processing", "progress": 0.5},
            {"success": True, "data": {"message": "Optimization cancelled"}},
            {"success": True, "data": [{"name": "iterative"}]},
            {"success": True, "data": {"optimization": {"id": 41}, "progress": {"status": "running"}, "timeline": []}},
            {"success": True, "data": {"id": 51}},
            {"success": True, "data": {"iterations": [{"iteration_number": 1}]}, "metadata": {"page": 1}},
            {"success": True, "data": {"optimization_ids": [41, 42], "job_ids": ["job-1", "job-2"]}},
            {
                "success": True,
                "data": {
                    "queue_depth": 1,
                    "processing": 2,
                    "leases": {"active": 2},
                    "by_status": {"queued": 1},
                    "by_type": {"optimization": 1},
                    "avg_processing_time_seconds": 3.0,
                    "success_rate": 99.0,
                },
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    evaluation = await client.create_prompt_studio_evaluation(
        PromptStudioEvaluationCreate(
            project_id=1,
            prompt_id=11,
            name="Baseline",
            config={"model_name": "gpt-4o-mini"},
            test_case_ids=[21],
            run_async=True,
        )
    )
    evaluations = await client.list_prompt_studio_evaluations(project_id=1, prompt_id=11, limit=50, offset=0)
    fetched_evaluation = await client.get_prompt_studio_evaluation(31)
    deleted_evaluation = await client.delete_prompt_studio_evaluation(31)
    optimization = await client.create_prompt_studio_optimization(
        PromptStudioOptimizationCreate(
            project_id=1,
            initial_prompt_id=11,
            optimization_config=PromptStudioOptimizationConfig(
                optimizer_type="iterative",
                max_iterations=5,
                target_metric="accuracy",
            ),
            test_case_ids=[21],
            name="Refine",
        ),
        idempotency_key="opt-key",
    )
    simple_optimization = await client.create_prompt_studio_optimization_simple(
        PromptStudioOptimizationSimpleCreateRequest(prompt_id=11, strategy="iterative", config={"max_iterations": 2})
    )
    optimizations = await client.list_prompt_studio_optimizations(1, status="pending")
    fetched_optimization = await client.get_prompt_studio_optimization(41)
    job_status = await client.get_prompt_studio_optimization_job_status("job-1")
    cancelled = await client.cancel_prompt_studio_optimization(41, reason="stop")
    strategies = await client.get_prompt_studio_optimization_strategies()
    history = await client.get_prompt_studio_optimization_history(41)
    iteration = await client.add_prompt_studio_optimization_iteration(
        41,
        PromptStudioOptimizationIterationCreate(iteration_number=1, metrics={"accuracy": 0.8}),
    )
    iterations = await client.list_prompt_studio_optimization_iterations(41, page=1, per_page=25)
    comparison = await client.compare_prompt_studio_optimization_strategies(
        PromptStudioCompareStrategiesRequest(
            prompt_id=11,
            test_case_ids=[21],
            strategies=["iterative", "random_search"],
            model_configuration={"model": "gpt-4o-mini"},
        )
    )
    status = await client.get_prompt_studio_status(warn_seconds=60)

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/prompt-studio/evaluations")
    assert mocked.await_args_list[0].kwargs["json_data"]["run_async"] is True
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/prompt-studio/evaluations")
    assert mocked.await_args_list[1].kwargs["params"] == {"project_id": 1, "prompt_id": 11, "limit": 50, "offset": 0}
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/prompt-studio/optimizations/create")
    assert mocked.await_args_list[4].kwargs["headers"] == {"Idempotency-Key": "opt-key"}
    assert mocked.await_args_list[5].args[:2] == ("POST", "/api/v1/prompt-studio/optimizations")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/prompt-studio/optimizations/list/1")
    assert mocked.await_args_list[9].args[:2] == ("POST", "/api/v1/prompt-studio/optimizations/cancel/41")
    assert mocked.await_args_list[9].kwargs["json_data"] == "stop"
    assert mocked.await_args_list[13].args[:2] == ("GET", "/api/v1/prompt-studio/optimizations/iterations/41")
    assert mocked.await_args_list[14].args[:2] == ("POST", "/api/v1/prompt-studio/optimizations/compare")
    assert mocked.await_args_list[15].args[:2] == ("GET", "/api/v1/prompt-studio/status")
    assert isinstance(evaluation, PromptStudioEvaluationResponse)
    assert isinstance(evaluations, PromptStudioEvaluationListResponse)
    assert fetched_evaluation.status == "completed"
    assert deleted_evaluation.message.endswith("successfully")
    assert optimization.data["job_id"] == "job-1"
    assert simple_optimization.id == "job-2"
    assert optimizations.data[0]["status"] == "pending"
    assert fetched_optimization.data["status"] == "running"
    assert job_status["progress"] == 0.5
    assert cancelled.data["message"] == "Optimization cancelled"
    assert strategies.data[0]["name"] == "iterative"
    assert history.data["progress"]["status"] == "running"
    assert iteration.data["id"] == 51
    assert iterations.data["iterations"][0]["iteration_number"] == 1
    assert comparison.data["optimization_ids"] == [41, 42]
    assert isinstance(status, PromptStudioStatusResponse)
    assert status.data["queue_depth"] == 1
