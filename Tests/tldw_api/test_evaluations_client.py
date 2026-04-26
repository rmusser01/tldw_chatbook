from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.tldw_api import (
    CreateEvaluationRequest,
    EvaluationDatasetCreateRequest,
    EvaluationDatasetListResponse,
    EvaluationDatasetResponse,
    EvaluationListResponse,
    EvaluationResponse,
    EvaluationRunCreateRequest,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    EvaluationSpec,
    RecipeDatasetValidationRequest,
    RecipeDatasetValidationResponse,
    RecipeLaunchReadiness,
    RecipeManifest,
    RecipeRunCreateRequest,
    RecipeRunRecord,
    SyntheticEvalGenerationRequest,
    SyntheticEvalGenerationResponse,
    SyntheticEvalPromotionRequest,
    SyntheticEvalPromotionResponse,
    SyntheticEvalQueueResponse,
    SyntheticEvalReviewRequest,
    SyntheticEvalReviewActionRecord,
    TLDWAPIClient,
    UpdateEvaluationRequest,
    WebhookRegistrationRequest,
    WebhookRegistrationResponse,
    WebhookStatusResponse,
    WebhookTestRequest,
    WebhookTestResponse,
)


@pytest.mark.asyncio
async def test_evaluation_dataset_crud_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": "dataset_123",
                "object": "dataset",
                "name": "demo_dataset",
                "description": "Demo dataset",
                "sample_count": 2,
                "samples": [
                    {"input": "What is 2+2?", "expected": "4", "metadata": {"difficulty": "easy"}},
                    {"input": "What is 3+3?", "expected": "6", "metadata": {}},
                ],
                "created": 1713571200,
                "created_at": 1713571200,
                "created_by": "u1",
                "metadata": {"source": "local"},
            },
            {
                "object": "list",
                "data": [
                    {
                        "id": "dataset_123",
                        "object": "dataset",
                        "name": "demo_dataset",
                        "description": "Demo dataset",
                        "sample_count": 2,
                        "created": 1713571200,
                        "created_at": 1713571200,
                        "created_by": "u1",
                        "metadata": {"source": "local"},
                    }
                ],
                "has_more": False,
                "first_id": "dataset_123",
                "last_id": "dataset_123",
            },
            {
                "id": "dataset_123",
                "object": "dataset",
                "name": "demo_dataset",
                "description": "Demo dataset",
                "sample_count": 2,
                "samples": [
                    {"input": "What is 2+2?", "expected": "4", "metadata": {"difficulty": "easy"}},
                    {"input": "What is 3+3?", "expected": "6", "metadata": {}},
                ],
                "created": 1713571200,
                "created_at": 1713571200,
                "created_by": "u1",
                "metadata": {"source": "local"},
            },
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_evaluation_dataset(
        EvaluationDatasetCreateRequest(
            name="demo_dataset",
            description="Demo dataset",
            samples=[
                {"input": "What is 2+2?", "expected": "4", "metadata": {"difficulty": "easy"}},
                {"input": "What is 3+3?", "expected": "6"},
            ],
            metadata={"source": "local"},
        )
    )
    listed = await client.list_evaluation_datasets(limit=10, offset=5)
    fetched = await client.get_evaluation_dataset(
        "dataset_123",
        include_samples=False,
        limit=25,
        offset=3,
    )
    await client.delete_evaluation_dataset("dataset_123")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations/datasets")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/datasets")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/datasets/dataset_123")
    assert mocked.await_args_list[2].kwargs["params"] == {
        "include_samples": False,
        "limit": 25,
        "offset": 3,
    }
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/evaluations/datasets/dataset_123")

    assert isinstance(created, EvaluationDatasetResponse)
    assert isinstance(listed, EvaluationDatasetListResponse)
    assert isinstance(fetched, EvaluationDatasetResponse)
    assert fetched.samples[0].expected == "4"


@pytest.mark.asyncio
async def test_evaluation_crud_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": "eval_123",
                "object": "evaluation",
                "name": "demo_eval",
                "description": "Demo evaluation",
                "eval_type": "exact_match",
                "eval_spec": {"metrics": ["accuracy"], "threshold": 0.8},
                "dataset_id": "dataset_123",
                "created": 1713571200,
                "created_at": 1713571200,
                "created_by": "u1",
                "updated": 1713571300,
                "updated_at": 1713571300,
                "metadata": {"project": "demo"},
            },
            {
                "object": "list",
                "data": [
                    {
                        "id": "eval_123",
                        "object": "evaluation",
                        "name": "demo_eval",
                        "description": "Demo evaluation",
                        "eval_type": "exact_match",
                        "eval_spec": {"metrics": ["accuracy"], "threshold": 0.8},
                        "dataset_id": "dataset_123",
                        "created": 1713571200,
                        "created_at": 1713571200,
                        "created_by": "u1",
                        "updated": 1713571300,
                        "updated_at": 1713571300,
                        "metadata": {"project": "demo"},
                    }
                ],
                "has_more": False,
                "first_id": "eval_123",
                "last_id": "eval_123",
            },
            {
                "id": "eval_123",
                "object": "evaluation",
                "name": "demo_eval",
                "description": "Demo evaluation",
                "eval_type": "exact_match",
                "eval_spec": {"metrics": ["accuracy"], "threshold": 0.8},
                "dataset_id": "dataset_123",
                "created": 1713571200,
                "created_at": 1713571200,
                "created_by": "u1",
                "updated": 1713571300,
                "updated_at": 1713571300,
                "metadata": {"project": "demo"},
            },
            {
                "id": "eval_123",
                "object": "evaluation",
                "name": "demo_eval",
                "description": "Updated evaluation",
                "eval_type": "exact_match",
                "eval_spec": {"metrics": ["accuracy"], "threshold": 0.9},
                "dataset_id": "dataset_123",
                "created": 1713571200,
                "created_at": 1713571200,
                "created_by": "u1",
                "updated": 1713571400,
                "updated_at": 1713571400,
                "metadata": {"project": "demo", "phase": "updated"},
            },
            {},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_evaluation(
        CreateEvaluationRequest(
            name="demo_eval",
            description="Demo evaluation",
            eval_type="exact_match",
            eval_spec=EvaluationSpec(metrics=["accuracy"], threshold=0.8),
            dataset_id="dataset_123",
            metadata={"project": "demo"},
        )
    )
    listed = await client.list_evaluations(limit=15, after="eval_100", eval_type="exact_match")
    fetched = await client.get_evaluation("eval_123")
    updated = await client.update_evaluation(
        "eval_123",
        UpdateEvaluationRequest(
            description="Updated evaluation",
            eval_spec=EvaluationSpec(metrics=["accuracy"], threshold=0.9),
            metadata={"project": "demo", "phase": "updated"},
        ),
    )
    await client.delete_evaluation("eval_123")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "limit": 15,
        "after": "eval_100",
        "eval_type": "exact_match",
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/eval_123")
    assert mocked.await_args_list[3].args[:2] == ("PATCH", "/api/v1/evaluations/eval_123")
    assert mocked.await_args_list[4].args[:2] == ("DELETE", "/api/v1/evaluations/eval_123")

    assert isinstance(created, EvaluationResponse)
    assert isinstance(listed, EvaluationListResponse)
    assert isinstance(fetched, EvaluationResponse)
    assert isinstance(updated, EvaluationResponse)
    assert updated.metadata["phase"] == "updated"


@pytest.mark.asyncio
async def test_evaluation_run_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "id": "run_123",
                "object": "run",
                "eval_id": "eval_123",
                "status": "pending",
                "target_model": "gpt-4.1-mini",
                "created": 1713571200,
                "created_at": 1713571200,
                "started_at": None,
                "completed_at": None,
                "progress": {"completed_samples": 0, "total_samples": 20, "percent_complete": 0.0},
                "error_message": None,
                "results": None,
                "usage": None,
            },
            {
                "object": "list",
                "data": [
                    {
                        "id": "run_123",
                        "object": "run",
                        "eval_id": "eval_123",
                        "status": "running",
                        "target_model": "gpt-4.1-mini",
                        "created": 1713571200,
                        "created_at": 1713571200,
                        "started_at": 1713571210,
                        "completed_at": None,
                        "progress": {"completed_samples": 5, "total_samples": 20, "percent_complete": 25.0},
                        "error_message": None,
                        "results": None,
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    }
                ],
                "has_more": False,
                "first_id": "run_123",
                "last_id": "run_123",
            },
            {
                "id": "run_123",
                "object": "run",
                "eval_id": "eval_123",
                "status": "completed",
                "target_model": "gpt-4.1-mini",
                "created": 1713571200,
                "created_at": 1713571200,
                "started_at": 1713571210,
                "completed_at": 1713571260,
                "progress": {"completed_samples": 20, "total_samples": 20, "percent_complete": 100.0},
                "error_message": None,
                "results": {"accuracy": 0.9},
                "usage": {"input_tokens": 300, "output_tokens": 70},
            },
            {"status": "cancelled", "run_id": "run_123"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_evaluation_run(
        "eval_123",
        EvaluationRunCreateRequest(
            target_model="gpt-4.1-mini",
            config={"temperature": 0.2, "max_workers": 2},
        ),
    )
    listed = await client.list_evaluation_runs("eval_123", limit=20, after="run_100", status="running")
    fetched = await client.get_evaluation_run("run_123")
    cancelled = await client.cancel_evaluation_run("run_123")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations/eval_123/runs")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/eval_123/runs")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "limit": 20,
        "after": "run_100",
        "status": "running",
    }
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/runs/run_123")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/evaluations/runs/run_123/cancel")

    assert isinstance(created, EvaluationRunResponse)
    assert isinstance(listed, EvaluationRunListResponse)
    assert isinstance(fetched, EvaluationRunResponse)
    assert cancelled == {"status": "cancelled", "run_id": "run_123"}
    assert fetched.progress.percent_complete == 100.0


@pytest.mark.asyncio
async def test_evaluation_rag_pipeline_preset_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"name": "fast", "config": {"retriever": "hybrid"}, "created_at": 1713571200},
            {"items": [{"name": "fast", "config": {"retriever": "hybrid"}}], "total": 1},
            {"name": "fast", "config": {"retriever": "hybrid"}, "updated_at": 1713571260},
            {},
            {"expired_count": 3, "deleted_count": 2, "errors": ["stale: missing"]},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_or_update_evaluation_rag_pipeline_preset(
        name="fast",
        config={"retriever": "hybrid"},
    )
    listed = await client.list_evaluation_rag_pipeline_presets(limit=10, offset=5)
    fetched = await client.get_evaluation_rag_pipeline_preset("fast")
    await client.delete_evaluation_rag_pipeline_preset("fast")
    cleanup = await client.cleanup_evaluation_rag_pipeline()

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations/rag/pipeline/presets")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "fast",
        "config": {"retriever": "hybrid"},
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/rag/pipeline/presets")
    assert mocked.await_args_list[1].kwargs["params"] == {"limit": 10, "offset": 5}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/rag/pipeline/presets/fast")
    assert mocked.await_args_list[3].args[:2] == ("DELETE", "/api/v1/evaluations/rag/pipeline/presets/fast")
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/evaluations/rag/pipeline/cleanup")
    assert created["name"] == "fast"
    assert listed["total"] == 1
    assert fetched["updated_at"] == 1713571260
    assert cleanup["deleted_count"] == 2


@pytest.mark.asyncio
async def test_evaluation_embeddings_abtest_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    config = {
        "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
        "retrieval": {"k": 10},
        "queries": [{"text": "What is RAG?"}],
    }
    mocked = AsyncMock(
        side_effect=[
            {"test_id": "ab_123", "status": "created"},
            {"test_id": "ab_123", "status": "running", "progress": {"phase": 0.1}},
            {"test_id": "ab_123", "status": "completed", "arms": []},
            {"summary": {"test_id": "ab_123", "status": "completed", "arms": []}, "results": [], "total": 0},
            {"metric": "ndcg", "p_value": 0.05},
            {"test_id": "ab_123", "total": 0, "results": []},
            {"status": "deleted", "test_id": "ab_123"},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    created = await client.create_evaluation_embeddings_abtest(
        name="embed-test",
        config=config,
        run_immediately=True,
    )
    launched = await client.run_evaluation_embeddings_abtest("ab_123", config=config)
    status = await client.get_evaluation_embeddings_abtest_status("ab_123")
    results = await client.get_evaluation_embeddings_abtest_results("ab_123", page=2, page_size=25)
    significance = await client.get_evaluation_embeddings_abtest_significance("ab_123", metric="mrr")
    exported = await client.export_evaluation_embeddings_abtest("ab_123", format="json")
    deleted = await client.delete_evaluation_embeddings_abtest("ab_123")

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations/embeddings/abtest")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "name": "embed-test",
        "config": config,
        "run_immediately": True,
    }
    assert mocked.await_args_list[1].args[:2] == ("POST", "/api/v1/evaluations/embeddings/abtest/ab_123/run")
    assert mocked.await_args_list[1].kwargs["json_data"] == {"config": config}
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/embeddings/abtest/ab_123")
    assert mocked.await_args_list[3].args[:2] == ("GET", "/api/v1/evaluations/embeddings/abtest/ab_123/results")
    assert mocked.await_args_list[3].kwargs["params"] == {"page": 2, "page_size": 25}
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/evaluations/embeddings/abtest/ab_123/significance")
    assert mocked.await_args_list[4].kwargs["params"] == {"metric": "mrr"}
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/evaluations/embeddings/abtest/ab_123/export")
    assert mocked.await_args_list[5].kwargs["params"] == {"format": "json"}
    assert mocked.await_args_list[6].args[:2] == ("DELETE", "/api/v1/evaluations/embeddings/abtest/ab_123")
    assert created["test_id"] == "ab_123"
    assert launched["progress"]["phase"] == 0.1
    assert status["status"] == "completed"
    assert results["total"] == 0
    assert significance["metric"] == "ndcg"
    assert exported["test_id"] == "ab_123"
    assert deleted["status"] == "deleted"


@pytest.mark.asyncio
async def test_evaluation_synthetic_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    sample = {
        "sample_id": "sample_1",
        "recipe_kind": "rag_answer_quality",
        "provenance": "synthetic_from_corpus",
        "review_state": "draft",
        "sample_payload": {"input": "What is RAG?", "expected": "Retrieval augmented generation."},
        "sample_metadata": {"topic": "rag"},
        "source_kind": "corpus",
        "created_by": "user_1",
        "created_at": "2026-04-21T00:00:00Z",
        "updated_at": "2026-04-21T00:01:00Z",
    }
    mocked = AsyncMock(
        side_effect=[
            {
                "generation_batch_id": "batch_1",
                "samples": [sample],
                "source_breakdown": {"corpus": 1},
                "coverage": {"topic": ["rag"]},
                "missing_coverage": {},
                "corpus_scope": {"collection": "docs"},
            },
            {"data": [sample], "total": 1},
            {
                "action_id": "review_1",
                "sample_id": "sample_1",
                "action": "approve",
                "reviewer_id": "user_1",
                "notes": "Looks usable",
                "action_payload": {},
                "resulting_review_state": "approved",
                "created_at": "2026-04-21T00:02:00Z",
            },
            {
                "dataset_id": "dataset_1",
                "dataset_snapshot_ref": "snapshot_1",
                "promotion_ids": ["promotion_1"],
                "sample_count": 1,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    generated = await client.generate_synthetic_evaluation_drafts(
        SyntheticEvalGenerationRequest(
            recipe_kind="rag_answer_quality",
            corpus_scope={"collection": "docs"},
            target_sample_count=1,
        )
    )
    queue = await client.list_synthetic_evaluation_queue(
        recipe_kind="rag_answer_quality",
        review_state="draft",
        source_kind="corpus",
        generation_batch_id="batch_1",
        limit=25,
        offset=5,
    )
    reviewed = await client.review_synthetic_evaluation_sample(
        "sample_1",
        SyntheticEvalReviewRequest(
            action="approve",
            notes="Looks usable",
            resulting_review_state="approved",
        ),
    )
    promoted = await client.promote_synthetic_evaluation_samples(
        SyntheticEvalPromotionRequest(
            sample_ids=["sample_1"],
            dataset_name="approved_synthetic",
            dataset_description="Approved synthetic samples",
            promotion_reason="seed dataset",
        )
    )

    assert mocked.await_args_list[0].args[:2] == ("POST", "/api/v1/evaluations/synthetic/drafts/generate")
    assert mocked.await_args_list[0].kwargs["json_data"] == {
        "recipe_kind": "rag_answer_quality",
        "corpus_scope": {"collection": "docs"},
        "generation_metadata": {},
        "real_examples": [],
        "seed_examples": [],
        "target_sample_count": 1,
    }
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/synthetic/queue")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "recipe_kind": "rag_answer_quality",
        "review_state": "draft",
        "source_kind": "corpus",
        "generation_batch_id": "batch_1",
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/evaluations/synthetic/queue/sample_1/review")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/evaluations/synthetic/promotions")

    assert isinstance(generated, SyntheticEvalGenerationResponse)
    assert isinstance(queue, SyntheticEvalQueueResponse)
    assert isinstance(reviewed, SyntheticEvalReviewActionRecord)
    assert isinstance(promoted, SyntheticEvalPromotionResponse)
    assert generated.samples[0].sample_id == "sample_1"
    assert queue.total == 1
    assert reviewed.resulting_review_state == "approved"
    assert promoted.dataset_snapshot_ref == "snapshot_1"


@pytest.mark.asyncio
async def test_evaluation_benchmark_and_webhook_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {"object": "list", "data": [{"name": "mmlu", "evaluation_type": "qa"}], "total": 1},
            {"name": "mmlu", "evaluation_type": "qa", "description": "Benchmark"},
            {
                "benchmark": "mmlu",
                "total_samples": 2,
                "results_summary": {"average_score": 0.75},
                "evaluation_id": "eval_1",
            },
            {
                "webhook_id": 10,
                "url": "https://example.com/evals",
                "events": ["evaluation.completed"],
                "secret": "x" * 32,
                "created_at": "2026-04-21T00:00:00Z",
                "status": "active",
                "retry_count": 2,
                "timeout_seconds": 10,
            },
            [
                {
                    "webhook_id": 10,
                    "url": "https://example.com/evals",
                    "events": ["evaluation.completed"],
                    "status": "active",
                    "retry_count": 2,
                    "timeout_seconds": 10,
                    "created_at": "2026-04-21T00:00:00Z",
                    "failure_count": 0,
                }
            ],
            {"status": "unregistered", "url": "https://example.com/evals"},
            {"success": True, "status_code": 204, "response_time_ms": 12.5},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    benchmarks = await client.list_evaluation_benchmarks()
    benchmark = await client.get_evaluation_benchmark("mmlu")
    run = await client.run_evaluation_benchmark(
        "mmlu",
        limit=2,
        api_name="openai",
        parallel=2,
        save_results=True,
        filter_categories=["math"],
    )
    registered = await client.register_evaluation_webhook(
        WebhookRegistrationRequest(
            url="https://example.com/evals",
            events=["evaluation.completed"],
            secret="x" * 32,
            retry_count=2,
            timeout_seconds=10,
        )
    )
    webhooks = await client.list_evaluation_webhooks()
    unregistered = await client.unregister_evaluation_webhook("https://example.com/evals")
    tested = await client.test_evaluation_webhook(WebhookTestRequest(url="https://example.com/evals"))

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/evaluations/benchmarks")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/benchmarks/mmlu")
    assert mocked.await_args_list[2].args[:2] == ("POST", "/api/v1/evaluations/benchmarks/mmlu/run")
    assert mocked.await_args_list[2].kwargs["json_data"] == {
        "limit": 2,
        "api_name": "openai",
        "parallel": 2,
        "save_results": True,
        "filter_categories": ["math"],
    }
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/evaluations/webhooks")
    assert mocked.await_args_list[4].args[:2] == ("GET", "/api/v1/evaluations/webhooks")
    assert mocked.await_args_list[5].args[:2] == ("DELETE", "/api/v1/evaluations/webhooks")
    assert mocked.await_args_list[5].kwargs["params"] == {"url": "https://example.com/evals"}
    assert mocked.await_args_list[6].args[:2] == ("POST", "/api/v1/evaluations/webhooks/test")

    assert benchmarks["total"] == 1
    assert benchmark["name"] == "mmlu"
    assert run["evaluation_id"] == "eval_1"
    assert isinstance(registered, WebhookRegistrationResponse)
    assert isinstance(webhooks[0], WebhookStatusResponse)
    assert unregistered["status"] == "unregistered"
    assert isinstance(tested, WebhookTestResponse)
    assert tested.success is True


@pytest.mark.asyncio
async def test_evaluation_recipe_routes_wire(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    run_record = {
        "run_id": "recipe_run_1",
        "recipe_id": "rag_answer_quality",
        "recipe_version": "1.0.0",
        "status": "pending",
        "review_state": "not_required",
        "created_at": "2026-04-22T00:00:00Z",
        "updated_at": "2026-04-22T00:00:00Z",
        "metadata": {},
    }
    mocked = AsyncMock(
        side_effect=[
            [
                {
                    "recipe_id": "rag_answer_quality",
                    "recipe_version": "1.0.0",
                    "name": "RAG Answer Quality",
                    "description": "Evaluate RAG answers.",
                    "launchable": True,
                    "supported_modes": ["labeled"],
                    "tags": ["rag"],
                    "capabilities": {},
                    "default_run_config": {},
                }
            ],
            {
                "recipe_id": "rag_answer_quality",
                "recipe_version": "1.0.0",
                "name": "RAG Answer Quality",
                "description": "Evaluate RAG answers.",
                "launchable": True,
                "supported_modes": ["labeled"],
                "tags": [],
                "capabilities": {},
                "default_run_config": {},
            },
            {
                "recipe_id": "rag_answer_quality",
                "ready": True,
                "can_enqueue_runs": True,
                "can_reuse_completed_runs": True,
                "runtime_checks": {"recipe_launchable": True},
            },
            {
                "valid": True,
                "errors": [],
                "dataset_mode": "labeled",
                "sample_count": 2,
                "dataset_snapshot_ref": "snapshot_1",
                "dataset_content_hash": "hash_1",
            },
            run_record,
            {**run_record, "status": "completed", "review_state": "approved"},
            {"run": run_record, "summary": {"score": 0.91}},
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    manifests = await client.list_evaluation_recipe_manifests()
    manifest = await client.get_evaluation_recipe_manifest("rag_answer_quality")
    readiness = await client.get_evaluation_recipe_launch_readiness("rag_answer_quality")
    validation = await client.validate_evaluation_recipe_dataset(
        "rag_answer_quality",
        RecipeDatasetValidationRequest(dataset_id="dataset_1", run_config={"mode": "fast"}),
    )
    created = await client.create_evaluation_recipe_run(
        "rag_answer_quality",
        RecipeRunCreateRequest(dataset_id="dataset_1", run_config={"mode": "fast"}, force_rerun=True),
    )
    fetched = await client.get_evaluation_recipe_run("recipe_run_1")
    report = await client.get_evaluation_recipe_run_report("recipe_run_1")

    assert mocked.await_args_list[0].args[:2] == ("GET", "/api/v1/evaluations/recipes")
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/recipes/rag_answer_quality")
    assert mocked.await_args_list[2].args[:2] == ("GET", "/api/v1/evaluations/recipes/rag_answer_quality/launch-readiness")
    assert mocked.await_args_list[3].args[:2] == ("POST", "/api/v1/evaluations/recipes/rag_answer_quality/validate-dataset")
    assert mocked.await_args_list[3].kwargs["json_data"] == {
        "dataset_id": "dataset_1",
        "run_config": {"mode": "fast"},
    }
    assert mocked.await_args_list[4].args[:2] == ("POST", "/api/v1/evaluations/recipes/rag_answer_quality/runs")
    assert mocked.await_args_list[4].kwargs["json_data"] == {
        "dataset_id": "dataset_1",
        "run_config": {"mode": "fast"},
        "force_rerun": True,
    }
    assert mocked.await_args_list[5].args[:2] == ("GET", "/api/v1/evaluations/recipe-runs/recipe_run_1")
    assert mocked.await_args_list[6].args[:2] == ("GET", "/api/v1/evaluations/recipe-runs/recipe_run_1/report")

    assert isinstance(manifests[0], RecipeManifest)
    assert isinstance(manifest, RecipeManifest)
    assert isinstance(readiness, RecipeLaunchReadiness)
    assert isinstance(validation, RecipeDatasetValidationResponse)
    assert isinstance(created, RecipeRunRecord)
    assert isinstance(fetched, RecipeRunRecord)
    assert manifests[0].recipe_id == "rag_answer_quality"
    assert manifest.name == "RAG Answer Quality"
    assert readiness.ready is True
    assert validation.valid is True
    assert created.run_id == "recipe_run_1"
    assert fetched.status == "completed"
    assert report["summary"]["score"] == 0.91
