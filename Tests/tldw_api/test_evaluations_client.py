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
    SyntheticEvalGenerationRequest,
    SyntheticEvalGenerationResponse,
    SyntheticEvalPromotionRequest,
    SyntheticEvalPromotionResponse,
    SyntheticEvalQueueResponse,
    SyntheticEvalReviewActionRecord,
    SyntheticEvalReviewRequest,
    TLDWAPIClient,
    UpdateEvaluationRequest,
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
async def test_synthetic_evaluation_routes_wire_and_return_typed_models(monkeypatch):
    client = TLDWAPIClient("http://localhost:8000")
    mocked = AsyncMock(
        side_effect=[
            {
                "generation_batch_id": "batch_123",
                "samples": [
                    {
                        "sample_id": "sample_123",
                        "recipe_kind": "rag_answer_quality",
                        "provenance": "synthetic_from_seed_examples",
                        "review_state": "draft",
                        "sample_payload": {"question": "What changed?"},
                        "sample_metadata": {"source": "seed"},
                        "source_kind": "seed",
                        "created_by": "u1",
                    }
                ],
                "source_breakdown": {"seed": 1},
                "coverage": {"topics": ["sync"]},
                "missing_coverage": {},
                "corpus_scope": {"workspace_id": "ws_1"},
            },
            {
                "data": [
                    {
                        "sample_id": "sample_123",
                        "recipe_kind": "rag_answer_quality",
                        "provenance": "synthetic_from_seed_examples",
                        "review_state": "in_review",
                        "sample_payload": {"question": "What changed?"},
                        "sample_metadata": {"source": "seed"},
                    }
                ],
                "total": 1,
            },
            {
                "action_id": "action_123",
                "sample_id": "sample_123",
                "action": "approve",
                "reviewer_id": "u1",
                "notes": "Looks usable",
                "action_payload": {},
                "resulting_review_state": "approved",
            },
            {
                "dataset_id": "dataset_123",
                "dataset_snapshot_ref": "snapshot_123",
                "promotion_ids": ["promotion_123"],
                "sample_count": 1,
            },
        ]
    )
    monkeypatch.setattr(client, "_request", mocked)

    generated = await client.generate_synthetic_evaluation_drafts(
        SyntheticEvalGenerationRequest(
            recipe_kind="rag_answer_quality",
            corpus_scope={"workspace_id": "ws_1"},
            seed_examples=[{"question": "What changed?", "answer": "Sync support"}],
            target_sample_count=1,
        )
    )
    queue = await client.list_synthetic_evaluation_queue(
        recipe_kind="rag_answer_quality",
        review_state="in_review",
        source_kind="seed",
        generation_batch_id="batch_123",
        limit=25,
        offset=5,
    )
    review = await client.review_synthetic_evaluation_sample(
        "sample_123",
        SyntheticEvalReviewRequest(
            action="approve",
            reviewer_id="u1",
            notes="Looks usable",
            resulting_review_state="approved",
        ),
    )
    promoted = await client.promote_synthetic_evaluation_samples(
        SyntheticEvalPromotionRequest(
            sample_ids=["sample_123"],
            dataset_name="Approved RAG samples",
            dataset_metadata={"project": "parity"},
            promotion_reason="manual_review",
        )
    )

    assert mocked.await_args_list[0].args[:2] == (
        "POST",
        "/api/v1/evaluations/synthetic/drafts/generate",
    )
    assert mocked.await_args_list[1].args[:2] == ("GET", "/api/v1/evaluations/synthetic/queue")
    assert mocked.await_args_list[1].kwargs["params"] == {
        "recipe_kind": "rag_answer_quality",
        "review_state": "in_review",
        "source_kind": "seed",
        "generation_batch_id": "batch_123",
        "limit": 25,
        "offset": 5,
    }
    assert mocked.await_args_list[2].args[:2] == (
        "POST",
        "/api/v1/evaluations/synthetic/queue/sample_123/review",
    )
    assert mocked.await_args_list[3].args[:2] == (
        "POST",
        "/api/v1/evaluations/synthetic/promotions",
    )

    assert isinstance(generated, SyntheticEvalGenerationResponse)
    assert isinstance(queue, SyntheticEvalQueueResponse)
    assert isinstance(review, SyntheticEvalReviewActionRecord)
    assert isinstance(promoted, SyntheticEvalPromotionResponse)
    assert generated.samples[0].sample_id == "sample_123"
    assert queue.total == 1
    assert review.resulting_review_state == "approved"
    assert promoted.dataset_snapshot_ref == "snapshot_123"
