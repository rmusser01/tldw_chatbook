import pytest

from tldw_chatbook.Evaluations_Interop.server_evaluations_service import ServerEvaluationsService


class FakeEvaluationClient:
    def __init__(self):
        self.calls = []

    async def run_evaluation_benchmark(self, benchmark_name, request_data):
        self.calls.append(("run_evaluation_benchmark", benchmark_name, request_data))
        return {
            "benchmark": benchmark_name,
            "total_samples": 2,
            "results_summary": {"average_score": 0.75},
            "evaluation_id": "eval_bench_1",
        }

    async def create_evaluation_recipe_run(self, recipe_id, request_data):
        self.calls.append(("create_evaluation_recipe_run", recipe_id, request_data))
        return {
            "run_id": "recipe_run_1",
            "recipe_id": recipe_id,
            "recipe_version": "1.0.0",
            "status": "pending",
            "review_state": "not_required",
            "child_run_ids": [],
            "created_at": "2026-04-23T12:00:00Z",
            "metadata": {"job_id": "job-1"},
        }

    async def get_evaluation_recipe_run(self, run_id):
        self.calls.append(("get_evaluation_recipe_run", run_id))
        return {
            "run_id": run_id,
            "recipe_id": "rag_answer_quality",
            "recipe_version": "1.0.0",
            "status": "running",
            "review_state": "not_required",
            "child_run_ids": [],
            "created_at": "2026-04-23T12:00:00Z",
        }

    async def get_evaluation_recipe_run_report(self, run_id):
        self.calls.append(("get_evaluation_recipe_run_report", run_id))
        return {
            "run": {
                "run_id": run_id,
                "recipe_id": "rag_answer_quality",
                "recipe_version": "1.0.0",
                "status": "completed",
                "review_state": "approved",
                "child_run_ids": [],
                "created_at": "2026-04-23T12:00:00Z",
            },
            "confidence_summary": {
                "kind": "aggregate",
                "confidence": 0.82,
                "sample_count": 2,
            },
            "recommendation_slots": {},
        }


@pytest.mark.asyncio
async def test_server_evaluations_service_wraps_benchmark_and_recipe_run_routes():
    client = FakeEvaluationClient()
    service = ServerEvaluationsService(client=client)

    benchmark_run = await service.run_benchmark(
        "truthfulqa",
        limit=2,
        parallel=2,
        filter_categories=["truthful"],
    )
    created_recipe_run = await service.create_recipe_run(
        "rag_answer_quality",
        dataset_id="dataset_1",
        run_config={"evaluation_mode": "fixed_context"},
        force_rerun=True,
    )
    fetched_recipe_run = await service.get_recipe_run("recipe_run_1")
    report = await service.get_recipe_run_report("recipe_run_1")

    assert benchmark_run["evaluation_id"] == "eval_bench_1"
    assert created_recipe_run["metadata"] == {"job_id": "job-1"}
    assert fetched_recipe_run["status"] == "running"
    assert report["confidence_summary"]["confidence"] == 0.82

    benchmark_request = client.calls[0][2]
    recipe_request = client.calls[1][2]
    assert client.calls[0][:2] == ("run_evaluation_benchmark", "truthfulqa")
    assert benchmark_request.model_dump(exclude_none=True, mode="json") == {
        "limit": 2,
        "api_name": "openai",
        "parallel": 2,
        "save_results": True,
        "filter_categories": ["truthful"],
    }
    assert client.calls[1][:2] == ("create_evaluation_recipe_run", "rag_answer_quality")
    assert recipe_request.model_dump(exclude_none=True, mode="json") == {
        "dataset_id": "dataset_1",
        "run_config": {"evaluation_mode": "fixed_context"},
        "force_rerun": True,
    }
    assert client.calls[2] == ("get_evaluation_recipe_run", "recipe_run_1")
    assert client.calls[3] == ("get_evaluation_recipe_run_report", "recipe_run_1")
