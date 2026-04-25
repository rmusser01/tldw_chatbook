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

    async def create_or_update_evaluation_rag_pipeline_preset(self, *, name, config):
        self.calls.append(("create_or_update_evaluation_rag_pipeline_preset", name, config))
        return {"name": name, "config": config}

    async def list_evaluation_rag_pipeline_presets(self, **kwargs):
        self.calls.append(("list_evaluation_rag_pipeline_presets", kwargs))
        return {"items": [], "total": 0}

    async def get_evaluation_rag_pipeline_preset(self, name):
        self.calls.append(("get_evaluation_rag_pipeline_preset", name))
        return {"name": name, "config": {"retriever": "hybrid"}}

    async def delete_evaluation_rag_pipeline_preset(self, name):
        self.calls.append(("delete_evaluation_rag_pipeline_preset", name))
        return None

    async def cleanup_evaluation_rag_pipeline(self):
        self.calls.append(("cleanup_evaluation_rag_pipeline",))
        return {"expired_count": 0, "deleted_count": 0}

    async def create_evaluation_embeddings_abtest(self, *, name, config, run_immediately=False):
        self.calls.append(("create_evaluation_embeddings_abtest", name, config, run_immediately))
        return {"test_id": "ab_1", "status": "created"}

    async def run_evaluation_embeddings_abtest(self, test_id, *, config):
        self.calls.append(("run_evaluation_embeddings_abtest", test_id, config))
        return {"test_id": test_id, "status": "running"}

    async def get_evaluation_embeddings_abtest_status(self, test_id):
        self.calls.append(("get_evaluation_embeddings_abtest_status", test_id))
        return {"test_id": test_id, "status": "completed", "arms": []}

    async def get_evaluation_embeddings_abtest_results(self, test_id, **kwargs):
        self.calls.append(("get_evaluation_embeddings_abtest_results", test_id, kwargs))
        return {"summary": {"test_id": test_id, "status": "completed", "arms": []}, "results": []}

    async def get_evaluation_embeddings_abtest_significance(self, test_id, **kwargs):
        self.calls.append(("get_evaluation_embeddings_abtest_significance", test_id, kwargs))
        return {"metric": kwargs.get("metric"), "p_value": 0.05}

    async def export_evaluation_embeddings_abtest(self, test_id, **kwargs):
        self.calls.append(("export_evaluation_embeddings_abtest", test_id, kwargs))
        return {"test_id": test_id, "total": 0, "results": []}

    async def delete_evaluation_embeddings_abtest(self, test_id):
        self.calls.append(("delete_evaluation_embeddings_abtest", test_id))
        return {"status": "deleted", "test_id": test_id}

    async def generate_synthetic_evaluation_drafts(self, request_data):
        self.calls.append(("generate_synthetic_evaluation_drafts", request_data.model_dump(exclude_none=True, mode="json")))
        return {"generation_batch_id": "batch_1", "samples": [], "source_breakdown": {}, "coverage": {}, "missing_coverage": {}, "corpus_scope": {}}

    async def list_synthetic_evaluation_queue(self, **kwargs):
        self.calls.append(("list_synthetic_evaluation_queue", kwargs))
        return {"data": [{"sample_id": "sample_1"}], "total": 1}

    async def review_synthetic_evaluation_sample(self, sample_id, request_data):
        self.calls.append(("review_synthetic_evaluation_sample", sample_id, request_data.model_dump(exclude_none=True, mode="json")))
        return {"action_id": "review_1", "sample_id": sample_id, "action": "approve"}

    async def promote_synthetic_evaluation_samples(self, request_data):
        self.calls.append(("promote_synthetic_evaluation_samples", request_data.model_dump(exclude_none=True, mode="json")))
        return {"dataset_id": "dataset_1", "dataset_snapshot_ref": "snapshot_1", "promotion_ids": [], "sample_count": 1}

    async def list_evaluation_benchmarks(self):
        self.calls.append(("list_evaluation_benchmarks",))
        return {"object": "list", "data": [{"name": "mmlu"}], "total": 1}

    async def get_evaluation_benchmark(self, benchmark_name):
        self.calls.append(("get_evaluation_benchmark", benchmark_name))
        return {"name": benchmark_name}

    async def run_evaluation_benchmark(self, benchmark_name, **kwargs):
        self.calls.append(("run_evaluation_benchmark", benchmark_name, kwargs))
        return {"benchmark": benchmark_name, "total_samples": 1, "results_summary": {}, "evaluation_id": "eval_1"}

    async def register_evaluation_webhook(self, request_data):
        self.calls.append(("register_evaluation_webhook", request_data.model_dump(exclude_none=True, mode="json")))
        return {"webhook_id": 10, "url": "https://example.com/evals", "events": ["evaluation.completed"], "secret": "x" * 32, "created_at": "2026-04-21T00:00:00Z"}

    async def list_evaluation_webhooks(self):
        self.calls.append(("list_evaluation_webhooks",))
        return []

    async def unregister_evaluation_webhook(self, url):
        self.calls.append(("unregister_evaluation_webhook", url))
        return {"status": "unregistered", "url": url}

    async def test_evaluation_webhook(self, request_data):
        self.calls.append(("test_evaluation_webhook", request_data.model_dump(exclude_none=True, mode="json")))
        return {"success": True}


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
    await service.create_or_update_rag_pipeline_preset(name="fast", config={"retriever": "hybrid"})
    await service.list_rag_pipeline_presets(limit=10, offset=5)
    await service.get_rag_pipeline_preset("fast")
    await service.delete_rag_pipeline_preset("fast")
    await service.cleanup_rag_pipeline()
    await service.create_embeddings_abtest(
        name="embed-test",
        config={"retrieval": {"k": 10}},
        run_immediately=True,
    )
    await service.run_embeddings_abtest("ab_1", config={"retrieval": {"k": 10}})
    await service.get_embeddings_abtest_status("ab_1")
    await service.get_embeddings_abtest_results("ab_1", page=2, page_size=25)
    await service.get_embeddings_abtest_significance("ab_1", metric="mrr")
    await service.export_embeddings_abtest("ab_1", format="json")
    await service.delete_embeddings_abtest("ab_1")
    await service.generate_synthetic_drafts(
        recipe_kind="rag_answer_quality",
        target_sample_count=2,
        corpus_scope={"collection": "docs"},
    )
    await service.list_synthetic_queue(recipe_kind="rag_answer_quality", limit=25, offset=5)
    await service.review_synthetic_sample("sample_1", action="approve", notes="usable")
    await service.promote_synthetic_samples(
        sample_ids=["sample_1"],
        dataset_name="approved_synthetic",
    )
    await service.list_benchmarks()
    await service.get_benchmark("mmlu")
    await service.run_benchmark("mmlu", limit=5, api_name="openai", parallel=2)
    await service.register_webhook(
        url="https://example.com/evals",
        events=["evaluation.completed"],
        secret="x" * 32,
    )
    await service.list_webhooks()
    await service.unregister_webhook("https://example.com/evals")
    await service.test_webhook("https://example.com/evals")

    assert [call.kwargs["action_id"] for call in policy.require_allowed.call_args_list] == [
        "evaluations.dataset.list.server",
        "evaluations.dataset.create.server",
        "evaluations.dataset.list.server",
        "evaluations.dataset.create.server",
        "evaluations.dataset.delete.server",
        "evaluations.run.launch.server",
        "evaluations.run.update.server",
        "evaluations.rag_pipeline.create.server",
        "evaluations.rag_pipeline.list.server",
        "evaluations.rag_pipeline.detail.server",
        "evaluations.rag_pipeline.delete.server",
        "evaluations.rag_pipeline.launch.server",
        "evaluations.embeddings_abtest.create.server",
        "evaluations.embeddings_abtest.launch.server",
        "evaluations.embeddings_abtest.detail.server",
        "evaluations.embeddings_abtest.observe.server",
        "evaluations.embeddings_abtest.observe.server",
        "evaluations.embeddings_abtest.export.server",
        "evaluations.embeddings_abtest.delete.server",
        "evaluations.synthetic.launch.server",
        "evaluations.synthetic.list.server",
        "evaluations.synthetic.update.server",
        "evaluations.synthetic.create.server",
        "evaluations.benchmarks.list.server",
        "evaluations.benchmarks.detail.server",
        "evaluations.benchmarks.launch.server",
        "evaluations.webhooks.create.server",
        "evaluations.webhooks.list.server",
        "evaluations.webhooks.delete.server",
        "evaluations.webhooks.launch.server",
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
