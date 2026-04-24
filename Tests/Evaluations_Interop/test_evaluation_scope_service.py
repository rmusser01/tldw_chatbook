import pytest

from tldw_chatbook.Evaluations_Interop.evaluation_scope_service import EvaluationScopeService


class FakePolicyEnforcer:
    def __init__(self):
        self.actions = []

    def require_allowed(self, *, action_id):
        self.actions.append(action_id)


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

    def get_dataset(self, dataset_id):
        self.calls.append(("get_dataset", dataset_id))
        return {
            "id": dataset_id,
            "name": "local_dataset",
            "description": "Local dataset",
            "format": "custom",
            "source_path": "inline:local_dataset",
            "samples": [{"input": "Q", "expected": "A"}],
            "sample_count": 1,
            "metadata": {"project": "offline"},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:02:00Z",
            "version": 1,
            "client_id": "local_client",
        }

    def list_datasets(self, *, limit=100, offset=0):
        self.calls.append(("list_datasets", limit, offset))
        return [
            {
                "id": "dataset_123",
                "name": "local_dataset",
                "description": "Local dataset",
                "format": "custom",
                "source_path": "inline:local_dataset",
                "samples": [{"input": "Q", "expected": "A"}],
                "sample_count": 1,
                "metadata": {"project": "offline"},
                "created_at": "2026-04-20T00:00:00Z",
                "updated_at": "2026-04-20T00:02:00Z",
                "version": 1,
                "client_id": "local_client",
            }
        ]

    def create_dataset(self, *, name, samples, description=None, metadata=None, format=None, source_path=None):
        self.calls.append(("create_dataset", name, samples, description, metadata, format, source_path))
        return "dataset_local"

    def update_dataset(
        self,
        dataset_id,
        *,
        name=None,
        samples=None,
        description=None,
        metadata=None,
        format=None,
        source_path=None,
    ):
        self.calls.append(("update_dataset", dataset_id, name, samples, description, metadata, format, source_path))
        return {
            "id": dataset_id,
            "name": name or "local_dataset",
            "description": description,
            "format": format or "custom",
            "source_path": source_path or "inline:local_dataset",
            "samples": list(samples or []),
            "sample_count": len(samples or []),
            "metadata": metadata or {},
            "created_at": "2026-04-20T00:00:00Z",
            "updated_at": "2026-04-20T00:02:00Z",
            "version": 2,
            "client_id": "local_client",
        }

    def delete_dataset(self, dataset_id):
        self.calls.append(("delete_dataset", dataset_id))
        return None

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

    def delete_evaluation(self, eval_id):
        self.calls.append(("delete_evaluation", eval_id))
        return None

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

    def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {
            "id": run_id,
            "task_id": "task_123",
            "name": "local_run",
            "status": "completed",
            "model_id": "model_123",
            "model_name": "gpt-4.1-mini",
            "created_at": "2026-04-20T00:00:00Z",
        }

    def cancel_run(self, run_id):
        self.calls.append(("cancel_run", run_id))
        return {"status": "cancelled", "id": run_id}


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

    async def get_evaluation(self, eval_id):
        self.calls.append(("get_evaluation", eval_id))
        return {
            "id": eval_id,
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

    async def create_evaluation(self, **kwargs):
        self.calls.append(("create_evaluation", kwargs))
        return {
            "id": "eval_created",
            "object": "evaluation",
            "name": kwargs["name"],
            "description": kwargs.get("description"),
            "eval_type": kwargs["eval_type"],
            "eval_spec": kwargs["eval_spec"],
            "dataset_id": kwargs.get("dataset_id"),
            "created": 1713571200,
            "updated": 1713571260,
            "created_by": "user_1",
            "metadata": kwargs.get("metadata") or {},
        }

    async def update_evaluation(self, eval_id, **kwargs):
        self.calls.append(("update_evaluation", eval_id, kwargs))
        return {
            "id": eval_id,
            "object": "evaluation",
            "name": "server_eval",
            "description": kwargs.get("description"),
            "eval_type": "classification",
            "eval_spec": kwargs.get("eval_spec") or {"metrics": ["f1"]},
            "dataset_id": "dataset_999",
            "created": 1713571200,
            "updated": 1713571260,
            "created_by": "user_1",
            "metadata": kwargs.get("metadata") or {},
        }

    async def delete_evaluation(self, eval_id):
        self.calls.append(("delete_evaluation", eval_id))
        return None

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

    async def get_dataset(self, dataset_id):
        self.calls.append(("get_dataset", dataset_id))
        return {
            "id": dataset_id,
            "object": "dataset",
            "name": "server_dataset",
            "description": "Server dataset",
            "sample_count": 1,
            "samples": [{"input": "Q", "expected": "A"}],
            "created": 1713571200,
            "metadata": {"project": "server"},
        }

    async def list_datasets(self, *, limit=100, offset=0):
        self.calls.append(("list_datasets", limit, offset))
        return [
            {
                "id": "dataset_server",
                "object": "dataset",
                "name": "server_dataset",
                "description": "Server dataset",
                "sample_count": 1,
                "samples": [{"input": "Q", "expected": "A"}],
                "created": 1713571200,
                "metadata": {"project": "server"},
            }
        ]

    async def create_dataset(self, *, name, samples, description=None, metadata=None):
        self.calls.append(("create_dataset", name, samples, description, metadata))
        return {
            "id": "dataset_server",
            "object": "dataset",
            "name": name,
            "description": description,
            "sample_count": len(samples),
            "samples": samples,
            "created": 1713571200,
            "metadata": metadata or {},
        }

    async def delete_dataset(self, dataset_id):
        self.calls.append(("delete_dataset", dataset_id))
        return None

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

    async def get_run(self, run_id):
        self.calls.append(("get_run", run_id))
        return {
            "id": run_id,
            "object": "run",
            "eval_id": "eval_123",
            "status": "completed",
            "target_model": "openai:gpt-4.1-mini",
            "created": 1713571200,
            "results": {"accuracy": 0.91},
        }

    async def cancel_run(self, run_id):
        self.calls.append(("cancel_run", run_id))
        return {"status": "cancellation_requested", "id": run_id}

    async def evaluate_geval(self, **kwargs):
        self.calls.append(("evaluate_geval", kwargs))
        return {
            "metrics": {"fluency": {"score": 0.91}},
            "average_score": 0.91,
            "summary_assessment": "Strong summary",
            "evaluation_time": 1.2,
            "metadata": {"evaluation_id": "geval_1"},
        }

    async def evaluate_rag(self, **kwargs):
        self.calls.append(("evaluate_rag", kwargs))
        return {
            "metrics": {"faithfulness": {"score": 0.86}},
            "overall_score": 0.86,
            "retrieval_quality": 0.8,
            "generation_quality": 0.9,
            "suggestions": ["Add source coverage"],
        }

    async def evaluate_response_quality(self, **kwargs):
        self.calls.append(("evaluate_response_quality", kwargs))
        return {
            "metrics": {"relevance": {"score": 0.88}},
            "overall_quality": 0.88,
            "format_compliance": {"json": True},
        }

    async def evaluate_propositions(self, **kwargs):
        self.calls.append(("evaluate_propositions", kwargs))
        return {
            "precision": 0.8,
            "recall": 0.75,
            "f1": 0.77,
            "matched": 3,
            "total_extracted": 4,
            "total_reference": 4,
            "claim_density_per_100_tokens": 2.5,
            "avg_prop_len_tokens": 8.0,
            "dedup_rate": 0.0,
        }

    async def evaluate_batch(self, **kwargs):
        self.calls.append(("evaluate_batch", kwargs))
        return {
            "total_items": 2,
            "successful": 2,
            "failed": 0,
            "results": [{"id": "item_1", "score": 0.9}],
            "aggregate_metrics": {"average_score": 0.9},
            "processing_time": 2.4,
        }

    async def evaluate_ocr(self, **kwargs):
        self.calls.append(("evaluate_ocr", kwargs))
        return {
            "evaluation_id": "ocr_1",
            "results": {"items": [{"id": "doc_1", "cer": 0.02}]},
            "evaluation_time": 0.5,
        }

    async def evaluate_ocr_pdf(self, **kwargs):
        self.calls.append(("evaluate_ocr_pdf", kwargs))
        return {
            "evaluation_id": "ocr_pdf_1",
            "results": {"items": [{"id": "doc.pdf", "cer": 0.02}]},
            "evaluation_time": 1.5,
        }

    async def get_evaluation_history(self, **kwargs):
        self.calls.append(("get_evaluation_history", kwargs))
        return {
            "total_count": 1,
            "items": [{"evaluation_id": "geval_1", "evaluation_type": "geval"}],
            "aggregations": {"geval": 1},
        }

    async def generate_synthetic_drafts(self, **kwargs):
        self.calls.append(("generate_synthetic_drafts", kwargs))
        return {
            "generation_batch_id": "batch_123",
            "samples": [{"sample_id": "sample_123", "review_state": "draft"}],
        }

    async def list_synthetic_queue(
        self,
        *,
        recipe_kind=None,
        review_state=None,
        source_kind=None,
        generation_batch_id=None,
        limit=50,
        offset=0,
    ):
        self.calls.append(
            (
                "list_synthetic_queue",
                recipe_kind,
                review_state,
                source_kind,
                generation_batch_id,
                limit,
                offset,
            )
        )
        return {"data": [{"sample_id": "sample_123", "review_state": "in_review"}], "total": 1}

    async def review_synthetic_sample(
        self,
        sample_id,
        *,
        action,
        reviewer_id=None,
        notes=None,
        action_payload=None,
        resulting_review_state=None,
    ):
        self.calls.append(
            (
                "review_synthetic_sample",
                sample_id,
                action,
                reviewer_id,
                notes,
                action_payload,
                resulting_review_state,
            )
        )
        return {
            "action_id": "action_123",
            "sample_id": sample_id,
            "action": action,
            "reviewer_id": reviewer_id,
            "notes": notes,
            "resulting_review_state": resulting_review_state,
        }

    async def promote_synthetic_samples(
        self,
        *,
        sample_ids,
        dataset_name,
        dataset_description=None,
        dataset_metadata=None,
        promoted_by=None,
        promotion_reason=None,
    ):
        self.calls.append(
            (
                "promote_synthetic_samples",
                sample_ids,
                dataset_name,
                dataset_description,
                dataset_metadata,
                promoted_by,
                promotion_reason,
            )
        )
        return {
            "dataset_id": "dataset_123",
            "dataset_snapshot_ref": "snapshot_123",
            "promotion_ids": ["promotion_123"],
            "sample_count": len(sample_ids),
        }

    async def create_embeddings_abtest(self, **kwargs):
        self.calls.append(("create_embeddings_abtest", kwargs))
        return {"test_id": "abtest_123", "status": "created"}

    async def run_embeddings_abtest(self, test_id, **kwargs):
        self.calls.append(("run_embeddings_abtest", test_id, kwargs))
        return {"test_id": test_id, "status": "running", "progress": {"phase": 0.05}}

    async def get_embeddings_abtest_summary(self, test_id):
        self.calls.append(("get_embeddings_abtest_summary", test_id))
        return {"test_id": test_id, "status": "completed", "arms": []}

    async def get_embeddings_abtest_results(self, test_id, *, page=1, page_size=50):
        self.calls.append(("get_embeddings_abtest_results", test_id, page, page_size))
        return {
            "summary": {"test_id": test_id, "status": "completed", "arms": []},
            "results": [],
            "page": page,
            "page_size": page_size,
            "total": 0,
        }

    async def get_embeddings_abtest_significance(self, test_id, *, metric="ndcg"):
        self.calls.append(("get_embeddings_abtest_significance", test_id, metric))
        return {"metric": metric, "significant": True}

    async def list_benchmarks(self):
        self.calls.append(("list_benchmarks",))
        return {"object": "list", "data": [{"name": "truthfulqa"}], "total": 1}

    async def get_benchmark(self, benchmark_name):
        self.calls.append(("get_benchmark", benchmark_name))
        return {"name": benchmark_name, "description": "Truthfulness benchmark"}

    async def list_recipes(self):
        self.calls.append(("list_recipes",))
        return [{"recipe_id": "rag_answer_quality", "launchable": True}]

    async def get_recipe(self, recipe_id):
        self.calls.append(("get_recipe", recipe_id))
        return {"recipe_id": recipe_id, "launchable": True}

    async def get_recipe_launch_readiness(self, recipe_id):
        self.calls.append(("get_recipe_launch_readiness", recipe_id))
        return {"recipe_id": recipe_id, "ready": True, "can_enqueue_runs": True}

    async def validate_recipe_dataset(self, recipe_id, *, dataset_id=None, dataset=None, run_config=None):
        self.calls.append(("validate_recipe_dataset", recipe_id, dataset_id, dataset, run_config))
        return {"valid": True, "errors": [], "dataset_id": dataset_id, "sample_count": 2}

    async def run_benchmark(self, benchmark_name, **kwargs):
        self.calls.append(("run_benchmark", benchmark_name, kwargs))
        return {
            "benchmark": benchmark_name,
            "total_samples": kwargs.get("limit") or 2,
            "results_summary": {"average_score": 0.75},
            "evaluation_id": "eval_bench_1",
        }

    async def create_recipe_run(self, recipe_id, **kwargs):
        self.calls.append(("create_recipe_run", recipe_id, kwargs))
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

    async def get_recipe_run(self, run_id):
        self.calls.append(("get_recipe_run", run_id))
        return {"run_id": run_id, "status": "running"}

    async def get_recipe_run_report(self, run_id):
        self.calls.append(("get_recipe_run_report", run_id))
        return {
            "run": {"run_id": run_id, "status": "completed"},
            "confidence_summary": {"confidence": 0.82},
            "recommendation_slots": {},
        }

    async def save_pipeline_preset(self, *, name, config):
        self.calls.append(("save_pipeline_preset", name, config))
        return {"name": name, "config": config, "created_at": 1, "updated_at": 2}

    async def list_pipeline_presets(self, *, limit=50, offset=0):
        self.calls.append(("list_pipeline_presets", limit, offset))
        return {"items": [{"name": "baseline", "config": {}}], "total": 1}

    async def get_pipeline_preset(self, name):
        self.calls.append(("get_pipeline_preset", name))
        return {"name": name, "config": {}}

    async def delete_pipeline_preset(self, name):
        self.calls.append(("delete_pipeline_preset", name))

    async def cleanup_pipeline_collections(self):
        self.calls.append(("cleanup_pipeline_collections",))
        return {"expired_count": 2, "deleted_count": 1, "errors": ["collection_b: locked"]}

    async def register_webhook(self, *, url, events, secret=None, retry_count=3, timeout_seconds=30):
        self.calls.append(("register_webhook", url, events, secret, retry_count, timeout_seconds))
        return {"webhook_id": 7, "url": url, "events": events, "secret": secret or "x" * 32}

    async def list_webhooks(self):
        self.calls.append(("list_webhooks",))
        return [{"webhook_id": 7, "url": "https://example.com/evals", "events": ["evaluation.completed"]}]

    async def unregister_webhook(self, url):
        self.calls.append(("unregister_webhook", url))
        return {"status": "unregistered", "url": url}

    async def test_webhook(self, *, url):
        self.calls.append(("test_webhook", url))
        return {"success": True, "status_code": 200}


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
async def test_scope_service_enforces_policy_for_core_evaluation_surfaces():
    policy = FakePolicyEnforcer()
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=FakeServerEvaluationService(),
        policy_enforcer=policy,
    )

    await scope.list_evaluations(mode="local")
    await scope.get_evaluation(mode="server", eval_id="eval_123")
    await scope.create_evaluation(
        mode="local",
        name="local_eval",
        eval_type="question_answer",
        eval_spec={"metrics": ["accuracy"]},
    )
    await scope.update_evaluation(
        "eval_123",
        mode="server",
        description="Updated",
    )
    await scope.delete_evaluation(mode="local", eval_id="task_123")
    await scope.list_datasets(mode="server")
    await scope.get_dataset(mode="local", dataset_id="dataset_123")
    await scope.create_dataset(
        mode="server",
        name="server_dataset",
        samples=[{"input": "Q", "expected": "A"}],
    )
    await scope.update_dataset(
        mode="local",
        dataset_id="dataset_123",
        name="renamed",
    )
    await scope.delete_dataset(mode="server", dataset_id="dataset_server")
    await scope.list_targets(mode="local")
    await scope.list_targets(mode="server")
    await scope.list_runs(mode="local", eval_id="task_123")
    await scope.get_run(mode="server", run_id="run_999")
    await scope.create_run(
        mode="server",
        eval_id="eval_123",
        target_model="openai:gpt-4.1-mini",
    )
    await scope.get_run_artifacts(mode="local", run_id="run_123")
    await scope.cancel_run(mode="server", run_id="run_999")

    assert policy.actions == [
        "evaluations.evaluation.list.local",
        "evaluations.evaluation.detail.server",
        "evaluations.evaluation.create.local",
        "evaluations.evaluation.update.server",
        "evaluations.evaluation.delete.local",
        "evaluations.dataset.list.server",
        "evaluations.dataset.detail.local",
        "evaluations.dataset.create.server",
        "evaluations.dataset.update.local",
        "evaluations.dataset.delete.server",
        "evaluations.target.list.local",
        "evaluations.target.list.server",
        "evaluations.run.list.local",
        "evaluations.run.detail.server",
        "evaluations.run.launch.server",
        "evaluations.run.detail.local",
        "evaluations.run.update.server",
    ]


@pytest.mark.asyncio
async def test_scope_service_enforces_policy_for_server_evaluation_adjuncts():
    policy = FakePolicyEnforcer()
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=FakeServerEvaluationService(),
        policy_enforcer=policy,
    )

    await scope.evaluate_geval(mode="server", source_text="source", summary="summary")
    await scope.get_evaluation_history(mode="server")
    await scope.generate_synthetic_drafts(
        mode="server",
        recipe_kind="rag_answer_quality",
        target_sample_count=1,
    )
    await scope.list_synthetic_queue(mode="server")
    await scope.review_synthetic_sample(mode="server", sample_id="sample_123", action="approve")
    await scope.promote_synthetic_samples(
        mode="server",
        sample_ids=["sample_123"],
        dataset_name="Promoted",
    )
    await scope.create_embeddings_abtest(mode="server", name="A/B", config={"arms": []})
    await scope.run_embeddings_abtest(mode="server", test_id="abtest_123", config={})
    await scope.get_embeddings_abtest_summary(mode="server", test_id="abtest_123")
    await scope.get_embeddings_abtest_results(mode="server", test_id="abtest_123")
    await scope.get_embeddings_abtest_significance(mode="server", test_id="abtest_123")
    await scope.list_benchmarks(mode="server")
    await scope.get_benchmark(mode="server", benchmark_name="truthfulqa")
    await scope.run_benchmark(mode="server", benchmark_name="truthfulqa")
    await scope.list_recipes(mode="server")
    await scope.get_recipe(mode="server", recipe_id="rag_answer_quality")
    await scope.get_recipe_launch_readiness(mode="server", recipe_id="rag_answer_quality")
    await scope.validate_recipe_dataset(mode="server", recipe_id="rag_answer_quality", dataset_id="dataset_123")
    await scope.create_recipe_run(mode="server", recipe_id="rag_answer_quality")
    await scope.get_recipe_run(mode="server", run_id="recipe_run_1")
    await scope.get_recipe_run_report(mode="server", run_id="recipe_run_1")
    await scope.save_pipeline_preset(mode="server", name="baseline", config={})
    await scope.list_pipeline_presets(mode="server")
    await scope.get_pipeline_preset(mode="server", name="baseline")
    await scope.delete_pipeline_preset(mode="server", name="baseline")
    await scope.cleanup_pipeline_collections(mode="server")
    await scope.register_webhook(mode="server", url="https://example.com/evals", events=["evaluation.completed"])
    await scope.list_webhooks(mode="server")
    await scope.unregister_webhook(mode="server", url="https://example.com/evals")
    await scope.test_webhook(mode="server", url="https://example.com/evals")

    assert policy.actions == [
        "evaluations.immediate.launch.server",
        "evaluations.immediate.list.server",
        "evaluations.synthetic.create.server",
        "evaluations.synthetic.list.server",
        "evaluations.synthetic.update.server",
        "evaluations.synthetic.create.server",
        "evaluations.abtest.create.server",
        "evaluations.abtest.launch.server",
        "evaluations.abtest.detail.server",
        "evaluations.abtest.detail.server",
        "evaluations.abtest.detail.server",
        "evaluations.benchmark.list.server",
        "evaluations.benchmark.detail.server",
        "evaluations.benchmark.launch.server",
        "evaluations.recipe.list.server",
        "evaluations.recipe.detail.server",
        "evaluations.recipe.detail.server",
        "evaluations.recipe.detail.server",
        "evaluations.recipe.launch.server",
        "evaluations.recipe.detail.server",
        "evaluations.recipe.detail.server",
        "evaluations.pipeline_preset.create.server",
        "evaluations.pipeline_preset.list.server",
        "evaluations.pipeline_preset.detail.server",
        "evaluations.pipeline_preset.delete.server",
        "evaluations.pipeline_preset.delete.server",
        "evaluations.webhook.create.server",
        "evaluations.webhook.list.server",
        "evaluations.webhook.delete.server",
        "evaluations.webhook.launch.server",
    ]


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
async def test_scope_service_routes_dataset_create_and_delete_by_backend():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    local_created = await scope.create_dataset(
        mode="local",
        name="local_dataset",
        samples=[{"input": "Q", "expected": "A"}],
        description="Local dataset",
        metadata={"project": "offline"},
    )
    server_created = await scope.create_dataset(
        mode="server",
        name="server_dataset",
        samples=[{"input": "Q", "expected": "A"}],
        description="Server dataset",
        metadata={"project": "server"},
    )
    await scope.delete_dataset(mode="local", dataset_id="dataset_local")
    await scope.delete_dataset(mode="server", dataset_id="dataset_server")

    assert local_created["record_id"] == "local:evaluation_dataset:dataset_local"
    assert local_created["sample_count"] == 1
    assert server_created["record_id"] == "server:evaluation_dataset:dataset_server"
    assert server_created["sample_count"] == 1
    assert (
        "create_dataset",
        "local_dataset",
        [{"input": "Q", "expected": "A"}],
        "Local dataset",
        {"project": "offline"},
        None,
        None,
    ) in local.calls
    assert ("delete_dataset", "dataset_local") in local.calls
    assert (
        "create_dataset",
        "server_dataset",
        [{"input": "Q", "expected": "A"}],
        "Server dataset",
        {"project": "server"},
    ) in server.calls
    assert ("delete_dataset", "dataset_server") in server.calls


@pytest.mark.asyncio
async def test_scope_service_updates_local_dataset_but_rejects_server_dataset_update():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    updated = await scope.update_dataset(
        mode="local",
        dataset_id="dataset_local",
        name="renamed_dataset",
        samples=[{"input": "Q2", "expected": "A2"}],
        metadata={"project": "offline-v2"},
    )

    assert updated["record_id"] == "local:evaluation_dataset:dataset_local"
    assert updated["name"] == "renamed_dataset"
    assert updated["sample_count"] == 1
    assert (
        "update_dataset",
        "dataset_local",
        "renamed_dataset",
        [{"input": "Q2", "expected": "A2"}],
        None,
        {"project": "offline-v2"},
        None,
        None,
    ) in local.calls
    with pytest.raises(ValueError, match="dataset update is not available"):
        await scope.update_dataset(
            mode="server",
            dataset_id="dataset_server",
            name="unsupported",
        )
    assert not any(call[0] == "update_dataset" for call in server.calls)


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
async def test_scope_service_routes_unified_immediate_evaluations_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    geval = await scope.evaluate_geval(
        mode="server",
        source_text="Original text long enough",
        summary="Summary text long enough",
    )
    rag = await scope.evaluate_rag(
        mode="server",
        query="What changed?",
        retrieved_contexts=["Context"],
        generated_response="Answer",
        ground_truth="Expected answer",
    )
    quality = await scope.evaluate_response_quality(
        mode="server",
        prompt="Explain this",
        response="A complete answer",
    )
    propositions = await scope.evaluate_propositions(
        mode="server",
        extracted=["Claim A"],
        reference=["Claim A"],
    )
    batch = await scope.evaluate_batch(
        mode="server",
        evaluation_type="geval",
        items=[{"source_text": "A", "summary": "B"}],
    )
    ocr = await scope.evaluate_ocr(
        mode="server",
        items=[
            {
                "id": "doc_1",
                "extracted_text": "hello world",
                "ground_truth_text": "hello world",
            }
        ],
    )
    ocr_pdf = await scope.evaluate_ocr_pdf(
        mode="server",
        file_paths=["/tmp/doc.pdf"],
        ground_truths=["hello world"],
        metrics=["cer"],
    )
    history = await scope.get_evaluation_history(mode="server", evaluation_type="geval", limit=10)

    assert geval["average_score"] == 0.91
    assert rag["suggestions"] == ["Add source coverage"]
    assert quality["format_compliance"] == {"json": True}
    assert propositions["f1"] == 0.77
    assert batch["aggregate_metrics"] == {"average_score": 0.9}
    assert ocr["results"]["items"][0]["cer"] == 0.02
    assert ocr_pdf["evaluation_id"] == "ocr_pdf_1"
    assert history["aggregations"] == {"geval": 1}
    assert server.calls[-8] == (
        "evaluate_geval",
        {"source_text": "Original text long enough", "summary": "Summary text long enough"},
    )
    assert server.calls[-5] == (
        "evaluate_propositions",
        {"extracted": ["Claim A"], "reference": ["Claim A"]},
    )
    assert server.calls[-2] == (
        "evaluate_ocr_pdf",
        {"file_paths": ["/tmp/doc.pdf"], "ground_truths": ["hello world"], "metrics": ["cer"]},
    )
    assert server.calls[-1] == (
        "get_evaluation_history",
        {"evaluation_type": "geval", "limit": 10},
    )

    with pytest.raises(ValueError, match="server-only"):
        await scope.evaluate_geval(
            mode="local",
            source_text="Original text long enough",
            summary="Summary text long enough",
        )


@pytest.mark.asyncio
async def test_scope_service_routes_synthetic_evaluation_actions_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    generated = await scope.generate_synthetic_drafts(
        mode="server",
        recipe_kind="rag_answer_quality",
        corpus_scope={"workspace_id": "ws_1"},
        seed_examples=[{"question": "What changed?"}],
        target_sample_count=1,
    )
    queue = await scope.list_synthetic_queue(
        mode="server",
        recipe_kind="rag_answer_quality",
        review_state="in_review",
        source_kind="seed",
        generation_batch_id="batch_123",
        limit=25,
        offset=5,
    )
    review = await scope.review_synthetic_sample(
        mode="server",
        sample_id="sample_123",
        action="approve",
        reviewer_id="u1",
        notes="Looks usable",
        resulting_review_state="approved",
    )
    promoted = await scope.promote_synthetic_samples(
        mode="server",
        sample_ids=["sample_123"],
        dataset_name="Approved RAG samples",
        dataset_metadata={"project": "parity"},
        promotion_reason="manual_review",
    )

    assert generated["generation_batch_id"] == "batch_123"
    assert queue["total"] == 1
    assert review["resulting_review_state"] == "approved"
    assert promoted["dataset_snapshot_ref"] == "snapshot_123"
    assert server.calls[-4][0] == "generate_synthetic_drafts"
    assert server.calls[-3] == (
        "list_synthetic_queue",
        "rag_answer_quality",
        "in_review",
        "seed",
        "batch_123",
        25,
        5,
    )

    with pytest.raises(ValueError, match="server-only"):
        await scope.generate_synthetic_drafts(
            mode="local",
            recipe_kind="rag_answer_quality",
        )


@pytest.mark.asyncio
async def test_scope_service_routes_embeddings_abtest_actions_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)
    config = {
        "arms": [{"provider": "openai", "model": "text-embedding-3-small"}],
        "media_ids": [42],
        "retrieval": {"k": 10, "search_mode": "vector"},
        "queries": [{"text": "What changed?", "expected_ids": [42]}],
    }

    created = await scope.create_embeddings_abtest(
        mode="server",
        name="embedding comparison",
        config=config,
        idempotency_key="create-key",
    )
    run_status = await scope.run_embeddings_abtest(
        mode="server",
        test_id="abtest_123",
        config=config,
        idempotency_key="run-key",
    )
    summary = await scope.get_embeddings_abtest_summary(mode="server", test_id="abtest_123")
    results = await scope.get_embeddings_abtest_results(
        mode="server",
        test_id="abtest_123",
        page=2,
        page_size=10,
    )
    significance = await scope.get_embeddings_abtest_significance(
        mode="server",
        test_id="abtest_123",
        metric="recall",
    )

    assert created["test_id"] == "abtest_123"
    assert run_status["progress"]["phase"] == 0.05
    assert summary["status"] == "completed"
    assert results["page"] == 2
    assert significance["metric"] == "recall"
    assert server.calls[-5][0] == "create_embeddings_abtest"
    assert server.calls[-4] == (
        "run_embeddings_abtest",
        "abtest_123",
        {"config": config, "idempotency_key": "run-key"},
    )

    with pytest.raises(ValueError, match="server-only"):
        await scope.get_embeddings_abtest_summary(mode="local", test_id="abtest_123")


@pytest.mark.asyncio
async def test_scope_service_routes_evaluation_catalog_actions_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    benchmarks = await scope.list_benchmarks(mode="server")
    benchmark = await scope.get_benchmark(mode="server", benchmark_name="truthfulqa")
    recipes = await scope.list_recipes(mode="server")
    recipe = await scope.get_recipe(mode="server", recipe_id="rag_answer_quality")
    readiness = await scope.get_recipe_launch_readiness(
        mode="server",
        recipe_id="rag_answer_quality",
    )
    validation = await scope.validate_recipe_dataset(
        mode="server",
        recipe_id="rag_answer_quality",
        dataset_id="dataset_123",
        run_config={"evaluation_mode": "fixed_context"},
    )

    assert benchmarks["total"] == 1
    assert benchmark["name"] == "truthfulqa"
    assert recipes[0]["recipe_id"] == "rag_answer_quality"
    assert recipe["recipe_id"] == "rag_answer_quality"
    assert readiness["ready"] is True
    assert validation["valid"] is True
    assert server.calls[-1] == (
        "validate_recipe_dataset",
        "rag_answer_quality",
        "dataset_123",
        None,
        {"evaluation_mode": "fixed_context"},
    )

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_benchmarks(mode="local")


@pytest.mark.asyncio
async def test_scope_service_routes_evaluation_run_launch_actions_to_server_only():
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=server,
    )

    benchmark_run = await scope.run_benchmark(
        mode="server",
        benchmark_name="truthfulqa",
        limit=2,
        parallel=2,
        filter_categories=["truthful"],
    )
    recipe_run = await scope.create_recipe_run(
        mode="server",
        recipe_id="rag_answer_quality",
        dataset_id="dataset_1",
        run_config={"evaluation_mode": "fixed_context"},
        force_rerun=True,
    )
    fetched_recipe_run = await scope.get_recipe_run(
        mode="server",
        run_id="recipe_run_1",
    )
    report = await scope.get_recipe_run_report(
        mode="server",
        run_id="recipe_run_1",
    )

    assert benchmark_run["evaluation_id"] == "eval_bench_1"
    assert recipe_run["metadata"] == {"job_id": "job-1"}
    assert fetched_recipe_run["status"] == "running"
    assert report["confidence_summary"]["confidence"] == 0.82
    assert server.calls[-4] == (
        "run_benchmark",
        "truthfulqa",
        {"limit": 2, "parallel": 2, "filter_categories": ["truthful"]},
    )
    assert server.calls[-3] == (
        "create_recipe_run",
        "rag_answer_quality",
        {
            "dataset_id": "dataset_1",
            "dataset": None,
            "run_config": {"evaluation_mode": "fixed_context"},
            "force_rerun": True,
        },
    )
    assert server.calls[-2] == ("get_recipe_run", "recipe_run_1")
    assert server.calls[-1] == ("get_recipe_run_report", "recipe_run_1")

    with pytest.raises(ValueError) as benchmark_exc:
        await scope.run_benchmark(mode="local", benchmark_name="truthfulqa")
    with pytest.raises(ValueError) as recipe_exc:
        await scope.create_recipe_run(mode="local", recipe_id="rag_answer_quality")

    assert "Evaluation benchmark run is server-only" in str(benchmark_exc.value)
    assert "Evaluation recipe run launch is server-only" in str(recipe_exc.value)


@pytest.mark.asyncio
async def test_scope_service_routes_rag_pipeline_preset_actions_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    saved = await scope.save_pipeline_preset(
        mode="server",
        name="baseline",
        config={"chunking": {"method": "sentences"}},
    )
    listed = await scope.list_pipeline_presets(mode="server", limit=25, offset=5)
    fetched = await scope.get_pipeline_preset(mode="server", name="baseline")
    deleted = await scope.delete_pipeline_preset(mode="server", name="baseline")
    cleanup = await scope.cleanup_pipeline_collections(mode="server")

    assert saved["name"] == "baseline"
    assert listed["total"] == 1
    assert fetched["name"] == "baseline"
    assert deleted["status"] == "deleted"
    assert cleanup["deleted_count"] == 1
    assert server.calls[-4] == ("list_pipeline_presets", 25, 5)
    assert server.calls[-1] == ("cleanup_pipeline_collections",)

    with pytest.raises(ValueError, match="server-only"):
        await scope.save_pipeline_preset(mode="local", name="baseline", config={})


@pytest.mark.asyncio
async def test_scope_service_routes_evaluation_webhook_actions_to_server_only():
    local = FakeLocalEvaluationService()
    server = FakeServerEvaluationService()
    scope = EvaluationScopeService(local_service=local, server_service=server)

    registered = await scope.register_webhook(
        mode="server",
        url="https://example.com/evals",
        events=["evaluation.completed"],
        secret="x" * 32,
    )
    webhooks = await scope.list_webhooks(mode="server")
    unregistered = await scope.unregister_webhook(mode="server", url="https://example.com/evals")
    tested = await scope.test_webhook(mode="server", url="https://example.com/evals")

    assert registered["webhook_id"] == 7
    assert webhooks[0]["webhook_id"] == 7
    assert unregistered["status"] == "unregistered"
    assert tested["success"] is True
    assert server.calls[-4] == (
        "register_webhook",
        "https://example.com/evals",
        ["evaluation.completed"],
        "x" * 32,
        3,
        30,
    )

    with pytest.raises(ValueError, match="server-only"):
        await scope.list_webhooks(mode="local")
