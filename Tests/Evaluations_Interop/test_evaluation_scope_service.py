import pytest

from tldw_chatbook.Evaluations_Interop.evaluation_scope_service import EvaluationScopeService


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
