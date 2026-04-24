"""Thin server-backed evaluation service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..Chatbooks.server_chatbook_service import build_tldw_api_client_from_config
from ..tldw_api import (
    CreateEvaluationRequest,
    EmbeddingsABTestConfig,
    EmbeddingsABTestCreateRequest,
    EmbeddingsABTestRunRequest,
    EvaluationBenchmarkRunRequest,
    EvaluationDatasetCreateRequest,
    EvaluationRecipeDatasetValidationRequest,
    EvaluationRecipeRunCreateRequest,
    EvaluationRunCreateRequest,
    EvaluationWebhookRegistrationRequest,
    EvaluationWebhookTestRequest,
    PipelinePresetCreate,
    SyntheticEvalGenerationRequest,
    SyntheticEvalPromotionRequest,
    SyntheticEvalReviewRequest,
    TLDWAPIClient,
    UpdateEvaluationRequest,
)


class ServerEvaluationsService:
    """Wrap server evaluation endpoints with plain dict/list payloads."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerEvaluationsService":
        return cls(client=build_tldw_api_client_from_config(app_config))

    def _require_client(self) -> TLDWAPIClient:
        if self.client is None:
            raise ValueError("TLDW API client is required for server evaluation operations.")
        return self.client

    def _dump_model(self, value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        if isinstance(value, list):
            return [self._dump_model(item) for item in value]
        return value

    def _flatten_metrics(self, metrics: Any) -> dict[str, Any]:
        if isinstance(metrics, Mapping):
            flattened: dict[str, Any] = {}
            for key, value in metrics.items():
                if isinstance(value, Mapping) and "value" in value:
                    flattened[key] = value.get("value")
                else:
                    flattened[key] = value
            return flattened
        return {}

    async def list_evaluations(
        self,
        *,
        limit: int = 100,
        after: str | None = None,
        eval_type: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._dump_model(
            await self._require_client().list_evaluations(
                limit=limit,
                after=after,
                eval_type=eval_type,
            )
        )
        return list(payload.get("data", []))

    async def get_evaluation(self, eval_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation(eval_id))

    async def create_evaluation(
        self,
        *,
        name: str,
        eval_type: str,
        eval_spec: Any,
        description: str | None = None,
        dataset_id: str | None = None,
        dataset: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        request = CreateEvaluationRequest(
            name=name,
            description=description,
            eval_type=eval_type,
            eval_spec=eval_spec,
            dataset_id=dataset_id,
            dataset=dataset,
            metadata=metadata,
        )
        return self._dump_model(await self._require_client().create_evaluation(request))

    async def update_evaluation(
        self,
        eval_id: str,
        *,
        description: str | None = None,
        eval_spec: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        request = UpdateEvaluationRequest(
            description=description,
            eval_spec=eval_spec,
            metadata=metadata,
        )
        return self._dump_model(await self._require_client().update_evaluation(eval_id, request))

    async def delete_evaluation(self, eval_id: str) -> None:
        await self._require_client().delete_evaluation(eval_id)

    async def list_datasets(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        payload = self._dump_model(
            await self._require_client().list_evaluation_datasets(limit=limit, offset=offset)
        )
        return list(payload.get("data", []))

    async def get_dataset(
        self,
        dataset_id: str,
        *,
        include_samples: bool = True,
        limit: int | None = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().get_evaluation_dataset(
                dataset_id,
                include_samples=include_samples,
                limit=limit,
                offset=offset,
            )
        )

    async def create_dataset(
        self,
        *,
        name: str,
        samples: list[Any],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = EvaluationDatasetCreateRequest(
            name=name,
            description=description,
            samples=samples,
            metadata=metadata,
        )
        return self._dump_model(await self._require_client().create_evaluation_dataset(request))

    async def delete_dataset(self, dataset_id: str) -> None:
        await self._require_client().delete_evaluation_dataset(dataset_id)

    async def list_runs(
        self,
        *,
        eval_id: str,
        limit: int = 100,
        after: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._dump_model(
            await self._require_client().list_evaluation_runs(
                eval_id,
                limit=limit,
                after=after,
                status=status,
            )
        )
        return list(payload.get("data", []))

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation_run(run_id))

    async def create_run(
        self,
        eval_id: str,
        *,
        target_id: str | None = None,
        target_model: str | None = None,
        dataset_override: Any = None,
        config: dict[str, Any] | None = None,
        webhook_url: str | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        del target_id, run_name
        request = EvaluationRunCreateRequest(
            target_model=target_model,
            dataset_override=dataset_override,
            config=config or {},
            webhook_url=webhook_url,
        )
        return self._dump_model(await self._require_client().create_evaluation_run(eval_id, request))

    async def get_run_artifacts(self, run_id: str) -> dict[str, Any]:
        run = await self.get_run(run_id)
        metrics = self._flatten_metrics(run.get("results"))
        return {
            "run": run,
            "metrics": metrics,
            "results": None,
            "detail_available": False,
        }

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().cancel_evaluation_run(run_id))

    async def generate_synthetic_drafts(
        self,
        *,
        recipe_kind: str,
        corpus_scope: dict[str, Any] | list[str] | None = None,
        generation_metadata: dict[str, Any] | None = None,
        context_snapshot_ref: str | None = None,
        retrieval_baseline_ref: str | None = None,
        reference_answer: str | None = None,
        real_examples: list[dict[str, Any]] | None = None,
        seed_examples: list[dict[str, Any]] | None = None,
        target_sample_count: int = 0,
    ) -> dict[str, Any]:
        request = SyntheticEvalGenerationRequest(
            recipe_kind=recipe_kind,
            corpus_scope=corpus_scope,
            generation_metadata=generation_metadata or {},
            context_snapshot_ref=context_snapshot_ref,
            retrieval_baseline_ref=retrieval_baseline_ref,
            reference_answer=reference_answer,
            real_examples=real_examples or [],
            seed_examples=seed_examples or [],
            target_sample_count=target_sample_count,
        )
        return self._dump_model(
            await self._require_client().generate_synthetic_evaluation_drafts(request)
        )

    async def list_synthetic_queue(
        self,
        *,
        recipe_kind: str | None = None,
        review_state: str | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().list_synthetic_evaluation_queue(
                recipe_kind=recipe_kind,
                review_state=review_state,
                source_kind=source_kind,
                generation_batch_id=generation_batch_id,
                limit=limit,
                offset=offset,
            )
        )

    async def review_synthetic_sample(
        self,
        sample_id: str,
        *,
        action: str,
        reviewer_id: str | None = None,
        notes: str | None = None,
        action_payload: dict[str, Any] | None = None,
        resulting_review_state: str | None = None,
    ) -> dict[str, Any]:
        request = SyntheticEvalReviewRequest(
            action=action,
            reviewer_id=reviewer_id,
            notes=notes,
            action_payload=action_payload or {},
            resulting_review_state=resulting_review_state,
        )
        return self._dump_model(
            await self._require_client().review_synthetic_evaluation_sample(sample_id, request)
        )

    async def promote_synthetic_samples(
        self,
        *,
        sample_ids: list[str],
        dataset_name: str,
        dataset_description: str | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        promoted_by: str | None = None,
        promotion_reason: str | None = None,
    ) -> dict[str, Any]:
        request = SyntheticEvalPromotionRequest(
            sample_ids=sample_ids,
            dataset_name=dataset_name,
            dataset_description=dataset_description,
            dataset_metadata=dataset_metadata or {},
            promoted_by=promoted_by,
            promotion_reason=promotion_reason,
        )
        return self._dump_model(
            await self._require_client().promote_synthetic_evaluation_samples(request)
        )

    async def create_embeddings_abtest(
        self,
        *,
        name: str,
        config: dict[str, Any] | EmbeddingsABTestConfig,
        run_immediately: bool | None = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        request = EmbeddingsABTestCreateRequest(
            name=name,
            config=config,
            run_immediately=run_immediately,
        )
        return self._dump_model(
            await self._require_client().create_embeddings_abtest(
                request,
                idempotency_key=idempotency_key,
            )
        )

    async def run_embeddings_abtest(
        self,
        test_id: str,
        *,
        config: dict[str, Any] | EmbeddingsABTestConfig,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        request = EmbeddingsABTestRunRequest(config=config)
        return self._dump_model(
            await self._require_client().run_embeddings_abtest(
                test_id,
                request,
                idempotency_key=idempotency_key,
            )
        )

    async def get_embeddings_abtest_summary(self, test_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_embeddings_abtest_summary(test_id))

    async def get_embeddings_abtest_results(
        self,
        test_id: str,
        *,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().get_embeddings_abtest_results(
                test_id,
                page=page,
                page_size=page_size,
            )
        )

    async def get_embeddings_abtest_significance(
        self,
        test_id: str,
        *,
        metric: str = "ndcg",
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().get_embeddings_abtest_significance(
                test_id,
                metric=metric,
            )
        )

    async def list_benchmarks(self) -> dict[str, Any]:
        return self._dump_model(await self._require_client().list_evaluation_benchmarks())

    async def get_benchmark(self, benchmark_name: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation_benchmark(benchmark_name))

    async def run_benchmark(
        self,
        benchmark_name: str,
        *,
        limit: int | None = None,
        api_name: str = "openai",
        parallel: int = 4,
        save_results: bool = True,
        filter_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        request = EvaluationBenchmarkRunRequest(
            limit=limit,
            api_name=api_name,
            parallel=parallel,
            save_results=save_results,
            filter_categories=filter_categories,
        )
        return self._dump_model(
            await self._require_client().run_evaluation_benchmark(benchmark_name, request)
        )

    async def list_recipes(self) -> list[dict[str, Any]]:
        return [
            self._dump_model(item)
            for item in await self._require_client().list_evaluation_recipes()
        ]

    async def get_recipe(self, recipe_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation_recipe(recipe_id))

    async def get_recipe_launch_readiness(self, recipe_id: str) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().get_evaluation_recipe_launch_readiness(recipe_id)
        )

    async def validate_recipe_dataset(
        self,
        recipe_id: str,
        *,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = EvaluationRecipeDatasetValidationRequest(
            dataset_id=dataset_id,
            dataset=dataset,
            run_config=run_config,
        )
        return self._dump_model(
            await self._require_client().validate_evaluation_recipe_dataset(recipe_id, request)
        )

    async def create_recipe_run(
        self,
        recipe_id: str,
        *,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any] | None = None,
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        request = EvaluationRecipeRunCreateRequest(
            dataset_id=dataset_id,
            dataset=dataset,
            run_config=run_config or {},
            force_rerun=force_rerun,
        )
        return self._dump_model(
            await self._require_client().create_evaluation_recipe_run(recipe_id, request)
        )

    async def get_recipe_run(self, run_id: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation_recipe_run(run_id))

    async def get_recipe_run_report(self, run_id: str) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().get_evaluation_recipe_run_report(run_id)
        )

    async def save_pipeline_preset(
        self,
        *,
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        request = PipelinePresetCreate(name=name, config=config)
        return self._dump_model(
            await self._require_client().save_evaluation_pipeline_preset(request)
        )

    async def list_pipeline_presets(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        return self._dump_model(
            await self._require_client().list_evaluation_pipeline_presets(
                limit=limit,
                offset=offset,
            )
        )

    async def get_pipeline_preset(self, name: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().get_evaluation_pipeline_preset(name))

    async def delete_pipeline_preset(self, name: str) -> None:
        await self._require_client().delete_evaluation_pipeline_preset(name)

    async def cleanup_pipeline_collections(self) -> dict[str, Any]:
        return self._dump_model(await self._require_client().cleanup_evaluation_pipeline_collections())

    async def register_webhook(
        self,
        *,
        url: str,
        events: list[str],
        secret: str | None = None,
        retry_count: int | None = 3,
        timeout_seconds: int | None = 30,
    ) -> dict[str, Any]:
        request = EvaluationWebhookRegistrationRequest(
            url=url,
            events=events,
            secret=secret,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )
        return self._dump_model(await self._require_client().register_evaluation_webhook(request))

    async def list_webhooks(self) -> list[dict[str, Any]]:
        return [
            self._dump_model(item)
            for item in await self._require_client().list_evaluation_webhooks()
        ]

    async def unregister_webhook(self, url: str) -> dict[str, Any]:
        return self._dump_model(await self._require_client().unregister_evaluation_webhook(url))

    async def test_webhook(self, *, url: str) -> dict[str, Any]:
        request = EvaluationWebhookTestRequest(url=url)
        return self._dump_model(await self._require_client().test_evaluation_webhook(request))
