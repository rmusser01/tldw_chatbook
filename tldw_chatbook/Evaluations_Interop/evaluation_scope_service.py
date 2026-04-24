"""Mode-aware routing for local/server evaluation surfaces."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any

from .evaluation_normalizers import (
    normalize_evaluation_dataset_record,
    normalize_evaluation_record,
    normalize_evaluation_run_record,
    normalize_evaluation_target_record,
)


class EvaluationBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


class EvaluationScopeService:
    """Route evaluation actions to local or server backends and normalize outputs."""

    def __init__(
        self,
        *,
        local_service: Any,
        server_service: Any,
        policy_enforcer: Any = None,
    ):
        self.local_service = local_service
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: EvaluationBackend | str | None) -> EvaluationBackend:
        if mode is None:
            return EvaluationBackend.LOCAL
        if isinstance(mode, EvaluationBackend):
            return mode
        try:
            return EvaluationBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid evaluation backend: {mode}") from exc

    def _service_for_mode(self, mode: EvaluationBackend) -> Any:
        if mode == EvaluationBackend.LOCAL:
            if self.local_service is None:
                raise ValueError("Local evaluations backend is unavailable.")
            return self.local_service
        if self.server_service is None:
            raise ValueError("Server evaluations backend is unavailable.")
        return self.server_service

    def _server_only_service(self, mode: EvaluationBackend | str | None, feature_name: str) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != EvaluationBackend.SERVER:
            raise ValueError(f"{feature_name} is server-only in this Chatbook parity slice.")
        return self._service_for_mode(normalized_mode)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def _resolve_record(self, record: Any, fetcher: Any, identifier: str) -> Any:
        if isinstance(record, Mapping):
            return record
        if isinstance(record, str) and record:
            return await self._maybe_await(fetcher(record))
        if isinstance(record, bool):
            return await self._maybe_await(fetcher(identifier))
        return record

    def _fallback_run_record(
        self,
        backend: EvaluationBackend,
        *,
        run_id: str,
        eval_id: str,
        target_model: str | None,
        config: dict[str, Any] | None,
        run_name: str | None,
    ) -> dict[str, Any]:
        if backend == EvaluationBackend.LOCAL:
            return {
                "id": run_id,
                "task_id": eval_id,
                "name": run_name or run_id,
                "status": "pending",
                "target_model": target_model,
                "config_overrides": config or {},
            }
        return {
            "id": run_id,
            "eval_id": eval_id,
            "name": run_name or run_id,
            "status": "pending",
            "target_model": target_model,
            "config": config or {},
        }

    async def list_evaluations(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
        after: str | None = None,
        eval_type: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == EvaluationBackend.LOCAL:
            records = await self._maybe_await(
                service.list_evaluations(limit=limit, offset=offset, eval_type=eval_type)
            )
        else:
            records = await self._maybe_await(
                service.list_evaluations(limit=limit, after=after, eval_type=eval_type)
            )
        return [
            normalize_evaluation_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_evaluation(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        eval_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(self._service_for_mode(normalized_mode).get_evaluation(eval_id))
        return normalize_evaluation_record(normalized_mode.value, record)

    async def create_evaluation(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        eval_type: str,
        eval_spec: Any,
        description: str | None = None,
        dataset_id: str | None = None,
        dataset: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.create_evaluation(
                name=name,
                description=description,
                eval_type=eval_type,
                eval_spec=eval_spec,
                dataset_id=dataset_id,
                dataset=dataset,
                metadata=metadata,
            )
        )
        resolved = await self._resolve_record(record, service.get_evaluation, name)
        return normalize_evaluation_record(normalized_mode.value, resolved)

    async def update_evaluation(
        self,
        eval_id: str,
        *,
        mode: EvaluationBackend | str | None = None,
        description: str | None = None,
        eval_spec: Any = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.update_evaluation(
                eval_id,
                description=description,
                eval_spec=eval_spec,
                metadata=metadata,
            )
        )
        resolved = await self._resolve_record(record, service.get_evaluation, eval_id)
        return normalize_evaluation_record(normalized_mode.value, resolved)

    async def delete_evaluation(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        eval_id: str,
    ) -> None:
        service = self._service_for_mode(self._normalize_mode(mode))
        await self._maybe_await(service.delete_evaluation(eval_id))

    async def list_datasets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        records = await self._maybe_await(
            self._service_for_mode(normalized_mode).list_datasets(limit=limit, offset=offset)
        )
        return [
            normalize_evaluation_dataset_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        dataset_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(self._service_for_mode(normalized_mode).get_dataset(dataset_id))
        return normalize_evaluation_dataset_record(normalized_mode.value, record)

    async def create_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        samples: list[dict[str, Any]],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        format: str | None = None,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == EvaluationBackend.LOCAL:
            record = await self._maybe_await(
                service.create_dataset(
                    name=name,
                    samples=samples,
                    description=description,
                    metadata=metadata,
                    format=format,
                    source_path=source_path,
                )
            )
        else:
            record = await self._maybe_await(
                service.create_dataset(
                    name=name,
                    samples=samples,
                    description=description,
                    metadata=metadata,
                )
            )
        resolved = await self._resolve_record(record, service.get_dataset, str(record or name))
        return normalize_evaluation_dataset_record(normalized_mode.value, resolved)

    async def delete_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        dataset_id: str,
    ) -> None:
        service = self._service_for_mode(self._normalize_mode(mode))
        await self._maybe_await(service.delete_dataset(dataset_id))

    async def list_targets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        provider: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        if not hasattr(service, "list_targets"):
            return []
        records = await self._maybe_await(
            service.list_targets(provider=provider, limit=limit, offset=offset)
        )
        return [
            normalize_evaluation_target_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def list_runs(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        eval_id: str,
        limit: int = 100,
        offset: int = 0,
        after: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == EvaluationBackend.LOCAL:
            records = await self._maybe_await(
                service.list_runs(eval_id=eval_id, status=status, limit=limit, offset=offset)
            )
        else:
            records = await self._maybe_await(
                service.list_runs(eval_id=eval_id, limit=limit, after=after, status=status)
            )
        return [
            normalize_evaluation_run_record(normalized_mode.value, record)
            for record in list(records or [])
        ]

    async def get_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        record = await self._maybe_await(self._service_for_mode(normalized_mode).get_run(run_id))
        return normalize_evaluation_run_record(normalized_mode.value, record)

    async def create_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        eval_id: str,
        target_id: str | None = None,
        target_model: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        dataset_override: Any = None,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        record = await self._maybe_await(
            service.create_run(
                eval_id,
                target_id=target_id,
                target_model=target_model,
                config=config,
                run_name=run_name,
                dataset_override=dataset_override,
                webhook_url=webhook_url,
            )
        )
        if isinstance(record, Mapping):
            resolved = record
        elif hasattr(service, "get_run"):
            resolved = await self._resolve_record(record, service.get_run, eval_id)
        elif isinstance(record, str) and record:
            resolved = self._fallback_run_record(
                normalized_mode,
                run_id=record,
                eval_id=eval_id,
                target_model=target_model or target_id,
                config=config,
                run_name=run_name,
            )
        else:
            resolved = record
        return normalize_evaluation_run_record(normalized_mode.value, resolved)

    async def get_run_artifacts(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        service = self._service_for_mode(normalized_mode)
        if hasattr(service, "get_run_artifacts"):
            payload = await self._maybe_await(service.get_run_artifacts(run_id))
        else:
            payload = {
                "run": await self._maybe_await(service.get_run(run_id)),
                "metrics": {},
                "results": None,
                "detail_available": normalized_mode == EvaluationBackend.LOCAL,
            }

        result = dict(payload or {})
        run_record = result.get("run")
        if run_record is None:
            run_record = await self._maybe_await(service.get_run(run_id))

        normalized_run = normalize_evaluation_run_record(normalized_mode.value, run_record)
        metrics = dict(result.get("metrics") or normalized_run.get("results") or {})

        return {
            "run": normalized_run,
            "metrics": metrics,
            "results": result.get("results"),
            "detail_available": bool(result.get("detail_available")),
        }

    async def cancel_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        payload = await self._maybe_await(self._service_for_mode(normalized_mode).cancel_run(run_id))
        result = dict(payload or {})
        result.setdefault("backend", normalized_mode.value)
        result.setdefault("id", run_id)
        return result

    async def evaluate_geval(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "G-Eval immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_geval(**payload)) or {})

    async def evaluate_rag(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "RAG immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_rag(**payload)) or {})

    async def evaluate_response_quality(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Response quality immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_response_quality(**payload)) or {})

    async def evaluate_propositions(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Proposition immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_propositions(**payload)) or {})

    async def evaluate_batch(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Batch immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_batch(**payload)) or {})

    async def evaluate_ocr(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "OCR immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_ocr(**payload)) or {})

    async def evaluate_ocr_pdf(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "OCR PDF immediate evaluation")
        return dict(await self._maybe_await(service.evaluate_ocr_pdf(**payload)) or {})

    async def get_evaluation_history(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation history")
        return dict(await self._maybe_await(service.get_evaluation_history(**payload)) or {})

    async def generate_synthetic_drafts(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
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
        service = self._server_only_service(mode, "Synthetic evaluation draft generation")
        return dict(
            await self._maybe_await(
                service.generate_synthetic_drafts(
                    recipe_kind=recipe_kind,
                    corpus_scope=corpus_scope,
                    generation_metadata=generation_metadata,
                    context_snapshot_ref=context_snapshot_ref,
                    retrieval_baseline_ref=retrieval_baseline_ref,
                    reference_answer=reference_answer,
                    real_examples=real_examples,
                    seed_examples=seed_examples,
                    target_sample_count=target_sample_count,
                )
            )
            or {}
        )

    async def list_synthetic_queue(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_kind: str | None = None,
        review_state: str | None = None,
        source_kind: str | None = None,
        generation_batch_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Synthetic evaluation queue")
        return dict(
            await self._maybe_await(
                service.list_synthetic_queue(
                    recipe_kind=recipe_kind,
                    review_state=review_state,
                    source_kind=source_kind,
                    generation_batch_id=generation_batch_id,
                    limit=limit,
                    offset=offset,
                )
            )
            or {}
        )

    async def review_synthetic_sample(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        sample_id: str,
        action: str,
        reviewer_id: str | None = None,
        notes: str | None = None,
        action_payload: dict[str, Any] | None = None,
        resulting_review_state: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Synthetic evaluation review")
        return dict(
            await self._maybe_await(
                service.review_synthetic_sample(
                    sample_id,
                    action=action,
                    reviewer_id=reviewer_id,
                    notes=notes,
                    action_payload=action_payload,
                    resulting_review_state=resulting_review_state,
                )
            )
            or {}
        )

    async def promote_synthetic_samples(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        sample_ids: list[str],
        dataset_name: str,
        dataset_description: str | None = None,
        dataset_metadata: dict[str, Any] | None = None,
        promoted_by: str | None = None,
        promotion_reason: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Synthetic evaluation promotion")
        return dict(
            await self._maybe_await(
                service.promote_synthetic_samples(
                    sample_ids=sample_ids,
                    dataset_name=dataset_name,
                    dataset_description=dataset_description,
                    dataset_metadata=dataset_metadata,
                    promoted_by=promoted_by,
                    promotion_reason=promotion_reason,
                )
            )
            or {}
        )

    async def create_embeddings_abtest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        config: dict[str, Any],
        run_immediately: bool | None = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Embedding A/B test creation")
        return dict(
            await self._maybe_await(
                service.create_embeddings_abtest(
                    name=name,
                    config=config,
                    run_immediately=run_immediately,
                    idempotency_key=idempotency_key,
                )
            )
            or {}
        )

    async def run_embeddings_abtest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
        config: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Embedding A/B test run")
        return dict(
            await self._maybe_await(
                service.run_embeddings_abtest(
                    test_id,
                    config=config,
                    idempotency_key=idempotency_key,
                )
            )
            or {}
        )

    async def get_embeddings_abtest_summary(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Embedding A/B test summary")
        return dict(
            await self._maybe_await(service.get_embeddings_abtest_summary(test_id))
            or {}
        )

    async def get_embeddings_abtest_results(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Embedding A/B test results")
        return dict(
            await self._maybe_await(
                service.get_embeddings_abtest_results(test_id, page=page, page_size=page_size)
            )
            or {}
        )

    async def get_embeddings_abtest_significance(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
        metric: str = "ndcg",
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Embedding A/B test significance")
        return dict(
            await self._maybe_await(
                service.get_embeddings_abtest_significance(test_id, metric=metric)
            )
            or {}
        )

    async def list_benchmarks(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation benchmark discovery")
        return dict(await self._maybe_await(service.list_benchmarks()) or {})

    async def get_benchmark(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        benchmark_name: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation benchmark detail")
        return dict(await self._maybe_await(service.get_benchmark(benchmark_name)) or {})

    async def list_recipes(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        service = self._server_only_service(mode, "Evaluation recipe discovery")
        return list(await self._maybe_await(service.list_recipes()) or [])

    async def get_recipe(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe detail")
        return dict(await self._maybe_await(service.get_recipe(recipe_id)) or {})

    async def get_recipe_launch_readiness(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe launch readiness")
        return dict(
            await self._maybe_await(service.get_recipe_launch_readiness(recipe_id))
            or {}
        )

    async def validate_recipe_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe dataset validation")
        return dict(
            await self._maybe_await(
                service.validate_recipe_dataset(
                    recipe_id,
                    dataset_id=dataset_id,
                    dataset=dataset,
                    run_config=run_config,
                )
            )
            or {}
        )

    async def run_benchmark(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        benchmark_name: str,
        limit: int | None = None,
        api_name: str | None = None,
        parallel: int | None = None,
        save_results: bool | None = None,
        filter_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation benchmark run")
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        if api_name is not None:
            kwargs["api_name"] = api_name
        if parallel is not None:
            kwargs["parallel"] = parallel
        if save_results is not None:
            kwargs["save_results"] = save_results
        if filter_categories is not None:
            kwargs["filter_categories"] = filter_categories
        return dict(
            await self._maybe_await(service.run_benchmark(benchmark_name, **kwargs))
            or {}
        )

    async def create_recipe_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
        dataset_id: str | None = None,
        dataset: list[dict[str, Any]] | None = None,
        run_config: dict[str, Any] | None = None,
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe run launch")
        return dict(
            await self._maybe_await(
                service.create_recipe_run(
                    recipe_id,
                    dataset_id=dataset_id,
                    dataset=dataset,
                    run_config=run_config,
                    force_rerun=force_rerun,
                )
            )
            or {}
        )

    async def get_recipe_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe run detail")
        return dict(await self._maybe_await(service.get_recipe_run(run_id)) or {})

    async def get_recipe_run_report(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation recipe run report")
        return dict(await self._maybe_await(service.get_recipe_run_report(run_id)) or {})

    async def save_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation RAG pipeline preset save")
        return dict(
            await self._maybe_await(service.save_pipeline_preset(name=name, config=config))
            or {}
        )

    async def list_pipeline_presets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation RAG pipeline preset list")
        return dict(
            await self._maybe_await(service.list_pipeline_presets(limit=limit, offset=offset))
            or {}
        )

    async def get_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation RAG pipeline preset detail")
        return dict(await self._maybe_await(service.get_pipeline_preset(name)) or {})

    async def delete_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation RAG pipeline preset delete")
        await self._maybe_await(service.delete_pipeline_preset(name))
        return {"status": "deleted", "name": name}

    async def cleanup_pipeline_collections(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation RAG pipeline cleanup")
        return dict(await self._maybe_await(service.cleanup_pipeline_collections()) or {})

    async def register_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
        events: list[str],
        secret: str | None = None,
        retry_count: int | None = 3,
        timeout_seconds: int | None = 30,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation webhook registration")
        return dict(
            await self._maybe_await(
                service.register_webhook(
                    url=url,
                    events=events,
                    secret=secret,
                    retry_count=retry_count,
                    timeout_seconds=timeout_seconds,
                )
            )
            or {}
        )

    async def list_webhooks(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        service = self._server_only_service(mode, "Evaluation webhook list")
        return list(await self._maybe_await(service.list_webhooks()) or [])

    async def unregister_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation webhook unregister")
        return dict(await self._maybe_await(service.unregister_webhook(url)) or {})

    async def test_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        service = self._server_only_service(mode, "Evaluation webhook test")
        return dict(await self._maybe_await(service.test_webhook(url=url)) or {})
