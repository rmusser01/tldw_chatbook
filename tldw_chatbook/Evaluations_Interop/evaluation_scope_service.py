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


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "evaluations.run.webhook_delivery.local",
        "source": "local",
        "supported": False,
        "reason_code": "local_contract_missing",
        "user_message": "Local evaluation runs can persist requested webhook URLs, but do not dispatch webhook callbacks yet; observe the local run record and artifacts instead.",
        "affected_action_ids": ["evaluations.run.observe.local", "evaluations.run.update.local"],
    },
    {
        "operation_id": "evaluations.server_auxiliary_controls.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server synthetic evaluation drafts, benchmark runs, webhook administration, and recipe-run controls are unavailable in local/offline mode.",
        "affected_action_ids": [],
    },
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "evaluations.targets.list.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server evaluation API does not expose a target catalog; server runs require an explicit target_model string.",
        "affected_action_ids": ["evaluations.run.list.server", "evaluations.run.launch.server"],
    },
    {
        "operation_id": "evaluations.run.results.detail.server",
        "source": "server",
        "supported": False,
        "reason_code": "server_contract_missing",
        "user_message": "The current server unified run detail exposes summary metrics, but not sample-level result artifacts.",
        "affected_action_ids": ["evaluations.run.detail.server", "evaluations.run.observe.server"],
    },
]


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

    def _enforce_policy(self, mode: EvaluationBackend, resource: str, action: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(
            action_id=f"evaluations.{resource}.{action}.{mode.value}"
        )

    def _server_only_service(
        self,
        mode: EvaluationBackend | str | None,
        feature_name: str,
        *,
        resource: str | None = None,
        action: str | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != EvaluationBackend.SERVER:
            raise ValueError(f"{feature_name} is server-only in this Chatbook parity slice.")
        if resource is not None and action is not None:
            self._enforce_policy(normalized_mode, resource, action)
        return self._service_for_mode(normalized_mode)

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    def _dataset_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.dataset.{action}.{mode.value}"

    @staticmethod
    def _run_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.run.{action}.{mode.value}"

    @staticmethod
    def _rag_pipeline_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.rag_pipeline.{action}.{mode.value}"

    @staticmethod
    def _abtest_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.embeddings_abtest.{action}.{mode.value}"

    @staticmethod
    def _synthetic_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.synthetic.{action}.{mode.value}"

    @staticmethod
    def _benchmark_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.benchmarks.{action}.{mode.value}"

    @staticmethod
    def _webhook_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.webhooks.{action}.{mode.value}"

    @staticmethod
    def _recipe_action_id(mode: EvaluationBackend, action: str) -> str:
        return f"evaluations.recipes.{action}.{mode.value}"

    def _require_server_only_mode(
        self,
        mode: EvaluationBackend | str | None,
        *,
        capability_name: str,
    ) -> EvaluationBackend:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode != EvaluationBackend.SERVER:
            raise ValueError(f"{capability_name} is only available from the server evaluation backend.")
        return normalized_mode

    @staticmethod
    def _raise_server_targets_unsupported() -> None:
        raise NotImplementedError(
            "The current server evaluation API does not expose a target catalog; "
            "server runs require an explicit target_model string."
        )

    def list_unsupported_capabilities(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == EvaluationBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        return [dict(item) for item in _SERVER_UNSUPPORTED_CAPABILITIES]

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
        self._enforce_policy(self._dataset_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._dataset_action_id(normalized_mode, "detail"))
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
        self._enforce_policy(self._dataset_action_id(normalized_mode, "create"))
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
        self._enforce_policy(self._dataset_action_id(normalized_mode, "update"))
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
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._dataset_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.delete_evaluation(eval_id))

    async def list_datasets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._dataset_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._dataset_action_id(normalized_mode, "detail"))
        record = await self._maybe_await(self._service_for_mode(normalized_mode).get_dataset(dataset_id))
        return normalize_evaluation_dataset_record(normalized_mode.value, record)

    async def create_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        samples: list[Any] | None = None,
        format: str = "custom",
        source_path: str | None = None,
        description: str | None = None,
        metadata: Any = None,
    ) -> dict[str, Any]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._dataset_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        if normalized_mode == EvaluationBackend.LOCAL:
            if not source_path:
                raise ValueError("source_path is required when creating a local evaluation dataset.")
            record = await self._maybe_await(
                service.create_dataset(
                    name=name,
                    format=format,
                    source_path=source_path,
                    description=description,
                    metadata=metadata,
                )
            )
        else:
            record = await self._maybe_await(
                service.create_dataset(
                    name=name,
                    samples=samples or [],
                    description=description,
                    metadata=metadata,
                )
            )
        if isinstance(record, Mapping):
            resolved = record
        else:
            resolved = await self._resolve_record(record, service.get_dataset, name)
        return normalize_evaluation_dataset_record(normalized_mode.value, resolved)

    async def delete_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        dataset_id: str,
    ) -> None:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._dataset_action_id(normalized_mode, "delete"))
        await self._maybe_await(self._service_for_mode(normalized_mode).delete_dataset(dataset_id))

    async def list_targets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        provider: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        self._enforce_policy(self._run_action_id(normalized_mode, "list"))
        if normalized_mode == EvaluationBackend.SERVER:
            self._raise_server_targets_unsupported()
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
        self._enforce_policy(self._run_action_id(normalized_mode, "list"))
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
        self._enforce_policy(self._run_action_id(normalized_mode, "detail"))
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
        self._enforce_policy(self._run_action_id(normalized_mode, "launch"))
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
        self._enforce_policy(self._run_action_id(normalized_mode, "observe"))
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
        self._enforce_policy(self._run_action_id(normalized_mode, "update"))
        payload = await self._maybe_await(self._service_for_mode(normalized_mode).cancel_run(run_id))
        result = dict(payload or {})
        result.setdefault("backend", normalized_mode.value)
        result.setdefault("id", run_id)
        return result

    async def create_or_update_rag_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="RAG pipeline preset administration",
        )
        self._enforce_policy(self._rag_pipeline_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.create_or_update_rag_pipeline_preset(name=name, config=config)
            )
            or {}
        )

    async def list_rag_pipeline_presets(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="RAG pipeline preset administration",
        )
        self._enforce_policy(self._rag_pipeline_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(service.list_rag_pipeline_presets(limit=limit, offset=offset))
            or {}
        )

    async def get_rag_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="RAG pipeline preset administration",
        )
        self._enforce_policy(self._rag_pipeline_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_rag_pipeline_preset(name)) or {})

    async def delete_rag_pipeline_preset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
    ) -> None:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="RAG pipeline preset administration",
        )
        self._enforce_policy(self._rag_pipeline_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        await self._maybe_await(service.delete_rag_pipeline_preset(name))

    async def cleanup_rag_pipeline(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="RAG pipeline cleanup",
        )
        self._enforce_policy(self._rag_pipeline_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.cleanup_rag_pipeline()) or {})

    async def create_embeddings_abtest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        name: str,
        config: dict[str, Any],
        run_immediately: bool | None = False,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.create_embeddings_abtest(
                    name=name,
                    config=config,
                    run_immediately=run_immediately,
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
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.run_embeddings_abtest(test_id, config=config)) or {})

    async def get_embeddings_abtest_status(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_embeddings_abtest_status(test_id)) or {})

    async def get_embeddings_abtest_results(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
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
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.get_embeddings_abtest_significance(test_id, metric=metric)
            )
            or {}
        )

    async def export_embeddings_abtest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
        format: str = "json",
    ) -> Any:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "export"))
        service = self._service_for_mode(normalized_mode)
        return await self._maybe_await(service.export_embeddings_abtest(test_id, format=format))

    async def delete_embeddings_abtest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        test_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Embeddings A/B test administration",
        )
        self._enforce_policy(self._abtest_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.delete_embeddings_abtest(test_id)) or {})

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
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Synthetic evaluation draft administration",
        )
        self._enforce_policy(self._synthetic_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
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
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Synthetic evaluation draft administration",
        )
        self._enforce_policy(self._synthetic_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
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
        notes: str | None = None,
        action_payload: dict[str, Any] | None = None,
        resulting_review_state: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Synthetic evaluation draft administration",
        )
        self._enforce_policy(self._synthetic_action_id(normalized_mode, "update"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.review_synthetic_sample(
                    sample_id,
                    action=action,
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
        promotion_reason: str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Synthetic evaluation draft administration",
        )
        self._enforce_policy(self._synthetic_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.promote_synthetic_samples(
                    sample_ids=sample_ids,
                    dataset_name=dataset_name,
                    dataset_description=dataset_description,
                    dataset_metadata=dataset_metadata,
                    promotion_reason=promotion_reason,
                )
            )
            or {}
        )

    async def list_benchmarks(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation benchmark administration",
        )
        self._enforce_policy(self._benchmark_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.list_benchmarks()) or {})

    async def get_benchmark(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        benchmark_name: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation benchmark administration",
        )
        self._enforce_policy(self._benchmark_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_benchmark(benchmark_name)) or {})

    async def run_benchmark(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        benchmark_name: str,
        limit: int | None = None,
        api_name: str = "openai",
        parallel: int = 4,
        save_results: bool = True,
        filter_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation benchmark administration",
        )
        self._enforce_policy(self._benchmark_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return dict(
            await self._maybe_await(
                service.run_benchmark(
                    benchmark_name,
                    limit=limit,
                    api_name=api_name,
                    parallel=parallel,
                    save_results=save_results,
                    filter_categories=filter_categories,
                )
            )
            or {}
        )

    async def register_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
        events: list[str],
        secret: str | None = None,
        retry_count: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation webhook administration",
        )
        self._enforce_policy(self._webhook_action_id(normalized_mode, "create"))
        service = self._service_for_mode(normalized_mode)
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
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation webhook administration",
        )
        self._enforce_policy(self._webhook_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return list(await self._maybe_await(service.list_webhooks()) or [])

    async def unregister_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation webhook administration",
        )
        self._enforce_policy(self._webhook_action_id(normalized_mode, "delete"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.unregister_webhook(url)) or {})

    async def test_webhook(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        url: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation webhook administration",
        )
        self._enforce_policy(self._webhook_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.test_webhook(url)) or {})

    async def list_recipe_manifests(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "list"))
        service = self._service_for_mode(normalized_mode)
        return list(await self._maybe_await(service.list_recipe_manifests()) or [])

    async def get_recipe_manifest(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "detail"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_recipe_manifest(recipe_id)) or {})

    async def get_recipe_launch_readiness(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_recipe_launch_readiness(recipe_id)) or {})

    async def validate_recipe_dataset(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
        dataset_id: str | None = None,
        dataset: Any = None,
        run_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
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

    async def create_recipe_run(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        recipe_id: str,
        dataset_id: str | None = None,
        dataset: Any = None,
        run_config: dict[str, Any] | None = None,
        force_rerun: bool = False,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "launch"))
        service = self._service_for_mode(normalized_mode)
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
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_recipe_run(run_id)) or {})

    async def get_recipe_run_report(
        self,
        *,
        mode: EvaluationBackend | str | None = None,
        run_id: str,
    ) -> dict[str, Any]:
        normalized_mode = self._require_server_only_mode(
            mode,
            capability_name="Evaluation recipe controls",
        )
        self._enforce_policy(self._recipe_action_id(normalized_mode, "observe"))
        service = self._service_for_mode(normalized_mode)
        return dict(await self._maybe_await(service.get_recipe_run_report(run_id)) or {})
