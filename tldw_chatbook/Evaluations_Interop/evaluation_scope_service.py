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
