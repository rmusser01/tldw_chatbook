"""Thin server-backed evaluation service around the shared API client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_from_config
from ..tldw_api import (
    CreateEvaluationRequest,
    EvaluationDatasetCreateRequest,
    EvaluationRunCreateRequest,
    TLDWAPIClient,
    UpdateEvaluationRequest,
)


class ServerEvaluationsService:
    """Wrap server evaluation endpoints with plain dict/list payloads."""

    def __init__(self, client: Optional[TLDWAPIClient]):
        self.client = client

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any]) -> "ServerEvaluationsService":
        return cls(client=build_runtime_api_client_from_config(app_config))

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
