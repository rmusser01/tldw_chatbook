"""Thin local adapter for evaluation tasks, datasets, and runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .evaluation_normalizers import RESERVED_LOCAL_DATASET_SAMPLES_KEY, RESERVED_LOCAL_METADATA_KEY


class LocalEvaluationsService:
    """Wrap the local eval DB with a compat-oriented evaluation surface."""

    def __init__(
        self,
        db: Any,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ):
        self.db = db
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def configure_notification_dispatch(
        self,
        *,
        notification_dispatch_service: Any = None,
        notification_app: Any = None,
    ) -> None:
        self.notification_dispatch_service = notification_dispatch_service
        self.notification_app = notification_app

    def _require_db(self) -> Any:
        if self.db is None:
            raise ValueError("Local evaluations DB is unavailable.")
        return self.db

    def _dispatch_local_notification(
        self,
        *,
        title: str,
        message: str,
        source_entity_id: str | None,
        source_entity_kind: str,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        dispatcher = getattr(self, "notification_dispatch_service", None)
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            return None
        try:
            return dispatch(
                app=getattr(self, "notification_app", None),
                category="evaluations",
                title=title,
                message=message,
                severity=severity,
                source_backend="local",
                source_entity_id=source_entity_id,
                source_entity_kind=source_entity_kind,
                payload=payload or {},
            )
        except Exception:
            return None

    def _model_dump(self, value: Any) -> Any:
        if hasattr(value, "model_dump") and callable(value.model_dump):
            return value.model_dump(mode="json")
        return value

    def _as_mapping(self, value: Any) -> dict[str, Any]:
        value = self._model_dump(value)
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _coerce_json_mapping(self, value: Any) -> dict[str, Any]:
        value = self._model_dump(value)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return {}
            if isinstance(parsed, Mapping):
                return dict(parsed)
        return {}

    def _coerce_json_list(self, value: Any) -> list[Any]:
        value = self._model_dump(value)
        if isinstance(value, list):
            return [self._model_dump(item) for item in value]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                return []
            if isinstance(parsed, list):
                return [self._model_dump(item) for item in parsed]
        return []

    def _normalize_dataset_override(self, dataset_override: Any) -> dict[str, Any]:
        value = self._model_dump(dataset_override)
        if isinstance(value, Mapping):
            normalized = dict(value)
            samples = self._coerce_json_list(normalized.get("samples"))
            if "samples" in normalized:
                normalized["samples"] = samples
            metadata = self._as_mapping(normalized.get("metadata"))
            if "metadata" in normalized or metadata:
                normalized["metadata"] = metadata
            return normalized
        if isinstance(value, list):
            return {"samples": self._coerce_json_list(value)}
        if isinstance(value, str) and value.strip():
            return {"dataset_name": value.strip()}
        raise ValueError("Local evaluation dataset_override must be a mapping, sample list, or dataset path string.")

    def _build_run_config_overrides(
        self,
        *,
        config: Any = None,
        dataset_override: Any = None,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        overrides = self._coerce_json_mapping(config)
        if dataset_override is not None:
            normalized_override = self._normalize_dataset_override(dataset_override)
            overrides["dataset_override"] = normalized_override
            for key in ("dataset_name", "dataset_path", "source_path", "path"):
                if normalized_override.get(key):
                    overrides.setdefault("dataset_name", normalized_override[key])
                    break
        if webhook_url is not None:
            normalized_webhook_url = str(webhook_url).strip()
            if not normalized_webhook_url:
                raise ValueError("Local evaluation webhook_url cannot be blank.")
            overrides["webhook_url"] = normalized_webhook_url
        return overrides

    def _split_task_payload(self, task_record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        config_data = self._coerce_json_mapping(task_record.get("config_data"))
        metadata = self._as_mapping(config_data.pop(RESERVED_LOCAL_METADATA_KEY, None))
        return config_data, metadata

    def _build_task_payload(self, *, eval_spec: Any, metadata: Any = None) -> dict[str, Any]:
        payload = self._coerce_json_mapping(eval_spec)
        metadata_payload = self._as_mapping(metadata)
        if metadata_payload:
            payload[RESERVED_LOCAL_METADATA_KEY] = metadata_payload
        return payload

    def _flatten_metrics(self, metrics: Any) -> dict[str, Any]:
        payload = self._coerce_json_mapping(metrics)
        flattened: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Mapping) and "value" in value:
                flattened[key] = value.get("value")
            else:
                flattened[key] = value
        return flattened

    def _normalize_inline_samples(self, samples: Any) -> list[dict[str, Any]]:
        samples = self._model_dump(samples)
        if samples is None:
            return []
        if not isinstance(samples, list):
            raise ValueError("Local evaluation dataset samples must be a list.")
        return [self._as_mapping(sample) for sample in samples]

    def _enrich_dataset(self, dataset_record: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(dataset_record)
        metadata = self._as_mapping(record.get("metadata"))
        samples = metadata.pop(RESERVED_LOCAL_DATASET_SAMPLES_KEY, None)
        sample_count = metadata.pop("sample_count", None)
        metadata.pop("inline_samples", None)
        record["metadata"] = metadata
        if isinstance(samples, list):
            record["samples"] = [self._as_mapping(sample) for sample in samples]
            record["sample_count"] = len(record["samples"])
        elif sample_count is not None:
            record["sample_count"] = sample_count
        return record

    def _target_model_string(self, model_record: Mapping[str, Any]) -> str:
        provider = model_record.get("provider")
        model_id = model_record.get("model_id")
        if provider and model_id:
            return f"{provider}:{model_id}"
        return str(model_id or model_record.get("name") or model_record.get("id") or "")

    def _find_model(self, *, target_id: str | None = None, target_model: str | None = None) -> dict[str, Any]:
        db = self._require_db()

        if target_id:
            record = db.get_model(target_id)
            if record:
                return dict(record)

        models = list(db.list_models(limit=500, offset=0) or [])
        normalized_target = str(target_model or "").strip().lower()
        normalized_target_id = str(target_id or "").strip().lower()

        for record in models:
            candidate = dict(record)
            if normalized_target_id and str(candidate.get("id") or "").strip().lower() == normalized_target_id:
                return candidate
            if normalized_target:
                aliases = {
                    str(candidate.get("model_id") or "").strip().lower(),
                    str(candidate.get("name") or "").strip().lower(),
                    self._target_model_string(candidate).strip().lower(),
                }
                aliases.discard("")
                if normalized_target in aliases:
                    return candidate

        raise ValueError("Local evaluation target was not found.")

    def _enrich_run(self, run_record: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(run_record)
        enriched["config_overrides"] = self._coerce_json_mapping(enriched.get("config_overrides"))

        model_id = enriched.get("model_id")
        if model_id:
            model_record = self._require_db().get_model(str(model_id))
            if model_record:
                enriched.setdefault("provider", model_record.get("provider"))
                enriched.setdefault("configured_model_id", model_record.get("model_id"))
                enriched.setdefault("target_model", self._target_model_string(model_record))
                if not enriched.get("model_name"):
                    enriched["model_name"] = model_record.get("name") or model_record.get("model_id")

        metrics_summary = self._flatten_metrics(enriched.get("metrics_summary"))
        if not metrics_summary:
            metrics_summary = self._flatten_metrics(
                self._require_db().get_run_metrics(str(enriched.get("id")))
            )
        enriched["metrics_summary"] = metrics_summary
        return enriched

    def list_evaluations(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        eval_type: str | None = None,
    ) -> list[dict[str, Any]]:
        return list(
            self._require_db().list_tasks(task_type=eval_type, limit=limit, offset=offset) or []
        )

    def get_evaluation(self, eval_id: str) -> dict[str, Any]:
        record = self._require_db().get_task(eval_id)
        if not record:
            raise ValueError(f"Local evaluation '{eval_id}' was not found.")
        return dict(record)

    def create_evaluation(
        self,
        *,
        name: str,
        eval_type: str,
        eval_spec: Any,
        description: str | None = None,
        dataset_id: str | None = None,
        dataset: Any = None,
        metadata: Any = None,
    ) -> str:
        if dataset is not None and not dataset_id:
            dataset_id = self.create_dataset(
                name=f"{name}_dataset",
                samples=dataset,
                description=f"Inline dataset for {name}",
            )
        task_id = self._require_db().create_task(
            name=name,
            description=description,
            task_type=eval_type,
            config_format="custom",
            config_data=self._build_task_payload(eval_spec=eval_spec, metadata=metadata),
            dataset_id=dataset_id,
        )
        self._dispatch_local_notification(
            title="Local evaluation created",
            message=f"Local evaluation created: {name}",
            source_entity_id=str(task_id),
            source_entity_kind="evaluation",
            payload={"action": "evaluation_created", "evaluation_id": str(task_id), "eval_type": eval_type},
        )
        return task_id

    def update_evaluation(
        self,
        eval_id: str,
        *,
        description: str | None = None,
        eval_spec: Any = None,
        metadata: Any = None,
    ) -> bool:
        existing = self.get_evaluation(eval_id)
        update_kwargs: dict[str, Any] = {}

        if description is not None:
            update_kwargs["description"] = description

        if eval_spec is not None or metadata is not None:
            existing_spec, existing_metadata = self._split_task_payload(existing)
            next_spec = self._coerce_json_mapping(eval_spec) if eval_spec is not None else existing_spec
            next_metadata = self._as_mapping(metadata) if metadata is not None else existing_metadata
            update_kwargs["config_data"] = self._build_task_payload(
                eval_spec=next_spec,
                metadata=next_metadata,
            )

        if not update_kwargs:
            return True

        updated = self._require_db().update_task(eval_id, **update_kwargs)
        if not updated:
            raise ValueError(f"Local evaluation '{eval_id}' could not be updated.")
        return updated

    def delete_evaluation(self, eval_id: str) -> None:
        deleted = self._require_db().delete_task(eval_id)
        if not deleted:
            raise ValueError(f"Local evaluation '{eval_id}' could not be deleted.")

    def list_datasets(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        return [
            self._enrich_dataset(record)
            for record in list(self._require_db().list_datasets(limit=limit, offset=offset) or [])
        ]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        record = self._require_db().get_dataset(dataset_id)
        if not record:
            raise ValueError(f"Local evaluation dataset '{dataset_id}' was not found.")
        return self._enrich_dataset(record)

    def create_dataset(
        self,
        *,
        name: str,
        samples: list[dict[str, Any]] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        format: str | None = None,
        source_path: str | None = None,
    ) -> str:
        metadata_payload = self._as_mapping(metadata)
        normalized_samples = self._normalize_inline_samples(samples)
        if normalized_samples:
            metadata_payload[RESERVED_LOCAL_DATASET_SAMPLES_KEY] = normalized_samples
            metadata_payload["sample_count"] = len(normalized_samples)
            metadata_payload["inline_samples"] = True
            source_path = source_path or f"inline:{name}"
        if not source_path:
            raise ValueError("Local evaluation dataset creation requires source_path or inline samples.")
        dataset_id = self._require_db().create_dataset(
            name=name,
            format=format or "custom",
            source_path=source_path,
            description=description,
            metadata=metadata_payload,
        )
        self._dispatch_local_notification(
            title="Local evaluation dataset created",
            message=f"Local evaluation dataset created: {name}",
            source_entity_id=str(dataset_id),
            source_entity_kind="evaluation_dataset",
            payload={
                "action": "dataset_created",
                "dataset_id": str(dataset_id),
                "sample_count": len(normalized_samples),
            },
        )
        return dataset_id

    def update_dataset(
        self,
        dataset_id: str,
        *,
        name: str | None = None,
        samples: list[dict[str, Any]] | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
        format: str | None = None,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        db = self._require_db()
        existing = db.get_dataset(dataset_id)
        if not existing:
            raise ValueError(f"Local evaluation dataset '{dataset_id}' was not found.")

        existing_metadata = self._as_mapping(existing.get("metadata"))
        existing_samples = existing_metadata.get(RESERVED_LOCAL_DATASET_SAMPLES_KEY)
        existing_sample_count = existing_metadata.get("sample_count")
        existing_inline_samples = existing_metadata.get("inline_samples")

        metadata_payload = self._as_mapping(metadata) if metadata is not None else dict(existing_metadata)
        metadata_payload.pop(RESERVED_LOCAL_DATASET_SAMPLES_KEY, None)
        metadata_payload.pop("sample_count", None)
        metadata_payload.pop("inline_samples", None)

        if samples is not None:
            normalized_samples = self._normalize_inline_samples(samples)
            metadata_payload[RESERVED_LOCAL_DATASET_SAMPLES_KEY] = normalized_samples
            metadata_payload["sample_count"] = len(normalized_samples)
            metadata_payload["inline_samples"] = True
            source_path = source_path or existing.get("source_path") or f"inline:{name or existing.get('name') or dataset_id}"
        elif isinstance(existing_samples, list):
            preserved_samples = [self._as_mapping(sample) for sample in existing_samples]
            metadata_payload[RESERVED_LOCAL_DATASET_SAMPLES_KEY] = preserved_samples
            metadata_payload["sample_count"] = len(preserved_samples)
            metadata_payload["inline_samples"] = True
        elif existing_sample_count is not None:
            metadata_payload["sample_count"] = existing_sample_count
            if existing_inline_samples is not None:
                metadata_payload["inline_samples"] = existing_inline_samples

        updated = db.update_dataset(
            dataset_id,
            name=name,
            description=description,
            format=format,
            source_path=source_path,
            metadata=metadata_payload,
        )
        if not updated:
            raise ValueError(f"Local evaluation dataset '{dataset_id}' could not be updated.")

        self._dispatch_local_notification(
            title="Local evaluation dataset updated",
            message=f"Local evaluation dataset updated: {name or existing.get('name') or dataset_id}",
            source_entity_id=str(dataset_id),
            source_entity_kind="evaluation_dataset",
            payload={"action": "dataset_updated", "dataset_id": str(dataset_id)},
        )
        return self.get_dataset(dataset_id)

    def delete_dataset(self, dataset_id: str) -> None:
        deleted = self._require_db().delete_dataset(dataset_id)
        if not deleted:
            raise ValueError(f"Local evaluation dataset '{dataset_id}' could not be deleted.")

    def list_targets(
        self,
        *,
        provider: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return list(self._require_db().list_models(provider=provider, limit=limit, offset=offset) or [])

    def list_runs(
        self,
        *,
        eval_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        records = self._require_db().list_runs(
            status=status,
            task_id=eval_id,
            limit=limit,
            offset=offset,
        )
        return [self._enrich_run(record) for record in list(records or [])]

    def get_run(self, run_id: str) -> dict[str, Any]:
        record = self._require_db().get_run(run_id)
        if not record:
            raise ValueError(f"Local evaluation run '{run_id}' was not found.")
        return self._enrich_run(record)

    def create_run(
        self,
        eval_id: str,
        *,
        target_id: str | None = None,
        target_model: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        dataset_override: Any = None,
        webhook_url: str | None = None,
    ) -> dict[str, Any]:
        model_record = self._find_model(target_id=target_id, target_model=target_model)
        run_id = self._require_db().create_run(
            name=run_name or f"{self.get_evaluation(eval_id).get('name') or 'eval_run'}",
            task_id=eval_id,
            model_id=str(model_record.get("id")),
            config_overrides=self._build_run_config_overrides(
                config=config,
                dataset_override=dataset_override,
                webhook_url=webhook_url,
            ),
        )
        record = self._require_db().get_run(run_id)
        if not record:
            raise ValueError(f"Local evaluation run '{run_id}' was not found after creation.")
        created_run = dict(record)
        created_run["config_overrides"] = self._coerce_json_mapping(created_run.get("config_overrides"))
        created_run.setdefault("target_model", self._target_model_string(model_record))
        self._dispatch_local_notification(
            title="Local evaluation run created",
            message=f"Local evaluation run created: {created_run.get('name') or run_id}",
            source_entity_id=str(run_id),
            source_entity_kind="evaluation_run",
            payload={
                "action": "run_created",
                "run_id": str(run_id),
                "evaluation_id": str(eval_id),
                "target_model": created_run.get("target_model"),
                "status": created_run.get("status"),
            },
        )
        return created_run

    def get_run_artifacts(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        return {
            "run": run,
            "metrics": self._flatten_metrics(self._require_db().get_run_metrics(run_id)),
            "results": list(self._require_db().get_run_results(run_id, limit=1000, offset=0) or []),
            "detail_available": True,
        }

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        self._require_db().update_run_status(run_id, "cancelled")
        self._dispatch_local_notification(
            title="Local evaluation run cancelled",
            message=f"Local evaluation run cancelled: {run_id}",
            source_entity_id=str(run_id),
            source_entity_kind="evaluation_run",
            payload={"action": "run_cancelled", "run_id": str(run_id), "status": "cancelled"},
        )
        return {"id": run_id, "status": "cancelled"}
