"""Normalization helpers for the shared local/server evaluations seam."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


RESERVED_LOCAL_METADATA_KEY = "__tldw_eval_metadata__"


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump(mode="json")
    return value


def _as_mapping(value: Any) -> dict[str, Any]:
    value = _model_dump(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_json_mapping(value: Any) -> dict[str, Any]:
    value = _model_dump(value)
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


def _clean_timestamp(*values: Any) -> str | None:
    for value in values:
        if value in (None, ""):
            continue
        return str(value)
    return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_dataset_samples(samples: Any) -> list[dict[str, Any]] | None:
    if samples in (None, ""):
        return None
    if not isinstance(samples, list):
        return None

    normalized: list[dict[str, Any]] = []
    for sample in samples:
        data = _as_mapping(sample)
        if not data:
            continue
        normalized.append(
            {
                "input": data.get("input"),
                "expected": data.get("expected"),
                "metadata": _as_mapping(data.get("metadata")),
            }
        )
    return normalized


def _normalize_local_eval_payload(data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_spec = _coerce_json_mapping(data.get("config_data"))
    metadata = _as_mapping(eval_spec.pop(RESERVED_LOCAL_METADATA_KEY, None))

    config_format = data.get("config_format")
    if config_format not in (None, ""):
        metadata.setdefault("config_format", str(config_format))

    return eval_spec, metadata


def _normalize_progress(value: Any, *, completed_samples: Any = None, total_samples: Any = None) -> dict[str, Any] | None:
    payload = _as_mapping(value)
    completed = _safe_int(payload.get("completed_samples"))
    total = _safe_int(payload.get("total_samples"))

    if completed is None:
        completed = _safe_int(completed_samples)
    if total is None:
        total = _safe_int(total_samples)

    percent = _safe_float(payload.get("percent_complete"))
    if percent is None and completed is not None and total not in (None, 0):
        percent = round((float(completed) / float(total)) * 100.0, 2)

    current_sample = _safe_int(payload.get("current_sample"))
    estimated_completion = _clean_timestamp(payload.get("estimated_completion"))

    if completed is None and total is None and percent is None and current_sample is None and estimated_completion is None:
        return None

    return {
        "completed_samples": completed or 0,
        "total_samples": total or 0,
        "percent_complete": percent or 0.0,
        "current_sample": current_sample,
        "estimated_completion": estimated_completion,
    }


def _normalize_results(value: Any) -> dict[str, Any] | None:
    payload = _coerce_json_mapping(value)
    if not payload:
        return None

    normalized: dict[str, Any] = {}
    for key, raw_value in payload.items():
        if isinstance(raw_value, Mapping) and "value" in raw_value:
            normalized[key] = raw_value.get("value")
        else:
            normalized[key] = raw_value
    return normalized


def normalize_evaluation_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _as_mapping(payload)
    if backend == "local":
        eval_spec, metadata = _normalize_local_eval_payload(data)
        eval_type = data.get("task_type")
    else:
        eval_spec = _coerce_json_mapping(data.get("eval_spec"))
        metadata = _as_mapping(data.get("metadata"))
        eval_type = data.get("eval_type")

    backing_id = data.get("id") or data.get("task_id")

    return {
        "record_id": f"{backend}:evaluation:{backing_id}",
        "record_type": "evaluation",
        "backend": backend,
        "backing_id": backing_id,
        "name": data.get("name") or "",
        "description": data.get("description") or "",
        "eval_type": eval_type,
        "eval_spec": eval_spec,
        "dataset_id": data.get("dataset_id"),
        "created_at": _clean_timestamp(data.get("created_at"), data.get("created")),
        "updated_at": _clean_timestamp(data.get("updated_at"), data.get("updated")),
        "version": _safe_int(data.get("version")),
        "metadata": metadata,
        "client_id": data.get("client_id"),
        "created_by": data.get("created_by"),
    }


def normalize_evaluation_dataset_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _as_mapping(payload)
    samples = _normalize_dataset_samples(data.get("samples"))
    sample_count = _safe_int(data.get("sample_count"))
    if sample_count is None and isinstance(samples, list):
        sample_count = len(samples)

    return {
        "record_id": f"{backend}:evaluation_dataset:{data.get('id')}",
        "record_type": "evaluation_dataset",
        "backend": backend,
        "backing_id": data.get("id"),
        "name": data.get("name") or "",
        "description": data.get("description") or "",
        "sample_count": sample_count,
        "samples": samples,
        "format": data.get("format"),
        "source_path": data.get("source_path"),
        "created_at": _clean_timestamp(data.get("created_at"), data.get("created")),
        "updated_at": _clean_timestamp(data.get("updated_at")),
        "version": _safe_int(data.get("version")),
        "created_by": data.get("created_by"),
        "metadata": _as_mapping(data.get("metadata")),
        "client_id": data.get("client_id"),
    }


def normalize_evaluation_target_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _as_mapping(payload)
    provider = data.get("provider")
    model_id = data.get("model_id")
    name = data.get("name") or model_id or data.get("id") or ""

    if provider and model_id:
        target_model = f"{provider}:{model_id}"
    else:
        target_model = model_id or name

    display_name = name
    if target_model and target_model != name:
        display_name = f"{name} ({target_model})"

    return {
        "record_id": f"{backend}:evaluation_target:{data.get('id')}",
        "record_type": "evaluation_target",
        "backend": backend,
        "backing_id": data.get("id"),
        "name": name,
        "display_name": display_name,
        "provider": provider,
        "model_id": model_id,
        "target_model": target_model,
        "config": _coerce_json_mapping(data.get("config")),
        "created_at": _clean_timestamp(data.get("created_at"), data.get("created")),
        "updated_at": _clean_timestamp(data.get("updated_at"), data.get("updated")),
        "version": _safe_int(data.get("version")),
        "client_id": data.get("client_id"),
    }


def normalize_evaluation_run_record(backend: str, payload: Any) -> dict[str, Any]:
    data = _as_mapping(payload)
    progress = _normalize_progress(
        data.get("progress"),
        completed_samples=data.get("completed_samples"),
        total_samples=data.get("total_samples"),
    )
    results = _normalize_results(data.get("results") or data.get("metrics_summary"))
    config = _coerce_json_mapping(data.get("config") or data.get("config_overrides"))
    usage = _coerce_json_mapping(data.get("usage"))

    if backend == "local":
        evaluation_id = data.get("task_id")
        target_model = data.get("target_model") or data.get("model_name") or data.get("model_id")
    else:
        evaluation_id = data.get("eval_id")
        target_model = data.get("target_model")

    return {
        "record_id": f"{backend}:evaluation_run:{data.get('id')}",
        "record_type": "evaluation_run",
        "backend": backend,
        "backing_id": data.get("id"),
        "evaluation_id": evaluation_id,
        "name": data.get("name") or "",
        "status": data.get("status"),
        "target_model": target_model,
        "created_at": _clean_timestamp(data.get("created_at"), data.get("created")),
        "started_at": _clean_timestamp(data.get("started_at"), data.get("start_time")),
        "completed_at": _clean_timestamp(data.get("completed_at"), data.get("end_time")),
        "progress": progress,
        "error_message": data.get("error_message"),
        "results": results,
        "usage": usage or None,
        "config": config or None,
        "metadata": _as_mapping(data.get("metadata")),
        "client_id": data.get("client_id"),
        "version": _safe_int(data.get("version")),
    }
