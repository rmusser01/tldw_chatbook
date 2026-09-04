"""Publication contracts for local Chunking Lab execution (ADR-118)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, model_serializer, model_validator


def canonical_json(value: Any) -> str:
    """Serialize JSON without coercing keys, nonfinite numbers, or objects."""

    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("Only JSON values can be published")

    check(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class _Snapshot(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", revalidate_instances="always"
    )


class RuntimeIdentity(_Snapshot):
    """Content-free identity of the backend and its already-local assets."""

    backend: str
    engine_version: str
    execution_version: str
    assets: tuple[dict, ...] = ()

    @model_serializer(mode="wrap")
    def validate_publication(self, handler: Any) -> Any:
        self.copy_assets({"assets": self.assets})
        return handler(self)

    @model_validator(mode="before")
    @classmethod
    def copy_assets(cls, value: Any) -> Any:
        if isinstance(value, dict) and "assets" in value:
            value = dict(value)
            value["assets"] = json.loads(canonical_json(value["assets"]))
            for asset in value["assets"]:
                if not isinstance(asset, dict) or set(asset) != {
                    "kind",
                    "name",
                    "version",
                    "content_digest",
                }:
                    raise ValueError(
                        "Assets require kind, name, version, and content_digest only"
                    )
                if any(
                    not isinstance(entry, str) or not entry for entry in asset.values()
                ):
                    raise ValueError("Asset identities must contain nonempty strings")
        return value


class PreparedRecipe(_Snapshot):
    """Immutable canonical authored and executable documents."""

    authored_json: str
    effective_json: str
    runtime: RuntimeIdentity
    recipe_hash: str


class ExecutionReport(_Snapshot):
    """Detached structured output; revalidate on each publication boundary."""

    chunks: tuple[dict, ...]
    transformed_text: str
    diagnostics: tuple[dict, ...] = ()

    @model_serializer(mode="wrap")
    def validate_publication(self, handler: Any) -> Any:
        self.copy_records(
            {
                "chunks": self.chunks,
                "transformed_text": self.transformed_text,
                "diagnostics": self.diagnostics,
            }
        )
        return handler(self)

    @model_validator(mode="before")
    @classmethod
    def copy_records(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = json.loads(canonical_json(value))
            for chunk in value.get("chunks", []):
                if not isinstance(chunk, dict) or not isinstance(
                    chunk.get("text"), str
                ):
                    raise ValueError("Chunk text must be a string")  # noqa: TRY004 - Pydantic validators require ValueError.
                for field in ("metadata", "provenance"):
                    if not isinstance(chunk.get(field), dict):
                        raise ValueError(f"Chunk {field} must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.
                span = chunk.get("span")
                if span is not None and (
                    not isinstance(span, dict)
                    or span.get("coordinate_space") not in ("source", "transformed")
                    or type(span.get("start")) is not int
                    or type(span.get("end")) is not int
                    or not 0 <= span["start"] <= span["end"]
                ):
                    raise ValueError("Invalid verified chunk span")
        return value
