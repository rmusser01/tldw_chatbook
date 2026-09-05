"""Publication contracts for local Chunking Lab execution (ADR-118)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Self

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

    @model_serializer(mode="wrap")
    def validate_snapshot_publication(self, handler: Any) -> Any:
        """Revalidate nested mutable values before crossing a dump boundary."""
        payload = handler(self)
        type(self).model_validate(payload)
        return payload


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


class DraftState(_Snapshot):
    """Lossless authoring state with one explicit editing authority."""

    raw_json: str
    parsed_json: str | None
    parse_error: dict | None
    pending_controls: dict[str, str]
    authority: str
    record_fields: dict
    expected_record: dict | None

    @model_validator(mode="before")
    @classmethod
    def copy_values(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = json.loads(canonical_json(value))
            if value.get("authority") not in {"json", "controls", "synced"}:
                raise ValueError("Draft authority must be json, controls, or synced")
            validate_record_fields(value.get("record_fields"))
            expected = value.get("expected_record")
            if expected is not None:
                if not isinstance(expected, dict) or any(
                    type(expected.get(key)) is not kind
                    for key, kind in (("id", int), ("uuid", str), ("version", int))
                ):
                    raise ValueError("Expected record requires ID, UUID and version")
                if (
                    expected["id"] < 1
                    or expected["version"] < 1
                    or not expected["uuid"]
                ):
                    raise ValueError("Expected record identity must be nonempty")
            error = value.get("parse_error")
            if error is not None and (
                not isinstance(error, dict)
                or not isinstance(error.get("message"), str)
                or any(
                    type(error.get(key)) is not int or error[key] < 1
                    for key in ("line", "column")
                )
            ):
                raise ValueError("Parse error requires message, line and column")
            parsed = value.get("parsed_json")
            if parsed is not None:
                if not isinstance(parsed, str):
                    raise ValueError("Last parsed JSON must be text")
                json.loads(parsed)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if self.parse_error is not None:
            if self.authority != "json" or self.pending_controls:
                raise ValueError("Invalid JSON must have sole JSON authority")
        elif self.pending_controls:
            if self.authority != "controls":
                raise ValueError("Pending control text must have controls authority")
            if self.parsed_json is None:
                raise ValueError("Pending controls require a last-valid document")
        elif self.authority == "controls":
            raise ValueError("Controls authority requires pending control text")
        return self


class SampleSnapshot(_Snapshot):
    """Exact copied sample text and its non-authoritative source description."""

    sample_hash: str
    text: str
    source: dict

    @model_validator(mode="before")
    @classmethod
    def copy_source(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return json.loads(canonical_json(value))
        return value

    @model_validator(mode="after")
    def validate_hash(self) -> Self:
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sample_hash != expected:
            raise ValueError("Sample hash does not match exact UTF-8 text")
        return self


class RunRequest(_Snapshot):
    """Detached, immutable inputs captured for one batch member."""

    run_id: str
    batch_id: str
    candidate_id: str
    epoch: str
    revision: int
    sample: SampleSnapshot
    recipe: PreparedRecipe
    template_record: dict | None = None

    @model_validator(mode="before")
    @classmethod
    def copy_template_record(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("template_record") is not None:
            value = dict(value)
            value["template_record"] = json.loads(
                canonical_json(value["template_record"])
            )
        return value


class RunResult(_Snapshot):
    """Terminal result for one exact request."""

    request: RunRequest
    status: Literal["completed", "failed", "canceled", "interrupted", "limited"]
    report: ExecutionReport | None
    started_at: str
    finished_at: str
    elapsed_ms: float
    error: dict | None

    @model_validator(mode="before")
    @classmethod
    def copy_error(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("error") is not None:
            value = dict(value)
            value["error"] = json.loads(canonical_json(value["error"]))
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.elapsed_ms < 0:
            raise ValueError("Elapsed time must be nonnegative")
        if self.status == "completed" and self.report is None:
            raise ValueError("A completed result requires a report")
        if self.status != "completed" and self.report is not None:
            raise ValueError("Only completed results may contain comparison output")
        return self


class LabSession(_Snapshot):
    """Detached active Lab state; transitions publish replacement instances."""

    profile_key: str
    epoch: str
    revision: int
    content_revision: int = 0
    candidates: dict
    samples: dict
    results: dict
    batch: dict | None
    view: dict
    undo: tuple[dict, ...]

    @model_validator(mode="before")
    @classmethod
    def copy_state(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return json.loads(canonical_json(value))
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        validate_session_references(self, validate_blobs=True)
        return self


def validate_record_fields(fields: Any) -> None:
    """Validate fields used by the editor, preserving unknown extension data."""
    if not isinstance(fields, dict):
        raise ValueError("Record fields must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.
    for key in ("name", "description", "tags_text"):
        if key in fields and not isinstance(fields[key], str):
            raise ValueError(f"Record {key} must be text")
    if "tags" in fields and (
        not isinstance(fields["tags"], list)
        or any(not isinstance(tag, str) for tag in fields["tags"])
    ):
        raise ValueError("Record tags must be a list of text")


def validate_view(view: Any) -> None:
    """Check known persisted UI shapes; unknown optional values remain opaque."""
    if not isinstance(view, dict):
        raise ValueError("Session view must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.
    if "region" in view and not isinstance(view["region"], str):
        raise ValueError("View region must be text")
    choices = view.get("result_choices", {})
    if not isinstance(choices, dict) or any(
        not isinstance(value, str) for value in choices.values()
    ):
        raise ValueError("Result choices must map candidate IDs to text")
    results = view.get("results", {})
    if not isinstance(results, dict):
        raise ValueError("Results view must be an object")  # noqa: TRY004 - Pydantic validators require ValueError.
    for key in ("active_view", "detail"):
        if key in results and not isinstance(results[key], str):
            raise ValueError(f"Results {key} must be text")
    if results.get("inspected_candidate") is not None and not isinstance(
        results["inspected_candidate"], str
    ):
        raise ValueError("Inspected candidate must be text")
    selections = results.get("selections", {})
    if not isinstance(selections, dict) or any(
        not isinstance(key, str) or type(index) is not int or index < 0
        for key, index in selections.items()
    ):
        raise ValueError(
            "Chunk selections must map candidate IDs to nonnegative indices"
        )


def validate_session_references(
    session: LabSession, *, validate_blobs: bool = False
) -> None:
    """Validate the small graph; stores may reuse privately validated blob values.

    ``validate_blobs=False`` is only for owners that have already captured and
    validated samples, results, and batch requests. Public model publication
    always performs the full validation, including nested mutable payloads.
    """
    if not session.profile_key or not session.epoch:
        raise ValueError("Session profile and epoch must be nonempty")
    if session.revision < 0:
        raise ValueError("Session revision must be nonnegative")
    if not 0 <= session.content_revision <= session.revision:
        raise ValueError("Content revision must be within the session revision")
    if not 1 <= len(session.candidates) <= 2:
        raise ValueError("Lab v1 supports at most two candidates")
    validate_view(session.view)

    editable_b = 0
    for candidate_id, candidate in session.candidates.items():
        if (
            not isinstance(candidate, dict)
            or candidate.get("candidate_id") != candidate_id
        ):
            raise ValueError("Candidate keys must match stable candidate IDs")
        role = candidate.get("role")
        if role == "B" and candidate.get("editable") is True:
            editable_b += 1
            DraftState.model_validate(candidate.get("draft"))
        elif role == "A" and candidate.get("editable") is False:
            PreparedRecipe.model_validate(candidate.get("pinned_recipe"))
            if candidate.get("template_record") is not None:
                validate_record_fields(candidate["template_record"])
        else:
            raise ValueError("Candidates require one editable B and optional frozen A")
    if editable_b != 1:
        raise ValueError("Lab requires exactly one editable B candidate")

    for sample_hash, sample in session.samples.items():
        if validate_blobs:
            SampleSnapshot.model_validate(sample)
        if sample["sample_hash"] != sample_hash:
            raise ValueError("Sample keys must match sample identities")
    if session.view.get("sample_hash") not in session.samples:
        raise ValueError("Active sample must reference a retained sample")

    for run_id, result in session.results.items():
        if validate_blobs:
            RunResult.model_validate(result)
        if result["request"]["run_id"] != run_id:
            raise ValueError("Result keys must match run identities")
    for candidate in session.candidates.values():
        for field in ("current_run_id", "previous_run_id"):
            run_id = candidate.get(field)
            if run_id is not None and run_id not in session.results:
                raise ValueError(f"Candidate {field} must reference a retained result")

    if session.batch is not None:
        if session.batch.get("epoch") != session.epoch:
            raise ValueError("Batch epoch must match the session")
        requests = session.batch.get("requests")
        if not isinstance(requests, dict) or not requests:
            raise ValueError("Batch requires captured request membership")
        batch_id = session.batch.get("batch_id")
        for run_id, request in requests.items():
            if validate_blobs:
                RunRequest.model_validate(request)
            if (
                request["run_id"] != run_id
                or request["batch_id"] != batch_id
                or request["epoch"] != session.epoch
                or request["candidate_id"] not in session.candidates
            ):
                raise ValueError("Invalid captured batch request")
