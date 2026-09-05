"""Pure captured-result measurements and comparison (ADR-118).

Call whole-result functions off the UI loop. Token counts are supplied by an
explicitly selected, available local measurement tokenizer; this module never
loads assets and never treats a chunking tokenizer as a measurement identity.
"""

from __future__ import annotations

import json
import math
import statistics
from typing import Any, NotRequired, TypedDict

from .lab_models import RunResult


class CountDistribution(TypedDict):
    """Chunk-size totals and quantiles; empty output has no quantiles."""

    minimum: int | None
    median: int | float | None
    p95: int | None
    maximum: int | None
    total: int


class MethodBudget(TypedDict):
    """Captured method budget, measured only for recognized counting units."""

    method: str | None
    limit: int | float | None
    unit: str | None
    oversized_chunks: int | None


class ResultSummary(TypedDict):
    """Output measurements and their definitions for one captured result."""

    status: str
    chunk_count: int
    characters: CountDistribution
    words: CountDistribution
    character_sizes: tuple[int, ...]
    word_sizes: tuple[int, ...]
    tokens: CountDistribution | None
    measurement_id: str | None
    budget: MethodBudget
    expansion_ratio: float | None
    overlap_characters: int | None
    elapsed_ms_observation: float
    definitions: dict[str, str]


class ComparisonDeltas(TypedDict):
    """B minus A counts, with tokens present only for matching measurements."""

    chunk_count: int
    characters: int
    words: int
    tokens: NotRequired[int]


def comparison_reason(a: RunResult, b: RunResult) -> str | None:
    """Return why direct comparison is unavailable, or None when compatible."""
    if a.status != "completed" or b.status != "completed":
        return "Comparison requires two successful results. Run both to compare."
    if a.request.sample.sample_hash != b.request.sample.sample_hash:
        return "Sample content differs. Run both on the same sample."
    for field in ("backend", "engine_version", "execution_version"):
        if getattr(a.request.recipe.runtime, field) != getattr(
            b.request.recipe.runtime, field
        ):
            return f"Captured {field.replace('_', ' ')} differs. Run both to compare."
    return None


def _distribution(values: tuple[int, ...]) -> CountDistribution:
    ordered = sorted(values)
    return {
        "minimum": ordered[0] if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
        "maximum": ordered[-1] if ordered else None,
        "total": sum(ordered),
    }


def chunk_mapping(result: RunResult, index: int) -> dict:
    """Expose only a verified, in-bounds map; never search for matching text."""
    chunk = result.report.chunks[index]
    span = chunk.get("span")
    mapping = chunk.get("provenance", {}).get("mapping", {})
    reason = mapping.get("reason", "Execution did not provide a verified map.")
    if span and mapping.get("status") == "exact":
        space = span["coordinate_space"]
        text = (
            result.request.sample.text
            if space == "source"
            else result.report.transformed_text
        )
        start, end = span["start"], span["end"]
        if 0 <= start <= end <= len(text) and text[start:end] == chunk["text"]:
            return {
                "coordinate_space": space,
                "start": start,
                "end": end,
                "reason": None
                if space == "source"
                else "Original-source alignment unavailable; coordinates refer to transformed text.",
            }
        reason = "Captured span does not match the captured text."
    return {"coordinate_space": None, "start": None, "end": None, "reason": reason}


def linked_chunks(a: RunResult, index: int, b: RunResult) -> tuple[int, ...]:
    """Find overlapping original-source spans only for compatible results."""
    if comparison_reason(a, b) is not None:
        return ()
    selected = chunk_mapping(a, index)
    if selected["coordinate_space"] != "source":
        return ()
    return tuple(
        i
        for i in range(len(b.report.chunks))
        if (span := chunk_mapping(b, i))["coordinate_space"] == "source"
        and span["start"] < selected["end"]
        and selected["start"] < span["end"]
    )


def summarize_result(
    result: RunResult,
    *,
    token_counts: tuple[int, ...] | None = None,
    measurement_id: str | None = None,
) -> ResultSummary:
    """Measure captured output shape using code points and whitespace words.

    Args:
        result: Immutable execution evidence.
        token_counts: Counts from an explicitly available local tokenizer.
        measurement_id: Stable measurement tokenizer identity, separate from execution.

    Returns:
        Counts, nearest-rank p95, labeled budgets, expansion and verified overlap.

    Raises:
        ValueError: Token counts lack an identity or do not match the chunks.
    """
    chunks = result.report.chunks if result.report else ()
    if token_counts is not None:
        if not measurement_id or not measurement_id.strip():
            raise ValueError("Token counts require a measurement identity")
        if len(token_counts) != len(chunks) or any(
            type(n) is not int or n < 0 for n in token_counts
        ):
            raise ValueError(
                "Token counts must contain one nonnegative count per chunk"
            )
    elif measurement_id is not None:
        raise ValueError("Measurement identity requires token counts")
    characters = tuple(len(chunk["text"]) for chunk in chunks)
    words = tuple(len(chunk["text"].split()) for chunk in chunks)
    document = json.loads(result.request.recipe.effective_json)
    config = document.get("chunking") if isinstance(document, dict) else None
    config = config if isinstance(config, dict) else {}
    method = config.get("method")
    method = method if isinstance(method, str) else None
    # Only units with a defined counting contract are measured. Future methods
    # remain inspectable without guessing their budget semantics.
    unit = {"words": "words", "fixed_size": "characters"}.get(method)
    options = config.get("config")
    limit = options.get("max_size") if isinstance(options, dict) else None
    if (
        not unit
        or type(limit) not in (int, float)
        or (isinstance(limit, float) and not math.isfinite(limit))
        or limit <= 0
    ):
        limit = None
    sizes = words if unit == "words" else characters
    budget: MethodBudget = {
        "method": method,
        "limit": limit,
        "unit": unit,
        "oversized_chunks": sum(n > limit for n in sizes)
        if unit and limit is not None
        else None,
    }
    spans = [chunk_mapping(result, i) for i in range(len(chunks))]
    overlap = None
    if chunks and all(span["coordinate_space"] == "source" for span in spans):
        covered = 0
        end = 0
        for span in sorted(spans, key=lambda s: s["start"]):
            covered += max(0, span["end"] - max(end, span["start"]))
            end = max(end, span["end"])
        overlap = sum(s["end"] - s["start"] for s in spans) - covered
    source_size = len(result.request.sample.text)
    return {
        "status": result.status,
        "chunk_count": len(chunks),
        "characters": _distribution(characters),
        "words": _distribution(words),
        "character_sizes": characters,
        "word_sizes": words,
        "tokens": _distribution(token_counts) if token_counts is not None else None,
        "measurement_id": measurement_id,
        "budget": budget,
        "expansion_ratio": sum(characters) / source_size if source_size else None,
        "overlap_characters": overlap,
        "elapsed_ms_observation": result.elapsed_ms,
        "definitions": {
            "characters": "Python Unicode code points: len(text)",
            "words": "Whitespace words: len(text.split())",
            "p95": "Nearest rank: ceil(0.95 * count)",
            "expansion_ratio": "Emitted/source characters; not measured overlap",
            "elapsed_ms_observation": "One execution observation; not a benchmark ranking",
        },
    }


def comparison_deltas(a: ResultSummary, b: ResultSummary) -> ComparisonDeltas:
    """Return B minus A common counts; caller first checks compatibility."""
    result: ComparisonDeltas = {
        "chunk_count": b["chunk_count"] - a["chunk_count"],
        "characters": b["characters"]["total"] - a["characters"]["total"],
        "words": b["words"]["total"] - a["words"]["total"],
    }
    if (
        a["measurement_id"]
        and a["measurement_id"] == b["measurement_id"]
        and a["tokens"]
        and b["tokens"]
    ):
        result["tokens"] = b["tokens"]["total"] - a["tokens"]["total"]
    return result


def diff_configs(
    a: RunResult, b: RunResult, *, authored: bool = False
) -> tuple[dict, ...]:
    """Diff captured JSON using JSON Pointer paths and positional arrays.

    Missing values use None and are distinguished from JSON null by kind.
    Complete values are retained; consumers may page their presentation.
    """
    left = json.loads(
        a.request.recipe.authored_json if authored else a.request.recipe.effective_json
    )
    right = json.loads(
        b.request.recipe.authored_json if authored else b.request.recipe.effective_json
    )
    changes = []
    missing = object()

    def walk(x: Any, y: Any, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(x.keys() | y.keys()):
                escaped = key.replace("~", "~0").replace("/", "~1")
                walk(x.get(key, missing), y.get(key, missing), f"{path}/{escaped}")
        elif isinstance(x, list) and isinstance(y, list):
            for i in range(max(len(x), len(y))):
                walk(
                    x[i] if i < len(x) else missing,
                    y[i] if i < len(y) else missing,
                    f"{path}/{i}",
                )
        elif type(x) is not type(y) or x != y:
            kind = "added" if x is missing else "removed" if y is missing else "changed"
            changes.append(
                {
                    "path": path,
                    "kind": kind,
                    "A": None if x is missing else x,
                    "B": None if y is missing else y,
                }
            )

    walk(left, right, "")
    return tuple(changes)
