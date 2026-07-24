#!/usr/bin/env python3
"""Network-free RAG citation provenance baseline and qualification runner."""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import io
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, NamedTuple, Sequence
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_ROOT = REPO_ROOT / "Tests" / "fixtures" / "rag_citation_provenance"
MANIFEST_PATH = FIXTURE_ROOT / "manifest_v1.json"
CORPUS_PATH = FIXTURE_ROOT / "corpus_v1.json"
FIXTURE_VERSION = "rag-citation-provenance-v1"
RESULT_SCHEMA_VERSION = 1
MOCK_PROVIDER = "mock-local-v1"
EXTERNAL_PROVIDER = "external-http-v1"
DEFAULT_SAMPLES = 30
DEFAULT_WARMUPS = 5
MOCK_FIRST_TOKEN_DELAY_SECONDS = 0.020
REPOSITORY_BENCHMARK_FINGERPRINT_SECRET = b"b" * 32

LIMITS = {
    "aggregate_json_bytes": 256 * 1024,
    "snapshot_utf8_bytes": 64 * 1024,
    "governed_trace_bytes": 4 * 1024 * 1024,
    "prompt_sets": 8,
    "evidence_per_prompt": 64,
    "answer_attempts": 8,
    "citation_occurrences": 512,
    "retrieval_candidates_per_run": 200,
    "locator_json_bytes": 16 * 1024,
    "observation_json_bytes": 8 * 1024,
    "error_code_characters": 256,
    "opaque_id_utf8_bytes": 256,
    "answer_attempt_body_utf8_bytes": 1024 * 1024,
    "legacy_sidecar_bytes": 32 * 1024 * 1024,
    "migration_batch_messages": 100,
}

BUDGETS = {
    "first_token": {
        "max_regression_percent": 10,
        "max_regression_ms": 25,
        "statistic": "p95",
    },
    "finalization": {
        "standard_p95_ms": 75,
        "maximum_p95_ms": 250,
        "standard_shape": "8 snapshots x 4 KiB",
        "maximum_shape": "64 snapshots x 64 KiB",
    },
    "inspector_load": {
        "cold_p95_ms": 100,
        "warm_p95_ms": 25,
    },
    "trace_size": {
        "aggregate_json_bytes": LIMITS["aggregate_json_bytes"],
        "snapshot_utf8_bytes": LIMITS["snapshot_utf8_bytes"],
        "governed_trace_bytes": LIMITS["governed_trace_bytes"],
    },
    "database_growth": {
        "governed_bytes_multiplier": 1.35,
        "fixed_allowance_bytes": 256 * 1024,
    },
    "migration": {
        "minimum_messages_per_second": 100,
        "maximum_duplicate_proxy_rows_after_restart": 0,
    },
}

_BASELINE_NON_NEGATIVE_METRIC_PATHS = (
    ("first_token", "control_ms", "median"),
    ("first_token", "control_ms", "p95"),
    ("first_token", "candidate_ms", "median"),
    ("finalization", "standard_ms", "median"),
    ("finalization", "standard_ms", "p95"),
    ("finalization", "maximum_ms", "median"),
    ("finalization", "maximum_ms", "p95"),
    ("inspector_load", "cold_ms", "median"),
    ("inspector_load", "cold_ms", "p95"),
    ("inspector_load", "warm_ms", "median"),
    ("inspector_load", "warm_ms", "p95"),
    ("trace_size", "aggregate_json_bytes"),
    ("trace_size", "maximum_snapshot_bytes"),
    ("trace_size", "governed_trace_bytes"),
    ("database_growth", "bytes", "median"),
    ("database_growth", "bytes", "p95"),
    ("database_growth", "governed_bytes_per_answer"),
    ("migration", "messages_per_second", "median"),
    ("migration", "messages_per_second", "p95"),
    ("migration", "duplicate_proxy_rows_after_restart", "median"),
    ("migration", "duplicate_proxy_rows_after_restart", "p95"),
)


class SampleWorkspace(NamedTuple):
    """Temporary paths used by one benchmark sample group."""

    root: Path
    db_path: Path
    sidecar_path: Path


class _DeterministicGateway:
    """In-process provider gateway that records the first streamed token."""

    def __init__(self, answer: str = "Synthetic answer [S1].") -> None:
        self.first_chunk_ns: int | None = None
        self.answer = answer

    async def resolve_for_send(self, _selection: Any) -> Any:
        return SimpleNamespace(
            ready=True,
            provider=MOCK_PROVIDER,
            model="deterministic-v1",
            base_url=None,
            max_tokens=64,
            visible_copy="",
        )

    async def stream_chat(self, _resolution: Any, _messages: Any) -> Any:
        await asyncio.sleep(MOCK_FIRST_TOKEN_DELAY_SECONDS)
        self.first_chunk_ns = time.perf_counter_ns()
        midpoint = max(1, len(self.answer) // 2)
        yield self.answer[:midpoint]
        yield self.answer[midpoint:]


class _SQLiteConsolePersistence:
    """Small SQLite-backed Console persistence adapter used by the TTFB path."""

    db = None

    def __init__(self, db_path: Path) -> None:
        self.connection = sqlite3.connect(db_path)

    def close(self) -> None:
        self.connection.close()

    def create_conversation(self, **kwargs: Any) -> str:
        cursor = self.connection.execute(
            "INSERT INTO conversations(title) VALUES (?)",
            (str(kwargs.get("title") or "Benchmark"),),
        )
        self.connection.commit()
        return str(cursor.lastrowid)

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        message_id: str | None = None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        attachments: Sequence[dict[str, Any]] | None = None,
    ) -> str:
        del image_data, image_mime_type, message_id, feedback, attachments
        cursor = self.connection.execute(
            """
            INSERT INTO messages(conversation_id, sender, content, parent_message_id)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, sender, content, parent_message_id),
        )
        self.connection.commit()
        return str(cursor.lastrowid)

    def update_message_content(
        self,
        *,
        message_id: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        update_parent: bool = False,
        update_feedback: bool = False,
        attachments: Sequence[dict[str, Any]] | None = None,
    ) -> bool:
        del (
            image_data,
            image_mime_type,
            feedback,
            update_feedback,
            attachments,
        )
        if update_parent:
            cursor = self.connection.execute(
                "UPDATE messages SET content = ?, parent_message_id = ? WHERE id = ?",
                (content, parent_message_id, message_id),
            )
        else:
            cursor = self.connection.execute(
                "UPDATE messages SET content = ? WHERE id = ?",
                (content, message_id),
            )
        self.connection.commit()
        return cursor.rowcount == 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the stable v1 runner command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("baseline", "qualification", "external"),
        default="baseline",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--provider", default=MOCK_PROVIDER)
    parser.add_argument("--base-url")
    parser.add_argument("--external-target")
    parser.add_argument("--external-timeout-seconds", type=float)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or incomplete local benchmark configuration."""

    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    if args.mode == "external":
        if not args.external_target:
            raise ValueError("external mode requires --external-target")
        if args.provider != EXTERNAL_PROVIDER:
            raise ValueError(f"external mode requires provider {EXTERNAL_PROVIDER!r}")
        if (
            args.external_timeout_seconds is not None
            and args.external_timeout_seconds <= 0
        ):
            raise ValueError("--external-timeout-seconds must be positive")
        _external_target_origin(args.external_target)
        return
    if args.external_target is not None or args.external_timeout_seconds is not None:
        raise ValueError("external-only arguments require --mode external")
    if args.provider != MOCK_PROVIDER or args.base_url:
        raise ValueError(
            "local benchmark modes are network-free and require provider "
            f"{MOCK_PROVIDER!r} with no base URL"
        )
    if args.mode == "qualification":
        if args.baseline is None:
            raise ValueError("qualification mode requires --baseline")
        if not args.baseline.is_file():
            raise ValueError(
                f"qualification --baseline does not exist: {args.baseline}"
            )


@contextmanager
def sample_group_workspace(
    scratch_root: Path, group_name: str
) -> Iterator[SampleWorkspace]:
    """Yield a fresh temporary ChaChaNotes DB and legacy sidecar."""

    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"ragcp-{group_name}-", dir=scratch_root
    ) as directory:
        root = Path(directory)
        db_path = root / "ChaChaNotes.db"
        sidecar_path = root / "chat_rag_context.json"
        db_path.touch()
        sidecar_path.write_text("{}\n", encoding="utf-8")
        yield SampleWorkspace(root, db_path, sidecar_path)


@contextmanager
def isolated_benchmark_host_state(root: Path) -> Iterator[None]:
    """Redirect production config/data access and temporarily hide host secrets."""

    root = root.resolve()
    home = root / "home"
    config_root = root / "config"
    data_root = root / "data"
    for path in (home, config_root, data_root):
        path.mkdir(parents=True, exist_ok=True)
    overrides = {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config_root),
        "XDG_DATA_HOME": str(data_root),
        "TLDW_CONFIG_PATH": str(config_root / "tldw_cli" / "config.toml"),
        "TLDW_TEST_CONFIG_ROOT": str(config_root),
        "TLDW_TEST_MODE": "1",
    }
    secret_markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    hidden = {
        key
        for key in os.environ
        if key == "DATABASE_URL"
        or any(marker in key.upper() for marker in secret_markers)
    }
    managed = set(overrides) | hidden
    previous = {key: os.environ.get(key) for key in managed}
    try:
        for key in hidden:
            os.environ.pop(key, None)
        os.environ.update(overrides)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Return median and nearest-rank p95; minima are intentionally excluded."""

    if not values:
        raise ValueError("at least one measurement is required")
    ordered = sorted(float(value) for value in values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median": round(float(statistics.median(ordered)), 6),
        "p95": round(ordered[p95_index], 6),
    }


def _load_fixture() -> dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    corpus_bytes = CORPUS_PATH.read_bytes()
    corpus_record = next(
        record for record in manifest["files"] if record["path"] == CORPUS_PATH.name
    )
    if hashlib.sha256(corpus_bytes).hexdigest() != corpus_record["sha256"]:
        raise ValueError("corpus_v1.json digest does not match manifest_v1.json")
    corpus = json.loads(corpus_bytes)
    if (
        manifest["fixture_version"] != FIXTURE_VERSION
        or corpus["fixture_version"] != FIXTURE_VERSION
    ):
        raise ValueError("unsupported citation provenance fixture version")
    return corpus


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _json_object_with_exact_size(
    target_bytes: int, value: dict[str, Any]
) -> dict[str, Any]:
    payload = {**value, "padding": ""}
    padding = target_bytes - len(_json_bytes(payload))
    if padding < 0:
        raise ValueError("target too small for a JSON object")
    payload["padding"] = "x" * padding
    if len(_json_bytes(payload)) != target_bytes:
        raise AssertionError("failed to materialize exact JSON byte size")
    return payload


def materialize_boundary(case: dict[str, Any]) -> Any:
    """Materialize one domain-shaped exact/over boundary descriptor."""

    units = int(case["units"])
    name = case["limit_name"]
    if name == "aggregate_json_bytes":
        return {
            "trace": _json_object_with_exact_size(
                units,
                {
                    "schema_version": 1,
                    "trace_id": "ragcp-boundary-trace",
                    "prompt_sets": [],
                    "answer_attempts": [],
                },
            )
        }
    if name == "snapshot_utf8_bytes":
        return {"snapshot_text": "x" * units}
    if name == "governed_trace_bytes":
        snapshots = []
        remaining = units
        while remaining:
            chunk_size = min(remaining, LIMITS["snapshot_utf8_bytes"])
            snapshots.append("x" * chunk_size)
            remaining -= chunk_size
        return {"governed_snapshots": snapshots}
    if name == "prompt_sets":
        return {"prompt_sets": [{"ordinal": index} for index in range(units)]}
    if name == "evidence_per_prompt":
        return {
            "prompt_set": {
                "evidence": [{"marker_ordinal": index} for index in range(units)]
            }
        }
    if name == "answer_attempts":
        return {"answer_attempts": [{"ordinal": index} for index in range(units)]}
    if name == "citation_occurrences":
        return {
            "selected_answer": {
                "occurrences": [{"ordinal": index} for index in range(units)]
            }
        }
    if name == "retrieval_candidates_per_run":
        return {"run": {"candidates": [{"rank": index} for index in range(units)]}}
    if name == "locator_json_bytes":
        return {
            "locator": _json_object_with_exact_size(
                units,
                {"kind": "synthetic", "resolver_version": 1},
            )
        }
    if name == "observation_json_bytes":
        return {
            "observation": _json_object_with_exact_size(
                units,
                {"availability": "available", "resolver_version": 1},
            )
        }
    if name == "error_code_characters":
        return {"error_code": "e" * units}
    if name == "opaque_id_utf8_bytes":
        return {"opaque_id": "o" * units}
    if name == "answer_attempt_body_utf8_bytes":
        return {"answer_attempt_body": "a" * units}
    if name == "legacy_sidecar_bytes":
        return {
            "legacy_sidecar": _json_object_with_exact_size(
                units,
                {"schema_version": 1, "conversations": []},
            )
        }
    if name == "migration_batch_messages":
        return {
            "legacy_messages": [
                {"legacy_message_id": f"message-{index}"} for index in range(units)
            ]
        }
    raise ValueError(f"unknown boundary limit: {name}")


def materialized_boundary_size(case: dict[str, Any]) -> int:
    """Measure a domain-shaped boundary through its validation field."""

    value = materialize_boundary(case)
    name = case["limit_name"]
    if name == "aggregate_json_bytes":
        return len(_json_bytes(value["trace"]))
    if name == "snapshot_utf8_bytes":
        return len(value["snapshot_text"].encode("utf-8"))
    if name == "governed_trace_bytes":
        return sum(
            len(snapshot.encode("utf-8")) for snapshot in value["governed_snapshots"]
        )
    if name == "prompt_sets":
        return len(value["prompt_sets"])
    if name == "evidence_per_prompt":
        return len(value["prompt_set"]["evidence"])
    if name == "answer_attempts":
        return len(value["answer_attempts"])
    if name == "citation_occurrences":
        return len(value["selected_answer"]["occurrences"])
    if name == "retrieval_candidates_per_run":
        return len(value["run"]["candidates"])
    if name == "locator_json_bytes":
        return len(_json_bytes(value["locator"]))
    if name == "observation_json_bytes":
        return len(_json_bytes(value["observation"]))
    if name == "error_code_characters":
        return len(value["error_code"])
    if name == "opaque_id_utf8_bytes":
        return len(value["opaque_id"].encode("utf-8"))
    if name == "answer_attempt_body_utf8_bytes":
        return len(value["answer_attempt_body"].encode("utf-8"))
    if name == "legacy_sidecar_bytes":
        return len(_json_bytes(value["legacy_sidecar"]))
    if name == "migration_batch_messages":
        return len(value["legacy_messages"])
    raise ValueError(f"unknown boundary limit: {name}")


def validate_boundary_case(case: dict[str, Any]) -> int:
    """Accept exact v1 domain values and reject the same value at limit + 1."""

    name = case["limit_name"]
    measured = materialized_boundary_size(case)
    if measured > LIMITS[name]:
        raise ValueError(f"{name} exceeds frozen v1 limit")
    return measured


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _snapshot_from_seed(seed: str, size: int) -> str:
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (token * math.ceil(size / len(token)))[:size]


def _snapshots_from_sources(
    sources: Sequence[dict[str, Any]], *, count: int, size: int
) -> tuple[str, ...]:
    return tuple(
        _snapshot_from_seed(
            f"{sources[index % len(sources)]['id']}:{sources[index % len(sources)]['text']}",
            size,
        )
        for index in range(count)
    )


def _validate_fixture_boundaries(corpus: dict[str, Any]) -> None:
    for case in corpus["boundary_cases"]:
        if case["expected"] == "accept":
            validate_boundary_case(case)
            continue
        try:
            validate_boundary_case(case)
        except ValueError:
            continue
        raise AssertionError(f"over-bound fixture was accepted: {case['id']}")


def _workload_from_corpus(corpus: dict[str, Any]) -> dict[str, Any]:
    """Select and materialize the representative v1 corpus workload."""

    _validate_fixture_boundaries(corpus)
    sources = corpus["sources"]
    answers = corpus["answers"]
    shapes = {
        int(shape["submitted_evidence_count"]): shape
        for shape in corpus["evidence_shapes"]
    }
    for count in (1, 8, 32, 64):
        if count not in shapes:
            raise ValueError(f"corpus is missing evidence shape {count}")

    source_inventory = [
        {
            "id": source["id"],
            "kind": source["kind"],
            "title": source["title"],
            "text": source["text"],
        }
        for source in sources
    ]
    answer_inventory = [
        {
            "id": answer["id"],
            "case": answer["case"],
            "body": answer["body"],
        }
        for answer in answers
    ]
    generation_cases = [
        {
            "source_id": source["id"],
            "answer_case": answer["case"],
            "evidence_count": 1,
            "prompt": (
                "Summarize this synthetic local evidence:\n"
                f"{source['kind']}: {source['title']} — {source['text']}"
            ),
            "answer": answer["body"],
        }
        for source in sources
        for answer in answers
    ]
    generation_input = {
        "shape": shapes[1],
        "cases": generation_cases,
    }
    finalization_input = {
        "shapes": [shapes[8], shapes[64]],
        "sources": source_inventory,
        "answers": answer_inventory,
    }
    legacy_input = corpus["legacy_records"]
    return {
        "generation_cases": generation_cases,
        "database_answer": "\n".join(answer["body"] for answer in answers),
        "standard_snapshots": _snapshots_from_sources(sources, count=8, size=4 * 1024),
        "inspector_snapshots": _snapshots_from_sources(
            sources, count=32, size=4 * 1024
        ),
        "maximum_snapshots": _snapshots_from_sources(
            sources,
            count=64,
            size=LIMITS["snapshot_utf8_bytes"],
        ),
        "storage_cases": corpus["storage_cases"],
        "legacy_records": legacy_input,
        "generation_sha256": _sha256_json(generation_input),
        "finalization_sha256": _sha256_json(finalization_input),
        "migration_sha256": _sha256_json(legacy_input),
        "coverage": {
            "generation": {
                "answer_cases": [answer["case"] for answer in answers],
                "source_kinds": [source["kind"] for source in sources],
                "evidence_count": 1,
            },
            "finalization": {"evidence_counts": [8, 64]},
            "inspector": {
                "evidence_count": 32,
                "storage_modes": [
                    storage_case["storage_mode"]
                    for storage_case in corpus["storage_cases"]
                ],
            },
            "migration": {
                "record_types": [
                    record["record_type"] for record in corpus["legacy_records"]
                ]
            },
        },
    }


def environment_metadata() -> dict[str, Any]:
    """Return non-sensitive reference and compatibility environment metadata."""

    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_major_minor": python_minor,
        "python_implementation": platform.python_implementation(),
        "sqlite": sqlite3.sqlite_version,
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "provider": MOCK_PROVIDER,
        "network": "disabled",
        "supported_envelope": {
            "fixture_version": FIXTURE_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "python_major_minor": python_minor,
            "operating_system": platform.system(),
            "machine": platform.machine(),
            "provider": MOCK_PROVIDER,
            "network": "disabled",
        },
    }


def check_baseline_compatibility(
    baseline: dict[str, Any],
    current_environment: dict[str, Any],
) -> dict[str, Any]:
    """Check fixture/result schema and the recorded reference envelope."""

    reasons: list[str] = []
    if baseline.get("fixture_version") != FIXTURE_VERSION:
        reasons.append("fixture_version")
    baseline_environment = baseline.get("environment")
    if not isinstance(baseline_environment, dict):
        reasons.append("environment")
        baseline_environment = {}
    for key in (
        "result_schema_version",
        "python_major_minor",
        "operating_system",
        "machine",
        "provider",
        "network",
    ):
        if baseline_environment.get(key) != current_environment.get(key):
            reasons.append(key)
    if not isinstance(baseline.get("metrics"), dict):
        reasons.append("metrics")
    return {"compatible": not reasons, "reasons": sorted(set(reasons))}


def _require_baseline_metric_number(
    metrics: dict[str, Any],
    path: tuple[str, ...],
    *,
    strictly_positive: bool = False,
) -> None:
    current: Any = metrics
    field = "metrics." + ".".join(path)
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(
                f"invalid qualification baseline: missing numeric field {field}"
            )
        current = current[key]
    if (
        isinstance(current, bool)
        or not isinstance(current, (int, float))
        or (isinstance(current, float) and not math.isfinite(current))
    ):
        raise ValueError(
            f"invalid qualification baseline: {field} must be a finite real number"
        )
    if strictly_positive:
        if current <= 0:
            raise ValueError(
                f"invalid qualification baseline: {field} must be positive"
            )
    elif current < 0:
        raise ValueError(
            f"invalid qualification baseline: {field} must be non-negative"
        )


def _validate_baseline_document(baseline: dict[str, Any]) -> None:
    """Refuse structurally incompatible historical baselines."""

    if not isinstance(baseline, dict):
        raise ValueError("invalid qualification baseline: document must be an object")
    for field, minimum in (
        ("samples", DEFAULT_SAMPLES),
        ("warmups", DEFAULT_WARMUPS),
    ):
        value = baseline.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"invalid qualification baseline: {field} must be an integer"
            )
        if value < minimum:
            raise ValueError(
                f"invalid qualification baseline: {field} must be at least {minimum}"
            )
    if baseline.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError(
            "invalid qualification baseline: fixture version is incompatible"
        )
    environment = baseline.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "invalid qualification baseline: result schema is incompatible"
        )
    metrics = baseline.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("invalid qualification baseline: metrics must be an object")
    for path in _BASELINE_NON_NEGATIVE_METRIC_PATHS:
        _require_baseline_metric_number(metrics, path)
    _require_baseline_metric_number(
        metrics,
        ("first_token", "candidate_ms", "p95"),
        strictly_positive=True,
    )


def _init_console_schema(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                parent_message_id TEXT
            );
            """
        )


async def _measure_console_ttfb_once(
    db_path: Path,
    *,
    prompt: str = "Summarize the synthetic evidence.",
    answer: str = "Synthetic answer [S1].",
) -> float:
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    persistence = _SQLiteConsolePersistence(db_path)
    gateway = _DeterministicGateway(answer)
    store = ConsoleChatStore(persistence=persistence)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider=MOCK_PROVIDER,
        model="deterministic-v1",
        base_url=None,
        agent_runtime_enabled=False,
        skill_substitution_enabled=False,
    )
    started_ns = time.perf_counter_ns()
    try:
        result = await controller.submit_draft(prompt)
    finally:
        persistence.close()
    if not result.accepted or gateway.first_chunk_ns is None:
        raise AssertionError("deterministic Console stream did not emit a first token")
    return (gateway.first_chunk_ns - started_ns) / 1_000_000


async def _measure_first_token(
    db_path: Path,
    *,
    samples: int,
    warmups: int,
    qualification: bool = False,
    workload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control: list[float] = []
    candidate: list[float] = []
    for index in range(warmups + samples):
        if workload is None:
            control_ms = await _measure_console_ttfb_once(db_path)
        else:
            generation_cases = workload["generation_cases"]
            generation_case = generation_cases[index % len(generation_cases)]
            control_ms = await _measure_console_ttfb_once(
                db_path,
                prompt=generation_case["prompt"],
                answer=generation_case["answer"],
            )
        candidate_ms = control_ms
        if index >= warmups:
            control.append(control_ms)
            candidate.append(candidate_ms)
    control_summary = summarize(control)
    candidate_summary = summarize(candidate)
    reference = control_summary["p95"]
    delta = candidate_summary["p95"] - reference
    return {
        "control_ms": control_summary,
        "candidate_ms": candidate_summary,
        "regression_vs_control": {
            "milliseconds": round(delta, 6),
            "percent": round((delta / reference * 100) if reference else 0.0, 6),
        },
        "comparison": (
            "prefeature-shared-control" if qualification else "baseline-current-control"
        ),
        "path": [
            "ConsoleChatController.submit_draft",
            "ConsoleChatController._stream_assistant_response",
        ],
        "corpus_input_sha256": (
            workload["generation_sha256"] if workload is not None else None
        ),
    }


def _finalize_snapshots(snapshots: Sequence[str]) -> dict[str, Any]:
    governed_bytes = 0
    refs = []
    for ordinal, snapshot in enumerate(snapshots):
        encoded = snapshot.encode("utf-8")
        if len(encoded) > LIMITS["snapshot_utf8_bytes"]:
            raise ValueError("snapshot exceeds v1 byte limit")
        governed_bytes += len(encoded)
        refs.append(
            {
                "ordinal": ordinal,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    if governed_bytes > LIMITS["governed_trace_bytes"]:
        raise ValueError("governed payload exceeds v1 trace byte limit")
    aggregate = json.dumps(
        {"schema_version": 1, "snapshots": refs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(aggregate) > LIMITS["aggregate_json_bytes"]:
        raise ValueError("immutable aggregate exceeds v1 byte limit")
    return {
        "aggregate_json_bytes": len(aggregate),
        "governed_trace_bytes": governed_bytes,
        "maximum_snapshot_bytes": max(len(item.encode("utf-8")) for item in snapshots),
    }


def _measure_finalization(
    *,
    samples: int,
    warmups: int,
    standard_snapshots: Sequence[str] | None = None,
    maximum_snapshots: Sequence[str] | None = None,
    corpus_input_sha256: str | None = None,
) -> dict[str, Any]:
    standard_snapshots = standard_snapshots or tuple("s" * (4 * 1024) for _ in range(8))
    maximum_snapshots = maximum_snapshots or tuple(
        "m" * LIMITS["snapshot_utf8_bytes"] for _ in range(64)
    )
    standard_times: list[float] = []
    maximum_times: list[float] = []
    maximum_result: dict[str, Any] | None = None
    for index in range(warmups + samples):
        started_ns = time.perf_counter_ns()
        _finalize_snapshots(standard_snapshots)
        standard_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

        started_ns = time.perf_counter_ns()
        maximum_result = _finalize_snapshots(maximum_snapshots)
        maximum_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if index >= warmups:
            standard_times.append(standard_ms)
            maximum_times.append(maximum_ms)
    assert maximum_result is not None
    return {
        "standard_ms": summarize(standard_times),
        "maximum_ms": summarize(maximum_times),
        "standard_shape": {"snapshots": 8, "bytes_per_snapshot": 4 * 1024},
        "maximum_shape": {
            "snapshots": 64,
            "bytes_per_snapshot": LIMITS["snapshot_utf8_bytes"],
        },
        "maximum_result": maximum_result,
        "corpus_input_sha256": corpus_input_sha256,
    }


def _init_inspector_schema(
    db_path: Path,
    snapshots: Sequence[str],
    storage_cases: Sequence[dict[str, Any]],
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE inspector_items (
                ordinal INTEGER PRIMARY KEY,
                snapshot_text TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO inspector_items(ordinal, snapshot_text, metadata_json)
            VALUES (?, ?, ?)
            """,
            (
                (
                    index,
                    snapshot,
                    json.dumps(
                        storage_cases[index % len(storage_cases)],
                        sort_keys=True,
                    ),
                )
                for index, snapshot in enumerate(snapshots)
            ),
        )


def _read_inspector(connection: sqlite3.Connection) -> int:
    rows = connection.execute(
        """
        SELECT ordinal, snapshot_text, metadata_json
        FROM inspector_items
        ORDER BY ordinal
        """
    ).fetchall()
    for _ordinal, snapshot_text, metadata_json in rows:
        len(snapshot_text)
        json.loads(metadata_json)
    return len(rows)


def _measure_inspector(
    db_path: Path, *, samples: int, warmups: int, expected_rows: int = 64
) -> dict[str, Any]:
    cold_times: list[float] = []
    warm_times: list[float] = []
    for index in range(warmups + samples):
        started_ns = time.perf_counter_ns()
        with sqlite3.connect(db_path) as cold_connection:
            if _read_inspector(cold_connection) != expected_rows:
                raise AssertionError("cold inspector read returned incomplete rows")
        cold_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

        with sqlite3.connect(db_path) as warm_connection:
            _read_inspector(warm_connection)
            started_ns = time.perf_counter_ns()
            if _read_inspector(warm_connection) != expected_rows:
                raise AssertionError("warm inspector read returned incomplete rows")
            warm_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if index >= warmups:
            cold_times.append(cold_ms)
            warm_times.append(warm_ms)
    return {"cold_ms": summarize(cold_times), "warm_ms": summarize(warm_times)}


def _checkpoint_and_size(db_path: Path, connection: sqlite3.Connection) -> int:
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    return db_path.stat().st_size


def _init_growth_schema(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE grounded_answers (
                id INTEGER PRIMARY KEY,
                body TEXT NOT NULL,
                aggregate_json TEXT NOT NULL
            );
            CREATE TABLE evidence_payloads (
                answer_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                snapshot_text TEXT NOT NULL,
                PRIMARY KEY(answer_id, ordinal)
            );
            """
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()


def _measure_database_growth(
    db_path: Path,
    *,
    samples: int,
    warmups: int,
    snapshots: Sequence[str] | None = None,
    answer_body: str = "Synthetic grounded answer [S1].",
    corpus_input_sha256: str | None = None,
) -> dict[str, Any]:
    snapshots = snapshots or tuple(
        "g" * LIMITS["snapshot_utf8_bytes"] for _ in range(64)
    )
    aggregate = json.dumps(
        {
            "schema_version": 1,
            "snapshot_count": len(snapshots),
            "storage": "governed",
            "corpus_input_sha256": corpus_input_sha256,
        },
        sort_keys=True,
    )
    governed_bytes = sum(len(item.encode("utf-8")) for item in snapshots)
    deltas: list[float] = []
    with sqlite3.connect(db_path) as connection:
        for index in range(warmups + samples):
            before = _checkpoint_and_size(db_path, connection)
            cursor = connection.execute(
                """
                INSERT INTO grounded_answers(body, aggregate_json)
                VALUES (?, ?)
                """,
                (answer_body, aggregate),
            )
            answer_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO evidence_payloads(answer_id, ordinal, snapshot_text)
                VALUES (?, ?, ?)
                """,
                (
                    (answer_id, ordinal, snapshot)
                    for ordinal, snapshot in enumerate(snapshots)
                ),
            )
            connection.commit()
            after = _checkpoint_and_size(db_path, connection)
            if index >= warmups:
                deltas.append(float(after - before))
    return {
        "bytes": summarize(deltas),
        "governed_bytes_per_answer": governed_bytes,
    }


def _sealed_repository_write(
    sample_index: int,
    snapshots: Sequence[str],
    *,
    authority_id: str,
) -> Any:
    """Build one complete sealed repository candidate outside timed writes."""

    from tldw_chatbook.Chat.citation_trace_identity import (
        CitationFingerprintCodec,
        CitationFingerprintDomain,
    )
    from tldw_chatbook.Chat.citation_trace_models import (
        AnswerAttempt,
        AnswerAttemptKind,
        AnswerAttemptPayload,
        CitationCompleteness,
        CitationOccurrence,
        CitationTrace,
        EvidenceRun,
        EvidenceRunPayload,
        EvidenceSnapshotPayload,
        EvidenceStorageMode,
        MarkerNamespace,
        PolicyCapability,
        PromptEvidenceEntry,
        PromptEvidenceSet,
        SealedCitationWrite,
        StructuralValidationState,
        TraceLifecycle,
        TraceOrigin,
    )

    prefix = f"repository-{sample_index}"
    timestamp = datetime(2026, 7, 24, 0, 0, tzinfo=UTC)
    answer = "Synthetic answer [S1]."
    marker_start = answer.index("[S1]")
    run = EvidenceRun(
        run_id=f"{prefix}-run",
        request_id=f"{prefix}-request",
        run_ordinal=1,
        stage="retrieval",
        payload_ref=f"{prefix}-run-payload",
        started_at=timestamp,
        ended_at=timestamp,
    )
    entries = tuple(
        PromptEvidenceEntry(
            evidence_ordinal=ordinal,
            marker_ordinal=ordinal,
            run_id=run.run_id,
            snapshot_payload_ref=f"{prefix}-snapshot-{ordinal}",
            storage_mode=EvidenceStorageMode.EMBEDDED,
        )
        for ordinal in range(1, len(snapshots) + 1)
    )
    prompt_set = PromptEvidenceSet(
        prompt_set_id=f"{prefix}-prompt",
        prompt_set_ordinal=1,
        marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
        entries=entries,
        created_at=timestamp,
    )
    attempt = AnswerAttempt(
        attempt_id=f"{prefix}-attempt",
        attempt_ordinal=1,
        kind=AnswerAttemptKind.INITIAL,
        prompt_evidence_set_id=prompt_set.prompt_set_id,
        answer_payload_ref=f"{prefix}-answer-payload",
        occurrences=(
            CitationOccurrence(
                occurrence_id=f"{prefix}-occurrence",
                occurrence_ordinal=1,
                raw_marker="[S1]",
                marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
                evidence_ordinal=1,
                marker_start=marker_start,
                marker_end=marker_start + len("[S1]"),
                claim_start=0,
                claim_end=marker_start - 1,
                structural_state=StructuralValidationState.VALID,
            ),
        ),
        created_at=timestamp,
    )
    trace = CitationTrace(
        trace_id=f"{prefix}-trace",
        request_id=run.request_id,
        generation_id=f"{prefix}-generation",
        origin=TraceOrigin.LOCAL,
        lifecycle=TraceLifecycle.SEALED,
        completeness_at_seal=CitationCompleteness.COMPLETE,
        evidence_runs=(run,),
        prompt_evidence_sets=(prompt_set,),
        answer_attempts=(attempt,),
        selected_attempt_id=attempt.attempt_id,
        policy_capabilities=(PolicyCapability.VIEW_SNAPSHOT,),
        policy_version="benchmark-policy-v1",
        created_at=timestamp,
        sealed_at=timestamp,
    )
    return SealedCitationWrite(
        trace=trace,
        evidence_run_payloads=(
            EvidenceRunPayload(
                payload_id=run.payload_ref,
                run_id=run.run_id,
                raw_query="synthetic benchmark query",
                query_fingerprint=f"{prefix}-query-fingerprint",
                authority_id=authority_id,
                retrieval_metadata={"kind": "network-free-benchmark"},
            ),
        ),
        evidence_snapshot_payloads=tuple(
            EvidenceSnapshotPayload(
                payload_id=entry.snapshot_payload_ref,
                storage_mode=EvidenceStorageMode.EMBEDDED,
                snapshot_text=snapshot,
                title=f"Synthetic source {entry.evidence_ordinal}",
                source_identity={"source_id": f"{prefix}-source"},
                locator={"source_kind": "benchmark"},
                lineage={"ordinal": entry.evidence_ordinal},
                content_hash=f"{prefix}-content-{entry.evidence_ordinal}",
                comparison_hash=f"{prefix}-comparison-{entry.evidence_ordinal}",
            )
            for entry, snapshot in zip(entries, snapshots, strict=True)
        ),
        answer_attempt_payloads=(
            AnswerAttemptPayload(
                payload_id=attempt.answer_payload_ref,
                attempt_id=attempt.attempt_id,
                answer_body=answer,
                body_integrity_hmac=CitationFingerprintCodec(
                    REPOSITORY_BENCHMARK_FINGERPRINT_SECRET
                ).fingerprint(
                    CitationFingerprintDomain.MESSAGE_BODY,
                    answer,
                ),
            ),
        ),
    )


def _measure_repository_storage(
    db_path: Path,
    *,
    samples: int,
    warmups: int,
    snapshots: Sequence[str],
) -> dict[str, Any]:
    """Measure the real sealed repository path beside message-only control."""

    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.citation_provenance_runtime import (
        CitationProvenanceRuntimePolicy,
    )
    from tldw_chatbook.Chat.citation_trace_identity import (
        CitationFingerprintCodec,
        local_trace_namespace,
    )
    from tldw_chatbook.Chat.citation_trace_repository import (
        CitationTraceRepository,
        load_local_citation_identity_context,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(db_path, client_id="ragcp-repository-benchmark")
    try:
        identity = load_local_citation_identity_context(db)
        if identity is None:
            raise AssertionError("repository benchmark identity is unavailable")
        repository = CitationTraceRepository(
            db,
            policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
            identity_context=identity,
            fingerprint_codec=CitationFingerprintCodec(
                REPOSITORY_BENCHMARK_FINGERPRINT_SECRET
            ),
        )
        service = ChatPersistenceService(db, citation_repository=repository)
        conversation_id = db.add_conversation(
            {"title": "Repository benchmark", "character_id": None}
        )
        control_times: list[float] = []
        write_times: list[float] = []
        read_times: list[float] = []
        for index in range(warmups + samples):
            sealed = _sealed_repository_write(
                index,
                snapshots,
                authority_id=identity.local_authority_id,
            )
            started_ns = time.perf_counter_ns()
            service.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                content="Synthetic answer [S1].",
                message_id=f"repository-control-{index}",
            )
            control_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

            started_ns = time.perf_counter_ns()
            service.create_message(
                conversation_id=conversation_id,
                sender="assistant",
                content="Synthetic answer [S1].",
                message_id=f"repository-candidate-{index}",
                citation_write=sealed,
            )
            write_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

            namespace = local_trace_namespace(
                identity,
                trace_id=sealed.trace.trace_id,
            )
            started_ns = time.perf_counter_ns()
            summary = repository.get_trace_summary(namespace)
            read_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
            if summary is None or summary.trace.trace_id != sealed.trace.trace_id:
                raise AssertionError("repository summary read returned the wrong trace")
            if index >= warmups:
                control_times.append(control_ms)
                write_times.append(write_ms)
                read_times.append(read_ms)

        connection = db.get_connection()
        trace_rows = int(
            connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        )
        owner_rows = int(
            connection.execute(
                "SELECT count(*) FROM rag_message_trace_owners"
            ).fetchone()[0]
        )
        return {
            "message_only_control_ms": summarize(control_times),
            "sealed_write_ms": summarize(write_times),
            "summary_read_ms": summarize(read_times),
            "persisted_trace_rows": trace_rows,
            "persisted_owner_rows": owner_rows,
            "control_path": (
                "ChatPersistenceService.create_message(citation_write=None)"
            ),
            "candidate_path": (
                "ChatPersistenceService.create_message(citation_write=sealed)"
            ),
        }
    finally:
        db.close_connection()


def _init_migration_schema(
    db_path: Path,
    sidecar_path: Path,
    legacy_records: Sequence[dict[str, Any]],
) -> None:
    del db_path
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fixture_scan_control": legacy_records,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _measure_migration_once(
    workspace: SampleWorkspace,
    legacy_records: Sequence[dict[str, Any]],
    *,
    sample_index: int,
) -> tuple[float, float, int, int, int]:
    from tldw_chatbook.Chat.citation_legacy_migration import (
        CitationLegacyMigrationService,
        LegacyMigrationState,
    )
    from tldw_chatbook.Chat.citation_provenance_runtime import (
        CitationProvenanceRuntimePolicy,
    )
    from tldw_chatbook.Chat.citation_trace_identity import CitationFingerprintCodec
    from tldw_chatbook.Chat.citation_trace_repository import (
        CitationTraceRepository,
        load_local_citation_identity_context,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db_path = workspace.root / f"migration-candidate-{sample_index}.sqlite"
    sidecar_path = workspace.root / f"migration-candidate-{sample_index}.json"
    db = CharactersRAGDB(db_path, client_id="ragcp-migration-benchmark")
    try:
        identity = load_local_citation_identity_context(db)
        if identity is None:
            raise AssertionError("migration benchmark identity is unavailable")
        codec = CitationFingerprintCodec(REPOSITORY_BENCHMARK_FINGERPRINT_SECRET)
        repository = CitationTraceRepository(
            db,
            policy=CitationProvenanceRuntimePolicy(canonical_writes_enabled=True),
            identity_context=identity,
            fingerprint_codec=codec,
        )
        conversation_id = db.add_conversation(
            {"title": "Legacy migration benchmark", "character_id": None}
        )
        records: dict[str, dict[str, Any]] = {}
        for index in range(100):
            message_id = f"legacy-message-{index:03d}"
            body = "Legacy benchmark answer [1]."
            db.add_message(
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": body,
                }
            )
            fixture = legacy_records[index % len(legacy_records)]
            records[message_id] = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "answer_body": body,
                "rag_context": {
                    "citation_validation": {"valid": True},
                    "legacy_fixture": fixture,
                },
                "citations": [],
            }
        raw_sidecar = json.dumps(
            {
                "version": 1,
                "conversations": {conversation_id: records},
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        sidecar_path.write_bytes(raw_sidecar)

        control_started_ns = time.perf_counter_ns()
        parsed = json.loads(sidecar_path.read_bytes().decode("utf-8"))
        scanned = parsed["conversations"][conversation_id]
        for message_id in sorted(scanned):
            json.dumps(
                scanned[message_id],
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        control_seconds = (time.perf_counter_ns() - control_started_ns) / 1_000_000_000

        interrupted = CitationLegacyMigrationService(
            db=db,
            repository=repository,
            sidecar_path=sidecar_path,
            fingerprint_codec=codec,
            batch_size=50,
        )
        started_ns = time.perf_counter_ns()
        first = interrupted.migrate_next_batch(conversation_id)
        if first.state is not LegacyMigrationState.RUNNING:
            raise AssertionError("migration benchmark interruption did not stage")
        restarted = CitationLegacyMigrationService(
            db=db,
            repository=repository,
            sidecar_path=sidecar_path,
            fingerprint_codec=codec,
        )
        second = restarted.migrate_next_batch(conversation_id)
        if second.state is not LegacyMigrationState.COMPLETE:
            raise AssertionError("migration benchmark restart did not complete")
        elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        connection = db.get_connection()
        trace_rows = int(
            connection.execute("SELECT count(*) FROM rag_citation_traces").fetchone()[0]
        )
        owner_rows = int(
            connection.execute(
                "SELECT count(*) FROM rag_message_trace_owners"
            ).fetchone()[0]
        )
        duplicate_count = int(
            connection.execute(
                """
                SELECT
                  (COUNT(*) - COUNT(DISTINCT trace_id))
                  + (
                    SELECT COUNT(*) - COUNT(DISTINCT message_id)
                    FROM rag_message_trace_owners
                  )
                FROM rag_citation_traces
                """
            ).fetchone()[0]
        )
        if trace_rows != 100 or owner_rows != 100:
            raise AssertionError("restart migration did not produce 100 canonical rows")
        return (
            100 / elapsed_seconds,
            100 / max(control_seconds, 1e-9),
            duplicate_count,
            trace_rows,
            owner_rows,
        )
    finally:
        db.close_connection()


def _measure_migration(
    workspace: SampleWorkspace,
    *,
    samples: int,
    warmups: int,
    legacy_records: Sequence[dict[str, Any]] | None = None,
    corpus_input_sha256: str | None = None,
) -> dict[str, Any]:
    legacy_records = legacy_records or (
        {
            "record_type": "CitationRef",
            "payload": {"source_id": "legacy-source"},
        },
    )
    throughputs: list[float] = []
    control_throughputs: list[float] = []
    duplicate_counts: list[float] = []
    persisted_trace_rows = 0
    persisted_owner_rows = 0
    for index in range(warmups + samples):
        (
            throughput,
            control_throughput,
            duplicates,
            persisted_trace_rows,
            persisted_owner_rows,
        ) = _measure_migration_once(
            workspace,
            legacy_records,
            sample_index=index,
        )
        if index >= warmups:
            throughputs.append(throughput)
            control_throughputs.append(control_throughput)
            duplicate_counts.append(float(duplicates))
    return {
        "messages_per_second": summarize(throughputs),
        "fixture_scan_control_messages_per_second": summarize(control_throughputs),
        "duplicate_proxy_rows_after_restart": summarize(duplicate_counts),
        "batch_messages": 100,
        "interrupted_after_messages": 50,
        "corpus_input_sha256": corpus_input_sha256,
        "persisted_trace_rows": persisted_trace_rows,
        "persisted_owner_rows": persisted_owner_rows,
        "control_path": "bounded legacy fixture scan",
        "candidate_path": ("CitationLegacyMigrationService.migrate_next_batch"),
    }


def _regression(candidate: float, reference: float) -> dict[str, float]:
    delta = candidate - reference
    percent = (
        delta / reference * 100 if reference else (0.0 if delta <= 0 else math.inf)
    )
    return {"milliseconds": round(delta, 6), "percent": round(percent, 6)}


def empty_passing_metrics() -> dict[str, Any]:
    """Return a complete synthetic passing metric shape for contract tests."""

    one = {"median": 1.0, "p95": 1.0}
    zero = {"median": 0.0, "p95": 0.0}
    return {
        "first_token": {
            "control_ms": dict(one),
            "candidate_ms": dict(one),
            "regression_vs_control": {"milliseconds": 0.0, "percent": 0.0},
        },
        "finalization": {
            "standard_ms": dict(one),
            "maximum_ms": dict(one),
        },
        "inspector_load": {"cold_ms": dict(one), "warm_ms": dict(one)},
        "trace_size": {
            "aggregate_json_bytes": 1,
            "maximum_snapshot_bytes": 1,
            "governed_trace_bytes": 1,
        },
        "database_growth": {
            "bytes": dict(one),
            "governed_bytes_per_answer": 1,
        },
        "migration": {
            "messages_per_second": {"median": 1000.0, "p95": 1000.0},
            "duplicate_proxy_rows_after_restart": dict(zero),
        },
    }


def evaluate_budgets(
    metrics: dict[str, Any],
    *,
    baseline: dict[str, Any] | None,
    mode: str,
    environment: dict[str, Any] | None = None,
    samples: int = DEFAULT_SAMPLES,
    warmups: int = DEFAULT_WARMUPS,
) -> dict[str, Any]:
    """Evaluate all six frozen budget families."""

    current_environment = environment or environment_metadata()
    compatibility = (
        check_baseline_compatibility(baseline, current_environment)
        if baseline is not None
        else {"compatible": mode == "baseline", "reasons": []}
    )
    candidate_p95 = metrics["first_token"]["candidate_ms"]["p95"]
    current_control_p95 = metrics["first_token"]["control_ms"]["p95"]
    regressions = {
        "in_process_control": _regression(candidate_p95, current_control_p95)
    }
    first_token_pass = all(
        regression["percent"] <= BUDGETS["first_token"]["max_regression_percent"]
        and regression["milliseconds"] <= BUDGETS["first_token"]["max_regression_ms"]
        for regression in regressions.values()
    )
    if baseline is not None and compatibility["compatible"]:
        historical = baseline["metrics"]["first_token"]
        regressions["historical_v1"] = _regression(
            candidate_p95, historical["candidate_ms"]["p95"]
        )
        historical = regressions["historical_v1"]
        first_token_pass = first_token_pass and (
            historical["percent"] <= BUDGETS["first_token"]["max_regression_percent"]
            and historical["milliseconds"]
            <= BUDGETS["first_token"]["max_regression_ms"]
        )

    finalization_pass = (
        metrics["finalization"]["standard_ms"]["p95"]
        <= BUDGETS["finalization"]["standard_p95_ms"]
        and metrics["finalization"]["maximum_ms"]["p95"]
        <= BUDGETS["finalization"]["maximum_p95_ms"]
    )
    inspector_pass = (
        metrics["inspector_load"]["cold_ms"]["p95"]
        <= BUDGETS["inspector_load"]["cold_p95_ms"]
        and metrics["inspector_load"]["warm_ms"]["p95"]
        <= BUDGETS["inspector_load"]["warm_p95_ms"]
    )
    trace_size_pass = (
        metrics["trace_size"]["aggregate_json_bytes"]
        <= BUDGETS["trace_size"]["aggregate_json_bytes"]
        and metrics["trace_size"]["maximum_snapshot_bytes"]
        <= BUDGETS["trace_size"]["snapshot_utf8_bytes"]
        and metrics["trace_size"]["governed_trace_bytes"]
        <= BUDGETS["trace_size"]["governed_trace_bytes"]
    )
    allowed_growth = (
        metrics["database_growth"]["governed_bytes_per_answer"]
        * BUDGETS["database_growth"]["governed_bytes_multiplier"]
        + BUDGETS["database_growth"]["fixed_allowance_bytes"]
    )
    database_growth_pass = metrics["database_growth"]["bytes"]["p95"] <= allowed_growth
    migration_pass = (
        metrics["migration"]["messages_per_second"]["median"]
        >= BUDGETS["migration"]["minimum_messages_per_second"]
        and metrics["migration"]["duplicate_proxy_rows_after_restart"]["p95"]
        <= BUDGETS["migration"]["maximum_duplicate_proxy_rows_after_restart"]
    )
    checks = {
        "first_token": {"pass": first_token_pass, "regressions": regressions},
        "finalization": {"pass": finalization_pass},
        "inspector_load": {"pass": inspector_pass},
        "trace_size": {"pass": trace_size_pass},
        "database_growth": {
            "pass": database_growth_pass,
            "allowed_bytes": round(allowed_growth),
        },
        "migration": {"pass": migration_pass},
    }
    repository_storage = metrics.get("repository_storage")
    if repository_storage is not None:
        repository_storage_pass = (
            repository_storage["sealed_write_ms"]["p95"]
            <= BUDGETS["finalization"]["standard_p95_ms"]
            and repository_storage["summary_read_ms"]["p95"]
            <= BUDGETS["inspector_load"]["warm_p95_ms"]
            and repository_storage["persisted_trace_rows"] == samples + warmups
            and repository_storage["persisted_owner_rows"] == samples + warmups
        )
        checks["repository_storage"] = {
            "pass": repository_storage_pass,
            "write_p95_budget_ms": BUDGETS["finalization"]["standard_p95_ms"],
            "read_p95_budget_ms": BUDGETS["inspector_load"]["warm_p95_ms"],
        }
    qualification_ineligibility_reasons = []
    if mode == "qualification" and samples < DEFAULT_SAMPLES:
        qualification_ineligibility_reasons.append("insufficient_samples")
    if mode == "qualification" and warmups < DEFAULT_WARMUPS:
        qualification_ineligibility_reasons.append("insufficient_warmups")
    qualification_eligible = not qualification_ineligibility_reasons
    environment_compatible = bool(compatibility["compatible"])
    return {
        "definitions": BUDGETS,
        "checks": checks,
        "qualification_eligible": qualification_eligible,
        "qualification_ineligibility_reasons": qualification_ineligibility_reasons,
        "environment_compatible": environment_compatible,
        "environment_incompatibilities": compatibility["reasons"],
        "overall_pass": environment_compatible
        and qualification_eligible
        and all(check["pass"] for check in checks.values()),
    }


def _external_target_origin(target: str) -> str:
    parsed = urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("external target must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("external target must not contain credentials")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


async def _default_external_resolver(target: str, timeout_seconds: float) -> None:
    def resolve() -> None:
        request = Request(
            target,
            headers={"User-Agent": "tldw-chatbook-ragcp-benchmark-v1"},
        )
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read(1)

    await asyncio.to_thread(resolve)


async def run_external_measurement(
    *,
    target: str,
    samples: int,
    warmups: int,
    timeout_seconds: float,
    resolver: Any | None = None,
) -> dict[str, Any]:
    """Measure explicit external resolution without evaluating local budgets."""

    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")
    if timeout_seconds <= 0:
        raise ValueError("external timeout must be positive")
    target_origin = _external_target_origin(target)
    resolve = resolver or _default_external_resolver
    latencies: list[float] = []
    for index in range(warmups + samples):
        started_ns = time.perf_counter_ns()
        await resolve(target, timeout_seconds)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        if index >= warmups:
            latencies.append(elapsed_ms)

    environment = {
        **environment_metadata(),
        "provider": EXTERNAL_PROVIDER,
        "network": "explicit-external-mode",
    }
    environment.pop("supported_envelope")
    return {
        "environment": environment,
        "fixture_version": FIXTURE_VERSION,
        "samples": samples,
        "warmups": warmups,
        "metrics": {},
        "budgets": {
            "definitions": BUDGETS,
            "checks": {},
            "qualification_eligible": None,
            "qualification_ineligibility_reasons": [],
            "environment_compatible": None,
            "environment_incompatibilities": [],
            "overall_pass": None,
        },
        "external_network": {
            "measured": True,
            "included_in_local_pass": False,
            "latency_ms": summarize(latencies),
            "target_origin": target_origin,
            "note": "External latency is informational and never a local budget gate.",
        },
    }


async def run_benchmark(
    *,
    mode: str,
    samples: int,
    warmups: int,
    scratch_root: Path | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all local benchmark groups without external resolution."""

    if mode not in {"baseline", "qualification"}:
        raise ValueError(f"unsupported mode: {mode}")
    if samples < 1 or warmups < 0:
        raise ValueError("samples must be positive and warmups non-negative")
    corpus = _load_fixture()
    workload = _workload_from_corpus(corpus)
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if scratch_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="ragcp-run-")
        root = Path(temporary_root.name)
    else:
        root = scratch_root
        root.mkdir(parents=True, exist_ok=True)

    repository_storage = None
    try:
        with isolated_benchmark_host_state(root / "host-state"):
            with sample_group_workspace(root, "first-token") as workspace:
                _init_console_schema(workspace.db_path)
                first_token = await _measure_first_token(
                    workspace.db_path,
                    samples=samples,
                    warmups=warmups,
                    qualification=mode == "qualification",
                    workload=workload,
                )
        with sample_group_workspace(root, "finalization"):
            finalization = _measure_finalization(
                samples=samples,
                warmups=warmups,
                standard_snapshots=workload["standard_snapshots"],
                maximum_snapshots=workload["maximum_snapshots"],
                corpus_input_sha256=workload["finalization_sha256"],
            )
        with sample_group_workspace(root, "inspector") as workspace:
            _init_inspector_schema(
                workspace.db_path,
                workload["inspector_snapshots"],
                workload["storage_cases"],
            )
            inspector = _measure_inspector(
                workspace.db_path,
                samples=samples,
                warmups=warmups,
                expected_rows=len(workload["inspector_snapshots"]),
            )
        with sample_group_workspace(root, "database-growth") as workspace:
            _init_growth_schema(workspace.db_path)
            database_growth = _measure_database_growth(
                workspace.db_path,
                samples=samples,
                warmups=warmups,
                snapshots=workload["maximum_snapshots"],
                answer_body=workload["database_answer"],
                corpus_input_sha256=workload["finalization_sha256"],
            )
        if mode == "qualification":
            with isolated_benchmark_host_state(root / "repository-host-state"):
                with sample_group_workspace(root, "repository-storage") as workspace:
                    repository_storage = _measure_repository_storage(
                        workspace.db_path,
                        samples=samples,
                        warmups=warmups,
                        snapshots=workload["standard_snapshots"],
                    )
        with sample_group_workspace(root, "migration") as workspace:
            _init_migration_schema(
                workspace.db_path,
                workspace.sidecar_path,
                workload["legacy_records"],
            )
            migration = _measure_migration(
                workspace,
                samples=samples,
                warmups=warmups,
                legacy_records=workload["legacy_records"],
                corpus_input_sha256=workload["migration_sha256"],
            )
    finally:
        if temporary_root is not None:
            temporary_root.cleanup()

    maximum = finalization["maximum_result"]
    metrics = {
        "first_token": first_token,
        "finalization": finalization,
        "inspector_load": inspector,
        "trace_size": {
            "aggregate_json_bytes": maximum["aggregate_json_bytes"],
            "maximum_snapshot_bytes": maximum["maximum_snapshot_bytes"],
            "governed_trace_bytes": maximum["governed_trace_bytes"],
        },
        "database_growth": database_growth,
        "migration": migration,
        "corpus_coverage": workload["coverage"],
    }
    if repository_storage is not None:
        metrics["repository_storage"] = repository_storage
    environment = environment_metadata()
    return {
        "environment": environment,
        "fixture_version": FIXTURE_VERSION,
        "samples": samples,
        "warmups": warmups,
        "metrics": metrics,
        "budgets": evaluate_budgets(
            metrics,
            baseline=baseline,
            mode=mode,
            environment=environment,
            samples=samples,
            warmups=warmups,
        ),
        "external_network": {
            "measured": False,
            "included_in_local_pass": False,
            "latency_ms": None,
            "note": (
                "External source resolution is a separate opt-in measurement "
                "and is never part of local rendering or persistence budgets."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and write deterministic-schema JSON output."""

    try:
        args = parse_args(argv)
        validate_args(args)
        baseline = None
        if args.mode == "external":
            result = asyncio.run(
                run_external_measurement(
                    target=args.external_target,
                    samples=args.samples,
                    warmups=args.warmups,
                    timeout_seconds=args.external_timeout_seconds or 10.0,
                )
            )
        else:
            if args.baseline is not None:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                _validate_baseline_document(baseline)
            result = asyncio.run(
                run_benchmark(
                    mode=args.mode,
                    samples=args.samples,
                    warmups=args.warmups,
                    baseline=baseline,
                )
            )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.output}")
        if args.mode == "external":
            return 0
        return 0 if result["budgets"]["overall_pass"] else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
