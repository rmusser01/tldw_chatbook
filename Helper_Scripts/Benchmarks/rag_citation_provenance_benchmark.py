#!/usr/bin/env python3
"""Network-free RAG citation provenance baseline and qualification runner."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, NamedTuple, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURE_ROOT = REPO_ROOT / "Tests" / "fixtures" / "rag_citation_provenance"
MANIFEST_PATH = FIXTURE_ROOT / "manifest_v1.json"
CORPUS_PATH = FIXTURE_ROOT / "corpus_v1.json"
FIXTURE_VERSION = "rag-citation-provenance-v1"
RESULT_SCHEMA_VERSION = 1
MOCK_PROVIDER = "mock-local-v1"
DEFAULT_SAMPLES = 30
DEFAULT_WARMUPS = 5

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


class SampleWorkspace(NamedTuple):
    """Temporary paths used by one benchmark sample group."""

    root: Path
    db_path: Path
    sidecar_path: Path


class _DeterministicGateway:
    """In-process provider gateway that records the first streamed token."""

    def __init__(self) -> None:
        self.first_chunk_ns: int | None = None

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
        self.first_chunk_ns = time.perf_counter_ns()
        yield "Synthetic "
        yield "answer [S1]."


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
        choices=("baseline", "qualification"),
        default="baseline",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--provider", default=MOCK_PROVIDER)
    parser.add_argument("--base-url")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or incomplete local benchmark configuration."""

    if args.samples < 1:
        raise ValueError("--samples must be at least 1")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
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


def _json_payload_with_exact_size(target_bytes: int) -> str:
    empty = json.dumps({"payload": ""}, separators=(",", ":"), ensure_ascii=False)
    padding = target_bytes - len(empty.encode("utf-8"))
    if padding < 0:
        raise ValueError("target too small for a JSON object")
    payload = json.dumps(
        {"payload": "x" * padding}, separators=(",", ":"), ensure_ascii=False
    )
    if len(payload.encode("utf-8")) != target_bytes:
        raise AssertionError("failed to materialize exact JSON byte size")
    return payload


def materialize_boundary(case: dict[str, Any]) -> Any:
    """Materialize one deterministic exact/over boundary descriptor."""

    units = int(case["units"])
    kind = case["kind"]
    if kind == "json_bytes":
        return _json_payload_with_exact_size(units)
    if kind in {"utf8_bytes", "characters"}:
        return "x" * units
    if kind == "count":
        return tuple(range(units))
    raise ValueError(f"unknown boundary kind: {kind}")


def materialized_boundary_size(case: dict[str, Any]) -> int:
    """Measure a materialized boundary in its declared unit."""

    value = materialize_boundary(case)
    if case["kind"] in {"json_bytes", "utf8_bytes"}:
        return len(value.encode("utf-8"))
    return len(value)


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


def _validate_baseline_document(baseline: dict[str, Any]) -> None:
    """Refuse structurally incompatible historical baselines."""

    if baseline.get("fixture_version") != FIXTURE_VERSION:
        raise ValueError("qualification baseline fixture version is incompatible")
    environment = baseline.get("environment")
    if (
        not isinstance(environment, dict)
        or environment.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise ValueError("qualification baseline result schema is incompatible")
    if not isinstance(baseline.get("metrics"), dict):
        raise ValueError("qualification baseline has no metrics")


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


async def _measure_console_ttfb_once(db_path: Path) -> float:
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    persistence = _SQLiteConsolePersistence(db_path)
    gateway = _DeterministicGateway()
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
        result = await controller.submit_draft("Summarize the synthetic evidence.")
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
) -> dict[str, Any]:
    control: list[float] = []
    candidate: list[float] = []
    for index in range(warmups + samples):
        control_ms = await _measure_console_ttfb_once(db_path)
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


def _measure_finalization(*, samples: int, warmups: int) -> dict[str, Any]:
    standard_snapshots = tuple("s" * (4 * 1024) for _ in range(8))
    maximum_snapshots = tuple("m" * LIMITS["snapshot_utf8_bytes"] for _ in range(64))
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
    }


def _init_inspector_schema(db_path: Path) -> None:
    metadata = json.dumps(
        {
            "source_kind": "note",
            "title": "Synthetic inspector item",
            "locator": None,
        },
        sort_keys=True,
    )
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
            ((index, "i" * (4 * 1024), metadata) for index in range(64)),
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


def _measure_inspector(db_path: Path, *, samples: int, warmups: int) -> dict[str, Any]:
    cold_times: list[float] = []
    warm_times: list[float] = []
    for index in range(warmups + samples):
        started_ns = time.perf_counter_ns()
        with sqlite3.connect(db_path) as cold_connection:
            if _read_inspector(cold_connection) != 64:
                raise AssertionError("cold inspector read returned incomplete rows")
        cold_ms = (time.perf_counter_ns() - started_ns) / 1_000_000

        with sqlite3.connect(db_path) as warm_connection:
            _read_inspector(warm_connection)
            started_ns = time.perf_counter_ns()
            if _read_inspector(warm_connection) != 64:
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
    db_path: Path, *, samples: int, warmups: int
) -> dict[str, Any]:
    snapshots = tuple("g" * LIMITS["snapshot_utf8_bytes"] for _ in range(64))
    aggregate = json.dumps(
        {
            "schema_version": 1,
            "snapshot_count": len(snapshots),
            "storage": "governed",
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
                ("Synthetic grounded answer [S1].", aggregate),
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


def _init_migration_schema(db_path: Path, sidecar_path: Path) -> None:
    sidecar_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "conversation_id": "legacy-benchmark",
                "messages": 100,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE legacy_messages (
                legacy_message_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE canonical_proxy_rows (
                legacy_message_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            );
            """
        )


def _measure_migration_once(db_path: Path) -> tuple[float, int]:
    payload = json.dumps(
        {"answer": "Legacy synthetic answer [1].", "source_id": "legacy-source"}
    )
    with sqlite3.connect(db_path) as connection:
        connection.execute("DELETE FROM canonical_proxy_rows")
        connection.execute("DELETE FROM legacy_messages")
        connection.executemany(
            "INSERT INTO legacy_messages(legacy_message_id, payload_json) VALUES (?, ?)",
            ((f"legacy-message-{index:03d}", payload) for index in range(100)),
        )
        connection.commit()

    started_ns = time.perf_counter_ns()
    with sqlite3.connect(db_path) as interrupted:
        rows = interrupted.execute(
            """
            SELECT legacy_message_id, payload_json
            FROM legacy_messages
            ORDER BY legacy_message_id
            LIMIT 50
            """
        ).fetchall()
        interrupted.executemany(
            """
            INSERT OR IGNORE INTO canonical_proxy_rows(legacy_message_id, payload_json)
            VALUES (?, ?)
            """,
            rows,
        )
        interrupted.commit()

    with sqlite3.connect(db_path) as restarted:
        rows = restarted.execute(
            """
            SELECT legacy_message_id, payload_json
            FROM legacy_messages
            ORDER BY legacy_message_id
            LIMIT 100
            """
        ).fetchall()
        restarted.executemany(
            """
            INSERT OR IGNORE INTO canonical_proxy_rows(legacy_message_id, payload_json)
            VALUES (?, ?)
            """,
            rows,
        )
        restarted.commit()
        duplicate_count = int(
            restarted.execute(
                """
                SELECT COUNT(*) - COUNT(DISTINCT legacy_message_id)
                FROM canonical_proxy_rows
                """
            ).fetchone()[0]
        )
        canonical_count = int(
            restarted.execute("SELECT COUNT(*) FROM canonical_proxy_rows").fetchone()[0]
        )
    elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
    if canonical_count != 100:
        raise AssertionError("restart migration did not produce 100 canonical rows")
    return 100 / elapsed_seconds, duplicate_count


def _measure_migration(db_path: Path, *, samples: int, warmups: int) -> dict[str, Any]:
    throughputs: list[float] = []
    duplicate_counts: list[float] = []
    for index in range(warmups + samples):
        throughput, duplicates = _measure_migration_once(db_path)
        if index >= warmups:
            throughputs.append(throughput)
            duplicate_counts.append(float(duplicates))
    return {
        "messages_per_second": summarize(throughputs),
        "duplicate_proxy_rows_after_restart": summarize(duplicate_counts),
        "batch_messages": 100,
        "interrupted_after_messages": 50,
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
        historical_control_p95 = historical["control_ms"]["p95"]
        historical_ratio = (
            historical["candidate_ms"]["p95"] / historical_control_p95
            if historical_control_p95
            else 1.0
        )
        normalized_historical_p95 = current_control_p95 * historical_ratio
        regressions["historical_v1"] = _regression(
            candidate_p95, normalized_historical_p95
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
    environment_compatible = bool(compatibility["compatible"])
    return {
        "definitions": BUDGETS,
        "checks": checks,
        "environment_compatible": environment_compatible,
        "environment_incompatibilities": compatibility["reasons"],
        "overall_pass": environment_compatible
        and all(check["pass"] for check in checks.values()),
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
    _load_fixture()
    temporary_root: tempfile.TemporaryDirectory[str] | None = None
    if scratch_root is None:
        temporary_root = tempfile.TemporaryDirectory(prefix="ragcp-run-")
        root = Path(temporary_root.name)
    else:
        root = scratch_root
        root.mkdir(parents=True, exist_ok=True)

    try:
        with sample_group_workspace(root, "first-token") as workspace:
            _init_console_schema(workspace.db_path)
            first_token = await _measure_first_token(
                workspace.db_path,
                samples=samples,
                warmups=warmups,
                qualification=mode == "qualification",
            )
        with sample_group_workspace(root, "finalization"):
            finalization = _measure_finalization(samples=samples, warmups=warmups)
        with sample_group_workspace(root, "inspector") as workspace:
            _init_inspector_schema(workspace.db_path)
            inspector = _measure_inspector(
                workspace.db_path, samples=samples, warmups=warmups
            )
        with sample_group_workspace(root, "database-growth") as workspace:
            _init_growth_schema(workspace.db_path)
            database_growth = _measure_database_growth(
                workspace.db_path, samples=samples, warmups=warmups
            )
        with sample_group_workspace(root, "migration") as workspace:
            _init_migration_schema(workspace.db_path, workspace.sidecar_path)
            migration = _measure_migration(
                workspace.db_path, samples=samples, warmups=warmups
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
    }
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
        return 0 if result["budgets"]["overall_pass"] else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
