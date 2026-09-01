"""Release gates for normalized Console trace storage growth."""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TypedDict

import pytest

from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_provider_gateway import (
    ConsoleProviderGateway,
    ConsoleProviderResolution,
)
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy, new_opaque_id
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenance,
    TraceProvenanceSource,
    request_route_provenance,
)
from tldw_chatbook.Chat.console_trace_runtime import ConsoleTraceBoundaryFactory
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.sql_validation import (
    escape_identifier,
    validate_column_name,
    validate_table_name,
)

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "console_trace_growth_v1.json"
FIXTURE_CHECKSUM_PATH = FIXTURE_PATH.with_suffix(".sha256")
ARTIFACT_NAME = "console_trace_growth.json"
SCENARIO_EVIDENCE_TIMEOUT_SECONDS = 300
EXPECTED_SCENARIOS = frozenset(
    {
        "edits",
        "regeneration",
        "retries",
        "tools",
        "rag_or_project_context",
        "forks",
        "failures",
        "credential_filtering",
        "legacy_migration",
        "logical_garbage_collection",
    }
)


class _TraceSize(TypedDict):
    rows: int
    bytes: int


class _RunResult(TypedDict):
    run_index: int
    first_half: _TraceSize
    second_half: _TraceSize
    total: _TraceSize
    adapter_call_count: int
    legacy_exchange_count: int


def _fixture() -> dict[str, object]:
    expected = FIXTURE_CHECKSUM_PATH.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == expected
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _semi_incompressible_text(seed: str, turn: int, size: int) -> str:
    blocks: list[str] = []
    block = 0
    while sum(map(len, blocks)) < size:
        blocks.append(
            hashlib.sha256(f"{seed}:{turn}:{block}".encode()).hexdigest()
        )
        block += 1
    return "".join(blocks)[:size]


def _trace_owned_size(database: CharactersRAGDB) -> _TraceSize:
    with database.transaction() as cursor:
        tables = tuple(
            str(row[0])
            for row in cursor.execute(
                """SELECT name FROM sqlite_master
                     WHERE type = 'table' AND name LIKE 'console_trace_%'
                     ORDER BY name"""
            ).fetchall()
        )
        rows = 0
        byte_count = 0
        for table in tables:
            if not validate_table_name(table, "chachanotes"):
                raise AssertionError(f"unsafe trace table name: {table}")
            quoted_table = escape_identifier(table)
            columns = tuple(
                str(row[1])
                for row in cursor.execute(
                    f"PRAGMA table_info({quoted_table})"
                ).fetchall()
            )
            rows += int(
                cursor.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0]
            )
            for column in columns:
                if not validate_column_name(column):
                    raise AssertionError(f"unsafe trace column name: {column}")
                quoted_column = escape_identifier(column)
                byte_count += int(
                    cursor.execute(
                        "SELECT COALESCE(SUM(length("
                        f"CAST({quoted_column} AS BLOB))), 0) FROM {quoted_table}"
                    ).fetchone()[0]
                )
    return {"rows": rows, "bytes": byte_count}


def test_trace_owned_size_rejects_unsafe_dynamic_column_name(tmp_path: Path) -> None:
    database = CharactersRAGDB(tmp_path / "unsafe-column.sqlite", "unsafe-column")
    try:
        with database.transaction(immediate=True) as cursor:
            cursor.execute(
                'ALTER TABLE console_trace_graph_epoch ADD COLUMN "bad""name" TEXT'
            )

        with pytest.raises(AssertionError, match="unsafe trace column name"):
            _trace_owned_size(database)
    finally:
        database.close()


def _delta(after: _TraceSize, before: _TraceSize) -> _TraceSize:
    return {
        "rows": after["rows"] - before["rows"],
        "bytes": after["bytes"] - before["bytes"],
    }


def _saved_message(
    database: CharactersRAGDB,
    conversation_id: str,
    content: str,
) -> tuple[dict[str, str], SavedRevisionTraceProvenance]:
    message_id = database.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": content,
        }
    )
    assert message_id is not None
    with database.transaction() as cursor:
        revision = cursor.execute(
            """SELECT revision_id FROM console_trace_semantic_revisions
                 WHERE source_message_id = ? ORDER BY revision_sequence DESC LIMIT 1""",
            (message_id,),
        ).fetchone()
    assert revision is not None
    return (
        {"role": "user", "content": content},
        SavedRevisionTraceProvenance(str(revision[0])),
    )


def _semantic_request(
    active: list[tuple[dict[str, str], TraceProvenance]],
    policy: FrozenTracePolicy,
):
    return build_console_request(
        [message for message, _descriptor in active],
        message_provenance=tuple(descriptor for _message, descriptor in active),
        memory_provenance=(),
        mandatory_provenance=(),
        tool_provenance=(),
        metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        capture_policy=policy,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )


async def _dispatch(
    gateway: ConsoleProviderGateway,
    resolution: ConsoleProviderResolution,
    active: list[tuple[dict[str, str], TraceProvenance]],
    policy: FrozenTracePolicy,
    boundary_errors: list[Exception],
) -> None:
    prepared = gateway.prepare_chat_request(
        resolution,
        _semantic_request(active, policy),
        route=ConsoleRequestRoute.FRESH,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
    )
    try:
        chunks = [
            chunk
            async for chunk in gateway.stream_chat(
                resolution,
                prepared,
                route=ConsoleRequestRoute.FRESH,
                capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            )
        ]
    except Exception:
        if boundary_errors:
            raise boundary_errors.pop() from None
        raise
    assert chunks == ["ok"]


async def _run_fixture(
    root: Path,
    fixture: dict[str, object],
    *,
    run_index: int,
    replacement_heavy: bool,
) -> _RunResult:
    turns = int(fixture["turns"])
    checkpoint_turn = int(fixture["checkpoint_turn"])
    message_bytes = int(fixture["message_bytes"])
    interval = int(fixture["replacement_interval"])
    replacement_fraction = float(fixture["replacement_fraction"])
    seed = str(fixture["seed"])
    database = CharactersRAGDB(
        root / f"trace-growth-{replacement_heavy}-{run_index}.sqlite",
        f"trace-growth-{run_index}",
    )
    async with AsyncExitStack() as resources:
        resources.callback(database.close)
        conversation_id = database.add_conversation({"title": "trace growth fixture"})
        assert conversation_id is not None
        policy = FrozenTracePolicy(new_opaque_id(), "credentials-v1", False, None)
        adapter_calls: list[dict[str, object]] = []

        def adapter(**kwargs):
            adapter_calls.append(kwargs)
            return {"choices": [{"message": {"content": "ok"}}]}

        boundary_errors: list[Exception] = []
        boundary_factory = ConsoleTraceBoundaryFactory(database)

        def diagnostic_boundary_factory(*args):
            try:
                return boundary_factory(*args)
            except Exception as exc:
                boundary_errors.append(exc)
                raise

        gateway = ConsoleProviderGateway(
            chat_api_call_fn=adapter,
            trace_call_boundary_factory=diagnostic_boundary_factory,
        )
        resources.push_async_callback(gateway.aclose)
        resolution = ConsoleProviderResolution(
            provider="openai",
            base_url="https://api.example.invalid/v1",
            model="gpt-growth",
            ready=True,
            execution_key="openai",
            api_key="fixture-secret-must-not-be-persisted",
            streaming=False,
        )
        active: list[tuple[dict[str, str], TraceProvenance]] = []
        start = _trace_owned_size(database)
        halfway: _TraceSize | None = None
        for turn in range(1, turns + 1):
            if replacement_heavy and turn % interval == 0 and active:
                replaced_count = max(1, int(len(active) * replacement_fraction))
                summary = _semi_incompressible_text(
                    f"{seed}:summary", turn, message_bytes
                )
                active = [
                    (
                        {"role": "user", "content": summary},
                        ProviderArtifactTraceProvenance(
                            TraceProvenanceSource.ACTIVE_REQUEST,
                            policy,
                        ),
                    ),
                    *active[replaced_count:],
                ]
                try:
                    await _dispatch(
                        gateway,
                        resolution,
                        active,
                        policy,
                        boundary_errors,
                    )
                except Exception as exc:
                    raise AssertionError(
                        "replacement dispatch failed at turn "
                        f"{turn} with {len(active)} active items"
                    ) from exc
            content = _semi_incompressible_text(seed, turn, message_bytes)
            active.append(_saved_message(database, conversation_id, content))
            try:
                await _dispatch(
                    gateway,
                    resolution,
                    active,
                    policy,
                    boundary_errors,
                )
            except Exception as exc:
                raise AssertionError(
                    f"append dispatch failed at turn {turn} "
                    f"with {len(active)} active items"
                ) from exc
            if turn == checkpoint_turn:
                halfway = _trace_owned_size(database)
        finish = _trace_owned_size(database)
        assert halfway is not None
        with database.transaction() as cursor:
            legacy_exchange_count = int(
                cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[
                    0
                ]
            )
            persisted = " ".join(
                str(value)
                for table in (
                    "console_trace_calls",
                    "console_trace_artifacts",
                    "console_trace_request_headers",
                )
                for row in cursor.execute(f'SELECT * FROM "{table}"').fetchall()
                for value in row
            )
        assert "fixture-secret-must-not-be-persisted" not in persisted
        return {
            "run_index": run_index,
            "first_half": _delta(halfway, start),
            "second_half": _delta(finish, halfway),
            "total": _delta(finish, start),
            "adapter_call_count": len(adapter_calls),
            "legacy_exchange_count": legacy_exchange_count,
        }


def _median(results: list[_RunResult], section: str, metric: str) -> float:
    return statistics.median(
        result[section][metric]  # type: ignore[literal-required]
        for result in results
    )


@pytest.mark.asyncio
async def test_run_fixture_closes_owned_resources_when_measurement_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    closed = {"database": False, "gateway": False}

    class Database:
        def __init__(self, *_args: object) -> None:
            pass

        def add_conversation(self, _values: object) -> str:
            return "conversation-id"

        def close(self) -> None:
            closed["database"] = True

    class Gateway:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def aclose(self) -> None:
            closed["gateway"] = True

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "CharactersRAGDB", Database)
    monkeypatch.setattr(module, "ConsoleProviderGateway", Gateway)

    def fail_measurement(_database: object) -> _TraceSize:
        raise RuntimeError("measurement failed")

    monkeypatch.setattr(module, "_trace_owned_size", fail_measurement)

    with pytest.raises(RuntimeError, match="measurement failed"):
        await _run_fixture(
            tmp_path,
            _fixture(),
            run_index=0,
            replacement_heavy=False,
        )

    assert closed == {"database": True, "gateway": True}


@pytest.fixture(scope="module")
def verified_scenario_evidence() -> tuple[str, ...]:
    fixture = _fixture()
    coverage = fixture.get("coverage")
    assert isinstance(coverage, dict)
    assert frozenset(coverage) == EXPECTED_SCENARIOS
    node_ids = tuple(coverage[scenario] for scenario in sorted(coverage))
    assert all(isinstance(node_id, str) and node_id for node_id in node_ids)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *node_ids],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=False,
        timeout=SCENARIO_EVIDENCE_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return node_ids


def test_verified_scenario_evidence_bounds_subprocess_runtime(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)

    verified_scenario_evidence.__wrapped__()

    assert observed["timeout"] == SCENARIO_EVIDENCE_TIMEOUT_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_heavy", [False, True], ids=["append", "replace"])
async def test_console_trace_growth_release_gate(
    tmp_path: Path,
    replacement_heavy: bool,
    verified_scenario_evidence: tuple[str, ...],
) -> None:
    fixture = _fixture()
    database_count = int(fixture["fresh_database_count"])
    thresholds = fixture["thresholds"]
    assert isinstance(thresholds, dict)
    assert verified_scenario_evidence
    results = [
        await _run_fixture(
            tmp_path,
            fixture,
            run_index=run_index,
            replacement_heavy=replacement_heavy,
        )
        for run_index in range(database_count)
    ]
    first_bytes = _median(results, "first_half", "bytes")
    second_bytes = _median(results, "second_half", "bytes")
    first_rows = _median(results, "first_half", "rows")
    second_rows = _median(results, "second_half", "rows")
    total_bytes = _median(results, "total", "bytes")
    artifact = {
        "fixture_version": fixture["fixture_version"],
        "replacement_heavy": replacement_heavy,
        "runs": results,
        "medians": {
            "first_half_bytes": first_bytes,
            "second_half_bytes": second_bytes,
            "first_half_rows": first_rows,
            "second_half_rows": second_rows,
            "total_trace_owned_bytes": total_bytes,
            "second_half_to_first_half_bytes": second_bytes / first_bytes,
            "second_half_to_first_half_rows": second_rows / first_rows,
        },
    }
    artifact_path = tmp_path / (
        ARTIFACT_NAME.replace(".json", f"-{'replace' if replacement_heavy else 'append'}.json")
    )
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    expected_calls = int(fixture["turns"])
    if replacement_heavy:
        expected_calls += expected_calls // int(fixture["replacement_interval"])
    assert all(result["adapter_call_count"] == expected_calls for result in results)
    assert all(result["legacy_exchange_count"] == 0 for result in results)
    assert second_bytes / first_bytes <= float(
        thresholds["second_half_to_first_half_bytes_max"]
    )
    assert second_rows / first_rows <= float(
        thresholds["second_half_to_first_half_rows_max"]
    )
    assert total_bytes <= int(thresholds["total_trace_owned_bytes_max"])
