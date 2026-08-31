"""Release gates for normalized Console trace storage growth."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
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


FIXTURE_PATH = Path(__file__).with_name("fixtures") / "console_trace_growth_v1.json"
FIXTURE_CHECKSUM_PATH = FIXTURE_PATH.with_suffix(".sha256")
ARTIFACT_NAME = "console_trace_growth.json"
SCENARIO_EVIDENCE = {
    "edits": "Tests/Chat/test_console_trace_runtime.py::test_production_factory_persists_one_item_bounded_replacement",
    "regeneration": "Tests/Chat/test_trajectory_capture.py::test_regenerate_replacement_identity_resolves_after_persistence",
    "retries": "Tests/Chat/test_console_request_provenance_routes.py::test_fresh_retry_and_agent_rag_absence_is_explicit",
    "tools": "Tests/Chat/test_console_trace_provenance.py::test_provider_tool_loop_is_an_exact_ordered_message_overlay",
    "rag_or_project_context": "Tests/Chat/test_console_trace_provenance.py::test_capture_metadata_survives_without_parallel_rag_values_and_reaches_provider",
    "forks": "Tests/Chat/test_console_trace_fork_lineage.py::test_fork_attaches_shared_boundary_without_copying_trace_payload_rows",
    "failures": "Tests/Chat/test_console_trace_call_lifecycle.py::test_real_sqlite_adapter_observes_dispatch_started_commit",
    "credential_filtering": "Tests/Chat/test_console_trace_privacy_owners.py::test_repository_filters_credentials_before_header_and_artifact_identity",
    "legacy_migration": "Tests/Chat/test_console_trace_legacy_migration.py::test_batches_resume_delete_only_verified_rows_and_finish_idempotently",
    "logical_garbage_collection": "Tests/Chat/test_console_trace_graph_gc.py::test_completed_request_is_idempotent_and_reports_logical_reclamation",
}


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
            if not table.replace("_", "").isalnum():
                raise AssertionError(f"unsafe trace table name: {table}")
            columns = tuple(
                str(row[1])
                for row in cursor.execute(f'PRAGMA table_info("{table}")').fetchall()
            )
            rows += int(cursor.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for column in columns:
                byte_count += int(
                    cursor.execute(
                        f'SELECT COALESCE(SUM(length(CAST("{column}" AS BLOB))), 0) '
                        f'FROM "{table}"'
                    ).fetchone()[0]
                )
    return {"rows": rows, "bytes": byte_count}


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
            summary = _semi_incompressible_text(f"{seed}:summary", turn, message_bytes)
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
                await _dispatch(gateway, resolution, active, policy, boundary_errors)
            except Exception as exc:
                raise AssertionError(
                    f"replacement dispatch failed at turn {turn} with {len(active)} active items"
                ) from exc
        content = _semi_incompressible_text(seed, turn, message_bytes)
        active.append(_saved_message(database, conversation_id, content))
        try:
            await _dispatch(gateway, resolution, active, policy, boundary_errors)
        except Exception as exc:
            raise AssertionError(
                f"append dispatch failed at turn {turn} with {len(active)} active items"
            ) from exc
        if turn == checkpoint_turn:
            halfway = _trace_owned_size(database)
    finish = _trace_owned_size(database)
    assert halfway is not None
    with database.transaction() as cursor:
        legacy_exchange_count = int(
            cursor.execute("SELECT COUNT(*) FROM message_exchanges").fetchone()[0]
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
    await gateway.aclose()
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
@pytest.mark.parametrize("replacement_heavy", [False, True], ids=["append", "replace"])
async def test_console_trace_growth_release_gate(tmp_path: Path, replacement_heavy: bool) -> None:
    fixture = _fixture()
    database_count = int(fixture["fresh_database_count"])
    thresholds = fixture["thresholds"]
    assert isinstance(thresholds, dict)
    assert fixture["coverage"] == SCENARIO_EVIDENCE
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
