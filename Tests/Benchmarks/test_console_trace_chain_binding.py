"""Opt-in benchmark: dispatch binding of a chained trace replacement.

TASK-40578 measured binding a rewritten tail stored as a chain (ADR-337
amendment 2026-11-06) with this probe; TASK-42130 reuses it for its before and
after. It is skipped unless ``TLDW_TRACE_CHAIN_BINDING_BENCH=1``, so no PR lane
runs it. A run at the 256-link cap takes several minutes.

Each shape is the Compact now shape: N recorded rows, a committed conversation
memory, then a send of 3 rows, so the whole history is replaced as a chain of
ceil(N / 256) links. A template database is copied with the SQLite backup API
before each run; a fresh service prepares the send and then binds it, warm, as
the runtime does. The bind call is timed with ``perf_counter`` and
``thread_time`` after ``gc.collect()``, with load1 before and after. Wrappers
time each ``_validate_replacement_range`` pass and its lineage reads.

Environment (all optional except the opt-in flag):
  TLDW_TRACE_CHAIN_BINDING_ROWS          comma list of N (default "1999")
  TLDW_TRACE_CHAIN_BINDING_RUNS          runs per shape (default 3)
  TLDW_TRACE_CHAIN_BINDING_DROP_TRIGGER  "1" drops the replacement lineage
                                         trigger from each run's copy
                                         (attribution only)
  TLDW_TRACE_CHAIN_BINDING_OUT           JSON file for the per-run results

Run, for example:
  TLDW_TRACE_CHAIN_BINDING_BENCH=1 TLDW_TRACE_CHAIN_BINDING_ROWS=1999,65536 \\
    .venv/bin/python -m pytest Tests/Benchmarks/test_console_trace_chain_binding.py \\
    -q -p no:cacheprovider -s
"""

from __future__ import annotations

import gc
import json
import os
import sqlite3
import statistics
import time
from dataclasses import dataclass, field

import pytest

from Tests.Chat import test_console_trace_chained_replacement as chained
from Tests.Chat import test_console_trace_rewritten_tail_planner as planner
from tldw_chatbook.Chat import console_send_diagnostics
from tldw_chatbook.Chat import console_trace_final_values as final_values
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(
        os.environ.get("TLDW_TRACE_CHAIN_BINDING_BENCH") != "1",
        reason="opt-in benchmark: set TLDW_TRACE_CHAIN_BINDING_BENCH=1",
    ),
]

_LINEAGE_TRIGGER = "console_trace_surface_replacements_lineage"


@dataclass
class _Passes:
    """Time spent in each link-validation pass of one bind."""

    active: bool = False
    depth: int = 0
    reads_wall: float = 0.0
    reads_cpu: float = 0.0
    rows_read: int = 0
    passes: list[tuple[float, float, float, float]] = field(default_factory=list)


def _instrument(monkeypatch: pytest.MonkeyPatch, timing: _Passes) -> None:
    real_validate = ConsoleTraceService._validate_replacement_range
    real_read = ConsoleTraceRepository.read_lineage_surface_nodes

    def validate(self, *args, **kwargs):
        if not timing.active:
            return real_validate(self, *args, **kwargs)
        timing.depth += 1
        timing.reads_wall = timing.reads_cpu = 0.0
        wall, cpu = time.perf_counter(), time.thread_time()
        try:
            return real_validate(self, *args, **kwargs)
        finally:
            timing.passes.append(
                (
                    time.perf_counter() - wall,
                    time.thread_time() - cpu,
                    timing.reads_wall,
                    timing.reads_cpu,
                )
            )
            timing.depth -= 1

    def read(self, *args, **kwargs):
        if not timing.depth:
            return real_read(self, *args, **kwargs)
        wall, cpu = time.perf_counter(), time.thread_time()
        rows = real_read(self, *args, **kwargs)
        timing.reads_wall += time.perf_counter() - wall
        timing.reads_cpu += time.thread_time() - cpu
        timing.rows_read += len(rows)
        return rows

    monkeypatch.setattr(ConsoleTraceService, "_validate_replacement_range", validate)
    monkeypatch.setattr(ConsoleTraceRepository, "read_lineage_surface_nodes", read)


def _template(tmp_path, monkeypatch, rows_n: int):
    """Record ``rows_n`` rows and the Compact now send that replaces them."""

    path = tmp_path / f"template-{rows_n}.sqlite"
    database = CharactersRAGDB(str(path), "trace-chain-bench")
    chat = planner._Chat(database, [])
    rows = chat.chain(rows_n + 1)
    with monkeypatch.context() as setup:
        # Setup only. One captured request carries at most about 20,000
        # messages (100,000 sanitizer nodes, 5 per message); a larger shape is
        # recorded live in one send as an upper bound.
        real_sanitizer = final_values.CredentialSanitizer
        setup.setattr(
            final_values,
            "CredentialSanitizer",
            lambda **kwargs: real_sanitizer(max_nodes=10**7, **kwargs),
        )
        chat.record(rows[:rows_n])
    chat.remember(rows[-3])
    current = chat.add(rows[-1], "after compaction")
    database.close_connection()
    return path, chat, [rows[-2], rows[-1], current], current


def _bind_once(tmp_path, template, chat, incoming, current, timing, label):
    copy = tmp_path / f"{label}.sqlite"
    source = sqlite3.connect(str(template))
    target = sqlite3.connect(str(copy))
    try:
        source.backup(target)
        if os.environ.get("TLDW_TRACE_CHAIN_BINDING_DROP_TRIGGER") == "1":
            target.execute(f"DROP TRIGGER {_LINEAGE_TRIGGER}")
            target.commit()
    finally:
        source.close()
        target.close()
    database = CharactersRAGDB(str(copy), "trace-chain-bench")
    try:
        chat.db = database
        chat.head = None
        service = ConsoleTraceService(chat.repository)
        admission, bind, _reserved, _sent = chained._prepare_send(
            chat, incoming, current, service=service
        )
        timing.passes.clear()
        timing.rows_read = 0
        timing.active = True
        gc.collect()
        load_before = os.getloadavg()[0]
        wall, cpu = time.perf_counter(), time.thread_time()
        call = bind()
        wall, cpu = time.perf_counter() - wall, time.thread_time() - cpu
        timing.active = False
        assert call.surface_node_id is not None
        return {
            "links": len(admission.replacement_range.chain),
            "load1_before": load_before,
            "load1_after": os.getloadavg()[0],
            "bind_wall_ms": wall * 1000,
            "bind_cpu_ms": cpu * 1000,
            "validate_cpu_ms": [item[1] * 1000 for item in timing.passes],
            "validate_reads_cpu_ms": [item[3] * 1000 for item in timing.passes],
            "rows_read": timing.rows_read,
        }
    finally:
        timing.active = False
        database.close_connection()


@pytest.mark.timeout(3600)
def test_console_trace_chain_binding_benchmark(tmp_path, monkeypatch):
    shapes = [
        int(value)
        for value in os.environ.get("TLDW_TRACE_CHAIN_BINDING_ROWS", "1999").split(",")
    ]
    runs = int(os.environ.get("TLDW_TRACE_CHAIN_BINDING_RUNS", "3"))
    monkeypatch.setattr(
        console_send_diagnostics, "record_send_stage", lambda *_args, **_kwargs: None
    )
    timing = _Passes()
    _instrument(monkeypatch, timing)
    report = []
    for rows_n in shapes:
        template, chat, incoming, current = _template(tmp_path, monkeypatch, rows_n)
        results = [
            _bind_once(
                tmp_path,
                template,
                chat,
                incoming,
                current,
                timing,
                f"run-{rows_n}-{run}",
            )
            for run in range(runs)
        ]
        cpu = [result["bind_cpu_ms"] for result in results]
        validation = [sum(result["validate_cpu_ms"]) for result in results]
        summary = {
            "rows": rows_n,
            "links": results[0]["links"],
            "runs": runs,
            "lineage_trigger_dropped": os.environ.get(
                "TLDW_TRACE_CHAIN_BINDING_DROP_TRIGGER"
            )
            == "1",
            "load1_min": min(result["load1_before"] for result in results),
            "load1_max": max(result["load1_after"] for result in results),
            "bind_wall_ms_median": statistics.median(
                result["bind_wall_ms"] for result in results
            ),
            "bind_cpu_ms_median": statistics.median(cpu),
            "validation_cpu_ms_median": statistics.median(validation),
            "validation_share_median": statistics.median(
                spent / total for spent, total in zip(validation, cpu, strict=True)
            ),
            "validation_passes": len(results[0]["validate_cpu_ms"]),
            "rows_read": results[0]["rows_read"],
            "results": results,
        }
        report.append(summary)
        print(
            "TRACE-CHAIN-BINDING "
            + json.dumps(
                {key: value for key, value in summary.items() if key != "results"}
            )
        )
    out = os.environ.get("TLDW_TRACE_CHAIN_BINDING_OUT")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    assert len(report) == len(shapes)
