"""Bounded-work benchmark for TASK-21351.

Run with the shared test plugin so repository test isolation is active:

    pytest -p Tests.conftest \\
      Docs/superpowers/qa/task-21351-console-switcher-activity/benchmark_evidence.py \\
      -q -s
"""

from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from types import SimpleNamespace

import pytest
from textual.widgets import Input

from Tests.UI.test_console_activity_switcher import (
    _ActivitySwitcherApp,
    _active_entry,
    _history_entry,
    _projection_controller,
)
from tldw_chatbook.Chat.console_activity_receipts import (
    ConsoleActivityReceiptService,
)
from tldw_chatbook.Chat.console_switcher_state import (
    ConsoleSwitcherHistoryPage,
    filter_console_active_results,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSessionSwitcherModal,
)

HERE = Path(__file__).resolve().parent
SIZES = {"small": 5, "representative": 50, "stress": 500}


def _summary(samples_ns: list[int]) -> dict[str, float | int]:
    ordered = sorted(samples_ns)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "samples": len(ordered),
        "median_ms": round(median(ordered) / 1_000_000, 3),
        "p95_ms": round(ordered[p95_index] / 1_000_000, 3),
    }


async def _wait_for_switcher_settle(
    modal: ConsoleSessionSwitcherModal, pilot
) -> None:  # type: ignore[no-untyped-def]
    for _ in range(100):
        await pilot.pause(0.01)
        if not modal._query_pending:
            return
    raise AssertionError("Switcher did not settle within the evidence bound")


async def _measure_open(active_results, history_loader) -> dict[str, float | int]:
    samples: list[int] = []
    for _ in range(7):
        app = _ActivitySwitcherApp(
            active_results=active_results,
            history_loader=history_loader,
        )
        started = perf_counter_ns()
        async with app.run_test(size=(120, 35)) as pilot:
            await pilot.pause()
            modal = app.screen
            assert isinstance(modal, ConsoleSessionSwitcherModal)
            assert not modal._query_pending
            samples.append(perf_counter_ns() - started)
    return _summary(samples)


async def _measure_modal_operations(active_results, history_loader):
    app = _ActivitySwitcherApp(
        active_results=active_results,
        history_loader=history_loader,
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleSessionSwitcherModal)
        query = modal.query_one("#console-switcher-query", Input)

        active_filter: list[int] = []
        zero_widen: list[int] = []
        f3_toggle: list[int] = []
        for index in range(20):
            started = perf_counter_ns()
            query.value = f"Agent {index % 5}"
            await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.02)
            await _wait_for_switcher_settle(modal, pilot)
            active_filter.append(perf_counter_ns() - started)

            started = perf_counter_ns()
            query.value = f"zero-match-{index}"
            await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.02)
            await _wait_for_switcher_settle(modal, pilot)
            zero_widen.append(perf_counter_ns() - started)

            started = perf_counter_ns()
            await pilot.press("f3")
            await _wait_for_switcher_settle(modal, pilot)
            f3_toggle.append(perf_counter_ns() - started)
            await pilot.press("f3")
            await _wait_for_switcher_settle(modal, pilot)

        return {
            "active_filter": _summary(active_filter),
            "zero_match_widening": _summary(zero_widen),
            "f3_toggle": _summary(f3_toggle),
            "mounted_selectable_widgets": len(
                app.screen.query(".console-switcher-result")
            ),
            "modal_rows": modal.region.height,
        }


class _BenchmarkRunsDB:
    receipt_capability_available = True

    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows

    def list_unseen_console_activity(self):
        return self.rows


def _receipt_rows(size: int) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "activity_id": f"activity-{index}",
            "origin": "ordinary",
            "logical_outcome_id": f"turn-{index}",
            "transition_revision": 1,
            "session_id": f"session-{index}",
            "conversation_id": None,
            "run_id": None,
            "assistant_message_id": f"message-{index}",
            "status": "done",
            "created_at": "2026-09-02T00:00:00+00:00",
        }
        for index in range(size)
    )


def _measure_receipt_refresh(size: int) -> dict[str, float | int]:
    rows = _receipt_rows(size)
    samples: list[int] = []
    for _ in range(30):
        service = ConsoleActivityReceiptService(_BenchmarkRunsDB(rows), None)
        started = perf_counter_ns()
        assert service.hydrate_from_storage() == size
        samples.append(perf_counter_ns() - started)
    result = _summary(samples)
    result["database_rows_materialized"] = size
    return result


async def _measure_history_page(size: int) -> dict[str, float | int]:
    def list_conversations(**kwargs):
        limit = int(kwargs["limit"])
        offset = int(kwargs["offset"])
        entries = [
            {
                "id": f"conversation-{offset + index}",
                "title": f"Conversation {offset + index}",
                "scope_type": "global",
                "state": "in-progress",
                "last_modified": "2026-09-02T00:00:00+00:00",
            }
            for index in range(min(limit, max(0, size - offset)))
        ]
        return {"items": entries, "pagination": {"total": size}}

    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-benchmark",
            authority_token="runtime-benchmark",
            activity_receipts=None,
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)
    samples: list[int] = []
    materialized = 0
    for _ in range(30):
        started = perf_counter_ns()
        page = await controller.load_console_session_switcher_history(
            query="conversation",
            offset=0,
            limit=50,
        )
        samples.append(perf_counter_ns() - started)
        materialized = len(page.entries)
    result = _summary(samples)
    result["database_rows_materialized"] = materialized
    result["reported_total"] = page.total
    return result


@pytest.mark.asyncio
async def test_record_bounded_switcher_work():
    evidence: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "method": (
            "perf_counter_ns; nearest-rank p95; warm in-process Textual CSS; "
            "no claim of improvement"
        ),
        "sizes": {},
    }
    for label, size in SIZES.items():
        active_results = tuple(
            _active_entry(
                f"session:{index}",
                f"Agent {index}",
                session_id=str(index),
            )
            for index in range(size)
        )

        async def load_history(*, query: str, offset: int, limit: int):
            entries = tuple(
                _history_entry(
                    f"conversation:{offset + index}",
                    f"Saved conversation {offset + index}",
                )
                for index in range(min(limit, max(0, size - offset)))
            )
            return ConsoleSwitcherHistoryPage(entries, offset, limit, size)

        pure_filter_samples: list[int] = []
        for _ in range(200):
            started = perf_counter_ns()
            filter_console_active_results(active_results, "Agent 1")
            pure_filter_samples.append(perf_counter_ns() - started)

        evidence["sizes"][label] = {  # type: ignore[index]
            "input_subjects": size,
            "modal_open": await _measure_open(active_results, load_history),
            "active_filter_pure": _summary(pure_filter_samples),
            "modal_operations": await _measure_modal_operations(
                active_results, load_history
            ),
            "history_page": await _measure_history_page(size),
            "receipt_cache_refresh": _measure_receipt_refresh(size),
        }

    output = HERE / "performance.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))

    for result in evidence["sizes"].values():  # type: ignore[union-attr]
        assert result["modal_operations"]["mounted_selectable_widgets"] <= 50
        assert result["modal_operations"]["modal_rows"] <= 35
        assert result["history_page"]["database_rows_materialized"] <= 50
