"""Deterministic large-history gate for completed speculative voice pairs."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_chatbook.Chat import chat_persistence_service as persistence_module
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_voice_promotion import (
    ConsoleSessionBindingOrigin,
    ResolvedVoicePromotionDestination,
    VoicePromotionContext,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


BASELINE_LOCATOR_ROWS = 1_000
LARGE_LOCATOR_ROWS = 100_000
HISTORY_WARMUPS = 10
HISTORY_TRIALS = 40
LARGE_P95_THRESHOLD_MS = 100.0
MEDIAN_DELTA_THRESHOLD_MS = 2.0
P95_DELTA_THRESHOLD_MS = 10.0

_TARGET_CONVERSATION_ID = "voice-history-target-conversation"
_TARGET_ROOT_MESSAGE_ID = "voice-history-target-root"
_CLIENT_ID = "voice-history-gate"
_CHECKPOINT_TABLE = "console_dispatch_checkpoints"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checked_locators() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            frozenset().union(
                persistence_module._VOICE_PROMOTION_MANDATORY_LOCATORS,
                persistence_module._VOICE_PROMOTION_FORBIDDEN_MESSAGE_LOCATORS,
                persistence_module._VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS,
            )
        )
    )


def _forbidden_locators() -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            persistence_module._VOICE_PROMOTION_FORBIDDEN_MESSAGE_LOCATORS
            | persistence_module._VOICE_PROMOTION_FORBIDDEN_REVISION_LOCATORS
        )
    )


def completed_pair_locator_inventory() -> dict[str, Any]:
    """Return the canonical completed-pair reconciliation locator inventory."""
    locators = [
        {"table": table, "column": column} for table, column in _checked_locators()
    ]
    return {
        "sha256": _canonical_sha256(locators),
        "count": len(locators),
        "locators": locators,
    }


def _locator_distribution() -> list[dict[str, Any]]:
    # Both checkpoint columns occur on the same physical row. Giving them 71
    # observations and the other 13 locators 66 yields exactly 1,000 locator
    # observations while keeping the two column populations identical.
    baseline = {
        locator: 71 if locator[0] == _CHECKPOINT_TABLE else 66
        for locator in _forbidden_locators()
    }
    if sum(baseline.values()) != BASELINE_LOCATOR_ROWS:
        raise AssertionError("completed-pair baseline distribution drifted")
    return [
        {
            "table": table,
            "column": column,
            "baseline_rows": baseline[(table, column)],
            "large_rows": baseline[(table, column)] * 100,
        }
        for table, column in _forbidden_locators()
    ]


class _TimedTransaction:
    def __init__(self, inner: Any, owner: "_TimedCharactersRAGDB") -> None:
        self._inner = inner
        self._owner = owner
        self._started_ns: int | None = None

    def __enter__(self) -> Any:
        cursor = self._inner.__enter__()
        # The wrapped transaction manager returns only after BEGIN IMMEDIATE
        # has successfully acquired its RESERVED lock.
        self._started_ns = time.perf_counter_ns()
        return cursor

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            return bool(self._inner.__exit__(exc_type, exc, traceback))
        finally:
            if self._started_ns is not None:
                self._owner._last_lock_held_ns = (
                    time.perf_counter_ns() - self._started_ns
                )
            self._owner._measure_next_immediate = False


class _TimedCharactersRAGDB(CharactersRAGDB):
    """CharactersRAGDB with benchmark-only outer transaction timing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._measure_next_immediate = False
        self._last_lock_held_ns: int | None = None
        super().__init__(*args, **kwargs)

    def transaction(self, *, immediate: bool = False) -> Any:
        inner = super().transaction(immediate=immediate)
        if immediate and self._measure_next_immediate:
            return _TimedTransaction(inner, self)
        return inner

    def arm_lock_timer(self) -> None:
        self._last_lock_held_ns = None
        self._measure_next_immediate = True

    def consume_lock_held_ms(self) -> float:
        value = self._last_lock_held_ns
        self._last_lock_held_ns = None
        if value is None or self._measure_next_immediate:
            raise RuntimeError("completed-pair transaction timing was not recorded")
        return value / 1_000_000


@dataclass(slots=True)
class _History:
    db: _TimedCharactersRAGDB
    service: ChatPersistenceService
    plans: list[dict[str, Any]]
    pragmas: dict[str, Any]
    physical_forbidden_rows: int

    def close(self) -> None:
        self.db.close_connection()


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _synthetic_expression(table: str, column: str, declared_type: str) -> str:
    prefix = _sql_text(f"history:{table}:{column}:")
    normalized_type = declared_type.upper()
    if "INT" in normalized_type or "BOOL" in normalized_type:
        return "n"
    if any(kind in normalized_type for kind in ("REAL", "FLOAT", "DOUBLE")):
        return "CAST(n AS REAL)"
    if "BLOB" in normalized_type:
        return f"CAST({prefix} || n AS BLOB)"
    return f"{prefix} || n"


def _seed_table(
    connection: sqlite3.Connection,
    *,
    table: str,
    locator_columns: tuple[str, ...],
    rows: int,
) -> None:
    columns = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    primary_keys = [column for column in columns if int(column[5]) > 0]
    selected: list[sqlite3.Row | tuple[Any, ...]] = []
    for column in columns:
        name = str(column[1])
        declared_type = str(column[2] or "TEXT")
        single_integer_primary_key = bool(
            len(primary_keys) == 1
            and int(column[5]) == 1
            and "INT" in declared_type.upper()
        )
        required_without_default = bool(column[3] and column[4] is None)
        if name in locator_columns or (
            not single_integer_primary_key
            and (required_without_default or int(column[5]) > 0)
        ):
            selected.append(column)

    names = ", ".join(f'"{column[1]}"' for column in selected)
    expressions = ", ".join(
        _synthetic_expression(table, str(column[1]), str(column[2] or "TEXT"))
        for column in selected
    )
    connection.execute(
        f"""WITH RECURSIVE counter(n) AS (
               VALUES(1)
               UNION ALL
               SELECT n + 1 FROM counter WHERE n < ?
             )
             INSERT INTO "{table}" ({names})
             SELECT {expressions} FROM counter""",
        (rows,),
    )


def _seed_unrelated_history(
    path: Path,
    distribution: list[dict[str, Any]],
    *,
    size_key: str,
) -> int:
    by_table: dict[str, dict[str, int]] = {}
    for locator in distribution:
        by_table.setdefault(str(locator["table"]), {})[str(locator["column"])] = int(
            locator[size_key]
        )

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("PRAGMA ignore_check_constraints = ON")
        placeholders = ", ".join("?" for _table in by_table)
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master "
            f"WHERE type = 'trigger' AND tbl_name IN ({placeholders})",
            tuple(sorted(by_table)),
        ).fetchall()
        connection.execute("BEGIN IMMEDIATE")
        for name, _sql in triggers:
            connection.execute(f'DROP TRIGGER "{name}"')
        for table, columns in sorted(by_table.items()):
            row_counts = set(columns.values())
            if len(row_counts) != 1:
                raise RuntimeError(f"incompatible locator counts for {table}")
            _seed_table(
                connection,
                table=table,
                locator_columns=tuple(sorted(columns)),
                rows=row_counts.pop(),
            )
        for _name, sql in triggers:
            if sql:
                connection.execute(str(sql))
        connection.execute("COMMIT")
        connection.execute("PRAGMA ignore_check_constraints = OFF")
        connection.execute("ANALYZE")
        return sum(next(iter(columns.values())) for columns in by_table.values())
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def _query_plans(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for table, column in _checked_locators():
        rows = connection.execute(
            f'EXPLAIN QUERY PLAN SELECT 1 FROM "{table}" '
            f'WHERE "{column}" IN (?, ?) LIMIT 1',
            ("absent-a", "absent-b"),
        ).fetchall()
        details = [str(row[3]) for row in rows]
        upper = " | ".join(details).upper()
        plans.append(
            {
                "table": table,
                "column": column,
                "details": details,
                "uses_search": "SEARCH" in upper,
                "uses_scan": "SCAN" in upper,
            }
        )
    return plans


def _sqlite_pragmas(connection: sqlite3.Connection) -> dict[str, Any]:
    names = (
        "journal_mode",
        "synchronous",
        "foreign_keys",
        "busy_timeout",
        "cache_size",
        "temp_store",
        "page_size",
        "auto_vacuum",
    )
    return {name: connection.execute(f"PRAGMA {name}").fetchone()[0] for name in names}


def _build_history(
    path: Path,
    distribution: list[dict[str, Any]],
    *,
    size_key: str,
) -> _History:
    initial = _TimedCharactersRAGDB(path, _CLIENT_ID)
    try:
        conversation_id = initial.add_conversation(
            {
                "id": _TARGET_CONVERSATION_ID,
                "title": "Completed pair history gate",
                "character_id": None,
            }
        )
        if conversation_id != _TARGET_CONVERSATION_ID:
            raise RuntimeError("history target conversation identity drifted")
        root_id = initial.add_message(
            {
                "id": _TARGET_ROOT_MESSAGE_ID,
                "conversation_id": conversation_id,
                "sender": "assistant",
                "role": "assistant",
                "content": "existing durable leaf",
            }
        )
        if root_id != _TARGET_ROOT_MESSAGE_ID:
            raise RuntimeError("history target root identity drifted")
        initial.set_conversation_active_leaf(conversation_id, root_id)
    finally:
        initial.close_connection()

    physical_rows = _seed_unrelated_history(path, distribution, size_key=size_key)
    db = _TimedCharactersRAGDB(path, _CLIENT_ID)
    connection = db.get_connection()
    return _History(
        db=db,
        service=ChatPersistenceService(db),
        plans=_query_plans(connection),
        pragmas=_sqlite_pragmas(connection),
        physical_forbidden_rows=physical_rows,
    )


def _active_leaf(history: _History) -> str:
    row = (
        history.db.get_connection()
        .execute(
            "SELECT active_leaf_message_id FROM conversations WHERE id = ?",
            (_TARGET_CONVERSATION_ID,),
        )
        .fetchone()
    )
    if row is None or type(row[0]) is not str:
        raise RuntimeError("completed-pair history target is unavailable")
    return row[0]


_USAGE_JSON = ProviderUsage(
    uncached_input=7,
    output=11,
    provider="history-gate",
    model="deterministic",
).to_json()


def _promotion_case(
    *,
    promotion_id: str,
    expected_leaf: str,
) -> tuple[ResolvedVoicePromotionDestination, VoicePromotionContext]:
    origin = ConsoleSessionBindingOrigin(
        session_id="voice-history-session",
        session_incarnation=1,
        persisted_conversation_id=_TARGET_CONVERSATION_ID,
        conversation_binding_revision=1,
    )
    context = VoicePromotionContext(
        promotion_id=promotion_id,
        attempt_id=f"attempt:{promotion_id}",
        origin=origin,
        expected_native_leaf_id="native-history-leaf",
        expected_persisted_leaf_id=expected_leaf,
        user_text="deterministic private transcript",
        assistant_text="deterministic completed response",
        usage_json=_USAGE_JSON,
        terminal_boundary_id=f"boundary:{promotion_id}",
        capture_eligible_at_dispatch=True,
    )
    destination = ResolvedVoicePromotionDestination(
        session_id=origin.session_id,
        session_incarnation=origin.session_incarnation,
        persisted_conversation_id=_TARGET_CONVERSATION_ID,
        expected_persisted_leaf_id=expected_leaf,
        capture_eligible_at_dispatch=True,
    )
    return destination, context


def _measure(history: _History, case: tuple[Any, Any], *, retry: bool) -> float:
    destination, context = case
    history.db.arm_lock_timer()
    result = history.service.commit_completed_voice_pair(
        destination=destination,
        context=context,
    )
    if result.already_committed is not retry:
        raise RuntimeError("completed-pair history shape was not deterministic")
    return history.db.consume_lock_held_ms()


def _measure_pair(
    baseline: _History,
    large: _History,
    case: tuple[Any, Any],
    *,
    baseline_first: bool,
    retry: bool,
) -> tuple[float, float]:
    ordered = (
        (("baseline", baseline), ("large", large))
        if baseline_first
        else (("large", large), ("baseline", baseline))
    )
    samples = {
        label: _measure(history, case, retry=retry) for label, history in ordered
    }
    return samples["baseline"], samples["large"]


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def _sample_summary(samples: list[float]) -> dict[str, Any]:
    rounded_samples = [round(value, 6) for value in samples]
    return {
        "samples_ms": rounded_samples,
        "median_ms": round(statistics.median(rounded_samples), 6),
        "p95_ms": round(_percentile(rounded_samples, 0.95), 6),
    }


def _operation_report(
    baseline_samples: list[float],
    large_samples: list[float],
) -> dict[str, Any]:
    deltas = [
        large - baseline
        for baseline, large in zip(baseline_samples, large_samples, strict=True)
    ]
    large_p95 = _percentile(large_samples, 0.95)
    median_delta = statistics.median(deltas)
    p95_delta = _percentile(deltas, 0.95)
    thresholds = {
        "large_p95_ms": LARGE_P95_THRESHOLD_MS,
        "median_paired_delta_ms": MEDIAN_DELTA_THRESHOLD_MS,
        "p95_paired_delta_ms": P95_DELTA_THRESHOLD_MS,
    }
    return {
        "baseline": _sample_summary(baseline_samples),
        "large": _sample_summary(large_samples),
        "paired_delta": _sample_summary(deltas),
        "thresholds": thresholds,
        "passed": bool(
            large_p95 <= LARGE_P95_THRESHOLD_MS
            and median_delta <= MEDIAN_DELTA_THRESHOLD_MS
            and p95_delta <= P95_DELTA_THRESHOLD_MS
        ),
    }


def run_completed_pair_history_gate(directory: Path) -> dict[str, Any]:
    """Run the exact v69 baseline/large completed-pair qualification gate."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    distribution = _locator_distribution()
    baseline = _build_history(
        directory / "completed-pair-baseline.sqlite",
        distribution,
        size_key="baseline_rows",
    )
    large = _build_history(
        directory / "completed-pair-large.sqlite",
        distribution,
        size_key="large_rows",
    )
    try:
        for index in range(HISTORY_WARMUPS):
            expected_leaf = _active_leaf(baseline)
            if _active_leaf(large) != expected_leaf:
                raise RuntimeError("completed-pair histories diverged")
            case = _promotion_case(
                promotion_id=f"history-warmup-{index}",
                expected_leaf=expected_leaf,
            )
            baseline_first = index % 2 == 0
            _measure_pair(
                baseline,
                large,
                case,
                baseline_first=baseline_first,
                retry=False,
            )
            _measure_pair(
                baseline,
                large,
                case,
                baseline_first=baseline_first,
                retry=True,
            )

        fresh_baseline: list[float] = []
        fresh_large: list[float] = []
        retry_baseline: list[float] = []
        retry_large: list[float] = []
        for index in range(HISTORY_TRIALS):
            expected_leaf = _active_leaf(baseline)
            if _active_leaf(large) != expected_leaf:
                raise RuntimeError("completed-pair histories diverged")
            case = _promotion_case(
                promotion_id=f"history-trial-{index}",
                expected_leaf=expected_leaf,
            )
            baseline_first = index % 2 == 0
            baseline_ms, large_ms = _measure_pair(
                baseline,
                large,
                case,
                baseline_first=baseline_first,
                retry=False,
            )
            fresh_baseline.append(baseline_ms)
            fresh_large.append(large_ms)
            baseline_ms, large_ms = _measure_pair(
                baseline,
                large,
                case,
                baseline_first=baseline_first,
                retry=True,
            )
            retry_baseline.append(baseline_ms)
            retry_large.append(large_ms)

        baseline_scan_count = sum(plan["uses_scan"] for plan in baseline.plans)
        large_scan_count = sum(plan["uses_scan"] for plan in large.plans)
        scan_count = baseline_scan_count + large_scan_count
        search_only = all(
            plan["uses_search"] and not plan["uses_scan"]
            for plan in baseline.plans + large.plans
        )
        fresh_report = _operation_report(fresh_baseline, fresh_large)
        retry_report = _operation_report(retry_baseline, retry_large)
        pragmas_match = baseline.pragmas == large.pragmas
        row_counts = {
            "baseline": BASELINE_LOCATOR_ROWS,
            "large": LARGE_LOCATOR_ROWS,
        }
        report = {
            "inventory": completed_pair_locator_inventory(),
            "query_plans": {
                "baseline": baseline.plans,
                "large": large.plans,
            },
            "scan_count": scan_count,
            "row_counts": row_counts,
            "physical_forbidden_rows": {
                "baseline": baseline.physical_forbidden_rows,
                "large": large.physical_forbidden_rows,
            },
            "distribution": distribution,
            "distribution_sha256": _canonical_sha256(distribution),
            "warmup_count": HISTORY_WARMUPS,
            "trial_count": HISTORY_TRIALS,
            "measurement_order": "ABBA",
            "sqlite": {
                "version": sqlite3.sqlite_version,
                "pragmas_match": pragmas_match,
                "baseline_pragmas": baseline.pragmas,
                "large_pragmas": large.pragmas,
            },
            "fresh_commit": fresh_report,
            "uncertain_retry": retry_report,
            "passed": bool(
                search_only
                and scan_count == 0
                and pragmas_match
                and row_counts
                == {"baseline": BASELINE_LOCATOR_ROWS, "large": LARGE_LOCATOR_ROWS}
                and fresh_report["passed"]
                and retry_report["passed"]
            ),
        }
        return report
    finally:
        baseline.close()
        large.close()


__all__ = [
    "BASELINE_LOCATOR_ROWS",
    "HISTORY_TRIALS",
    "HISTORY_WARMUPS",
    "LARGE_LOCATOR_ROWS",
    "completed_pair_locator_inventory",
    "run_completed_pair_history_gate",
]
