"""Normalize local subscriptions and server watchlist sources for one shell."""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Mapping

from .watchlist_failure import (
    LEGACY_FAILURE_MESSAGE,
    LEGACY_FAILURE_NEXT_ACTION,
    sanitize_watchlist_failure_stats,
    watchlist_failure_from_stats,
    watchlist_failure_stats,
)


def _model_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value or {})


def build_watchlist_item_id(backend: str, entity_kind: str, source_id: Any) -> str:
    """Build the canonical local/server watchlist item id."""
    return f"{backend}:{entity_kind}:{source_id}"


def _coerce_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _coerce_tags(parsed)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in stripped.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _json_mapping(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


#: Separator the run query's `group_concat` uses to pack a source's watchlist
#: names into one column. ASCII UNIT SEPARATOR rather than a comma: watchlist
#: names are user-typed and a comma in one would otherwise split it into two
#: watchlists that do not exist.
WATCHLIST_NAME_SEPARATOR = "\x1f"


def _coerce_watchlist_names(value: Any) -> list[str]:
    """The watchlists a run's source belongs to, as a list.

    Sorted HERE, not in SQL (review wave, Minor 1). SQLite documents
    `group_concat`'s element order as arbitrary; the `ORDER BY` subquery the
    run query uses is a widely-relied-on workaround, not a contract, and the
    guaranteed form (`group_concat(x, sep ORDER BY x)`) needs SQLite >= 3.44
    while this project's `requires-python = ">=3.11"` admits older runtimes.
    It matters because `RunsPane._run_identity` prints only `names[0] +N`: an
    arbitrary order would name a different watchlist on successive reads of
    the same unchanged run, in the one place a run is identified at all.
    Sorting the parsed list is guaranteed, free, and normalises the
    list-input branch, which had no order either.

    Args:
        value: A list, a `WATCHLIST_NAME_SEPARATOR`-joined string (what the
            run query returns), or `None`.

    Returns:
        Names in a stable (sorted) order, blanks dropped.
    """
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        names = [str(name).strip() for name in value if str(name).strip()]
    else:
        names = [
            part.strip()
            for part in str(value).split(WATCHLIST_NAME_SEPARATOR)
            if part.strip()
        ]
    return sorted(names)


#: Run statuses that mean the run did not succeed. Mirrors
#: `WatchlistsCollectionsScreen._FAILED_RUN_STATUSES`; used only to give a run
#: that recorded no error counter an honest 1 rather than a flattering 0.
_FAILED_RUN_STATUSES = frozenset({"failed", "error", "errored"})


def _run_stat(stats: Mapping[str, Any], *keys: str) -> int | None:
    """The first of `keys` present in `stats` as an int, or `None`.

    Several aliases per counter because two backends write these: the local
    check pipeline records `items_found`/`items_ingested`, while a server
    run's `stats` blob is the server's own shape.

    A malformed value is skipped, never raised (Qodo, PR #1348). A server's
    `stats` blob is not guaranteed well-formed, and the old parse was
    `int(float(text))` guarded by `ValueError` alone -- so `"inf"`, `"nan"` and
    `"1e400"` all reached `int()`, which raises `OverflowError`/`ValueError`
    out through `normalize_watchlist_run` and takes the whole Runs table down
    rather than one counter. Integer strings are parsed as integers FIRST, so
    an arbitrarily large one stays exact instead of round-tripping through a
    float that cannot hold it.

    Args:
        stats: A run's `stats` mapping.
        *keys: Candidate keys, most authoritative first.

    Returns:
        The value as an int, or `None` when no key held a usable number.
    """
    for key in keys:
        value = stats.get(key)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            # A non-finite float reaches `int()` the same way a non-finite
            # string does, and raises the same way.
            if not math.isfinite(value):
                continue
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(text)
            except ValueError:
                pass
            try:
                parsed = float(text)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(parsed):
                continue
            try:
                return int(parsed)
            except (OverflowError, ValueError):
                continue
    return None


def _run_accounting(
    stats: Mapping[str, Any], *, status: str, error_msg: Any
) -> dict[str, int]:
    """The four per-run counters the Runs pane displays.

    Args:
        stats: The run's persisted `stats` blob.
        status: The run's status, so a failed run with no error counter still
            reports one error rather than a flattering zero.
        error_msg: The run's own error text, same reason.

    Returns:
        `found_count`, `processed_count`, `filtered_count`, `error_count`.
    """
    found = _run_stat(stats, "items_found", "found_count", "found") or 0
    processed = (
        _run_stat(stats, "items_ingested", "processed_count", "new_items_found") or 0
    )
    filtered = _run_stat(stats, "items_filtered", "filtered_count")
    if filtered is None:
        # The local answer, always: the check pipeline records no filtered
        # counter (review wave, Minor 3 -- it could never differ from this),
        # so everything found and not ingested was dropped by a filter. The
        # two keys above are read for a server blob that does carry one.
        # Never negative -- a stats blob whose `items_ingested` exceeds its
        # `items_found` is malformed, not evidence of negative filtering.
        filtered = max(found - processed, 0)
    errors = _run_stat(stats, "error_count", "items_errored")
    if errors is None:
        dispositions = stats.get("dispositions")
        if isinstance(dispositions, Mapping):
            errors = _run_stat(dispositions, "error") or 0
        elif error_msg or status.strip().lower() in _FAILED_RUN_STATUSES:
            errors = 1
        else:
            errors = 0
    return {
        "found_count": found,
        "processed_count": processed,
        "filtered_count": filtered,
        "error_count": errors,
    }


def _parse_run_timestamp(value: Any) -> datetime | None:
    """An ISO-8601 run timestamp as a `datetime`, or `None` if unreadable."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _run_duration_text(started_at: Any, finished_at: Any, stats: Mapping[str, Any]) -> str | None:
    """How long a run took, as a short human string, or `None`.

    `None` (rendered `-` by the pane) is the honest answer for a run that has
    not finished: an elapsed time for a `running` row would be a number that
    changes every time the table is repainted and never for a row already on
    screen.

    Args:
        started_at: The run's `started_at`.
        finished_at: The run's `finished_at`.
        stats: Its stats blob, whose `response_time_ms` is the fallback when
            the two timestamps cannot be read (a server payload that omits
            one, or a pre-existing row with a malformed value).

    Returns:
        e.g. `"820ms"`, `"4.8s"`, `"2m 3s"`, `"1h 4m"`, or `None`.
    """
    start = _parse_run_timestamp(started_at)
    end = _parse_run_timestamp(finished_at)
    seconds: float | None = None
    if start is not None and end is not None:
        if (start.tzinfo is None) == (end.tzinfo is None):
            elapsed = (end - start).total_seconds()
            if elapsed >= 0:
                seconds = elapsed
    if seconds is None:
        elapsed_ms = _run_stat(stats, "response_time_ms")
        if elapsed_ms is not None and elapsed_ms >= 0:
            seconds = elapsed_ms / 1000
    if seconds is None:
        return None
    if seconds < 1:
        return f"{int(round(seconds * 1000))}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def _alert_match_count(value: Any) -> int:
    """How many content-alert rules an item matched.

    `subscription_items.alert_matches` is a JSON list written by
    `WatchlistContentAlertService.evaluate` (or `None` when nothing matched),
    but the column comes back as raw text, so it is decoded here rather than
    at each display site.

    Args:
        value: The stored `alert_matches` column, already-decoded list, or
            `None`.

    Returns:
        The number of matches, or 0 for anything unparseable.
    """
    if value in (None, ""):
        return 0
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return 0
        return len(parsed) if isinstance(parsed, (list, tuple)) else 0
    return 0


def _local_source_settings(row: Mapping[str, Any]) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    scalar_fields = (
        "check_frequency",
        "extraction_method",
        "change_threshold",
        "auto_ingest",
    )
    for field in scalar_fields:
        value = row.get(field)
        if value is not None:
            settings[field] = value

    for field in (
        "extraction_rules",
        "processing_options",
        "notification_config",
        "rate_limit_config",
    ):
        parsed = _json_mapping(row.get(field))
        if parsed:
            settings[field] = parsed

    ignore_selectors = row.get("ignore_selectors")
    if ignore_selectors:
        settings["ignore_selectors"] = [
            selector.strip()
            for selector in str(ignore_selectors).split("\n")
            if selector.strip()
        ]
    return settings


def normalize_local_subscription_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a local subscriptions DB row as a watch item.

    task-2050 (AC#1): `status_summary` precedence is
    ``paused > error > inactive > active``. A source auto-paused by repeated
    check failures (`SubscriptionsDB._advance_failure_and_maybe_pause`,
    task-1410) always still carries the `last_error` that caused the pause,
    so an error-first precedence would render ``"error (10)"`` on a source
    that has actually STOPPED being retried -- indistinguishable from one
    that is merely having a bad day but is still being checked on schedule.
    That is the one fact a paused source's status needs to lead with: it
    needs an explicit Resume, not just time. Note this is a real trade-off,
    not a free win: today NEITHER `last_error` NOR `error_count` is
    surfaced anywhere else in the watchlists UI for a source (only a run's
    own `error_count` renders, in the Runs pane) -- so for the window a
    source is both paused and carrying the error that caused it, this
    precedence trades away the only place that error text was visible at
    all, in exchange for the Status column no longer implying the source is
    still being retried when it is not. The underlying `last_error`/
    `error_count` columns are untouched by this normalizer either way, and
    remain available to a future source-detail affordance.

    Args:
        row: A `subscriptions` table row (or an equivalent mapping) as the
            local backend reads it.

    Returns:
        The normalized watch-item dict: namespaced ``id``, ``entity_kind``
        ``"subscription"``, the display fields, ``paused`` (task-2050), and
        ``status_summary`` per the precedence above.
    """
    source_id = row["id"]
    paused = bool(row.get("is_paused", False))
    active = bool(row.get("is_active", True)) and not paused
    error_count = int(row.get("error_count") or 0)
    last_error = row.get("last_error")
    if paused:
        status_summary = "paused"
    elif last_error:
        status_summary = f"error ({error_count})" if error_count else "error"
    else:
        status_summary = "active" if active else "inactive"

    return {
        "id": build_watchlist_item_id("local", "subscription", source_id),
        "backend": "local",
        "entity_kind": "subscription",
        "source_id": source_id,
        "title": row.get("name") or "Untitled subscription",
        "description": row.get("description"),
        "source_type": row.get("type"),
        "url": row.get("source"),
        "active": active,
        # task-2050 AC#1: an inactive (is_active=0) source and an
        # auto-paused one both read `active: False` and, before this task,
        # both fell into the same "inactive" status text -- nothing told
        # them apart, and a paused source's only real recourse (task-1410's
        # data-layer resume-on-success) had no UI trigger at all. Carried
        # as its own field, distinct from `status_summary`, so a consumer
        # that only cares about the boolean (e.g. the Resume button's
        # visibility gate) does not have to string-match status text.
        "paused": paused,
        "tags": _coerce_tags(row.get("tags")),
        "group_ids": [],
        "settings": _local_source_settings(row),
        "status_summary": status_summary,
        "last_checked_or_scraped_at": row.get("last_checked")
        or row.get("last_successful_check"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def normalize_server_watchlist_source(
    source: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Normalize a server watchlist source response as a watch item."""
    payload = _model_to_dict(source)
    source_id = payload["id"]
    return {
        "id": build_watchlist_item_id("server", "watchlist_source", source_id),
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": source_id,
        "title": payload.get("name") or "Untitled source",
        "description": payload.get("description"),
        "source_type": payload.get("source_type"),
        "url": payload.get("url"),
        "active": bool(payload.get("active", True)),
        # task-2050: the server watchlist source model has no auto-pause
        # concept (no `is_paused` equivalent on the response payload) --
        # always False here, which is also why the Resume affordance never
        # renders for a server-backed source
        # (`InspectorPane._is_paused_subscription` gates on `entity_kind ==
        # "subscription"`, the local-only kind this field is meaningful for).
        "paused": False,
        "tags": _coerce_tags(payload.get("tags")),
        "group_ids": list(payload.get("group_ids") or []),
        "settings": dict(payload.get("settings") or {}),
        "status_summary": "active" if payload.get("active", True) else "inactive",
        "last_checked_or_scraped_at": payload.get("last_checked_at")
        or payload.get("last_scraped_at"),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def normalize_server_delete_response(
    response: Mapping[str, Any] | Any, *, source_id: Any
) -> dict[str, Any]:
    """Normalize server reversible-delete metadata."""
    payload = _model_to_dict(response)
    return {
        "success": bool(payload.get("success", True)),
        "id": build_watchlist_item_id(
            "server", "watchlist_source", payload.get("source_id", source_id)
        ),
        "backend": "server",
        "entity_kind": "watchlist_source",
        "source_id": payload.get("source_id", source_id),
        "restore_window_seconds": payload.get("restore_window_seconds"),
        "restore_expires_at": payload.get("restore_expires_at"),
    }


def normalize_watchlist_run(
    source: str, run: Mapping[str, Any] | Any
) -> dict[str, Any]:
    """Normalize local or server watchlist run metadata."""
    payload = _model_to_dict(run)
    run_id = payload["id"]
    source_id = payload.get("source_id")
    job_id = payload.get("job_id")
    if source_id is None and source == "local":
        source_id = job_id
    stats = dict(payload.get("stats") or {})
    status = payload.get("status") or "unknown"
    failed = str(status).strip().lower() in _FAILED_RUN_STATUSES
    if failed:
        stats, failure = sanitize_watchlist_failure_stats(stats)
    else:
        failure = watchlist_failure_from_stats(stats)
    if failure is not None:
        stats.update(watchlist_failure_stats(failure))
        error_msg = failure.message
        log_text = f"{failure.message} {failure.next_action}"
        next_action = failure.next_action
    elif failed:
        error_msg = LEGACY_FAILURE_MESSAGE
        log_text = f"{LEGACY_FAILURE_MESSAGE} {LEGACY_FAILURE_NEXT_ACTION}"
        next_action = LEGACY_FAILURE_NEXT_ACTION
    else:
        error_msg = payload.get("error_msg")
        log_text = payload.get("log_text")
        next_action = None
    counts = _run_accounting(stats, status=str(status), error_msg=error_msg)
    normalized = {
        "id": build_watchlist_item_id(source, "watchlist_run", run_id),
        "backend": source,
        "entity_kind": "watchlist_run",
        "run_id": run_id,
        "job_id": job_id,
        "source_id": source_id,
        "status": status,
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        # TASK-2305. The Runs pane has always read `found_count` and friends
        # off the run's own top level, and no normalizer had ever written
        # them: the numbers the check pipeline records live nested under
        # `stats` as `items_found`/`items_ingested`. Every
        # run therefore displayed `Found 0 · Processed 0 · Filtered 0 ·
        # Errors 0` however much it had actually harvested, which reads as if
        # checks do nothing. Lifted here, once, rather than teaching each
        # display site the nesting.
        **counts,
        "duration": _run_duration_text(
            payload.get("started_at"), payload.get("finished_at"), stats
        ),
        # The source's own name, resolved by the query that reads the run
        # (see `LocalWatchlistsService._RUN_SELECT`); absent for a server run,
        # whose API carries no source name.
        "source_title": payload.get("source_title") or None,
        "watchlist_names": _coerce_watchlist_names(payload.get("watchlist_names")),
        "stats": stats,
        "failure_category": failure.category.value if failure is not None else None,
        "retryable": failure.retryable if failure is not None else False,
        "http_status": failure.http_status if failure is not None else None,
        "retry_after_seconds": (
            failure.retry_after_seconds if failure is not None else None
        ),
        "next_action": next_action,
        "error_msg": error_msg,
        "filter_tallies": payload.get("filter_tallies"),
        "log_text": log_text,
        "log_path": payload.get("log_path"),
        "truncated": bool(payload.get("truncated", False)),
        "filtered_sample": payload.get("filtered_sample"),
    }
    # TASK-1362 Task 7 (spec §4): lift a url-family run's check dispositions
    # from the nested `stats` blob onto the run dict's own top level, the
    # same way the Runs pane reads every other per-run counter (`found_count`
    # and friends) -- so `RunsPane._stats_text` can render them without also
    # knowing the `stats` nesting. Only added when present at all: a feed/API
    # run's `stats` never carries `dispositions` (see `test_feed_runs_record_
    # no_dispositions`), and a run dict with no key renders identically to
    # one whose key is an empty dict, so there is no reason to fabricate one.
    dispositions = stats.get("dispositions")
    if isinstance(dispositions, Mapping):
        normalized["dispositions"] = dict(dispositions)
    # Whole-branch review, Critical 1: the same lift for the withheld
    # magnitude, which lives beside `dispositions` rather than inside it (see
    # `_disposition_counts`, which returns integers only). Absent whenever the
    # run withheld nothing, so `_stats_text` appends the number only when it
    # has one.
    max_withheld = stats.get("max_withheld_pct")
    if isinstance(max_withheld, (int, float)) and not isinstance(max_withheld, bool):
        normalized["max_withheld_pct"] = float(max_withheld)
    return normalized


def _coerce_condition_value(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
    return {"value": value}


def normalize_watchlist_alert_rule(
    source: str, rule: Mapping[str, Any] | Any
) -> dict[str, Any]:
    """Normalize local or server watchlist alert-rule metadata."""
    payload = _model_to_dict(rule)
    rule_id = payload["id"]
    job_id = payload.get("job_id")
    return {
        "id": build_watchlist_item_id(source, "watchlist_alert_rule", rule_id),
        "backend": source,
        "entity_kind": "watchlist_alert_rule",
        "rule_id": rule_id,
        "user_id": payload.get("user_id") or ("local" if source == "local" else None),
        "job_id": job_id,
        "source_id": payload.get("source_id") or job_id,
        "name": payload.get("name") or "Untitled alert rule",
        "enabled": bool(payload.get("enabled", True)),
        "condition_type": payload.get("condition_type"),
        "condition_value": _coerce_condition_value(payload.get("condition_value")),
        "severity": payload.get("severity") or "warning",
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
    }


def normalize_watchlist_item(source: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a local subscriptions DB item row as a watchlist item."""
    item_id = row["id"]
    return {
        "id": build_watchlist_item_id(source, "watchlist_item", item_id),
        "backend": source,
        "entity_kind": "watchlist_item",
        "item_id": item_id,
        "source_id": row.get("subscription_id"),
        "source_name": row.get("subscription_name"),
        "source_type": row.get("subscription_type"),
        "title": row.get("title") or "Untitled item",
        "url": row.get("url") or row.get("canonical_url"),
        "status": row.get("status") or "new",
        # TASK-2306: which run produced this item, and how many content-alert
        # rules it matched. Both columns are on the row (`get_new_items`
        # selected `i.*` at the time this was written; TASK-15464 narrowed
        # that to an explicit column list, `SubscriptionsDB._LIST_ITEM_
        # COLUMNS`, which still names `run_id`/`alert_matches` -- both are
        # small scalars, not the large-payload columns that narrowing
        # dropped) and both are what the Runs tab's Items sub-region
        # displays -- its "Alerts" column had no source at all before this,
        # so it rendered `0` over every item however many alerts had fired.
        "run_id": row.get("run_id"),
        "alert_count": _alert_match_count(row.get("alert_matches")),
        "author": row.get("author"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "published_date": row.get("published_date"),
        "effective_date": row.get("effective_date"),
        # Phase D reader fields. Originally added because `get_new_items` was
        # `SELECT i.*` and these were already on the row but this dict simply
        # was not carrying them (title-only downstream regardless of what
        # Phase A persisted). TASK-15464: `get_new_items` is NO LONGER
        # `SELECT i.*` -- `content` is absent from a freshly-loaded LIST row
        # now (fetched separately, once, only for an opened item -- see
        # `SubscriptionsDB.get_item_content` and
        # `WatchlistsCollectionsScreen._load_item_content`). `row.get(...)`
        # here still does the right thing either way: `None` when absent,
        # the real value once `_load_item_content` has merged it in.
        "content": row.get("content"),
        # TASK-15464: the list row's OWN preview snippet source
        # (`article_list._render_row`'s `body_snippet`) -- a cheap `substr`
        # projection (`_LIST_ITEM_COLUMNS`), never the full body. Present on
        # every list row regardless of whether `content` itself is.
        "content_preview": row.get("content_preview"),
        "content_kind": row.get("content_kind"),
        # Read by `content_pane.render_article` to decide whether the body is
        # markdown source or plain text.
        "content_format": row.get("content_format"),
        # `change`-kind items render from these three
        # (`content_pane.render_change`).
        "change_percentage": row.get("change_percentage"),
        "change_type": row.get("change_type"),
        "diff_summary": row.get("diff_summary"),
        # Spec #2 phase 1 read-path lesson (Phase D's shape, repeated).
        # Coerce SQLite's 0/1 to an actual bool, or every downstream
        # consumer sees a truthy int instead of a real flag.
        "queued_for_briefing": bool(row.get("queued_for_briefing")),
        # task-3072: same column-present/bool-coercion shape for the star.
        "is_flagged": bool(row.get("is_flagged")),
        # `canonical_url` is deliberately NOT re-exported as its own key: it
        # is already folded into `url` two lines above (`row.get("url") or
        # row.get("canonical_url")`), and a second copy under a second name
        # had no consumer, so it was one more thing for a reader to have to
        # rule out.
    }
