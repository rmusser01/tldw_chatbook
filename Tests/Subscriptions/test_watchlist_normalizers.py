import pytest

pytestmark = pytest.mark.unit


def test_normalize_carries_the_reader_fields():
    """The reader cannot render what the read path drops.

    `get_new_items` is `SELECT i.*`, so the body is present in the row. This
    normalizer rebuilt an explicit dict and omitted it, which meant every
    downstream consumer saw a title-only item no matter what was persisted.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 7,
        "subscription_id": 3,
        "title": "Claude Opus 4.5 is now available",
        "url": "https://example.test/a",
        "content": "body text that must survive",
        "content_kind": "article",
        "content_format": "markdown",
        "change_percentage": None,
        "change_type": None,
        "diff_summary": None,
    }

    item = normalize_watchlist_item("local", row)

    assert item["content"] == "body text that must survive"
    assert item["content_kind"] == "article"
    assert item["content_format"] == "markdown"


def test_normalize_carries_the_change_fields():
    """A `change` item renders from these three and nothing else."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 8,
        "subscription_id": 3,
        "title": "anthropic.com/news",
        "url": "https://anthropic.test/news",
        "content": "+ added line\n- removed line",
        "content_kind": "change",
        "content_format": "diff",
        "change_percentage": 12.0,
        "change_type": "structural",
        "diff_summary": "2 lines changed",
    }

    item = normalize_watchlist_item("local", row)

    assert item["content_kind"] == "change"
    assert item["change_percentage"] == 12.0
    assert item["change_type"] == "structural"
    assert item["diff_summary"] == "2 lines changed"


def test_normalize_tolerates_a_row_with_no_body():
    """Every pre-existing item has `content` NULL; that must not raise."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    item = normalize_watchlist_item(
        "local", {"id": 9, "subscription_id": 3, "title": "Old item"}
    )

    assert item["content"] is None
    assert item["content_kind"] is None


def test_normalize_marks_a_paused_source_paused_and_says_so_in_status(
) -> None:
    """task-2050 AC#1: a paused source must be distinguishable from
    inactive/healthy -- both `paused` and `status_summary` carry it."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_local_subscription_row,
    )

    source = normalize_local_subscription_row(
        {
            "id": 1,
            "name": "Dead feed",
            "type": "rss",
            "source": "https://dead.example/feed",
            "is_active": 1,
            "is_paused": 1,
        }
    )

    assert source["paused"] is True
    assert source["status_summary"] == "paused"
    assert source["active"] is False, (
        "a paused source must not also read as active in the Active column"
    )


def test_normalize_a_paused_source_with_a_last_error_still_says_paused(
) -> None:
    """task-2050: precedence is paused > error, not error > paused.

    A source auto-paused by repeated failures (task-1410) always still
    carries the `last_error` that caused the pause. An error-first
    precedence would render "error (N)" -- indistinguishable from a source
    that is merely having a bad day but is STILL being retried on schedule.
    Paused has to win so the Status column's headline does not lie about
    that; the error detail itself is not lost, it stays on the entity's own
    `last_error`/`error_count` keys for the Inspector to show.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_local_subscription_row,
    )

    source = normalize_local_subscription_row(
        {
            "id": 2,
            "name": "Flaky feed",
            "type": "rss",
            "source": "https://flaky.example/feed",
            "is_active": 1,
            "is_paused": 1,
            "last_error": "connection refused",
            "error_count": 5,
        }
    )

    assert source["paused"] is True
    assert source["status_summary"] == "paused", (
        "paused must win over error in the status headline"
    )
    assert "error" not in source["status_summary"]


def test_normalize_a_healthy_or_merely_inactive_source_is_unchanged(
) -> None:
    """Regression: task-2050 must not touch the two pre-existing statuses."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_local_subscription_row,
    )

    healthy = normalize_local_subscription_row(
        {
            "id": 3,
            "name": "Healthy feed",
            "type": "rss",
            "source": "https://ok.example/feed",
            "is_active": 1,
        }
    )
    assert healthy["paused"] is False
    assert healthy["status_summary"] == "active"
    assert healthy["active"] is True

    inactive = normalize_local_subscription_row(
        {
            "id": 4,
            "name": "Disabled feed",
            "type": "rss",
            "source": "https://off.example/feed",
            "is_active": 0,
        }
    )
    assert inactive["paused"] is False
    assert inactive["status_summary"] == "inactive"
    assert inactive["active"] is False

    errored = normalize_local_subscription_row(
        {
            "id": 5,
            "name": "Errored feed",
            "type": "rss",
            "source": "https://err.example/feed",
            "is_active": 1,
            "last_error": "timeout",
            "error_count": 2,
        }
    )
    assert errored["paused"] is False
    assert errored["status_summary"] == "error (2)"


def test_normalize_server_watchlist_source_never_reports_paused() -> None:
    """task-2050: the server watchlist source model has no pause concept
    yet -- `paused` must always be False, never sourced from `active`."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_server_watchlist_source,
    )

    source = normalize_server_watchlist_source(
        {
            "id": 9,
            "name": "Server source",
            "url": "https://example.test/feed",
            "active": False,
        }
    )

    assert source["paused"] is False


# --- TASK-2305: run accounting is lifted out of the nested `stats` blob ----


def _run(**payload):
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_run,
    )

    base = {"id": 4, "source_id": 2, "job_id": 2, "status": "completed"}
    base.update(payload)
    return normalize_watchlist_run("local", base)


def test_run_counters_come_from_the_names_the_pipeline_actually_writes():
    """The whole of F33 in one assertion.

    `execute_run` records `items_found`/`items_ingested`/`items_filtered`
    inside `stats`; the Runs pane reads `found_count`/`processed_count`/
    `filtered_count` off the run's top level. Nothing bridged the two, so
    every run displayed four zeros.
    """
    run = _run(
        stats={
            "items_found": 30,
            "items_ingested": 28,
            "items_filtered": 2,
            "response_time_ms": 412,
        }
    )

    assert run["found_count"] == 30
    assert run["processed_count"] == 28
    assert run["filtered_count"] == 2
    assert run["error_count"] == 0
    # The nested blob is still carried verbatim for the consumers that read it.
    assert run["stats"]["items_found"] == 30


def test_a_run_recorded_before_the_filtered_count_existed_derives_it():
    """Rows written before TASK-2305 have no `items_filtered` to lift."""
    run = _run(stats={"items_found": 10, "items_ingested": 4})

    assert run["filtered_count"] == 6


def test_a_malformed_stats_blob_never_reports_negative_filtering():
    run = _run(stats={"items_found": 2, "items_ingested": 9})

    assert run["filtered_count"] == 0


def test_a_failed_run_with_no_error_counter_reports_one_error():
    run = _run(status="failed", error_msg="feed unreachable", stats={})

    assert run["error_count"] == 1
    assert run["found_count"] == 0


def test_a_url_family_runs_error_count_comes_from_its_dispositions():
    """`dispositions` is the per-URL truth for the url/url_list/sitemap arms."""
    run = _run(
        status="completed",
        stats={
            "items_found": 3,
            "items_ingested": 3,
            "dispositions": {"changed": 3, "error": 2},
        },
    )

    assert run["error_count"] == 2


def test_duration_is_measured_between_the_runs_own_timestamps():
    run = _run(
        started_at="2026-08-04T10:00:00+00:00",
        finished_at="2026-08-04T10:00:04.800000+00:00",
        stats={},
    )

    assert run["duration"] == "4.8s"


@pytest.mark.parametrize(
    "finished,expected",
    [
        ("2026-08-04T10:00:00.820000+00:00", "820ms"),
        ("2026-08-04T10:02:03+00:00", "2m 3s"),
        ("2026-08-04T11:04:00+00:00", "1h 4m"),
    ],
)
def test_duration_scales_its_units(finished, expected):
    run = _run(
        started_at="2026-08-04T10:00:00+00:00", finished_at=finished, stats={}
    )

    assert run["duration"] == expected


def test_an_unfinished_run_reports_no_duration():
    """`-` is honest for a queued or running row; an elapsed time is not."""
    run = _run(status="running", started_at="2026-08-04T10:00:00+00:00", stats={})

    assert run["duration"] is None


def test_duration_falls_back_to_the_recorded_response_time():
    """A payload missing a timestamp still has the pipeline's own measurement."""
    run = _run(status="completed", stats={"response_time_ms": 1500})

    assert run["duration"] == "1.5s"


def test_watchlist_names_split_on_the_unit_separator_not_a_comma():
    """Watchlist names are user-typed; a comma in one must not split it."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        WATCHLIST_NAME_SEPARATOR,
    )

    run = _run(
        source_title="Hacker News",
        watchlist_names=f"Morning read{WATCHLIST_NAME_SEPARATOR}Security, daily",
        stats={},
    )

    assert run["source_title"] == "Hacker News"
    assert run["watchlist_names"] == ["Morning read", "Security, daily"]


def test_a_server_run_carries_no_source_name_rather_than_a_wrong_one():
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_run,
    )

    run = normalize_watchlist_run(
        "server", {"id": 8, "job_id": 3, "status": "completed", "stats": {}}
    )

    assert run["source_title"] is None
    assert run["watchlist_names"] == []
