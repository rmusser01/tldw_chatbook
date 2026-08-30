import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from threading import Event

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Tools.watchlists_command_service import WatchlistsCommandService


def _service(
    *,
    runtime="local",
    create_sources=None,
    create_collection=None,
    update=None,
    accept_checks=None,
    accept_briefing=None,
    collection_sources=None,
    set_briefing_schedule=None,
    briefing_gate=lambda: True,
    scheduler_running=lambda: True,
    request_reload=None,
    wait_reload=None,
    default_provider=lambda: "openai",
):
    def default_sources(rows):
        return [
            {
                "input_index": index,
                "outcome": "created",
                "source": {"source_id": index + 1},
            }
            for index, _row in enumerate(rows)
        ]

    def default_collection(**kwargs):
        return {
            "outcome": "created",
            "watchlist": {"id": 7, "name": kwargs["name"]},
            "membership_count": len(kwargs["source_ids"]),
        }

    def default_update(**kwargs):
        return {
            "watchlist_id": kwargs["watchlist_id"],
            "added": len(kwargs["add_ids"]),
            "removed": len(kwargs["remove_ids"]),
            "membership_count": len(kwargs["add_ids"]),
        }

    return WatchlistsCommandService(
        runtime_source_loader=lambda: runtime,
        create_sources_batch=create_sources or default_sources,
        create_collection=create_collection or default_collection,
        update_collection_sources=update or default_update,
        accept_source_checks=accept_checks,
        accept_briefing=accept_briefing,
        resolve_collection_sources=collection_sources,
        set_briefing_schedule=set_briefing_schedule,
        briefing_schedules_enabled=briefing_gate,
        scheduler_running=scheduler_running,
        request_scheduler_reload=request_reload,
        wait_scheduler_reload=wait_reload,
        default_briefing_defaults=(
            (lambda: (default_provider(), "default-model"))
            if default_provider is not None
            else None
        ),
    )


def _schedule_receipt(**overrides):
    receipt = {
        "watchlist_id": 7,
        "name": "Threat intel",
        "briefing_selection_mode": "auto_featured",
        "default_briefing_preset_id": 3,
        "default_preset_name": "Analyst",
        "preset_provider": "anthropic",
        "preset_model": "claude-sonnet",
        "briefing_cadence_seconds": 86_400,
        "last_attempt_at": "2026-08-27 18:00:00",
        "last_success_at": "2026-08-26 18:00:00",
    }
    receipt.update(overrides)
    return receipt


def test_set_briefing_schedule_returns_durable_acknowledged_receipt():
    writes = []
    waits = []
    token = SimpleNamespace(value=41)

    result = json.loads(
        _service(
            set_briefing_schedule=lambda watchlist_id, **kwargs: (
                writes.append((watchlist_id, kwargs)) or _schedule_receipt()
            ),
            request_reload=lambda: token,
            wait_reload=lambda supplied, timeout: (
                waits.append((supplied, timeout)) or True
            ),
        ).set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
            }
        )
    )

    assert writes == [(7, {"briefing_cadence_seconds": 86_400})]
    assert waits == [(token, 1.0)]
    assert result == {
        "status": "ok",
        "retryable": False,
        "collection_id": "local:watchlist:7",
        "collection_name": "Threat intel",
        "cadence": "every_24_hours",
        "cadence_seconds": 86_400,
        "selection_mode": "auto_featured",
        "preset_id": 3,
        "preset_name": "Analyst",
        "global_gate_enabled": True,
        "scheduler_running": True,
        "reload_requested": True,
        "reload_token": 41,
        "reload_acknowledged": True,
        "next_eligible_at": "2026-08-28T18:00:00+00:00",
        "last_attempt_at": "2026-08-27 18:00:00",
        "last_success_at": "2026-08-26 18:00:00",
        "briefing_route_ready": True,
        "preset_resolution_source": "stored_preset",
        "provider": "anthropic",
        "provider_resolution_source": "preset",
        "model": "claude-sonnet",
        "model_resolution_source": "preset",
        "recovery": "Chatbook schedules run while the app is open. Review the global gate in Settings and this collection's cadence in Artifacts.",
    }


@pytest.mark.parametrize(
    ("cadence", "stored_seconds", "canonical"),
    [
        ("every_12_hours", 43_200, "every_12_hours"),
        ("every_7_days", 604_800, "every_7_days"),
        ("off", None, "off"),
        (3_600, 3_600, "advanced"),
        (2_678_400, 2_678_400, "advanced"),
    ],
)
def test_set_briefing_schedule_accepts_exact_interval_vocabulary(
    cadence, stored_seconds, canonical
):
    writes = []

    result = json.loads(
        _service(
            set_briefing_schedule=lambda watchlist_id, **kwargs: (
                writes.append((watchlist_id, kwargs))
                or _schedule_receipt(
                    briefing_cadence_seconds=stored_seconds,
                    last_attempt_at=None,
                    last_success_at=None,
                )
            ),
            request_reload=lambda: SimpleNamespace(value=1),
            wait_reload=lambda _token, _timeout: True,
        ).set_briefing_schedule(
            {"collection_id": "local:watchlist:7", "cadence": cadence}
        )
    )

    assert writes == [(7, {"briefing_cadence_seconds": stored_seconds})]
    assert result["cadence"] == canonical
    if stored_seconds is None:
        assert result["next_eligible_at"] is None
    else:
        assert result["next_eligible_at"] is not None


@pytest.mark.parametrize(
    "cadence",
    [True, False, 3_599, 2_678_401, 86_400.0, "daily", "86400"],
)
def test_set_briefing_schedule_rejects_noncanonical_intervals_before_write(cadence):
    writes = []

    result = json.loads(
        _service(set_briefing_schedule=lambda *args, **kwargs: writes.append((args, kwargs)))
        .set_briefing_schedule(
            {"collection_id": "local:watchlist:7", "cadence": cadence}
        )
    )

    assert result["status"] == "invalid_argument"
    assert writes == []


def test_set_briefing_schedule_preserves_omitted_choices_and_commits_supplied_choices():
    writes = []

    json.loads(
        _service(
            set_briefing_schedule=lambda watchlist_id, **kwargs: (
                writes.append((watchlist_id, kwargs)) or _schedule_receipt()
            ),
            request_reload=lambda: SimpleNamespace(value=1),
            wait_reload=lambda _token, _timeout: True,
        ).set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "off",
                "preset_id": None,
                "selection_mode": "curated",
            }
        )
    )

    assert writes == [
        (
            7,
            {
                "briefing_cadence_seconds": None,
                "default_preset_id": None,
                "selection_mode": "curated",
            },
        )
    ]


@pytest.mark.parametrize(
    ("gate", "running", "wait_result", "requested", "acknowledged"),
    [
        (False, True, True, True, False),
        (True, False, False, True, False),
        (True, True, False, True, False),
    ],
)
def test_set_briefing_schedule_reports_disabled_stopped_and_timeout_honestly(
    gate, running, wait_result, requested, acknowledged
):
    request_count = []

    result = json.loads(
        _service(
            set_briefing_schedule=lambda _watchlist_id, **_kwargs: _schedule_receipt(),
            briefing_gate=lambda: gate,
            scheduler_running=lambda: running,
            request_reload=lambda: (
                request_count.append(True) or SimpleNamespace(value=9)
            ),
            wait_reload=lambda _token, _timeout: wait_result,
        ).set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
            }
        )
    )

    assert result["reload_requested"] is requested
    assert result["reload_acknowledged"] is acknowledged
    assert result["global_gate_enabled"] is gate
    assert result["scheduler_running"] is running
    assert len(request_count) == int(requested)


@pytest.mark.parametrize(
    "selection_mode",
    [
        pytest.param([], id="array"),
        pytest.param({}, id="object"),
        pytest.param(True, id="boolean"),
        pytest.param(1, id="integer"),
        pytest.param(None, id="null"),
    ],
)
def test_set_briefing_schedule_rejects_non_string_selection_mode_without_side_effects(
    selection_mode,
):
    """A malformed JSON value cannot escape the fixed command boundary."""
    writes = []
    reloads = []

    raw = _service(
        set_briefing_schedule=lambda *args, **kwargs: writes.append((args, kwargs)),
        request_reload=lambda: reloads.append(True),
    ).set_briefing_schedule(
        {
            "collection_id": "local:watchlist:7",
            "cadence": "every_24_hours",
            "selection_mode": selection_mode,
        }
    )

    assert json.loads(raw) == {
        "status": "invalid_argument",
        "retryable": False,
        "message": "Briefing selection mode is invalid.",
    }
    assert "unhashable" not in raw
    assert writes == []
    assert reloads == []


def test_set_briefing_schedule_refuses_server_mode_before_persistence():
    writes = []
    result = json.loads(
        _service(
            runtime="server",
            set_briefing_schedule=lambda *args, **kwargs: writes.append((args, kwargs)),
        ).set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
            }
        )
    )

    assert result["status"] == "unsupported"
    assert writes == []


def test_set_briefing_schedule_scrubs_persistence_failure_and_does_not_reload():
    reloads = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("token=secret /private/user.db")

    raw = _service(
        set_briefing_schedule=fail,
        request_reload=lambda: reloads.append(True),
    ).set_briefing_schedule(
        {
            "collection_id": "local:watchlist:7",
            "cadence": "every_24_hours",
        }
    )
    result = json.loads(raw)

    assert result == {
        "status": "feature_unavailable",
        "retryable": True,
        "message": "Watchlists storage is temporarily unavailable. Try again.",
    }
    assert "secret" not in raw and "user.db" not in raw
    assert reloads == []


def test_set_briefing_schedule_uses_pipeline_defaults_not_console_model_arguments():
    default_calls = []
    writes = []
    service = _service(
        set_briefing_schedule=lambda watchlist_id, **kwargs: (
            writes.append((watchlist_id, kwargs))
            or _schedule_receipt(
                default_briefing_preset_id=None,
                default_preset_name=None,
                preset_provider=None,
                preset_model=None,
            )
        ),
        request_reload=lambda: SimpleNamespace(value=1),
        wait_reload=lambda _token, _timeout: True,
        default_provider=lambda: default_calls.append(True) or "app-default",
    )

    result = json.loads(
        service.set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
            }
        )
    )
    rejected = json.loads(
        service.set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
                "model": "current-console-model",
            }
        )
    )

    assert result["provider"] == "app-default"
    assert result["provider_resolution_source"] == "app_default"
    assert result["model"] == "default-model"
    assert result["model_resolution_source"] == "app_default"
    assert default_calls == [True]
    assert writes == [(7, {"briefing_cadence_seconds": 86_400})]
    assert rejected["status"] == "invalid_argument"


def test_set_briefing_schedule_reports_committed_state_when_defaults_are_unavailable():
    writes = []
    reloads = []
    waits = []
    token = SimpleNamespace(value=17)

    def unavailable_default():
        raise RuntimeError("token=secret /private/config.toml")

    result = json.loads(
        _service(
            set_briefing_schedule=lambda watchlist_id, **kwargs: (
                writes.append((watchlist_id, kwargs))
                or _schedule_receipt(
                    default_briefing_preset_id=None,
                    default_preset_name=None,
                    preset_provider=None,
                    preset_model=None,
                )
            ),
            request_reload=lambda: reloads.append(True) or token,
            wait_reload=lambda supplied, timeout: (
                waits.append((supplied, timeout)) or True
            ),
            default_provider=unavailable_default,
        ).set_briefing_schedule(
            {
                "collection_id": "local:watchlist:7",
                "cadence": "every_24_hours",
            }
        )
    )

    assert writes == [(7, {"briefing_cadence_seconds": 86_400})]
    assert reloads == [True]
    assert waits == [(token, 1.0)]
    assert result["status"] == "ok"
    assert result["retryable"] is False
    assert result["cadence_seconds"] == 86_400
    assert result["reload_requested"] is True
    assert result["reload_acknowledged"] is True
    assert result["briefing_route_ready"] is False
    assert result["provider"] is None
    assert result["model"] is None
    assert result["provider_resolution_source"] == "unavailable"
    assert result["model_resolution_source"] == "unavailable"
    assert result["recovery"] == (
        "Schedule saved, but the briefing provider/model route is not ready. "
        "Configure it in Settings before briefings can run."
    )


def test_check_sources_accepts_exact_receipts_with_poll_contract():
    calls = []

    def accept(source_ids):
        calls.append(source_ids)
        return [
            {"run_id": 31, "source_id": 4, "status": "queued"},
            {"run_id": 32, "source_id": 8, "status": "queued"},
        ]

    result = json.loads(
        _service(accept_checks=accept).check_sources(
            {
                "source_ids": [
                    "local:subscription:4",
                    "local:subscription:8",
                ]
            }
        )
    )

    assert calls == [[4, 8]]
    assert result == {
        "status": "accepted",
        "retryable": False,
        "operations": [
            {
                "operation_id": "local:watchlist_run:31",
                "source_id": "local:subscription:4",
                "status": "queued",
                "poll_tool": "watchlists_get_operation_status",
                "poll_arguments": {"operation_id": "local:watchlist_run:31"},
                "suggested_poll_seconds": 2,
                "maximum_poll_seconds": 8,
                "terminal_states": ["completed", "failed", "cancelled"],
            },
            {
                "operation_id": "local:watchlist_run:32",
                "source_id": "local:subscription:8",
                "status": "queued",
                "poll_tool": "watchlists_get_operation_status",
                "poll_arguments": {"operation_id": "local:watchlist_run:32"},
                "suggested_poll_seconds": 2,
                "maximum_poll_seconds": 8,
                "terminal_states": ["completed", "failed", "cancelled"],
            },
        ],
    }


def test_check_sources_resolves_one_collection_and_rejects_oversize_before_accept():
    accepted = []

    def resolve(watchlist_id):
        assert watchlist_id == 7
        return list(range(1, 52))

    result = json.loads(
        _service(
            accept_checks=lambda ids: accepted.append(ids),
            collection_sources=resolve,
        ).check_sources({"collection_id": "local:watchlist:7"})
    )

    assert result["status"] == "invalid_argument"
    assert "at most 50" in result["message"]
    assert accepted == []


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"source_ids": [], "collection_id": "local:watchlist:1"},
        {"source_ids": ["local:subscription:1"] * 2},
        {"source_ids": [f"local:subscription:{number}" for number in range(1, 52)]},
        {"collection_id": "local:watchlist:0"},
    ],
)
def test_check_sources_rejects_ambiguous_or_unbounded_scope(arguments):
    calls = []
    result = json.loads(
        _service(
            accept_checks=lambda ids: calls.append(ids),
            collection_sources=lambda _watchlist_id: [1],
        ).check_sources(arguments)
    )

    assert result["status"] == "invalid_argument"
    assert calls == []


def test_generate_briefing_returns_exact_durable_poll_contract():
    calls = []

    def accept(watchlist_id, preset_id):
        calls.append((watchlist_id, preset_id))
        return {"id": 42, "watchlist_id": watchlist_id, "status": "generating"}

    result = json.loads(
        _service(accept_briefing=accept).generate_briefing(
            {"collection_id": "local:watchlist:7", "preset_id": 3}
        )
    )

    assert calls == [(7, 3)]
    assert result == {
        "status": "accepted",
        "retryable": False,
        "operation_id": "local:briefing:42",
        "collection_id": "local:watchlist:7",
        "receipt_status": "generating",
        "poll_tool": "watchlists_get_operation_status",
        "poll_arguments": {"operation_id": "local:briefing:42"},
        "suggested_poll_seconds": 2,
        "maximum_poll_seconds": 8,
        "terminal_states": ["complete", "empty", "failed"],
    }


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("check_sources", {"source_ids": ["local:subscription:1"]}),
        ("generate_briefing", {"collection_id": "local:watchlist:1"}),
    ],
)
def test_long_commands_refuse_server_mode_before_coordinator(method, arguments):
    calls = []
    service = _service(
        runtime="server",
        accept_checks=lambda ids: calls.append(ids),
        accept_briefing=lambda watchlist_id, preset_id: calls.append(
            (watchlist_id, preset_id)
        ),
    )

    result = json.loads(getattr(service, method)(arguments))

    assert result["status"] == "unsupported"
    assert calls == []


def test_delayed_collection_mutation_returns_only_after_definitive_commit(tmp_path):
    database = SubscriptionsDB(tmp_path / "subscriptions.db", "test")
    owner = WatchlistBundleService(database)
    entered = Event()
    release = Event()

    def delayed_create(**kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return owner.create_with_sources(**kwargs)

    try:
        service = WatchlistsCommandService(
            runtime_source_loader=lambda: "local",
            create_sources_batch=lambda _rows: [],
            create_collection=delayed_create,
            update_collection_sources=lambda **_kwargs: {},
        )
    except TypeError:
        pytest.fail("short mutations still require the unsafe app-loop bridge")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            service.create_collection,
            {"name": "Threat intel", "if_exists": "auto_suffix"},
        )
        assert entered.wait(timeout=2)
        assert not future.done(), "a failure response preceded the pending commit"
        release.set()
        first = json.loads(future.result(timeout=2))

    second = json.loads(
        service.create_collection(
            {"name": "Threat intel", "if_exists": "auto_suffix"}
        )
    )

    assert first["status"] == second["status"] == "ok"
    assert [row["name"] for row in owner.list_watchlists()] == [
        "Threat intel",
        "Threat intel (2)",
    ]


def test_create_sources_validates_rows_before_one_bounded_write_and_redacts_urls():
    calls = []

    def create(rows):
        calls.append(rows)
        return [
            {"input_index": 0, "outcome": "created", "source": {"source_id": 11}},
            {"input_index": 1, "outcome": "existing", "source": {"source_id": 11}},
        ]

    result = json.loads(
        _service(create_sources=create).create_sources(
            {
                "sources": [
                    {
                        "name": "Primary",
                        "url": "https://feeds.example/a?token=secret#part",
                        "type": "rss",
                    },
                    {"url": "https://feeds.example/a?token=secret#part", "type": "rss"},
                    {"url": "https://user:secret@feeds.example/private", "type": "rss"},
                ]
            }
        )
    )

    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert result["status"] == "partial_success"
    assert result["follow_on_confirmation_required"] is True
    assert result["results"] == [
        {"input_index": 0, "outcome": "created", "source_id": "local:subscription:11"},
        {"input_index": 1, "outcome": "existing", "source_id": "local:subscription:11"},
        {"input_index": 2, "outcome": "invalid", "message": "Source URL must not include credentials."},
    ]
    serialized = json.dumps(result)
    assert "token" not in serialized
    assert "secret" not in serialized
    assert "#part" not in serialized


def test_create_sources_rejects_boolean_integer_and_over_50_without_storage():
    calls = []

    def create(rows):
        calls.append(rows)
        return []

    service = _service(create_sources=create)
    bad_integer = json.loads(
        service.create_sources(
            {"sources": [{"url": "https://example.com/feed", "check_frequency": True}]}
        )
    )
    too_many = json.loads(
        service.create_sources(
            {"sources": [{"url": f"https://example.com/{index}"} for index in range(51)]}
        )
    )

    assert bad_integer["status"] == "invalid_argument"
    assert too_many["status"] == "invalid_argument"
    assert calls == []


def test_unhashable_enum_values_are_invalid_without_storage():
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    service = _service(create_sources=create, create_collection=create)

    source_result = json.loads(
        service.create_sources(
            {"sources": [{"url": "https://example.com/feed", "type": []}]}
        )
    )
    collection_result = json.loads(
        service.create_collection({"name": "Security", "if_exists": []})
    )

    assert source_result["status"] == "invalid_argument"
    assert collection_result["status"] == "invalid_argument"
    assert calls == []


def test_create_sources_rejects_parser_ambiguous_urls_without_storage():
    calls = []

    def create(rows):
        calls.append(rows)
        return []

    result = json.loads(
        _service(create_sources=create).create_sources(
            {
                "sources": [
                    {"url": "https://example.com:bad/feed"},
                    {"url": "https://example.com\\@internal.test/feed"},
                    {"url": "https://example.com/a feed"},
                ]
            }
        )
    )

    assert result["status"] == "invalid_argument"
    assert [row["outcome"] for row in result["results"]] == [
        "invalid",
        "invalid",
        "invalid",
    ]
    assert calls == []


@pytest.mark.parametrize(
    "url",
    [
        "https://.example.com/feed",
        "https://example..com/feed",
        "https://bad_label.example/feed",
        "https://-bad.example/feed",
        "https://bad-.example/feed",
        "https://\ud800.example/feed",
        "https://example.com/\x00feed",
        "https://example.com/\x1ffeed",
        "https://example.com/\x7ffeed",
        "https://example.com/\x85feed",
    ],
)
def test_create_sources_rejects_invalid_host_labels_and_controls_before_storage(url):
    calls = []

    def create(rows):
        calls.append(rows)
        return []

    result = json.loads(
        _service(create_sources=create).create_sources({"sources": [{"url": url}]})
    )

    assert result["status"] == "invalid_argument"
    assert result["results"][0]["outcome"] == "invalid"
    assert calls == []


def test_create_sources_trims_only_outer_url_whitespace():
    calls = []

    def create(rows):
        calls.append(rows)
        return [
            {"input_index": 0, "outcome": "created", "source": {"source_id": 3}}
        ]

    result = json.loads(
        _service(create_sources=create).create_sources(
            {"sources": [{"url": "  https://example.com/Feed?b=2&a=1  "}]}
        )
    )

    assert result["status"] == "ok"
    assert calls[0][0]["url"] == "https://example.com/Feed?b=2&a=1"


@pytest.mark.parametrize(
    ("method", "arguments"),
    [
        ("create_sources", {"sources": [{"url": "https://example.com/feed"}]}),
        ("create_collection", {"name": "Security"}),
        (
            "update_collection_sources",
            {
                "collection_id": "local:watchlist:1",
                "add_source_ids": ["local:subscription:1"],
            },
        ),
    ],
)
def test_server_mode_refuses_before_storage_resolution(method, arguments):
    calls = []

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    service = WatchlistsCommandService(
        runtime_source_loader=lambda: "server",
        create_sources_batch=create,
        create_collection=create,
        update_collection_sources=create,
    )

    result = json.loads(getattr(service, method)(arguments))

    assert result["status"] == "unsupported"
    assert calls == []


def test_create_collection_shapes_explicit_policy_and_canonical_ids():
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return {
            "outcome": "existing",
            "watchlist": {"id": 9, "name": "Existing"},
            "membership_count": 3,
        }

    result = json.loads(
        _service(create_collection=create).create_collection(
            {
                "name": "Existing",
                "source_ids": ["local:subscription:2", "local:subscription:3"],
                "if_exists": "return_existing",
            }
        )
    )

    assert calls == [
        {
            "name": "Existing",
            "description": None,
            "tags": None,
            "source_ids": [2, 3],
            "if_exists": "return_existing",
        }
    ]
    assert result == {
        "status": "ok",
        "retryable": False,
        "outcome": "existing",
        "collection_id": "local:watchlist:9",
        "collision_policy": "return_existing",
        "membership_count": 3,
    }


def test_update_collection_sources_rejects_overlap_before_mutation():
    calls = []

    def update(**kwargs):
        calls.append(kwargs)
        return {}

    result = json.loads(
        _service(update=update).update_collection_sources(
            {
                "collection_id": "local:watchlist:4",
                "add_source_ids": ["local:subscription:2"],
                "remove_source_ids": ["local:subscription:2"],
            }
        )
    )

    assert result["status"] == "invalid_argument"
    assert calls == []


def test_unexpected_errors_are_scrubbed_and_retryable():
    def create(_rows):
        raise RuntimeError(
            "db /Users/alice/private/subs.db https://example.com/?signed=secret"
        )

    result = json.loads(
        _service(create_sources=create).create_sources(
            {"sources": [{"url": "https://example.com/feed?token=secret"}]}
        )
    )

    assert result == {
        "status": "feature_unavailable",
        "retryable": True,
        "message": "Watchlists storage is temporarily unavailable. Try again.",
    }


@pytest.mark.parametrize("command", ["sources", "collection", "update"])
def test_malformed_domain_success_is_scrubbed(command):
    def incomplete_sources(_rows):
        return []

    def incomplete_collection(**_kwargs):
        return {"outcome": "created"}

    def incomplete_update(**_kwargs):
        return {"added": 1}

    service = _service(
        create_sources=incomplete_sources,
        create_collection=incomplete_collection,
        update=incomplete_update,
    )
    if command == "sources":
        raw = service.create_sources(
            {"sources": [{"url": "https://example.com/feed"}]}
        )
    elif command == "collection":
        raw = service.create_collection({"name": "Security"})
    else:
        raw = service.update_collection_sources(
            {
                "collection_id": "local:watchlist:1",
                "add_source_ids": ["local:subscription:1"],
            }
        )

    assert json.loads(raw) == {
        "status": "feature_unavailable",
        "retryable": True,
        "message": "Watchlists storage is temporarily unavailable. Try again.",
    }
