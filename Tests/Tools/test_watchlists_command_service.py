import asyncio
import json

import pytest

from tldw_chatbook.Tools.watchlists_command_service import WatchlistsCommandService


def _bridge(factory):
    with asyncio.Runner() as runner:
        return runner.run(factory())


def _service(*, runtime="local", create_sources=None, create_collection=None, update=None):
    async def default_sources(rows):
        return [
            {
                "input_index": index,
                "outcome": "created",
                "source": {"source_id": index + 1},
            }
            for index, _row in enumerate(rows)
        ]

    async def default_collection(**kwargs):
        return {
            "outcome": "created",
            "watchlist": {"id": 7, "name": kwargs["name"]},
            "membership_count": len(kwargs["source_ids"]),
        }

    async def default_update(**kwargs):
        return {
            "watchlist_id": kwargs["watchlist_id"],
            "added": len(kwargs["add_ids"]),
            "removed": len(kwargs["remove_ids"]),
            "membership_count": len(kwargs["add_ids"]),
        }

    return WatchlistsCommandService(
        runtime_source_loader=lambda: runtime,
        app_loop_bridge=_bridge,
        create_sources_batch=create_sources or default_sources,
        create_collection=create_collection or default_collection,
        update_collection_sources=update or default_update,
    )


def test_create_sources_validates_rows_before_one_bounded_write_and_redacts_urls():
    calls = []

    async def create(rows):
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

    async def create(rows):
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

    async def create(*args, **kwargs):
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

    async def create(rows):
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


def test_create_sources_trims_only_outer_url_whitespace():
    calls = []

    async def create(rows):
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
def test_server_mode_refuses_before_storage_or_bridge_resolution(method, arguments):
    calls = []

    async def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {}

    service = WatchlistsCommandService(
        runtime_source_loader=lambda: "server",
        app_loop_bridge=lambda factory: (_ for _ in ()).throw(AssertionError("bridge")),
        create_sources_batch=create,
        create_collection=create,
        update_collection_sources=create,
    )

    result = json.loads(getattr(service, method)(arguments))

    assert result["status"] == "unsupported"
    assert calls == []


def test_create_collection_shapes_explicit_policy_and_canonical_ids():
    calls = []

    async def create(**kwargs):
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

    async def update(**kwargs):
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
    async def create(_rows):
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
    async def incomplete_sources(_rows):
        return []

    async def incomplete_collection(**_kwargs):
        return {"outcome": "created"}

    async def incomplete_update(**_kwargs):
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
