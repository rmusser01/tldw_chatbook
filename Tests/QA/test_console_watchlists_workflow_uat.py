"""Deterministic public-seam UAT for the Console Watchlists workflow."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.Chat.provider_setup_persistence import persist_provider_setup
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.MCP import local_server_tools
from tldw_chatbook.MCP.local_server_tools import (
    _local_agent_tool_registrations,
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import (
    MCPPermissionStore,
    definition_hash,
    resolve_effective_state,
)
from tldw_chatbook.Scheduling.services.briefing_projection import BriefingProjection
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)
from tldw_chatbook.Skills_Interop.skill_package_inspection import inspect_skill_directory
from tldw_chatbook.Subscriptions import (
    watchlists_operation_coordinator as coordinator_module,
)
from tldw_chatbook.Subscriptions.briefing_service import (
    execute_accepted_briefing,
    generate_briefing,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.Subscriptions.watchlists_operation_coordinator import (
    WatchlistsOperationCoordinator,
)
from tldw_chatbook.Tools.watchlists_command_service import WatchlistsCommandService
from tldw_chatbook.Tools.watchlists_tool_service import WatchlistsToolService
from tldw_chatbook.UI.Library_Modules.library_skill_import_controller import (
    LibrarySkillImportCoordinator,
)
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state


BRIEFING_ONLY_MARKER = "BRIEFING-ONLY-UAT-MARKER-7F31"
_CONSOLE_TOOL_NAMES = (
    "watchlists_create_sources",
    "watchlists_create_collection",
    "watchlists_check_sources",
    "watchlists_get_operation_status",
    "watchlists_generate_briefing",
    "watchlists_set_briefing_schedule",
    "watchlists_list_briefings",
    "watchlists_get_briefing",
)
_CONSOLE_ONLY_TOOLS = {
    "watchlists_create_sources",
    "watchlists_create_collection",
    "watchlists_update_collection_sources",
    "watchlists_check_sources",
    "watchlists_generate_briefing",
    "watchlists_set_briefing_schedule",
    "watchlists_search_items",
    "watchlists_get_item",
    "watchlists_get_briefing",
}
_SHARED_TOOLS = {
    "watchlists_list_sources",
    "watchlists_list_collections",
    "watchlists_list_briefings",
    "watchlists_get_operations_status",
    "watchlists_get_operation_status",
}


def _tool_fence(name: str, arguments: dict) -> str:
    return (
        "```tool_call\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n```"
    )


def _operation_status(messages: list[dict], operation_id: str) -> str:
    for message in reversed(messages):
        content = str(message.get("content", ""))
        if operation_id not in content:
            continue
        match = re.search(r'"status_detail"\s*:\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return ""


class _FixtureFeedHandler(BaseHTTPRequestHandler):
    feeds: dict[str, bytes] = {}
    active = 0
    peak = 0
    lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        body = self.feeds.get(self.path)
        if body is None:
            self.send_error(404)
            return
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
        try:
            time.sleep(0.03)
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self.lock:
                type(self).active -= 1

    def log_message(self, _format: str, *_args) -> None:
        return


def _feed_xml(index: int) -> bytes:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<rss version='2.0'><channel>"
        f"<title>Threat feed {index}</title>"
        f"<link>https://public.example/feed-{index}</link>"
        f"<description>Fixture {index}</description>"
        "<item>"
        f"<guid>fixture-{index}</guid>"
        f"<title>Campaign signal {index}</title>"
        f"<link>https://public.example/item-{index}</link>"
        f"<description>Observed indicator family {index}.</description>"
        "<pubDate>Thu, 27 Aug 2026 18:00:00 GMT</pubDate>"
        "</item></channel></rss>"
    ).encode()


class _ScriptedWatchlistsGateway:
    """Script only model planning; every tool and durable effect stays real."""

    def __init__(
        self,
        feed_urls: list[str],
        *,
        receipt_ready: Callable[[str], bool] | None = None,
    ) -> None:
        self.feed_urls = feed_urls
        self.receipt_ready = receipt_ready
        self.calls: list[list[dict]] = []
        self.stage = 0
        self.check_index = 0
        self.check_receipt_polled = False
        self.briefing_receipt_polled = False

    async def resolve_for_send(self, selection):
        """Resolve through the same typed destination contract as production."""
        resolution = ConsoleProviderResolution(
            provider=selection.provider,
            base_url=selection.base_url or "http://127.0.0.1:8791",
            model=selection.explicit_model
            or selection.configured_model
            or "scripted-mounted-model",
            ready=True,
            execution_key=selection.provider,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    async def stream_chat(self, _resolution, messages, tools=None, **_kwargs):
        del tools
        self.calls.append(copy.deepcopy(messages))
        text = "\n".join(str(message.get("content", "")) for message in messages)
        if self.stage == 0:
            self.stage += 1
            yield _tool_fence("find_tools", {"query": "Watchlists briefing workflow"})
            return
        if self.stage == 1:
            self.stage += 1
            yield _tool_fence(
                "load_tools", {"ids": [f"local:{name}" for name in _CONSOLE_TOOL_NAMES]}
            )
            return
        if self.stage == 2:
            self.stage += 1
            yield _tool_fence(
                "watchlists_create_sources",
                {
                    "sources": [
                        {"name": f"Threat feed {index}", "url": url, "type": "rss"}
                        for index, url in enumerate(self.feed_urls, 1)
                    ]
                },
            )
            return
        if self.stage == 3:
            self.stage += 1
            yield _tool_fence(
                "watchlists_create_collection",
                {
                    "name": "Daily threat intelligence",
                    "source_ids": [
                        "local:subscription:1",
                        "local:subscription:2",
                        "local:subscription:3",
                    ],
                },
            )
            return
        if self.stage == 4:
            self.stage += 1
            yield _tool_fence(
                "watchlists_check_sources",
                {"collection_id": "local:watchlist:1"},
            )
            return
        if self.stage == 5:
            if not self.check_receipt_polled and self.receipt_ready is not None:
                for _ in range(200):
                    if self.receipt_ready("checks"):
                        self.check_receipt_polled = True
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("source-check receipts did not complete")
            operation_id = f"local:watchlist_run:{self.check_index + 1}"
            if _operation_status(messages, operation_id) == "completed":
                self.check_index += 1
                if self.check_index == 3:
                    self.stage += 1
                else:
                    operation_id = f"local:watchlist_run:{self.check_index + 1}"
            if self.stage == 5:
                yield _tool_fence(
                    "watchlists_get_operation_status",
                    {"operation_id": operation_id},
                )
                return
        if self.stage == 6:
            self.stage += 1
            yield _tool_fence(
                "watchlists_generate_briefing",
                {"collection_id": "local:watchlist:1"},
            )
            return
        if self.stage == 7:
            if not self.briefing_receipt_polled and self.receipt_ready is not None:
                for _ in range(200):
                    if self.receipt_ready("briefing"):
                        self.briefing_receipt_polled = True
                        break
                    await asyncio.sleep(0.05)
                else:
                    raise AssertionError("briefing receipt did not complete")
            if _operation_status(messages, "local:briefing:1") == "complete":
                self.stage += 1
            else:
                yield _tool_fence(
                    "watchlists_get_operation_status",
                    {"operation_id": "local:briefing:1"},
                )
                return
        if self.stage == 8:
            self.stage += 1
            yield _tool_fence(
                "watchlists_set_briefing_schedule",
                {
                    "collection_id": "local:watchlist:1",
                    "cadence": "every_24_hours",
                },
            )
            return
        if self.stage == 9:
            self.stage += 1
            yield _tool_fence(
                "watchlists_list_briefings",
                {"collection": "local:watchlist:1", "statuses": ["complete"]},
            )
            return
        if self.stage == 10:
            self.stage += 1
            yield _tool_fence(
                "watchlists_get_briefing", {"briefing_id": "local:briefing:1"}
            )
            return
        assert BRIEFING_ONLY_MARKER in text
        self.stage += 1
        yield (
            f"The completed briefing reports three campaign signals "
            f"({BRIEFING_ONLY_MARKER}) with exact cited provenance."
        )


def _grant_tools(
    store: MCPPermissionStore, provider: LocalToolProvider, names: set[str]
) -> None:
    for name in sorted(names):
        hub = provider.hub_tool_for(name)
        store.set_tool_state(
            hub.server_key,
            hub.name,
            "allow",
            definition_hash=definition_hash(hub.description, hub.input_schema),
        )


async def _run_console_round_trip(tmp_path: Path, monkeypatch):
    (tmp_path / "profile").mkdir()
    database = SubscriptionsDB(tmp_path / "profile" / "subscriptions.sqlite", "uat")
    local_service = LocalWatchlistsService(db_factory=lambda: database)
    bundle_service = WatchlistBundleService(database)
    coordinator = WatchlistsOperationCoordinator(
        local_service=local_service,
        briefing_db=database,
    )
    coordinator.bind_running_loop()

    async def scripted_briefing(db, briefing_id, **_kwargs):
        return await execute_accepted_briefing(
            db,
            briefing_id,
            chat=lambda **_chat_kwargs: (
                f"## Daily signals\n\n{BRIEFING_ONLY_MARKER} "
                "links the three observed campaigns [item 1] [item 2] [item 3]."
            ),
        )

    monkeypatch.setattr(
        coordinator_module, "execute_accepted_briefing", scripted_briefing
    )
    reload_token = SimpleNamespace(value=73)
    command_service = WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=local_service.create_sources_exact_batch_sync,
        create_collection=bundle_service.create_with_sources,
        update_collection_sources=bundle_service.update_sources,
        accept_source_checks=coordinator.submit_checks,
        accept_briefing=coordinator.submit_briefing,
        resolve_collection_sources=bundle_service.list_sources,
        set_briefing_schedule=database.set_watchlist_briefing_settings,
        briefing_schedules_enabled=lambda: True,
        scheduler_running=lambda: True,
        request_scheduler_reload=lambda: reload_token,
        wait_scheduler_reload=lambda token, timeout: token is reload_token and timeout == 1.0,
        default_briefing_defaults=lambda: (
            "scripted-existing-provider",
            "scripted-existing-model",
        ),
    )
    query_service = WatchlistsToolService(
        db_resolver=lambda: database,
        runtime_source_loader=lambda: "local",
    )
    permission_store = MCPPermissionStore(tmp_path / "profile" / "permissions.json")
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        watchlists_service=query_service,
        watchlists_command_service=command_service,
        resolve_state=lambda hub: resolve_effective_state(permission_store.load(), hub),
    )
    _grant_tools(permission_store, provider, set(_CONSOLE_TOOL_NAMES))

    _FixtureFeedHandler.feeds = {
        f"/feed-{index}.xml": _feed_xml(index) for index in range(1, 4)
    }
    _FixtureFeedHandler.active = 0
    _FixtureFeedHandler.peak = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FixtureFeedHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    feed_urls = [
        f"http://127.0.0.1:{server.server_port}/feed-{index}.xml"
        for index in range(1, 4)
    ]

    def receipt_ready(kind: str) -> bool:
        if kind == "checks":
            runs = database.list_operations_for_agent(limit=10)["source_runs"]
            return len(runs) == 3 and all(
                row["status"] == "completed" for row in runs
            )
        rows = database.list_briefings(1)
        return bool(rows and rows[0]["status"] == "complete")

    gateway = _ScriptedWatchlistsGateway(
        feed_urls,
        receipt_ready=receipt_ready,
    )
    store = ConsoleChatStore()
    session = store.ensure_session()
    prompt = "Create a daily threat-intel Watchlist, run it, brief it, and read it."
    store.append_message(session.id, role=ConsoleMessageRole.USER, content=prompt)
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=AgentRunsDB(
            tmp_path / "profile" / "agent-runs.sqlite", client_id="uat"
        ),
        store=store,
        provider_gateway=gateway,
        native_tools_enabled=lambda: False,
    )
    bridge_result: list[tuple] = []
    bridge_error: list[BaseException] = []

    def run_bridge() -> None:
        try:
            bridge_result.append(
                bridge.run_reply(
                    conversation_id="uat-console-watchlists",
                    session_id=session.id,
                    resolution=ConsoleProviderResolution(
                        provider="scripted",
                        base_url="",
                        model="existing-selected-model",
                        ready=True,
                        execution_key="scripted",
                    ),
                    assistant_message_id=assistant.id,
                    model="existing-selected-model",
                    session_system_prompt=(
                        "Use the Watchlists tools and follow receipts."
                    ),
                    agent_messages=[{"role": "user", "content": prompt}],
                    should_cancel=lambda: False,
                    local_provider=provider,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted in caller
            bridge_error.append(exc)

    bridge_thread = threading.Thread(target=run_bridge, daemon=True)
    try:
        bridge_thread.start()
        for _ in range(100):
            if bridge_result or bridge_error:
                break
            await asyncio.sleep(0.1)
        if bridge_result or bridge_error:
            await asyncio.to_thread(bridge_thread.join, 1.0)
        assert not bridge_thread.is_alive(), "Console bridge exceeded 10 seconds"
        if bridge_error:
            raise bridge_error[0]
        _run_id, outcome = bridge_result[0]
        await coordinator.wait_idle(timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    assert outcome.status == "done", outcome.steps
    briefing_payload = json.loads(
        query_service.get_briefing({"briefing_id": "local:briefing:1"})
    )["briefing"]
    schedule_payload = next(
        json.loads(step.result)
        for step in outcome.steps
        if step.kind == "tool_result"
        and step.tool_name == "watchlists_set_briefing_schedule"
    )
    sources = await local_service.list_sources()
    collections = bundle_service.list_watchlists()
    scheduled_jobs = BriefingProjection(database).list_jobs(owner_id="local")
    invoked = {
        step.tool_name
        for step in outcome.steps
        if step.kind == "tool_call" and step.tool_name.startswith("watchlists_")
    }
    permission_payload = permission_store.load()
    explicit = set(
        permission_payload["profiles"]["default"]["servers"]["local:__local__"][
            "tools"
        ]
    )
    return {
        "source_ids": [source["id"] for source in sources],
        "collection_id": f"local:watchlist:{collections[0]['id']}",
        "check_receipt_ids": [
            "local:watchlist_run:1",
            "local:watchlist_run:2",
            "local:watchlist_run:3",
        ],
        "briefing_receipt_id": "local:briefing:1",
        "cadence_seconds": schedule_payload["cadence_seconds"],
        "reload_acknowledged": schedule_payload["reload_acknowledged"],
        "briefing_status": briefing_payload["status"],
        "ordered_selected_item_ids": [
            row["id"] for row in briefing_payload["selected_items"]
        ],
        "ordered_cited_item_ids": [
            row["id"] for row in briefing_payload["cited_items"]
        ],
        "agent_consumed_marker": BRIEFING_ONLY_MARKER in outcome.final_text,
        "watchlists_surface_matches": (
            [source["source_id"] for source in sources] == [1, 2, 3]
            and bundle_service.list_sources(1) == [1, 2, 3]
        ),
        "settings_surface_matches": (
            len(scheduled_jobs) == 1 and scheduled_jobs[0].id == "briefing:1"
        ),
        "explicit_permission_tools": explicit,
        "invoked_tools": invoked,
        "max_check_concurrency": _FixtureFeedHandler.peak,
    }


async def _seed_boundary_database(path: Path) -> None:
    path.parent.mkdir()
    database = SubscriptionsDB(path, "uat")
    source_id = database.add_subscription(
        name="Boundary feed", type="rss", source="https://public.example/feed"
    )
    bundles = WatchlistBundleService(database)
    watchlist_id = bundles.create("Boundary collection")["id"]
    bundles.add_source(watchlist_id, source_id)
    with database.transaction() as connection:
        persist_subscription_item(
            connection,
            source_id,
            {
                "url": "https://public.example/item",
                "title": "Boundary item",
                "content": "private article body",
                "content_hash": "boundary-item-hash",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now="2026-08-27T18:00:00+00:00",
        )
    await generate_briefing(
        database,
        watchlist_id,
        chat=lambda **_kwargs: f"{BRIEFING_ONLY_MARKER} [item 1]",
    )
    database.close()


def _sqlite_logical_snapshot(path: Path) -> tuple[str, ...]:
    """Return an exact read-only schema-and-row snapshot of a settled fixture."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return tuple(connection.iterdump())
    finally:
        connection.close()


def _database_file_snapshot(path: Path) -> dict[str, str]:
    return {
        candidate.name: hashlib.sha256(candidate.read_bytes()).hexdigest()
        for candidate in path.parent.iterdir()
        if candidate.is_file() and candidate.name != "mcp-permissions.json"
    }


async def _run_external_mcp_boundary(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "profile" / "subscriptions.sqlite"
    await _seed_boundary_database(database_path)
    permission_store = MCPPermissionStore(
        tmp_path / "profile" / "mcp-permissions.json"
    )
    monkeypatch.setattr(
        local_server_tools, "get_subscriptions_db_path", lambda: database_path
    )
    monkeypatch.setattr(
        local_server_tools, "load_default_runtime_source_state", lambda: "local"
    )
    provider = build_server_local_provider(tmp_path, permission_store)
    _grant_tools(permission_store, provider, _SHARED_TOOLS)
    registrations = {
        registration.name: registration
        for registration in _local_agent_tool_registrations(provider)
    }
    warmup = registrations["watchlists_list_sources"].handler({})
    assert warmup.ok, warmup
    before_files = _database_file_snapshot(database_path)
    before_logical = _sqlite_logical_snapshot(database_path)
    serialized_surfaces: list[str] = [
        json.dumps(
            {
                name: {
                    "description": registration.description,
                    "parameters": registration.parameters,
                }
                for name, registration in registrations.items()
            },
            sort_keys=True,
        ),
        warmup.content,
    ]
    for name, arguments in (
        ("watchlists_list_sources", {}),
        ("watchlists_list_collections", {}),
        ("watchlists_list_briefings", {}),
        ("watchlists_get_operations_status", {}),
        (
            "watchlists_get_operation_status",
            {"operation_id": "local:briefing:1"},
        ),
    ):
        result = registrations[name].handler(arguments)
        assert result.ok, (name, result)
        serialized_surfaces.append(result.content)
    refused = provider.invoke(
        "local:watchlists_get_briefing", {"briefing_id": "local:briefing:1"}
    )
    serialized_surfaces.append(repr(refused))
    serialized_surfaces.append(json.dumps(permission_store.load(), sort_keys=True))
    after_files = _database_file_snapshot(database_path)
    after_logical = _sqlite_logical_snapshot(database_path)
    return {
        "shared_tools": _SHARED_TOOLS & registrations.keys(),
        "console_only_tools": _CONSOLE_ONLY_TOOLS,
        "published_tools": set(registrations),
        "direct_dispatch_refused": refused.ok is False,
        "private_marker_absent": all(
            BRIEFING_ONLY_MARKER not in surface
            and "private article body" not in surface
            for surface in serialized_surfaces
        ),
        "database_unchanged": (
            before_files == after_files and before_logical == after_logical
        ),
    }


async def _run_skill_classification_regression(tmp_path: Path, _monkeypatch):
    root = tmp_path / "root-skill"
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\nname: root-skill\ndescription: Root.\n---\n", encoding="utf-8"
    )
    multi = tmp_path / "multi"
    for name in ("a", "b"):
        skill = multi / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Skill {name}.\n---\n",
            encoding="utf-8",
        )
    framework = tmp_path / "framework"
    framework.mkdir()
    (framework / "README.md").write_text(
        "# A generic agent framework\n", encoding="utf-8"
    )
    (framework / "pyproject.toml").write_text(
        "[project]\nname='framework'\n", encoding="utf-8"
    )

    started = threading.Event()
    release = threading.Event()
    trust_values: list[bool] = []

    class _SkillsService:
        def import_skill_directory(self, _path, **kwargs):
            trust_values.append(kwargs["trust_approved"])
            started.set()
            assert release.wait(timeout=5)
            return {"name": "root-skill"}

        def import_skill_file(self, *_args, **_kwargs):
            raise AssertionError("directory fixture must not use file import")

    app = SimpleNamespace(skills_scope_service=_SkillsService())
    coordinator = LibrarySkillImportCoordinator(app)
    runtime_app = SimpleNamespace(
        screen=SimpleNamespace(
            _present_library_skills_import_snapshot=lambda **_kwargs: None
        )
    )
    assert coordinator.open_draft()
    assert coordinator.claim(str(root))
    owner = asyncio.create_task(
        coordinator.run(str(root), runtime_app=runtime_app)
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
    except TimeoutError:
        release.set()
        await owner
        pytest.fail(f"skill import never reached trust review: {coordinator.snapshot!r}")
    second_submit_refused = coordinator.claim(str(framework)) is False
    release.set()
    await owner
    return {
        "root": inspect_skill_directory(root).kind.value,
        "multi": inspect_skill_directory(multi).candidates,
        "framework": inspect_skill_directory(framework).kind.value,
        "trust_approved": trust_values[0],
        "second_submit_refused": second_submit_refused,
        "actual_result_reported": (
            coordinator.snapshot.review_name == "root-skill"
            and coordinator.snapshot.status.startswith('Imported "root-skill"')
        ),
    }


@pytest.mark.asyncio
@pytest.mark.allow_network
async def test_console_round_trip_uses_receipts_schedule_and_briefing_provenance(
    tmp_path, monkeypatch
):
    evidence = await _run_console_round_trip(tmp_path, monkeypatch)

    assert evidence["source_ids"] == [
        "local:subscription:1",
        "local:subscription:2",
        "local:subscription:3",
    ]
    assert evidence["collection_id"] == "local:watchlist:1"
    assert len(evidence["check_receipt_ids"]) == 3
    assert evidence["briefing_receipt_id"] == "local:briefing:1"
    assert evidence["cadence_seconds"] == 86_400
    assert evidence["reload_acknowledged"] is True
    assert evidence["briefing_status"] == "complete"
    assert evidence["ordered_selected_item_ids"]
    assert evidence["ordered_cited_item_ids"]
    assert evidence["agent_consumed_marker"] is True
    assert evidence["watchlists_surface_matches"] is True
    assert evidence["settings_surface_matches"] is True
    assert evidence["explicit_permission_tools"] == evidence["invoked_tools"]
    assert evidence["max_check_concurrency"] <= 4


@pytest.mark.asyncio
async def test_external_mcp_boundary_is_metadata_and_receipts_only(
    tmp_path, monkeypatch
):
    evidence = await _run_external_mcp_boundary(tmp_path, monkeypatch)

    assert evidence["shared_tools"] == {
        "watchlists_list_sources",
        "watchlists_list_collections",
        "watchlists_list_briefings",
        "watchlists_get_operations_status",
        "watchlists_get_operation_status",
    }
    assert evidence["console_only_tools"].isdisjoint(evidence["published_tools"])
    assert evidence["direct_dispatch_refused"] is True
    assert evidence["private_marker_absent"] is True
    assert evidence["database_unchanged"] is True


@pytest.mark.asyncio
async def test_skill_framework_and_single_flight_regressions(tmp_path, monkeypatch):
    evidence = await _run_skill_classification_regression(tmp_path, monkeypatch)

    assert evidence == {
        "root": "root_skill",
        "multi": ("skills/a", "skills/b"),
        "framework": "framework_repository",
        "trust_approved": False,
        "second_submit_refused": True,
        "actual_result_reported": True,
    }


@pytest.mark.asyncio
async def test_no_preset_briefings_use_first_run_persisted_defaults_everywhere(
    tmp_path, monkeypatch
):
    """Manual, scheduled, and schedule receipts share persisted defaults."""
    from tldw_chatbook import config as app_config
    from tldw_chatbook.Subscriptions import briefing_service

    config_path = tmp_path / "profile" / "config.toml"
    config_path.parent.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    loaded = app_config.load_cli_config_and_ensure_existence(force_reload=True)
    mutation = wizard_state.build_first_run_provider_commit(
        wizard_state.FirstRunProviderDraft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8791/v1/chat/completions",
            credential=wizard_state.ProviderCredentialDraft("none", "", 0),
        ),
        "persisted-first-run-model",
        loaded,
    )
    assert persist_provider_setup(mutation).fully_applied is True

    # Neither legacy import-time state nor a conversation-scoped selection may
    # affect briefing egress or its durable/displayed provenance.
    monkeypatch.setattr(app_config, "default_api_endpoint", "legacy-conflict")
    active_conversation = ConsoleProviderResolution(
        provider="openai",
        base_url="",
        model="active-conversation-model",
        ready=True,
        execution_key="active-conversation",
    )
    assert active_conversation.model == "active-conversation-model"

    database = SubscriptionsDB(tmp_path / "profile" / "subscriptions.sqlite", "uat")
    bundles = WatchlistBundleService(database)
    watchlist_id = bundles.create("Persisted defaults")['id']
    source_id = database.add_subscription(
        name="Fixture feed", type="rss", source="https://public.example/feed"
    )
    bundles.add_source(watchlist_id, source_id)

    def add_item(number: int) -> None:
        with database.transaction() as connection:
            persist_subscription_item(
                connection,
                source_id,
                {
                    "url": f"https://public.example/item-{number}",
                    "title": f"Persisted default signal {number}",
                    "content": f"Fixture body {number}",
                    "content_hash": f"persisted-default-{number}",
                    "content_kind": "article",
                    "content_format": "text",
                },
                run_id=None,
                now=f"2026-08-29T0{number}:00:00+00:00",
            )

    calls: list[dict] = []

    def scripted_chat(**kwargs):
        calls.append(kwargs)
        return "Persisted default briefing [item 1] [item 2]"

    add_item(1)
    manual = await generate_briefing(database, watchlist_id, chat=scripted_chat)

    command_service = WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=lambda _rows: None,
        create_collection=lambda **_kwargs: None,
        update_collection_sources=lambda **_kwargs: None,
        set_briefing_schedule=database.set_watchlist_briefing_settings,
        briefing_schedules_enabled=lambda: True,
        scheduler_running=lambda: True,
        default_briefing_defaults=briefing_service.resolve_persisted_briefing_defaults,
    )
    schedule = json.loads(
        command_service.set_briefing_schedule(
            {
                "collection_id": f"local:watchlist:{watchlist_id}",
                "cadence": "every_24_hours",
            }
        )
    )

    add_item(2)
    handler = BriefingJobHandler(
        subscriptions_db=database,
        generate=lambda db, target, **kwargs: generate_briefing(
            db, target, chat=scripted_chat, **kwargs
        ),
    )
    await handler.handle({"id": f"briefing:{watchlist_id}"})
    async with asyncio.timeout(5):
        while True:
            rows = database.list_briefings(watchlist_id)
            if len(rows) == 2 and rows[0]["status"] == "complete":
                break
            await asyncio.sleep(0.01)

    assert [(call["api_endpoint"], call["model"]) for call in calls] == [
        ("llama_cpp", "persisted-first-run-model"),
        ("llama_cpp", "persisted-first-run-model"),
    ]
    assert manual["model_used"] == "llama_cpp/persisted-first-run-model"
    assert rows[0]["model_used"] == "llama_cpp/persisted-first-run-model"
    assert schedule["provider"] == "llama_cpp"
    assert schedule["model"] == "persisted-first-run-model"
    assert schedule["provider_resolution_source"] == "app_default"
    assert schedule["model_resolution_source"] == "app_default"
    for row in (manual, rows[0]):
        provenance = database.get_briefing_provenance_for_agent(row["id"])
        assert provenance["selected"]
        assert provenance["cited"]


def test_redaction_checker_rejects_cli_targets_outside_repository(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[2]
    checker = (
        repository_root
        / "Docs"
        / "superpowers"
        / "qa"
        / "console-watchlists-workflow-2026-08"
        / "redaction_check.py"
    )
    outside = tmp_path / "outside-scan-root"
    outside.mkdir()
    (outside / "clean.txt").write_text("public fixture", encoding="utf-8")
    environment = dict(os.environ)
    environment["TASK22868_PRIVATE_SENTINEL"] = "OUT-OF-BAND-PRIVATE-CANARY"

    result = subprocess.run(
        [sys.executable, str(checker), str(outside)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "ERROR: Path is outside the allowed directory" in result.stderr
    assert str(outside) not in result.stdout + result.stderr


def test_seeded_capture_cleans_sandbox_when_rendering_fails() -> None:
    repository_root = Path(__file__).parents[2]
    capture_script = (
        repository_root
        / "Docs"
        / "superpowers"
        / "qa"
        / "console-watchlists-workflow-2026-08"
        / "capture_uat.py"
    )
    probe = f"""
import asyncio
import importlib.util

spec = importlib.util.spec_from_file_location("task22868_capture_uat", {str(capture_script)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sandbox = module.SANDBOX

async def fail(_size):
    raise RuntimeError("injected capture failure")

module._capture_console = fail
module.drain_active_service_patches = lambda: (_ for _ in ()).throw(
    RuntimeError("injected cleanup failure")
)
try:
    asyncio.run(module.main())
except RuntimeError:
    pass
assert not sandbox.exists(), sandbox
"""

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
