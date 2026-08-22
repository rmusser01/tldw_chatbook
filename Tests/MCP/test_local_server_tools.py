"""Tests for MCP/local_server_tools.py — external-client local tool composition.

The composition builds a LocalToolProvider for non-Console MCP serving:
permission state resolved FRESH per call from a real MCPPermissionStore,
no approval callback (ask fails closed with the external refusal copy),
kill switch honored, and no Console-only seams (todo store, session
approvals, persistence).
"""

from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
import threading
import time

import pytest
from loguru import logger

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
)
from tldw_chatbook.DB.Subscriptions_DB import (
    SubscriptionsDB,
    SubscriptionsDBReadError,
)
import tldw_chatbook.MCP.local_server_tools as local_server_tools
from tldw_chatbook.MCP.local_server_tools import (
    EXTERNAL_NO_CALLBACK_REFUSAL,
    LocalToolRegistration,
    _local_agent_tool_registrations,
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import MCPPermissionStore, definition_hash
from tldw_chatbook.MCP.server import TldwMCPServer, _describe_local_tools
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


BUILTIN_TOOL_NAMES = [descriptor["name"] for descriptor in _describe_local_tools()]
TASK_TOOL_NAMES = {"todo_create", "todo_update", "todo_get", "todo_list"}


def _pin_runtime_source(monkeypatch, source):
    """Pin the runtime source the composed watchlists service will read.

    The seam is ``local_server_tools.load_default_runtime_source_state`` --
    the owner-module loader that ``build_server_local_provider`` injects as
    ``runtime_source_loader=`` (TASK-18609). It used to be a
    ``RuntimeSourceStateStore`` constructed in-module, and these tests went
    on patching that vanished name for weeks (TASK-19569): four of them
    errored at the monkeypatch line, and the two that passed
    ``raising=False`` installed a never-read attribute and failed
    downstream on a scrubbed ``ToolResult`` instead.

    ``source`` may be a literal ``"local"``/``"server"`` or a zero-arg
    callable, so a test can flip the source between calls. The loader
    returns a real ``RuntimeSourceState`` -- the production shape, which
    exercises ``WatchlistsToolService._runtime_source``'s attribute branch
    rather than the bare-string convenience branch the old fakes used.

    Deliberately NOT ``raising=False``: if this name is renamed or removed
    again, every caller must fail loudly at the patch line.
    """
    resolve = source if callable(source) else (lambda: source)
    monkeypatch.setattr(
        local_server_tools,
        "load_default_runtime_source_state",
        lambda: RuntimeSourceState(active_source=resolve()),
    )


def _context():
    gateway = pytest.importorskip("mcp_unified.gateway")
    return gateway.GatewayRequestContext(request_id="test-request")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "hello.txt").write_text("hello world\n")
    return root


@pytest.fixture
def store(tmp_path):
    return MCPPermissionStore(tmp_path / "mcp_permissions.json")


def _grant(store, provider, name):
    """Persist an operator allow the way Console "Always allow" does."""
    hub = provider.hub_tool_for(name)
    store.set_tool_state(
        hub.server_key,
        hub.name,
        "allow",
        definition_hash=definition_hash(hub.description, hub.input_schema),
    )


def test_granted_tool_executes(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert r.ok and "hello world" in r.content


def test_default_ask_fails_closed_with_external_refusal(workspace, store):
    # Nothing granted (an untouched store file) -> global default "ask",
    # no approval callback -> external no-callback refusal.
    provider = build_server_local_provider(workspace, store)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_missing_store_file_fails_closed(workspace, tmp_path):
    missing = MCPPermissionStore(tmp_path / "no-such-dir" / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, missing)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_kill_switch_refuses_even_granted_tools(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    store.set_kill_switch(True)
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


def test_kill_switch_read_failure_fails_closed(workspace):
    sentinel = "SENTINEL /private/path API_KEY=secret"

    class RaisingStore:
        def load(self):
            return {}

        def get_kill_switch(self):
            raise RuntimeError(sentinel)

    provider = build_server_local_provider(workspace, RaisingStore())
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    finally:
        logger.remove(sink_id)
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL
    assert all(sentinel not in record for record in records)


@pytest.mark.parametrize("failure_stage", ["load", "resolution"])
def test_permission_store_read_failure_is_payload_free(workspace, failure_stage):
    sentinel = "SENTINEL /private/path API_KEY=secret"

    class RaisingPayload(dict):
        def get(self, _key, _default=None):
            raise RuntimeError(sentinel)

    class RaisingStore:
        def load(self):
            if failure_stage == "load":
                raise RuntimeError(sentinel)
            return RaisingPayload()

        def get_kill_switch(self):
            return False

    provider = build_server_local_provider(workspace, RaisingStore())
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        result = provider.invoke("local:fs_read", {"path": sentinel})
    finally:
        logger.remove(sink_id)

    assert not result.ok and result.error == LOCAL_GATE_ERROR_REFUSAL
    assert all(sentinel not in record for record in records)


def test_deny_state_refuses(workspace, store):
    provider = build_server_local_provider(workspace, store)
    hub = provider.hub_tool_for("fs_read")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_resolve_state_reads_store_fresh_per_call(workspace, store):
    # Operator changes take effect immediately: grant -> executes, revoke
    # -> fails closed, all against the same composed provider.
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    assert provider.invoke("local:fs_read", {"path": "hello.txt"}).ok
    hub = provider.hub_tool_for("fs_read")
    store.set_tool_state(hub.server_key, hub.name, None)  # operator revokes
    r = provider.invoke("local:fs_read", {"path": "hello.txt"})
    assert not r.ok and r.error == EXTERNAL_NO_CALLBACK_REFUSAL


def test_session_task_tools_absent_from_external_catalog(workspace, store):
    # No Console SessionTodoStore is handed in, so neither the replacement
    # task tools nor the retired todo_write tool can be registered.
    provider = build_server_local_provider(workspace, store)
    names = {e.name for e in provider.list_catalog()}
    assert "todo_write" not in names
    assert TASK_TOOL_NAMES.isdisjoint(names)
    assert "fs_read" in names


def test_watchlists_registration_is_storage_lazy_and_server_mode_never_resolves_path(
    monkeypatch, workspace, store
):
    path_calls = 0

    def fail_path_resolution():
        nonlocal path_calls
        path_calls += 1
        raise AssertionError("server mode must not resolve the subscriptions path")

    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        fail_path_resolution,
    )
    _pin_runtime_source(monkeypatch, "server")
    provider = build_server_local_provider(workspace, store)
    provider.list_catalog()
    provider.load_schema("local:watchlists_search_items")
    provider.load_schema("local:watchlists_get_item")
    assert path_calls == 0
    _grant(store, provider, "watchlists_search_items")

    result = provider.invoke("local:watchlists_search_items", {})

    assert result.ok
    assert json.loads(result.content) == {
        "status": "unsupported",
        "retryable": False,
        "message": (
            "server Watchlists search is not supported; switch Watchlists to Local "
            "before retrying"
        ),
    }
    assert path_calls == 0


def test_watchlists_first_local_call_opens_one_read_only_database(
    monkeypatch, tmp_path, workspace, store
):
    db_path = tmp_path / "subscriptions.db"
    mutable = SubscriptionsDB(db_path)
    mutable.close()
    path_calls = 0
    constructions = []

    real_database = SubscriptionsDB

    def resolve_path():
        nonlocal path_calls
        path_calls += 1
        return db_path

    def construct_database(path, client_id="default", *, read_only=False):
        constructions.append((path, client_id, read_only))
        return real_database(path, client_id, read_only=read_only)

    monkeypatch.setattr(local_server_tools, "get_subscriptions_db_path", resolve_path)
    _pin_runtime_source(monkeypatch, "local")
    monkeypatch.setattr(local_server_tools, "SubscriptionsDB", construct_database)

    provider = build_server_local_provider(workspace, store)
    assert path_calls == 0
    assert constructions == []
    _grant(store, provider, "watchlists_search_items")

    first = provider.invoke("local:watchlists_search_items", {})
    second = provider.invoke("local:watchlists_search_items", {})

    assert json.loads(first.content)["status"] == "ok"
    assert json.loads(second.content)["status"] == "ok"
    assert path_calls == 1
    assert constructions == [(db_path, "default", True)]


def test_watchlists_lazy_resolver_closes_failure_and_retries(monkeypatch, tmp_path):
    candidates = []

    class Candidate:
        def __init__(self, *, fail):
            self.fail = fail
            self.closed = False

        def assert_agent_read_ready(self):
            if self.fail:
                raise SubscriptionsDBReadError()

        def close(self):
            self.closed = True

    def construct_database(_path, _client_id="default", *, read_only=False):
        assert read_only is True
        candidate = Candidate(fail=not candidates)
        candidates.append(candidate)
        return candidate

    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: tmp_path / "subscriptions.db",
    )
    monkeypatch.setattr(local_server_tools, "SubscriptionsDB", construct_database)
    resolver = local_server_tools._LazyWatchlistsDBResolver()

    with pytest.raises(SubscriptionsDBReadError):
        resolver()
    assert candidates[0].closed is True
    assert resolver() is candidates[1]
    assert candidates[1].closed is False


def test_watchlists_lazy_resolver_blocks_replacement_until_failed_close_succeeds(
    monkeypatch, tmp_path, workspace, store, caplog
):
    sentinel = "SENTINEL /private/leaked.db API_KEY=secret"
    constructions = []

    class FailedCandidate:
        def __init__(self):
            self.close_calls = 0

        def assert_agent_read_ready(self):
            raise SubscriptionsDBReadError()

        def close(self):
            self.close_calls += 1
            if self.close_calls < 3:
                raise RuntimeError(sentinel)

        def search_items_for_agent(self, **_kwargs):
            raise AssertionError("failed candidate must never execute a search")

    class ReadyCandidate:
        def assert_agent_read_ready(self):
            return None

        def close(self):
            raise AssertionError("successful candidate remains process-owned")

        def search_items_for_agent(self, **_kwargs):
            return {"items": [], "has_more": False, "snapshot_max_item_id": 0}

        def get_source_collection_memberships(self, _source_ids):
            return {}

    failed = FailedCandidate()
    ready = ReadyCandidate()

    def construct_database(_path, _client_id="default", *, read_only=False):
        assert read_only is True
        candidate = failed if not constructions else ready
        constructions.append(candidate)
        return candidate

    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: tmp_path / "subscriptions.db",
    )
    monkeypatch.setattr(local_server_tools, "SubscriptionsDB", construct_database)
    _pin_runtime_source(monkeypatch, "local")
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "watchlists_search_items")
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))

    try:
        first = provider.invoke("local:watchlists_search_items", {})
        second = provider.invoke("local:watchlists_search_items", {})
        assert constructions == [failed]
        third = provider.invoke("local:watchlists_search_items", {})
    finally:
        logger.remove(sink_id)

    assert failed.close_calls == 3
    assert constructions == [failed, ready]
    assert [json.loads(result.content)["status"] for result in (first, second)] == [
        "feature_unavailable",
        "feature_unavailable",
    ]
    assert all(
        json.loads(result.content)["retryable"] is True for result in (first, second)
    )
    assert json.loads(third.content)["status"] == "ok"
    assert sentinel not in first.content
    assert sentinel not in second.content
    assert sentinel not in caplog.text
    assert all(sentinel not in record for record in records)


def test_watchlists_lazy_resolver_concurrent_first_calls_retain_one_instance(
    monkeypatch, tmp_path
):
    constructions = []
    entered = threading.Event()

    class Candidate:
        def assert_agent_read_ready(self):
            return None

        def close(self):
            raise AssertionError("successful candidate must remain open")

    def construct_database(_path, _client_id="default", *, read_only=False):
        assert read_only is True
        candidate = Candidate()
        constructions.append(candidate)
        entered.set()
        time.sleep(0.02)
        return candidate

    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: tmp_path / "subscriptions.db",
    )
    monkeypatch.setattr(local_server_tools, "SubscriptionsDB", construct_database)
    resolver = local_server_tools._LazyWatchlistsDBResolver()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(resolver) for _ in range(8)]
        assert entered.wait(timeout=1)
        resolved = [future.result() for future in futures]

    assert len(constructions) == 1
    assert all(candidate is constructions[0] for candidate in resolved)


@pytest.mark.parametrize("storage_state", ["missing", "pre_migration"])
def test_watchlists_unready_database_is_bounded_and_keeps_other_tools(
    monkeypatch, tmp_path, workspace, store, storage_state
):
    database_path = tmp_path / "subscriptions.db"
    if storage_state == "pre_migration":
        with sqlite3.connect(database_path) as connection:
            connection.execute("CREATE TABLE legacy_only (id INTEGER PRIMARY KEY)")
        before = database_path.read_bytes()
    else:
        before = None

    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: database_path,
    )
    _pin_runtime_source(monkeypatch, "local")
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "watchlists_search_items")

    result = provider.invoke("local:watchlists_search_items", {})

    payload = json.loads(result.content)
    assert payload["status"] == "feature_unavailable"
    assert payload["retryable"] is False
    assert str(database_path) not in result.content
    if before is None:
        assert not database_path.exists()
    else:
        assert database_path.read_bytes() == before
    names = {entry.name for entry in provider.list_catalog()}
    assert {"fs_read", "git_status", "web_fetch"} <= names


def test_watchlists_external_ask_refuses_before_storage_resolution(
    monkeypatch, workspace, store
):
    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve storage")),
    )
    provider = build_server_local_provider(workspace, store)

    result = provider.invoke("local:watchlists_search_items", {})

    assert result == ToolResult.blocked(EXTERNAL_NO_CALLBACK_REFUSAL)


# -- _local_agent_tool_registrations (pure builder, no mcp package needed) --


def _registrations(provider):
    return {r.name: r for r in _local_agent_tool_registrations(provider)}


def test_granted_tool_registration_handler_executes(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    regs = _registrations(provider)
    assert "fs_read" in regs
    reg = regs["fs_read"]
    # Introspection fields come from the provider's load_schema.
    assert reg.description
    assert reg.parameters.get("type") == "object"
    result = reg.handler({"path": "hello.txt"})
    assert isinstance(result, ToolResult)
    assert result.ok and "hello world" in result.content


def test_ask_state_handler_returns_tool_result(workspace, store):
    # fs_write with nothing granted -> ask default -> external refusal.
    provider = build_server_local_provider(workspace, store)
    result = _registrations(provider)["fs_write"].handler(
        {"path": "x.txt", "content": "y"}
    )
    assert result == ToolResult.blocked(EXTERNAL_NO_CALLBACK_REFUSAL)


def test_deny_state_handler_returns_tool_result(workspace, store):
    provider = build_server_local_provider(workspace, store)
    hub = provider.hub_tool_for("fs_glob")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    result = _registrations(provider)["fs_glob"].handler({"pattern": "*.py"})
    assert result == ToolResult.blocked(LOCAL_DENY_REFUSAL)


def test_kill_switch_handler_returns_tool_result(workspace, store):
    provider = build_server_local_provider(workspace, store)
    _grant(store, provider, "fs_read")
    store.set_kill_switch(True)
    result = _registrations(provider)["fs_read"].handler({"path": "hello.txt"})
    assert result == ToolResult.blocked(LOCAL_KILL_SWITCH_REFUSAL)


def test_session_task_tools_absent_from_external_registrations(workspace, store):
    provider = build_server_local_provider(workspace, store)
    regs = _registrations(provider)
    assert "todo_write" not in regs
    assert TASK_TOOL_NAMES.isdisjoint(regs)
    # The rest of the default catalog IS present.
    assert "fs_read" in regs
    assert "git_status" in regs
    assert "web_fetch" in regs


def _bare_server() -> TldwMCPServer:
    """Return a DB-free server with the real gateway registration seam."""
    pytest.importorskip("mcp_unified.gateway")
    from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime

    server = TldwMCPServer.__new__(TldwMCPServer)
    server.mcp = ChatbookGatewayRuntime(
        name="test",
        version="0.1.0",
        tool_descriptors=_describe_local_tools(),
    )
    server._register_tools()
    return server


@pytest.mark.asyncio
async def test_flag_on_registers_local_tool_names(monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("mcp", "expose_local_tools"):
            return True
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    server = _bare_server()
    server._register_local_agent_tools()
    server.mcp.finalize()

    names = {
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    }
    assert "fs_read" in names
    assert "fs_write" in names
    assert "git_status" in names
    assert "web_fetch" in names
    assert "watchlists_search_items" in names
    assert "watchlists_get_item" in names
    assert "todo_write" not in names
    assert TASK_TOOL_NAMES.isdisjoint(names)

    provider = build_server_local_provider(
        tmp_path, MCPPermissionStore(tmp_path / "mcp_permissions.json")
    )
    expected = {
        name: provider.load_schema(f"local:{name}").parameters
        for name in ("watchlists_search_items", "watchlists_get_item")
    }
    published = {
        descriptor["name"]: descriptor["inputSchema"]
        for descriptor in await server.mcp.list_tools(_context())
        if descriptor["name"] in expected
    }
    assert published == expected


@pytest.mark.asyncio
async def test_flag_off_registers_nothing(monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    monkeypatch.setattr(
        config, "get_cli_setting", lambda section, key, default=None: default
    )
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    server = _bare_server()
    server._register_local_agent_tools()
    server.mcp.finalize()

    assert [
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    ] == BUILTIN_TOOL_NAMES


@pytest.mark.parametrize("failure_stage", ["construction", "staging"])
@pytest.mark.asyncio
async def test_registration_failure_keeps_builtins_and_emits_fixed_stderr_once(
    failure_stage, monkeypatch, tmp_path, capsys
):
    import tldw_chatbook.config as config
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    sentinel = "SENTINEL /private/path API_KEY=secret"

    def fake_get_cli_setting(section, key, default=None):
        if (section, key) == ("mcp", "expose_local_tools"):
            return True
        return default

    monkeypatch.setattr(config, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(config, "get_user_data_dir", lambda: tmp_path)

    def boom(*_a, **_k):
        raise RuntimeError(sentinel)

    if failure_stage == "construction":
        monkeypatch.setattr(local_server_tools, "build_server_local_provider", boom)
    else:
        monkeypatch.setattr(
            local_server_tools,
            "_local_agent_tool_registrations",
            lambda _provider: [
                LocalToolRegistration(
                    "valid_first",
                    "valid",
                    {"type": "object", "properties": {}},
                    lambda _arguments: ToolResult(ok=True, content="ok"),
                ),
                LocalToolRegistration(
                    "invalid_second",
                    "invalid",
                    {"type": "array"},
                    lambda _arguments: ToolResult(ok=True, content="ok"),
                ),
            ],
        )

    server = _bare_server()
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))
    try:
        server._register_local_agent_tools()
    finally:
        logger.remove(sink_id)
    server.mcp.finalize()

    captured = capsys.readouterr()
    assert all("Local registration failed" not in record for record in records)
    assert all(sentinel not in record for record in records)
    assert captured.out == ""
    assert captured.err == (
        "Local MCP tools unavailable; continuing with built-in tools.\n"
    )
    assert sentinel not in captured.err
    assert [
        descriptor["name"] for descriptor in await server.mcp.list_tools(_context())
    ] == BUILTIN_TOOL_NAMES


def test_fastmcp_parameter_summary_workaround_is_removed() -> None:
    import tldw_chatbook.MCP.local_server_tools as local_server_tools

    assert not hasattr(local_server_tools, "_parameter_summary")
