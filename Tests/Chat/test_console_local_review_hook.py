"""Tests for the local-tool review hook + provider composition (Task 5).

The hook tests mirror build_mcp_review_hook's discipline: clear-first
stamps, ONE approval round trip per batch, verdicts only ever "proceed".
"""

import asyncio
import json
import weakref
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_chat_controller as controller_mod
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LocalApprovalEffect,
    LocalToolProvider,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_combined_review_hook,
    build_local_review_hook,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleLibraryItemScopeSnapshot,
    ConsoleProviderIntent,
    ConsoleResolvedDestination,
    ConsoleTurnLibraryAuthority,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_scratch_space import ConsoleScratchSpaceManager
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnExecutionContext,
)
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.runtime_policy.bootstrap import default_runtime_policy_path
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

ASK = EffectiveToolState(state="ask", origin="global_default")
ALLOW = EffectiveToolState(state="allow", origin="tool_override")

#: PR2a Task 5: the hook takes the reviewing run's id and every stamp it
#: writes is keyed by it. These tests each drive ONE run; the assertions
#: are unchanged apart from that key.
RUN = "run-1"


@pytest.fixture(autouse=True)
def _dispatching_run():
    """Bind ``RUN`` as the dispatching run for every test in this module.

    ``LocalToolProvider.invoke()`` reads the run whose call it is
    executing from ``run_context`` (bound in production by
    ``AgentService`` around each invocation), so a test that stamps for
    ``RUN`` and then invokes must be running as ``RUN``.
    """
    with use_run_id(RUN):
        yield


def provider(state, tmp_path):
    return LocalToolProvider(workspace_root=tmp_path, resolve_state=lambda hub: state)


def test_hook_clears_stamps_before_gating(tmp_path):
    p = provider(ASK, tmp_path)
    p.apply_batch_decisions(RUN, {"fs_list": "approve_once"})
    hook = build_local_review_hook(p, lambda pending: {})
    hook([], RUN)  # a turn with no calls still clears
    assert p._stamps == {}


def test_hook_gates_ask_calls_in_one_batch(tmp_path):
    p = provider(ASK, tmp_path)
    seen = []
    hook = build_local_review_hook(
        p, lambda pending: seen.append(pending) or {"fs_list": "approve_once"}
    )
    verdicts = hook(
        [
            ToolCall(name="fs_list", args={"path": "."}),
            ToolCall(name="fs_list", args={"path": "sub"}),
        ],
        RUN,
    )
    assert len(seen) == 1 and len(seen[0]) == 2  # ONE round trip for the batch
    assert verdicts == {"fs_list": "proceed"}
    assert p._stamps == {(RUN, "fs_list"): "approve_once"}


def test_local_pending_gate_carries_descriptor_owned_effects(tmp_path):
    gate = provider(ASK, tmp_path).pending_gate_for("fs_list", {"path": "."})

    assert gate is not None
    assert gate.effects == (LocalApprovalEffect.PRIVATE_READ,)


def test_mounted_approval_row_carries_exact_descriptor_effects(tmp_path):
    """The controller must pass the descriptor-owned effect through unchanged."""
    store = ConsoleChatStore()
    session = store.ensure_session()
    gate = provider(ASK, tmp_path).pending_gate_for(
        "fs_list", {"effects": ["network"], "path": "."}
    )
    assert gate is not None
    call = MCPPendingCall(
        llm_name=gate.llm_name,
        server_key=gate.server_key,
        tool_name=gate.tool_name,
        server_label=gate.server_label,
        arguments=gate.arguments,
        reason=gate.reason,
        effects=gate.effects,
    )
    controller = ConsoleChatController(store=store, provider_gateway=object())
    mounted: list[dict[str, object]] = []

    def _mount(payload: dict[str, object] | None) -> None:
        if payload is None:
            return
        mounted.append(payload)
        controller.resolve_pending_approval(
            {call.llm_name: "deny"}, round_id=str(payload["round_id"])
        )

    controller.app = SimpleNamespace(
        call_from_thread=lambda callback, *args: callback(*args)
    )
    controller.set_pending_approval = _mount
    controller.park_pending_approval = lambda _session_id: None

    assert controller.request_mcp_approvals([call], session_id=session.id) == {
        call.llm_name: "deny"
    }
    row = mounted[0]["calls"][0]
    assert row["effects"] == [LocalApprovalEffect.PRIVATE_READ]
    assert row["effects"] != row["arguments"]["effects"]


def test_hook_skips_non_ask_calls(tmp_path):
    p = provider(ALLOW, tmp_path)
    hook = build_local_review_hook(
        p, lambda pending: (_ for _ in ()).throw(AssertionError("must not ask"))
    )
    assert hook([ToolCall(name="fs_list", args={"path": "."})], RUN) == {}


def test_combined_hook_merges_verdicts(tmp_path):
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    hook = build_combined_review_hook(
        [
            build_local_review_hook(p1, lambda pending: {"fs_list": "approve_once"}),
            build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
        ]
    )
    # each provider only gates what it owns; both see the batch
    out = hook([ToolCall(name="fs_list", args={"path": "."})], RUN)
    assert out == {"fs_list": "proceed"}


def test_combined_hook_empty_list_is_noop():
    hook = build_combined_review_hook([])
    assert hook([ToolCall(name="fs_list", args={"path": "."})], RUN) == {}


def test_combined_hook_clears_later_providers_when_earlier_hook_raises(tmp_path):
    """I3 across providers: a raising hook must not strand a LATER provider's
    stale prior-turn stamp for the fail-open runtime to hand to invoke()."""
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    p1.apply_batch_decisions(RUN, {"fs_list": "approve_once"})  # stale, prior turn
    p2.apply_batch_decisions(RUN, {"fs_list": "approve_once"})  # stale, prior turn

    def raising_approvals(pending):
        raise RuntimeError("mid-shutdown")

    hook = build_combined_review_hook(
        [
            build_local_review_hook(p1, raising_approvals),
            build_local_review_hook(p2, raising_approvals),
        ]
    )
    with pytest.raises(RuntimeError):
        hook([ToolCall(name="fs_list", args={"path": "."})], RUN)
    # the exception propagates to run_agent_loop's fail-open handling, but
    # BOTH providers' stamps were cleared first -- no stale stamp survives.
    assert p1._stamps == {}
    assert p2._stamps == {}


def test_combined_hook_runs_remaining_hooks_after_a_raise(tmp_path):
    """A raise in one hook must not skip the remaining hooks entirely: hook 2
    still completes its own clear + round trip with this turn's decisions."""
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)

    def raising_approvals(pending):
        raise RuntimeError("mid-shutdown")

    hook = build_combined_review_hook(
        [
            build_local_review_hook(p1, raising_approvals),
            build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
        ]
    )
    with pytest.raises(RuntimeError):
        hook([ToolCall(name="fs_list", args={"path": "."})], RUN)
    assert p1._stamps == {}  # cleared at entry, round trip raised
    assert p2._stamps == {(RUN, "fs_list"): "deny"}  # fresh THIS-turn decision


# -- _compose_local_provider -------------------------------------------------


class _FakeService:
    """Minimal unified-control-plane stand-in for local provider composition."""

    def __init__(self, *, kill_switch=False, state=ASK):
        self._kill_switch = kill_switch
        self._state = state
        self.session_approvals = set()
        self.persisted_states = []
        self.recorded_decisions = []

    def get_kill_switch(self):
        return self._kill_switch

    def gate_tool_test(self, hub):
        return self._state

    def is_session_approved(self, server_key, tool_name):
        return (server_key, tool_name) in self.session_approvals

    def approve_for_session(self, server_key, tool_name):
        self.session_approvals.add((server_key, tool_name))

    def set_tool_state(self, server_key, tool_name, ui_state, *, tool=None):
        self.persisted_states.append((server_key, tool_name, ui_state))

    def record_tool_decision(
        self, server_key, tool_name, *, decision, initiator="agent", error=None
    ):
        self.recorded_decisions.append(
            (server_key, tool_name, decision, initiator, error)
        )


def _test_execution_context(
    scratch_snapshot,
    *,
    session_id="test-chat",
    tool_configuration=None,
):
    """Build the complete immutable turn authority production now requires."""
    provider_selection = ConsoleProviderSelection(provider="deepseek")
    return ConsoleTurnExecutionContext(
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session_id,
            provider_selection=provider_selection,
            scratch_space=scratch_snapshot,
            tool_configuration=tool_configuration or {},
        ),
        library_authority=ConsoleTurnLibraryAuthority(
            policy=ConsoleLibraryPolicySnapshot(
                auto_retrieve=ConsoleAutoRetrieve.NEVER,
                assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                policy_revision=0,
                source="test",
            ),
            direct_library_tools=False,
            source_types=(),
            scope_snapshot=ConsoleLibraryItemScopeSnapshot((), (), False),
            provider_intent=ConsoleProviderIntent("deepseek", None, None),
            attempt_id="test-attempt",
        ),
        resolved_destination=ConsoleResolvedDestination(
            provider="deepseek",
            model=None,
            endpoint_identity="test",
            egress_class=ConsoleEgressClass.PUBLIC_NETWORK,
        ),
    )


def _bare_controller(app):
    """A controller instance with only what _compose_local_provider touches."""
    controller = object.__new__(ConsoleChatController)
    controller.app = app
    controller._agent_bridge = None
    controller._pending_approval_event = None
    controller._pending_approval_decisions = None
    scratch_spaces = ConsoleScratchSpaceManager()
    scratch_snapshot = scratch_spaces.snapshot("test-chat")
    controller._scratch_spaces = scratch_spaces
    controller._test_turn_context = _test_execution_context(
        scratch_snapshot,
        tool_configuration={
            "local_tools_enabled": controller_mod.get_cli_setting(
                "console",
                "local_tools_enabled",
                True,
            )
        },
    )
    weakref.finalize(controller, scratch_spaces.dispose)
    return controller


def _compose_local_provider(controller, *args, **kwargs):
    """Call the production composer with this harness's captured scratch."""
    kwargs.setdefault("turn_context", controller._test_turn_context)
    return ConsoleChatController._compose_local_provider(
        controller,
        *args,
        **kwargs,
    )


def _console_settings(enabled=True, workspace_root=""):
    values = {
        ("console", "local_tools_enabled"): enabled,
        ("console", "workspace_root"): workspace_root,
    }

    def get_cli_setting(section, key=None, default=None):
        return values.get((section, key), default)

    return get_cli_setting


def test_compose_local_provider_disabled_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        controller_mod, "get_cli_setting", _console_settings(enabled=False)
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    assert _compose_local_provider(
        controller,
    ) == (None, None)


def test_compose_local_provider_missing_master_key_defaults_enabled(
    monkeypatch, tmp_path
):
    values = {("console", "workspace_root"): str(tmp_path)}

    def missing_master_setting(section, key=None, default=None):
        return values.get((section, key), default)

    monkeypatch.setattr(controller_mod, "get_cli_setting", missing_master_setting)
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, hook = _compose_local_provider(
        controller,
    )

    assert isinstance(local_provider, LocalToolProvider)
    assert callable(hook)


def test_compose_local_provider_coerces_quoted_false_to_disabled(monkeypatch, tmp_path):
    """task-3240 fix round 1 (Critical 2). `get_cli_setting` returns the
    RAW TOML value -- a hand-typed quoted "false" is a non-empty string
    and therefore truthy under a bare `not get_cli_setting(...)` read, so
    it would COMPOSE the entire local tool group while the MCP-hub gate
    checkbox (`Agents/builtin_tool_gate.py`'s `all_tool_gates()`) and
    `mcp_workbench.py`'s own `[console] local_tools_enabled` read both
    show it OFF -- the exact lie-class task-3240 exists to close, on the
    very gate it added. Must coerce identically to every other
    `[tools]`/`[console]` gate read in the codebase.
    """
    monkeypatch.setattr(
        controller_mod, "get_cli_setting", _console_settings(enabled="false")
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    assert _compose_local_provider(
        controller,
    ) == (None, None)


def test_compose_local_provider_coerces_quoted_true_to_enabled(monkeypatch, tmp_path):
    """Mirror case: a quoted "true" must still compose the provider."""
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(enabled="true", workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    local_provider, hook = _compose_local_provider(
        controller,
    )
    assert isinstance(local_provider, LocalToolProvider)
    assert callable(hook)


def test_compose_local_provider_no_service(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    controller = _bare_controller(SimpleNamespace())  # no unified_mcp_service
    assert _compose_local_provider(
        controller,
    ) == (None, None)


def test_compose_local_provider_kill_switch_on(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    app = SimpleNamespace(unified_mcp_service=_FakeService(kill_switch=True))
    controller = _bare_controller(app)
    assert _compose_local_provider(
        controller,
    ) == (None, None)


def test_compose_local_provider_kill_switch_read_failure_fails_closed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())

    class _RaisingService(_FakeService):
        def get_kill_switch(self):
            raise RuntimeError("store unavailable")

    controller = _bare_controller(
        SimpleNamespace(unified_mcp_service=_RaisingService())
    )
    assert _compose_local_provider(
        controller,
    ) == (None, None)


def test_compose_local_provider_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService()
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))

    local_provider, hook = _compose_local_provider(
        controller,
    )

    assert isinstance(local_provider, LocalToolProvider)
    assert local_provider.workspace_root == (
        controller._test_turn_context.scratch_space.root
    )
    assert callable(hook)
    catalog_ids = {entry.id for entry in local_provider.list_catalog()}
    assert {
        "local:web_search",
        "local:web_fetch",
        "local:web_crawl",
        "local:watchlists_search_items",
        "local:watchlists_get_item",
    } <= catalog_ids
    # resolve_state is the same payload source the MCP gate uses.
    gate = local_provider.pending_gate_for("fs_list", {"path": "."})
    assert gate is not None and gate.server_key == "local:__local__"


def test_default_chat_local_provider_uses_scratch_not_config_or_cwd(
    monkeypatch,
    tmp_path,
):
    configured = tmp_path / "configured"
    cwd = tmp_path / "cwd"
    configured.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(enabled=True, workspace_root=str(configured)),
    )
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = scratch_spaces.snapshot("chat-a")
    context = _test_execution_context(
        snapshot,
        session_id="chat-a",
        tool_configuration={
            "local_tools_enabled": True,
            "workspace_root": str(configured),
        },
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    controller._scratch_spaces = scratch_spaces

    provider, review = _compose_local_provider(
        controller,
        session_id="chat-a",
        turn_context=context,
    )

    assert provider.workspace_root == snapshot.root
    assert callable(review)
    assert scratch_spaces.dispose()


def test_default_chat_local_provider_rejects_after_scratch_close(tmp_path):
    scratch_spaces = ConsoleScratchSpaceManager(temp_parent=tmp_path)
    snapshot = scratch_spaces.snapshot("chat-a")
    context = _test_execution_context(
        snapshot,
        session_id="chat-a",
        tool_configuration={"local_tools_enabled": True},
    )
    controller = _bare_controller(
        SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    )
    controller._scratch_spaces = scratch_spaces
    provider, _review = _compose_local_provider(
        controller,
        session_id="chat-a",
        turn_context=context,
    )

    scratch_spaces.close("chat-a")
    result = provider.invoke("local:fs_list", {"path": "."})

    assert result.ok is False
    assert result.error == LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL
    assert scratch_spaces.wait_for_cleanup(timeout_seconds=2.0)


def test_compose_local_provider_reuses_app_database_and_loads_runtime_source_per_call(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    profile = tmp_path / "profile" / "config.toml"
    profile.parent.mkdir()
    profile.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile))

    class AppDatabase:
        def __init__(self):
            self.searches = 0

        def assert_agent_read_ready(self):
            return None

        def search_items_for_agent(self, **_kwargs):
            self.searches += 1
            return {"items": [], "has_more": False, "snapshot_max_item_id": 0}

        def get_source_collection_memberships(self, _source_ids):
            return {}

    database = AppDatabase()
    app = SimpleNamespace(
        unified_mcp_service=_FakeService(state=ALLOW),
        subscriptions_db=database,
    )
    controller = _bare_controller(app)
    provider, hook = _compose_local_provider(
        controller,
    )
    watchlists_service = provider._specs["watchlists_search_items"].handler.__self__
    assert watchlists_service._db_resolver() is database

    monkeypatch.setattr(
        controller_mod.asyncio,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sync Watchlists handlers must not create an event loop")
        ),
    )

    assert callable(hook)
    local_result = provider.invoke("local:watchlists_search_items", {})
    assert json.loads(local_result.content)["status"] == "ok"
    assert database.searches == 1

    RuntimeSourceStateStore(default_runtime_policy_path()).save(
        RuntimeSourceState(active_source="server")
    )
    server_result = provider.invoke("local:watchlists_search_items", {})
    assert json.loads(server_result.content) == {
        "status": "unsupported",
        "retryable": False,
        "message": (
            "server Watchlists search is not supported; switch Watchlists to Local "
            "before retrying"
        ),
    }
    assert database.searches == 1


@pytest.mark.asyncio
async def test_compose_local_provider_wires_transactional_watchlists_commands(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    profile = tmp_path / "profile" / "config.toml"
    profile.parent.mkdir()
    profile.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile))
    RuntimeSourceStateStore(default_runtime_policy_path()).save(RuntimeSourceState())
    database = SubscriptionsDB(tmp_path / "subscriptions.db")
    local_service = LocalWatchlistsService(db_factory=lambda: database)
    bundle_service = WatchlistBundleService(database)
    controller = _bare_controller(
        SimpleNamespace(
            unified_mcp_service=_FakeService(state=ALLOW),
            subscriptions_db=database,
            local_watchlists_service=local_service,
            watchlist_bundle_service=bundle_service,
        )
    )

    provider, _hook = _compose_local_provider(controller)
    result = await asyncio.to_thread(
        provider.invoke,
        "local:watchlists_create_sources",
        {"sources": [{"url": "https://example.test/feed?token=private"}]},
    )

    assert result.ok is True
    payload = json.loads(result.content)
    assert payload.get("results") == [
        {
            "input_index": 0,
            "outcome": "created",
            "source_id": "local:subscription:1",
        }
    ]
    assert database.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1

    created_collection = await asyncio.to_thread(
        provider.invoke,
        "local:watchlists_create_collection",
        {"name": "Threat intel", "source_ids": ["local:subscription:1"]},
    )
    conflict = await asyncio.to_thread(
        provider.invoke,
        "local:watchlists_create_collection",
        {"name": "threat INTEL", "if_exists": "conflict"},
    )
    assert json.loads(created_collection.content)["status"] == "ok"
    assert json.loads(conflict.content) == {
        "status": "conflict",
        "retryable": False,
        "message": "A collection with that name already exists.",
    }


def test_console_watchlists_real_reads_leave_app_owned_state_unchanged(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    profile = tmp_path / "profile" / "config.toml"
    profile.parent.mkdir()
    profile.write_text("", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(profile))
    policy_store = RuntimeSourceStateStore(default_runtime_policy_path())
    policy_store.save(RuntimeSourceState())

    database = SubscriptionsDB(tmp_path / "subscriptions.db")
    try:
        source_id = database.add_subscription(
            name="Evidence feed",
            type="rss",
            source="https://example.test/feed?token=private",
        )
        with database.transaction() as conn:
            collection_id = conn.execute(
                "INSERT INTO watchlists (name) VALUES ('Evidence collection')"
            ).lastrowid
            conn.execute(
                "INSERT INTO watchlist_sources (watchlist_id, subscription_id) "
                "VALUES (?, ?)",
                (collection_id, source_id),
            )
            item_id = conn.execute(
                """
                INSERT INTO subscription_items (
                    subscription_id, url, title, content, status, is_flagged,
                    published_date
                ) VALUES (?, ?, ?, ?, 'reviewed', 1, ?)
                """,
                (
                    source_id,
                    "https://example.test/item?secret=value",
                    "Read-only evidence",
                    "needle body",
                    "2026-08-14T12:00:00Z",
                ),
            ).lastrowid

        def snapshot():
            tables = (
                "schema_version",
                "subscriptions",
                "subscription_items",
                "watchlists",
                "watchlist_sources",
            )
            return {
                table: [
                    tuple(row)
                    for row in database.conn.execute(f"SELECT * FROM {table}")
                ]
                for table in tables
            }

        before_database = snapshot()
        before_policy = policy_store.path.read_bytes()
        controller = _bare_controller(
            SimpleNamespace(
                unified_mcp_service=_FakeService(state=ALLOW),
                subscriptions_db=database,
            )
        )
        provider, _hook = _compose_local_provider(
            controller,
        )

        search = provider.invoke("local:watchlists_search_items", {"query": "needle"})
        detail = provider.invoke(
            "local:watchlists_get_item",
            {"item_id": f"local:watchlist_item:{item_id}"},
        )

        assert json.loads(search.content)["status"] == "ok"
        assert json.loads(detail.content)["status"] == "ok"
        assert snapshot() == before_database
        assert policy_store.path.read_bytes() == before_policy
    finally:
        database.close()


def test_compose_local_provider_empty_workspace_root_uses_scratch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    monkeypatch.chdir(tmp_path)
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = _compose_local_provider(
        controller,
    )

    assert local_provider.workspace_root == (
        controller._test_turn_context.scratch_space.root
    )
    assert local_provider.workspace_root != tmp_path.resolve()


def test_local_provider_read_only_filters_write_specs_without_global_mutation(tmp_path):
    before = LocalToolProvider(workspace_root=tmp_path)
    read_only = LocalToolProvider(workspace_root=tmp_path, allow_write=False)
    after = LocalToolProvider(workspace_root=tmp_path)

    before_names = {entry.name for entry in before.list_catalog()}
    read_only_names = {entry.name for entry in read_only.list_catalog()}
    after_names = {entry.name for entry in after.list_catalog()}
    assert {"fs_write", "fs_edit", "fs_patch"} <= before_names
    assert {"fs_write", "fs_edit", "fs_patch"}.isdisjoint(read_only_names)
    assert {"fs_read", "fs_list", "git_status"} <= read_only_names
    assert after_names == before_names


def test_compose_local_provider_selected_root_overrides_disabled_fallback(
    monkeypatch, tmp_path
):
    fallback = tmp_path / "fallback"
    selected = tmp_path / "selected"
    fallback.mkdir()
    selected.mkdir()
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(fallback)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    legacy_provider, _ = _compose_local_provider(
        controller,
    )
    selected_provider, _ = _compose_local_provider(
        controller, project_root=selected, allow_write=False
    )

    assert legacy_provider.workspace_root == (
        controller._test_turn_context.scratch_space.root
    )
    assert legacy_provider.workspace_root != fallback.resolve()
    assert selected_provider.workspace_root == selected.resolve()
    selected_names = {entry.name for entry in selected_provider.list_catalog()}
    assert {"fs_write", "fs_edit", "fs_patch"}.isdisjoint(selected_names)


def test_selected_root_swap_fails_closed_before_local_invoke(monkeypatch, tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "secret.txt").write_text("inside")
    identity = controller_mod._capture_project_root_identity(selected)
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    controller = _bare_controller(
        SimpleNamespace(unified_mcp_service=_FakeService(state=ALLOW))
    )
    local_provider, review = _compose_local_provider(
        controller,
        project_root=selected,
        project_root_identity=identity,
    )

    moved = tmp_path / "moved"
    selected.rename(moved)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")
    selected.symlink_to(outside, target_is_directory=True)

    assert (
        review(
            [ToolCall(name="fs_read", args={"path": "secret.txt"})],
            "run-root-swap",
        )
        == {}
    )
    result = local_provider.invoke("fs_read", {"path": "secret.txt"})
    assert result.ok is False
    assert "root changed" in result.error.lower()
    assert "outside" not in result.error


def test_compose_local_provider_tilde_workspace_root_does_not_grant_home_access(
    monkeypatch, tmp_path
):
    """The retired configured root cannot replace a chat's private scratch."""
    home = tmp_path / "home"
    (home / "repo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root="~/repo"),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = _compose_local_provider(
        controller,
    )

    assert local_provider.workspace_root == (
        controller._test_turn_context.scratch_space.root
    )
    assert local_provider.workspace_root != (home / "repo").resolve()


def test_compose_local_provider_persists_session_and_always_allow(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService()
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    local_provider, _hook = _compose_local_provider(
        controller,
    )

    (tmp_path / "a.txt").write_text("a")

    local_provider.apply_batch_decisions(RUN, {"fs_list": "approve_session"})
    assert local_provider.invoke("local:fs_list", {"path": "."}).ok
    assert ("local:__local__", "fs_list") in service.session_approvals

    local_provider.apply_batch_decisions(RUN, {"fs_list": "always_allow"})
    assert local_provider.invoke("local:fs_list", {"path": "."}).ok
    assert service.persisted_states == [("local:__local__", "fs_list", "allow")]


def test_compose_local_provider_session_approval_skips_reprompt(monkeypatch, tmp_path):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService()
    service.approve_for_session("local:__local__", "fs_list")
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    local_provider, _hook = _compose_local_provider(
        controller,
    )

    assert local_provider.pending_gate_for("fs_list", {"path": "."}) is None
    (tmp_path / "a.txt").write_text("a")
    assert local_provider.invoke("local:fs_list", {"path": "."}).ok


# -- audit recording wiring (Task 7) -------------------------------------------


def _composed(monkeypatch, tmp_path, service):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    local_provider, _hook = _compose_local_provider(
        controller,
    )
    assert local_provider is not None
    return local_provider


def test_compose_local_provider_records_deny_via_service(monkeypatch, tmp_path):
    service = _FakeService(
        state=EffectiveToolState(state="deny", origin="tool_override")
    )
    local_provider = _composed(monkeypatch, tmp_path, service)

    r = local_provider.invoke("local:fs_list", {"path": "."})

    assert not r.ok
    assert service.recorded_decisions == [
        ("local:__local__", "fs_list", "denied", "agent", None)
    ]


def test_compose_local_provider_records_timeout_via_service(monkeypatch, tmp_path):
    service = _FakeService()  # ASK state
    local_provider = _composed(monkeypatch, tmp_path, service)
    local_provider.apply_batch_decisions(RUN, {"fs_list": "timeout"})

    r = local_provider.invoke("local:fs_list", {"path": "."})

    assert not r.ok
    assert service.recorded_decisions == [
        ("local:__local__", "fs_list", "denied-timeout", "agent", None)
    ]


def test_compose_local_provider_allow_records_no_refusal(monkeypatch, tmp_path):
    service = _FakeService(state=ALLOW)
    local_provider = _composed(monkeypatch, tmp_path, service)
    (tmp_path / "a.txt").write_text("a")

    assert local_provider.invoke("local:fs_list", {"path": "."}).ok
    assert service.recorded_decisions == []


def test_compose_local_provider_recording_failure_does_not_break_invoke(
    monkeypatch, tmp_path
):
    class _RaisingRecordService(_FakeService):
        def __init__(self):
            super().__init__(
                state=EffectiveToolState(state="deny", origin="tool_override")
            )

        def record_tool_decision(self, *args, **kwargs):
            raise RuntimeError("audit store down")

    local_provider = _composed(monkeypatch, tmp_path, _RaisingRecordService())

    r = local_provider.invoke("local:fs_list", {"path": "."})
    assert not r.ok  # refusal still returned; the raise was swallowed


# -- stable task session wiring (TASK-13216 Task 5) -----------------------------


_TASK_TOOL_NAMES = {
    "todo_create",
    "todo_update",
    "todo_get",
    "todo_list",
}


def _registered_task_tools(provider: LocalToolProvider) -> set[str]:
    return {
        entry.name
        for entry in provider.list_catalog()
        if entry.name in _TASK_TOOL_NAMES
    }


def test_compose_local_provider_without_session_registers_no_todo_spec(
    monkeypatch, tmp_path
):
    """No session context keeps all four stable task tools absent."""
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = _compose_local_provider(
        controller,
    )

    assert _registered_task_tools(local_provider) == set()
    assert "todo_write" not in {entry.name for entry in local_provider.list_catalog()}


def test_compose_local_provider_wires_the_sessions_exact_todo_store(
    monkeypatch, tmp_path
):
    """An inactive target, not the active session, owns provider task state."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService(state=ALLOW)
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    controller.store = ConsoleChatStore()
    target = controller.store.create_session(title="Target", workspace_id="ws")
    active = controller.store.create_session(title="Active", workspace_id="ws")
    assert controller.store.active_session_id == active.id
    markers = []
    controller._agent_bridge = SimpleNamespace(
        append_todo_marker=lambda session_id, todos: markers.append(
            (session_id, list(todos))
        )
    )

    local_provider, _hook = _compose_local_provider(controller, session_id=target.id)

    created = local_provider.invoke("local:todo_create", {"content": "Ship it"})

    assert created.ok
    assert target.todo_store.get("1")["content"] == "Ship it"
    assert active.todo_store.list_after(None) == []
    assert markers == [(target.id, target.todo_store.list_after(None))]


def test_compose_local_provider_unknown_session_registers_no_todo_spec(
    monkeypatch, tmp_path
):
    """A session_id the store does not know must not create todo state."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    controller.store = ConsoleChatStore()
    controller._agent_bridge = SimpleNamespace(append_todo_marker=lambda *a: None)

    local_provider, _hook = _compose_local_provider(controller, session_id="ghost")

    assert _registered_task_tools(local_provider) == set()


def test_compose_local_provider_without_bridge_registers_no_todo_spec(
    monkeypatch, tmp_path
):
    """A live session without a transcript bridge exposes no task capability."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))
    controller.store = ConsoleChatStore()
    session = controller.store.create_session(workspace_id="ws")
    controller._agent_bridge = None

    local_provider, _hook = _compose_local_provider(controller, session_id=session.id)

    assert _registered_task_tools(local_provider) == set()
