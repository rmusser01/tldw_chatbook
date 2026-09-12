"""Tests for the local-tool review hook + provider composition (Task 5).

The hook tests mirror build_mcp_review_hook's discipline: clear-first
stamps, ONE approval round trip per batch, verdicts only ever "proceed".
"""

import asyncio
import contextlib
import json
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_chat_controller as controller_mod
from tldw_chatbook.Agents.agent_models import ToolCall, ToolResult
from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL,
    LocalApprovalEffect,
    LocalToolProvider,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    USER_DENIED_REFUSAL,
    watchlists_operation_receipt_ids,
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


def test_watchlists_receipt_capture_accepts_only_structured_canonical_ids():
    secret = "private article body and arguments"
    check_result = ToolResult(
        ok=True,
        content=json.dumps(
            {
                "status": "accepted",
                "operations": [
                    {"operation_id": "local:watchlist_run:7", "body": secret},
                    {"operation_id": "local:watchlist_run:0"},
                    {"operation_id": "external:mcp:8"},
                ],
                "arguments": {"source_ids": [secret]},
            }
        ),
    )
    briefing_result = ToolResult(
        ok=True,
        content=json.dumps(
            {
                "status": "accepted",
                "operation_id": "local:briefing:9",
                "body": secret,
            }
        ),
    )

    assert watchlists_operation_receipt_ids(
        "watchlists_check_sources", check_result
    ) == ("local:watchlist_run:7",)
    assert watchlists_operation_receipt_ids(
        "watchlists_generate_briefing", briefing_result
    ) == ("local:briefing:9",)
    assert (
        watchlists_operation_receipt_ids(
            "watchlists_check_sources",
            ToolResult(ok=False, content=check_result.content),
        )
        == ()
    )
    assert watchlists_operation_receipt_ids("calculator", check_result) == ()
    assert secret not in repr(
        watchlists_operation_receipt_ids("watchlists_check_sources", check_result)
    )


def test_controller_retains_and_publishes_only_canonical_receipt_identity():
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=object(),
    )
    published: list[tuple[str, ...]] = []
    controller.follow_watchlists_operations = published.append
    controller.app = SimpleNamespace(call_from_thread=lambda callback: callback())

    controller.observe_watchlists_operation_result(
        "run-1",
        "call-1",
        "watchlists_generate_briefing",
        ToolResult(
            ok=True,
            content=json.dumps(
                {
                    "status": "accepted",
                    "operation_id": "local:briefing:9",
                    "body": "never retain me",
                }
            ),
        ),
    )

    assert controller._followed_watchlists_operation_ids == ("local:briefing:9",)
    assert published == [("local:briefing:9",)]
    assert "never retain me" not in repr(controller._followed_watchlists_operation_ids)


def test_controller_unfollow_survives_view_detach_and_remount():
    controller = ConsoleChatController(
        store=ConsoleChatStore(),
        provider_gateway=object(),
    )
    controller.app = SimpleNamespace(call_from_thread=lambda callback: callback())
    controller.observe_watchlists_operation_result(
        "run-1",
        "call-1",
        "watchlists_generate_briefing",
        ToolResult(
            ok=True,
            content=json.dumps(
                {
                    "status": "accepted",
                    "operation_id": "local:briefing:9",
                }
            ),
        ),
    )

    assert controller.unfollow_watchlists_operation("local:briefing:9") is True
    assert controller.unfollow_watchlists_operation("server:briefing:9") is False

    published: list[tuple[str, ...]] = []
    controller.follow_watchlists_operations = None
    controller.follow_watchlists_operations = published.append
    controller.remount_watchlists_operation_receipts()

    assert published == [()]
    assert controller._followed_watchlists_operation_ids == ()


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


def test_hook_keeps_same_name_watchlist_decisions_per_call(tmp_path):
    """One denied collection target must not cancel its approved sibling."""
    p = provider(ASK, tmp_path)
    seen: list[list[MCPPendingCall]] = []

    def decide(pending: list[MCPPendingCall]) -> dict[str, str]:
        seen.append(pending)
        return {
            "call-denied": "deny",
            "call-session": "approve_session",
        }

    hook = build_local_review_hook(p, decide)
    verdicts = hook(
        [
            ToolCall(
                name="watchlists_create_collection",
                args={"name": "Blocked target", "if_exists": "conflict"},
                call_id="call-denied",
            ),
            ToolCall(
                name="watchlists_create_collection",
                args={"name": "Approved target", "if_exists": "conflict"},
                call_id="call-session",
            ),
        ],
        RUN,
    )

    assert [row.call_id for row in seen[0]] == ["call-denied", "call-session"]
    assert verdicts == {
        "watchlists_create_collection": "proceed",
        "call-denied": USER_DENIED_REFUSAL.format(name="watchlists_create_collection"),
    }
    assert p._stamps == {(RUN, "watchlists_create_collection"): "approve_session"}


@pytest.mark.parametrize(
    ("broad", "narrow"),
    [
        ("always_allow", "approve_session"),
        ("always_allow", "approve_once"),
        ("approve_session", "approve_once"),
    ],
)
def test_local_same_name_broad_approval_scope_survives_later_narrow_scope(
    tmp_path, broad, narrow
):
    """A later per-call approval must not downgrade the tool-level grant."""
    p = provider(ASK, tmp_path)
    hook = build_local_review_hook(
        p,
        lambda _pending: {
            "call-broad": broad,
            "call-narrow": narrow,
        },
    )

    verdicts = hook(
        [
            ToolCall(
                name="watchlists_create_collection",
                args={"name": "Broad", "if_exists": "conflict"},
                call_id="call-broad",
            ),
            ToolCall(
                name="watchlists_create_collection",
                args={"name": "Narrow", "if_exists": "conflict"},
                call_id="call-narrow",
            ),
        ],
        RUN,
    )

    assert verdicts == {"watchlists_create_collection": "proceed"}
    assert p.stamped(RUN, "watchlists_create_collection") == broad


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


def test_combined_hook_does_not_overwrite_a_local_refusal(tmp_path):
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    hook = build_combined_review_hook(
        [
            build_local_review_hook(p1, lambda pending: {"fs_list": "approve_once"}),
            build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
        ]
    )
    # This deliberately impossible double-owner arrangement pins the merge
    # safety rule: a later refusal cannot be weakened to proceed.
    out = hook([ToolCall(name="fs_list", args={"path": "."})], RUN)
    assert out == {"fs_list": USER_DENIED_REFUSAL.format(name="fs_list")}


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


def test_hook_level_card_deny_lands_in_the_execution_log_exactly_once(tmp_path):
    """task-32280 fix round (Critical review finding).

    Mirrors `test_mcp_tool_provider.py::
    test_hook_level_card_deny_lands_in_the_execution_log_exactly_once`: a
    hook-level deny is turned straight into the call's result by
    `run_agent_loop`, which skips dispatch entirely, so
    `LocalToolProvider.invoke_detailed()` -- the only thing that otherwise
    records a local refusal -- never runs for it. Drives the REAL provider
    through the REAL `build_local_review_hook` so a fake cannot paper over
    the gap; the split `denied`/`denied-policy`/`denied-unresolved` tokens
    added to `invoke_detailed()` are dead code for this path otherwise.
    """
    recorded: list[tuple[str, str]] = []
    p = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: ASK,
        record_decision=lambda hub, decision: recorded.append((hub.name, decision)),
    )
    hook = build_local_review_hook(p, lambda pending: {"fs_list": "deny"})

    verdicts = hook(
        [ToolCall(name="fs_list", args={"path": "."}, call_id="c-1")], RUN
    )

    assert verdicts.get("c-1", "proceed") != "proceed", (
        f"precondition: the denied call must not be dispatched: {verdicts}"
    )
    assert recorded == [("fs_list", "denied")], (
        "the user's Deny left no row in the execution log: " f"{recorded}"
    )


def test_stop_mid_approval_records_only_the_unresolved_row(tmp_path):
    """R23, local half: mirrors `test_mcp_tool_provider.py::
    test_stop_mid_approval_records_only_the_unresolved_row`. A Stop while
    the card is up already writes the honest `denied-unresolved` row from
    the controller; the hook must not add a "Denied by you" one on top."""
    from tldw_chatbook.MCP.execution_log import UNRESOLVED_DENIED_DECISION

    recorded: list[tuple[str, str]] = []
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: ASK,
        record_decision=lambda hub, decision: recorded.append((hub.name, decision)),
    )
    cancel_rows: list[tuple] = []
    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=object()
    )
    controller.app = SimpleNamespace(
        call_from_thread=lambda fn, *a, **kw: fn(*a, **kw),
        unified_mcp_service=SimpleNamespace(
            record_tool_decision=lambda server_key, tool_name, **kw: cancel_rows.append(
                (server_key, tool_name, kw.get("decision"))
            )
        ),
    )
    controller.set_pending_approval = lambda payload: None
    controller.mcp_approval_timeout_seconds = lambda: 30.0

    def _stop_soon() -> None:
        time.sleep(0.05)
        # begin_shutdown() runs its owner-thread queue teardown inline when
        # no owner loop is bound (none is, in this synchronous test), which
        # trips the queue's cross-thread guard -- it still denies the
        # unresolved approval first (in begin_shutdown's `finally`), so the
        # assertions below hold; only the thread-local exception is noise.
        with contextlib.suppress(Exception):
            controller.begin_shutdown()

    stopper = threading.Thread(target=_stop_soon)
    stopper.start()
    hook = build_local_review_hook(provider, controller.request_mcp_approvals)
    with use_run_id(RUN):
        verdicts = hook(
            [ToolCall(name="fs_list", args={"path": "."}, call_id="c-1")], RUN
        )
    stopper.join()

    assert verdicts.get("c-1", "proceed") != "proceed"
    assert recorded == [], (
        f"a Stop mid-approval recorded a user denial it never received: {recorded}"
    )
    assert [row[2] for row in cancel_rows] == [UNRESOLVED_DENIED_DECISION]


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
    controller.set_pending_question = None
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
    assert (
        database.conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0] == 1
    )

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


def test_compose_local_provider_routes_schedule_through_shared_app_command_service(
    monkeypatch, tmp_path
):
    """Console and Artifacts use the same app-owned schedule command seam."""
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    calls = []

    def _record_schedule(arguments):
        calls.append(dict(arguments))
        return '{"status":"ok","reload_requested":true,"reload_acknowledged":true}'

    def unavailable(_arguments):
        return '{"status":"feature_unavailable"}'

    commands = SimpleNamespace(
        create_sources=unavailable,
        create_collection=unavailable,
        update_collection_sources=unavailable,
        check_sources=unavailable,
        generate_briefing=unavailable,
        set_briefing_schedule=_record_schedule,
        approval_source_destinations=lambda _arguments: {},
    )
    controller = _bare_controller(
        SimpleNamespace(
            unified_mcp_service=_FakeService(state=ALLOW),
            watchlists_command_service=commands,
        )
    )

    provider, _hook = _compose_local_provider(controller)
    result = provider.invoke(
        "local:watchlists_set_briefing_schedule",
        {
            "collection_id": "local:watchlist:7",
            "cadence": "every_24_hours",
        },
    )

    assert result.ok is True
    assert json.loads(result.content)["reload_acknowledged"] is True
    assert calls == [
        {
            "collection_id": "local:watchlist:7",
            "cadence": "every_24_hours",
        }
    ]


@pytest.mark.asyncio
async def test_compose_local_provider_routes_long_watchlists_work_to_app_coordinator(
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

    class Coordinator:
        def __init__(self):
            self.checks = []
            self.briefings = []

        def submit_checks(self, source_ids):
            self.checks.append(source_ids)
            return [{"run_id": 7, "source_id": 3, "status": "queued"}]

        def submit_briefing(self, watchlist_id, preset_id):
            self.briefings.append((watchlist_id, preset_id))
            return {"id": 11, "status": "generating"}

    coordinator = Coordinator()
    bundle = SimpleNamespace(list_sources=lambda watchlist_id: [3])
    app = SimpleNamespace(
        unified_mcp_service=_FakeService(state=ALLOW),
        watchlists_operation_coordinator=coordinator,
        watchlist_bundle_service=bundle,
    )
    provider, _hook = _compose_local_provider(_bare_controller(app))

    check = await asyncio.to_thread(
        provider.invoke,
        "local:watchlists_check_sources",
        {"collection_id": "local:watchlist:5"},
    )
    briefing = await asyncio.to_thread(
        provider.invoke,
        "local:watchlists_generate_briefing",
        {"collection_id": "local:watchlist:5", "preset_id": 2},
    )

    assert json.loads(check.content)["operations"][0]["operation_id"] == (
        "local:watchlist_run:7"
    )
    assert json.loads(briefing.content)["operation_id"] == "local:briefing:11"
    assert coordinator.checks == [[3]]
    assert coordinator.briefings == [(5, 2)]


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


def test_console_run_without_admitted_roots_keeps_non_path_local_tools(tmp_path):
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    provider, review = _compose_local_provider(controller, admitted_roots=())

    names = {entry.name for entry in provider.list_catalog()}
    assert {
        "fs_list",
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_patch",
        "fs_glob",
        "fs_grep",
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
        "git_branches",
    }.isdisjoint(names)
    assert {"web_search", "web_fetch", "watchlists_search_items"} <= names
    assert callable(review)


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
    # task-32280 fix round: a tool configured Off is not a person saying no.
    # Through the REAL wiring (the controller's `record_decision` seam into
    # the service), not just the provider's own unit test.
    assert service.recorded_decisions == [
        ("local:__local__", "fs_list", "denied-policy", "agent", None)
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


def test_hook_passes_rationale_onto_local_pending_rows(tmp_path):
    """Qodo review #10: the local owner receives each call's rationale.

    Without this, every local approval row renders without the model's
    advisory context even though MCP and builtin rows carry it.
    """
    p = provider(ASK, tmp_path)
    seen: list[list[MCPPendingCall]] = []

    def decide(pending: list[MCPPendingCall]) -> dict[str, str]:
        seen.append(pending)
        return {"fs_list": "deny"}

    hook = build_local_review_hook(p, decide)
    hook(
        [
            ToolCall(
                name="fs_list",
                args={"path": "."},
                rationale="Listing the workspace to find the config",
            ),
            ToolCall(name="fs_list", args={"path": "sub"}),
        ],
        RUN,
    )
    assert seen[0][0].rationale == "Listing the workspace to find the config"
    assert seen[0][1].rationale == ""


def test_a_no_app_headless_round_is_not_recorded_as_a_local_user_denial(tmp_path):
    """task-32280 (Qodo #2597 #8): `request_mcp_approvals` fails CLOSED when
    no app is wired -- no card is ever shown, so no user decides anything.
    It returned a bare `{key: "deny"}` dict, which `approval_was_unanswered`
    reads as "answered", so THIS hook wrote `record_user_denial()` and the
    local audit trail claimed a person pressed Deny.

    Driven through the REAL `request_mcp_approvals` (not a stub returning
    the same shape), because the bug lived in what that method returns.
    """
    from tldw_chatbook.Chat.console_chat_controller import ApprovalDecisions

    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=object()
    )
    assert controller.app is None  # the branch under test

    p = provider(ASK, tmp_path)
    denials: list[str] = []
    p.record_user_denial = denials.append

    seen: list[dict] = []

    def _headless_round(pending):
        decisions = controller.request_mcp_approvals(pending)
        seen.append(decisions)
        return decisions

    hook = build_local_review_hook(p, _headless_round)
    verdicts = hook(
        [ToolCall(name="fs_list", args={"path": "."}, call_id="call-1")], RUN
    )

    # Fails closed, exactly as before.
    assert verdicts["call-1"] == USER_DENIED_REFUSAL.format(name="fs_list")
    assert isinstance(seen[0], ApprovalDecisions)
    assert seen[0].unresolved_keys == frozenset({"call-1"})
    assert denials == [], "a headless fail-closed deny was audited as the user's"


# -- PreToolUse hooks around the review chain (console run hooks Task 6) --------


def _deny_fs_tools_engine(tmp_path):
    """A RunHooksEngine with one exit-2 PreToolUse hook matching ``fs_*``.

    The stub command is deliberately a real ``sys.executable`` process (spec
    §5's protocol is subprocess-based; no in-process fakes here), and the
    exit code is the spec's deny shorthand (exit 2, no JSON stdout).
    """
    import sys as _sys

    from tldw_chatbook.Agents.run_hooks import HookSpec, RunHooksConfig, RunHooksEngine

    return RunHooksEngine(
        lambda: RunHooksConfig(
            enabled=True,
            hooks=(
                HookSpec(
                    "PreToolUse",
                    (_sys.executable, "-c", "raise SystemExit(2)"),
                    matcher="fs_*",
                ),
            ),
        ),
        lambda: str(tmp_path),
    )


def test_pretooluse_hook_denies_before_permission_store(tmp_path):
    """A configured exit-2 PreToolUse hook denies the matched call and the
    approval round never carries it (deny-only, spec §5).

    Composition-level: the engine's wrap_review layers the real controller
    review hook (build_local_review_hook) exactly the way the bridge does --
    the matched ``fs_*`` call comes back as a namespaced ``hook: `` refusal
    without ever entering the request_approvals round, while the
    non-matching tool runs the normal ask -> approve -> proceed chain.
    """
    engine = _deny_fs_tools_engine(tmp_path)
    p = provider(ASK, tmp_path)
    rounds = []

    def approvals(pending):
        rounds.append([call.tool_name for call in pending])
        return {call.tool_name: "approve_once" for call in pending}

    wrapped = engine.wrap_review(
        build_local_review_hook(p, approvals), session_id="session-1"
    )
    verdicts = wrapped(
        [
            ToolCall(name="fs_list", args={"path": "."}),
            ToolCall(name="git_status", args={"path": "."}),
        ],
        RUN,
    )
    assert verdicts["fs_list"] != "proceed"
    assert verdicts["fs_list"].startswith("hook: ")
    assert verdicts["git_status"] == "proceed"
    # ONE approval round trip, carrying only the non-matching call: the
    # hook-denied call never reaches the permission store.
    assert rounds == [["git_status"]]


def test_run_reply_wraps_the_review_chain_with_pretooluse_hooks(tmp_path):
    """Task 6 bridge wiring: run_reply must wrap the caller's review chain
    with the engine's PreToolUse layer when the bridge was built with an
    ``ensure_run_hooks`` accessor.

    Verified end-to-end through a real ``run_reply`` (scripted gateway, real
    LocalToolProvider + real build_local_review_hook as the caller-supplied
    chain): the matched ``fs_read`` call is refused with a namespaced
    ``hook: `` verdict and the approval round only ever carries the
    non-matching ``git_status`` call. Without the wrap, the batch arrives at
    the review chain whole and no ``hook: `` refusal exists.
    """
    import json as _json

    from tldw_chatbook.Agents.agent_models import STEP_TOOL_RESULT
    from tldw_chatbook.Agents.local_tool_provider import _default_specs
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderResolution,
        ProviderToolCalls,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    engine = _deny_fs_tools_engine(tmp_path)

    class _ScriptedGateway:
        """Streams one scripted chunk-list per model call (fakes only)."""

        def __init__(self, scripts):
            self._scripts = list(scripts)
            self.calls = 0

        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            chunks = self._scripts[self.calls]
            self.calls += 1
            for chunk in chunks:
                yield chunk

    def _native_batch(*pairs):
        return ProviderToolCalls(
            tool_calls=tuple(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": _json.dumps(args)},
                }
                for name, args, call_id in pairs
            )
        )

    gateway = _ScriptedGateway(
        [
            [
                _native_batch(
                    ("fs_read", {"path": "."}, "read-1"),
                    ("git_status", {"path": "."}, "git-1"),
                )
            ],
            ["done."],
        ]
    )
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    dispatched = []

    def execute(operation, arguments, *, intent):
        dispatched.append(operation)
        return "clean working tree"

    local = LocalToolProvider(
        workspace_root=tmp_path,
        specs=[
            spec
            for spec in _default_specs(
                tmp_path, workspace_executor=SimpleNamespace(execute=execute)
            )
            if spec.name in {"fs_read", "git_status"}
        ],
        resolve_state=lambda _hub: ASK,
    )
    rounds = []

    def approvals(pending):
        rounds.append([call.tool_name for call in pending])
        return {call.tool_name: "approve_once" for call in pending}

    bridge = ConsoleAgentBridge(
        agent_runs_db=db,
        store=store,
        provider_gateway=gateway,
        ensure_run_hooks=lambda: engine,
    )
    _run_id, outcome = bridge.run_reply(
        conversation_id="conv-1",
        session_id=session.id,
        resolution=ConsoleProviderResolution(
            provider="Groq", execution_key="groq", base_url="", model=None, ready=True
        ),
        assistant_message_id=assistant.id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
        local_provider=local,
        review_tool_calls=build_local_review_hook(local, approvals),
    )

    assert outcome.status == "done", outcome.steps
    # The hook denied fs_read BEFORE the review chain: the one approval
    # round carried only the non-matching call.
    assert rounds == [["git_status"]]
    results = {
        step.tool_name: step.result
        for step in outcome.steps
        if step.kind == STEP_TOOL_RESULT
    }
    assert results["fs_read"].startswith("hook: ")
    assert "git_status" in results  # ran the normal chain and dispatched
    assert dispatched == ["git_status"]


# -- ApprovalRequested at the approval-round registration (run hooks Task 8) --


class _RecordingHooksEngine:
    """Test double for the run-hooks engine: records every ``notify`` fire.

    The Task 8 events (ApprovalRequested / Stop / SubagentStop) are all
    non-blocking ``notify`` fires -- a recorder is the whole contract the
    fire sites consume. Runs nothing, unlike the real engine.
    """

    def __init__(self):
        self.notifications: list[tuple[str, dict]] = []

    def notify(self, event, **kwargs):
        self.notifications.append((event, kwargs))


class _InlineCallFromThreadApp:
    """``call_from_thread`` stand-in: run the marshalled callback inline."""

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def test_approval_requested_fires_with_round_payload():
    """One ask-state call through ``request_mcp_approvals`` fires
    ApprovalRequested exactly once, the moment the round is registered --
    before any bridge slot (mount/park) is consulted -- carrying the
    call's name and ``session_active=True`` for the viewed session."""
    import threading
    import time

    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    engine = _RecordingHooksEngine()
    store = ConsoleChatStore()
    session = store.ensure_session()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        ensure_run_hooks=lambda: engine,
    )
    mounted: list[dict | None] = []
    controller.app = _InlineCallFromThreadApp()
    controller.set_pending_approval = mounted.append
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    pending = MCPPendingCall(
        llm_name="local:__local__:fs_write",
        server_key="local:__local__",
        tool_name="fs_write",
        server_label="Local tools",
        arguments={"path": "notes.txt"},
        reason="ask",
    )

    def _resolve_soon() -> None:
        time.sleep(0.05)
        assert mounted and mounted[-1] is not None
        controller.resolve_pending_approval(
            {"local:__local__:fs_write": "approve_once"},
            round_id=mounted[-1]["round_id"],
        )

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals([pending], session_id=session.id)

    assert decisions == {"local:__local__:fs_write": "approve_once"}
    # Exactly ONE notify for the round: the event name, the round's owning
    # session, the call's name, and the active-session flag.
    assert [event for event, _ in engine.notifications] == ["ApprovalRequested"]
    _event, kwargs = engine.notifications[0]
    assert kwargs["session_id"] == session.id
    # Review fix R29: spec §4 promised an args summary per call -- a stable
    # small args dict serializes to exactly its JSON string.
    assert kwargs["data"] == {
        "calls": [
            {
                "name": "local:__local__:fs_write",
                "args_summary": '{"path": "notes.txt"}',
            }
        ],
        "session_active": True,
    }


def test_approval_requested_fires_for_view_detached_round():
    """A parked round armed with NO Console view anywhere (both bridge
    seams unwired) still fires ApprovalRequested exactly once, with
    ``session_active=False`` -- no view exists, so the round's owning
    session is not the viewed one. Review finding R25: the detached
    branch announced a toast but fired no event, while the ``elif
    is_parked:`` branch below it fired -- a detached background round is
    the notification hook's most valuable case and it got nothing."""
    import threading
    import time

    from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    engine = _RecordingHooksEngine()
    store = ConsoleChatStore()
    store.ensure_session()  # the active/viewed session
    background = store.create_session(title="fleet", activate=False)
    # Detachment is the constructor's own default: neither bridge seam
    # (`set_pending_approval` / `park_pending_approval`) is wired, so
    # `_approval_view_is_detached()` is True -- the announce toast is the
    # only surfacing this round could ever get (the app double below has
    # no `notify`, so the best-effort announce silently skips).
    controller = ConsoleChatController(
        store=store,
        provider_gateway=object(),
        ensure_run_hooks=lambda: engine,
    )
    controller.app = _InlineCallFromThreadApp()
    controller.mcp_approval_timeout_seconds = lambda: 30.0
    pending = MCPPendingCall(
        llm_name="local:__local__:fs_write",
        server_key="local:__local__",
        tool_name="fs_write",
        server_label="Local tools",
        arguments={"path": "notes.txt"},
        reason="ask",
    )

    def _resolve_soon() -> None:
        # No card mounts and nothing parks (detachment), so the round id
        # is read from the registered-rounds map itself -- under the same
        # lock the controller documents for exactly this cross-thread
        # read (the F2b guard comment on `_approval_state_lock`).
        round_id = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with controller._approval_state_lock:
                registered = list(controller._pending_approval_rounds)
            if registered:
                round_id = registered[0]
                break
            time.sleep(0.01)
        assert round_id is not None, "round never registered"
        controller.resolve_pending_approval(
            {"local:__local__:fs_write": "approve_once"}, round_id=round_id
        )

    threading.Thread(target=_resolve_soon).start()
    decisions = controller.request_mcp_approvals(
        [pending], session_id=background.id
    )

    assert decisions == {"local:__local__:fs_write": "approve_once"}
    # Exactly ONE notify for the round even though the detached branch
    # also runs: the registration-time fire is skipped (`is_parked`), so
    # the detached branch's fire is the round's only one.
    assert [event for event, _ in engine.notifications] == ["ApprovalRequested"]
    _event, kwargs = engine.notifications[0]
    assert kwargs["session_id"] == background.id
    # Review fix R29: args summary present on the detached path too -- the
    # detached branch builds the same per-call entry as the registration fire.
    assert kwargs["data"] == {
        "calls": [
            {
                "name": "local:__local__:fs_write",
                "args_summary": '{"path": "notes.txt"}',
            }
        ],
        "session_active": False,
    }
