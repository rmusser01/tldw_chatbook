"""Tests for the local-tool review hook + provider composition (Task 5).

The hook tests mirror build_mcp_review_hook's discipline: clear-first
stamps, ONE approval round trip per batch, verdicts only ever "proceed".
"""

from types import SimpleNamespace

import pytest

import tldw_chatbook.Chat.console_chat_controller as controller_mod
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    build_combined_review_hook,
    build_local_review_hook,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

ASK = EffectiveToolState(state="ask", origin="global_default")
ALLOW = EffectiveToolState(state="allow", origin="tool_override")


def provider(state, tmp_path):
    return LocalToolProvider(workspace_root=tmp_path, resolve_state=lambda hub: state)


def test_hook_clears_stamps_before_gating(tmp_path):
    p = provider(ASK, tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    hook = build_local_review_hook(p, lambda pending: {})
    hook([])  # a turn with no calls still clears
    assert p._stamps == {}


def test_hook_gates_ask_calls_in_one_batch(tmp_path):
    p = provider(ASK, tmp_path)
    seen = []
    hook = build_local_review_hook(p, lambda pending: seen.append(pending) or {"fs_list": "approve_once"})
    verdicts = hook([ToolCall(name="fs_list", args={"path": "."}),
                     ToolCall(name="fs_list", args={"path": "sub"})])
    assert len(seen) == 1 and len(seen[0]) == 2  # ONE round trip for the batch
    assert verdicts == {"fs_list": "proceed"}
    assert p._stamps == {"fs_list": "approve_once"}


def test_hook_skips_non_ask_calls(tmp_path):
    p = provider(ALLOW, tmp_path)
    hook = build_local_review_hook(p, lambda pending: (_ for _ in ()).throw(AssertionError("must not ask")))
    assert hook([ToolCall(name="fs_list", args={"path": "."})]) == {}


def test_combined_hook_merges_verdicts(tmp_path):
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    hook = build_combined_review_hook([
        build_local_review_hook(p1, lambda pending: {"fs_list": "approve_once"}),
        build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
    ])
    # each provider only gates what it owns; both see the batch
    out = hook([ToolCall(name="fs_list", args={"path": "."})])
    assert out == {"fs_list": "proceed"}


def test_combined_hook_empty_list_is_noop():
    hook = build_combined_review_hook([])
    assert hook([ToolCall(name="fs_list", args={"path": "."})]) == {}


def test_combined_hook_clears_later_providers_when_earlier_hook_raises(tmp_path):
    """I3 across providers: a raising hook must not strand a LATER provider's
    stale prior-turn stamp for the fail-open runtime to hand to invoke()."""
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    p1.apply_batch_decisions({"fs_list": "approve_once"})  # stale, prior turn
    p2.apply_batch_decisions({"fs_list": "approve_once"})  # stale, prior turn

    def raising_approvals(pending):
        raise RuntimeError("mid-shutdown")

    hook = build_combined_review_hook([
        build_local_review_hook(p1, raising_approvals),
        build_local_review_hook(p2, raising_approvals),
    ])
    with pytest.raises(RuntimeError):
        hook([ToolCall(name="fs_list", args={"path": "."})])
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

    hook = build_combined_review_hook([
        build_local_review_hook(p1, raising_approvals),
        build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
    ])
    with pytest.raises(RuntimeError):
        hook([ToolCall(name="fs_list", args={"path": "."})])
    assert p1._stamps == {}  # cleared at entry, round trip raised
    assert p2._stamps == {"fs_list": "deny"}  # fresh THIS-turn decision


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
        self.recorded_decisions.append((server_key, tool_name, decision, initiator, error))


def _bare_controller(app):
    """A controller instance with only what _compose_local_provider touches."""
    controller = object.__new__(ConsoleChatController)
    controller.app = app
    controller._pending_approval_event = None
    controller._pending_approval_decisions = None
    return controller


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
    assert controller._compose_local_provider() == (None, None)


def test_compose_local_provider_no_service(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    controller = _bare_controller(SimpleNamespace())  # no unified_mcp_service
    assert controller._compose_local_provider() == (None, None)


def test_compose_local_provider_kill_switch_on(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    app = SimpleNamespace(unified_mcp_service=_FakeService(kill_switch=True))
    controller = _bare_controller(app)
    assert controller._compose_local_provider() == (None, None)


def test_compose_local_provider_kill_switch_read_failure_fails_closed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())

    class _RaisingService(_FakeService):
        def get_kill_switch(self):
            raise RuntimeError("store unavailable")

    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_RaisingService()))
    assert controller._compose_local_provider() == (None, None)


def test_compose_local_provider_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService()
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))

    local_provider, hook = controller._compose_local_provider()

    assert isinstance(local_provider, LocalToolProvider)
    assert local_provider._root == tmp_path.resolve()
    assert callable(hook)
    # resolve_state is the same payload source the MCP gate uses.
    gate = local_provider.pending_gate_for("fs_list", {"path": "."})
    assert gate is not None and gate.server_key == "local:__local__"


def test_compose_local_provider_empty_workspace_root_uses_cwd(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(controller_mod, "get_cli_setting", _console_settings())
    monkeypatch.chdir(tmp_path)
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = controller._compose_local_provider()

    assert local_provider._root == tmp_path.resolve()


def test_compose_local_provider_tilde_workspace_root_expands_home(
    monkeypatch, tmp_path
):
    """A configured ``~/repo`` must expand against HOME (PR #1352 review):
    without expanduser() the root would resolve to a literal "~" directory
    under the cwd."""
    home = tmp_path / "home"
    (home / "repo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root="~/repo"),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = controller._compose_local_provider()

    assert local_provider._root == (home / "repo").resolve()


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
    local_provider, _hook = controller._compose_local_provider()

    (tmp_path / "a.txt").write_text("a")

    local_provider.apply_batch_decisions({"fs_list": "approve_session"})
    assert local_provider.invoke("local:fs_list", {"path": "."}).ok
    assert ("local:__local__", "fs_list") in service.session_approvals

    local_provider.apply_batch_decisions({"fs_list": "always_allow"})
    assert local_provider.invoke("local:fs_list", {"path": "."}).ok
    assert service.persisted_states == [("local:__local__", "fs_list", "allow")]


def test_compose_local_provider_session_approval_skips_reprompt(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService()
    service.approve_for_session("local:__local__", "fs_list")
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    local_provider, _hook = controller._compose_local_provider()

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
    local_provider, _hook = controller._compose_local_provider()
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
    local_provider.apply_batch_decisions({"fs_list": "timeout"})

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



# -- todo_write session wiring (phase-3a Task 4) --------------------------------


def test_compose_local_provider_without_session_registers_no_todo_spec(
    monkeypatch, tmp_path
):
    """No session context -> provider stays context-free: no todo_write."""
    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=_FakeService()))

    local_provider, _hook = controller._compose_local_provider()

    r = local_provider.invoke("local:todo_write", {"todos": []})
    assert not r.ok
    assert r.error == "Unknown local tool: todo_write"


def test_compose_local_provider_wires_the_sessions_own_todo_list(
    monkeypatch, tmp_path
):
    """The composed provider mutates the session's live ``todos`` list in
    place and surfaces the change through the bridge's todo marker seam."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    monkeypatch.setattr(
        controller_mod,
        "get_cli_setting",
        _console_settings(workspace_root=str(tmp_path)),
    )
    service = _FakeService(state=ALLOW)
    controller = _bare_controller(SimpleNamespace(unified_mcp_service=service))
    controller.store = ConsoleChatStore()
    session = controller.store.create_session(workspace_id="ws")
    markers = []
    controller._agent_bridge = SimpleNamespace(
        append_todo_marker=lambda session_id, todos: markers.append(
            (session_id, list(todos))
        )
    )

    local_provider, _hook = controller._compose_local_provider(session_id=session.id)

    todos = [
        {"content": "ship it", "status": "in_progress", "activeForm": "shipping"},
        {"content": "celebrate", "status": "pending"},
    ]
    r = local_provider.invoke("local:todo_write", {"todos": todos})

    assert r.ok
    assert r.content == "2 todos (1 in progress)"
    assert session.todos == todos  # the session's own list, mutated in place
    assert markers == [(session.id, todos)]


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

    local_provider, _hook = controller._compose_local_provider(session_id="ghost")

    r = local_provider.invoke("local:todo_write", {"todos": []})
    assert not r.ok
    assert r.error == "Unknown local tool: todo_write"
