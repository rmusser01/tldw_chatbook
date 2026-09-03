"""HITL confirm + bridge wiring for merge_agent_worktree/discard_agent_worktree.

TASK-28238 phase 2 Task 6. Covers two halves:

1. The controller-side worker-thread <-> UI-thread bridge
   (``ConsoleChatController.request_worktree_merge_confirm`` /
   ``resolve_pending_worktree_merge``) -- clones ``request_skill_script_
   confirm``'s round machinery (see ``Tests/Chat/test_console_skill_script_
   confirm.py``) but with a single-key ``{"allow": bool}`` decision instead
   of that method's two-part one: worktree merge/discard has no "remember"
   concept.
2. The bridge wiring: ``console_agent_bridge.run_reply`` forwards
   ``request_worktree_merge_confirm`` straight to ``AgentService.run_turn``
   -- unlike ``run_skill_script``, the merge/discard tool CLOSURES already
   live inside ``AgentService`` itself (TASK-28238 phase 2 Task 5), so there
   is nothing for the bridge to build; it only has to pass the callable on.
   ``Tests/Agents/test_fleet_runtime.py`` already exercises a deny decision
   propagating through those real closures as a refusal ``ToolResult``
   (Task 5's own suite), so this file does not repeat that end-to-end path
   -- it pins the bridge's OWN forwarding responsibility in isolation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import pytest

import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge_module
from tldw_chatbook.Agents.agent_service import AgentService as _RealAgentService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from Tests.console_provider_doubles import persisted_console_store


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout}s")


class _FakeApp:
    """`call_from_thread` stand-in: invokes the callback immediately."""

    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


@pytest.fixture
def make_controller() -> Callable[[], ConsoleChatController]:
    """Factory fixture: a fresh controller with a fake UI wired for the
    worktree-merge confirm, mirroring ``test_console_skill_script_confirm.
    make_controller``.
    """
    made: list[ConsoleChatController] = []

    def _make() -> ConsoleChatController:
        store = persisted_console_store()
        controller = ConsoleChatController(store=store, provider_gateway=object())
        controller.app = _FakeApp()
        controller.pending_worktree_merge_payloads = []
        controller.set_pending_worktree_merge = (
            controller.pending_worktree_merge_payloads.append
        )
        made.append(controller)
        return controller

    yield _make

    for controller in made:
        # ADR-067's no-deadline default means a round left armed at test
        # end would hang its non-daemon worker thread forever otherwise --
        # `begin_shutdown()` sets the signal the round's poll loop checks.
        controller.begin_shutdown()


# -- Controller round machinery -------------------------------------------


def test_no_ui_bridge_denies_immediately(make_controller):
    """Headless must fail closed at once, not block for the full timeout."""
    controller = make_controller()
    controller.app = None
    controller.set_pending_worktree_merge = None
    decision = controller.request_worktree_merge_confirm({"handle_id": "h1"})
    assert decision == {"allow": False}


def test_allow_round_trip(make_controller):
    controller = make_controller()
    result: dict[str, Any] = {}

    def worker():
        result["decision"] = controller.request_worktree_merge_confirm(
            {"handle_id": "h1", "mode": "apply", "branch": "agent/h1"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: bool(controller.pending_worktree_merge_ids()))
    controller.resolve_pending_worktree_merge(
        True, request_id=controller.pending_worktree_merge_ids()[0]
    )
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True}


def test_deny_round_trip(make_controller):
    controller = make_controller()
    result: dict[str, Any] = {}

    def worker():
        result["decision"] = controller.request_worktree_merge_confirm(
            {"handle_id": "h1", "action": "discard", "branch": "agent/h1"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: bool(controller.pending_worktree_merge_ids()))
    controller.resolve_pending_worktree_merge(
        False, request_id=controller.pending_worktree_merge_ids()[0]
    )
    thread.join(timeout=5)
    assert result["decision"] == {"allow": False}


def test_confirm_timeout_denies(make_controller):
    """DENY is the default action: an unresolved round times out closed."""
    controller = make_controller()
    controller.worktree_merge_confirm_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    decision = controller.request_worktree_merge_confirm({"handle_id": "h1"})
    elapsed = time.monotonic() - started
    assert decision == {"allow": False}
    assert elapsed < 2.5


def test_confirm_payload_carries_handle_mode_branch_diffstat_and_request_id(
    make_controller,
):
    """The payload actually marshaled to the UI sink must carry the fields
    the confirm card renders (handle_id, mode, branch, diffstat) plus the
    round's own request_id/timeout -- a card built from an under-described
    payload is a security defect, since this is exactly what the human
    approves on."""
    controller = make_controller()
    controller.worktree_merge_confirm_timeout_seconds = lambda: 45.0
    result: dict[str, Any] = {}

    def worker():
        result["decision"] = controller.request_worktree_merge_confirm(
            {
                "handle_id": "h1",
                "mode": "merge",
                "branch": "agent/h1",
                "worktree": "/tmp/wt",
                "diffstat": " a.txt | 1 +",
            }
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: bool(controller.pending_worktree_merge_ids()))
    shown = controller.pending_worktree_merge_payloads[0]
    assert shown is not None
    assert shown["handle_id"] == "h1"
    assert shown["mode"] == "merge"
    assert shown["branch"] == "agent/h1"
    assert shown["diffstat"] == " a.txt | 1 +"
    assert shown["timeout_seconds"] == 45.0
    assert shown["request_id"] == controller.pending_worktree_merge_ids()[0]
    assert shown["request_id"]  # non-empty
    controller.resolve_pending_worktree_merge(True, request_id=shown["request_id"])
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True}
    # The clearing call at teardown hands `None`, not a second payload.
    assert controller.pending_worktree_merge_payloads[-1] is None


def test_decision_shape_matches_what_agent_service_reads(make_controller):
    """Contract pin: `merge_agent_worktree_tool`/`discard_agent_worktree_
    tool` (agent_service.py) read the decision via `decision.get("allow",
    False)` and nothing else -- the returned dict's keys must be exactly
    `{"allow"}`, not the two-part `{"allow", "remember"}` shape
    `request_skill_script_confirm` returns. This is the reconciliation
    Task 5's `ponytail:` comment (now removed) asked Task 6 to settle."""
    controller = make_controller()
    result: dict[str, Any] = {}

    def worker():
        result["decision"] = controller.request_worktree_merge_confirm(
            {"handle_id": "h1"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: bool(controller.pending_worktree_merge_ids()))
    controller.resolve_pending_worktree_merge(
        True, request_id=controller.pending_worktree_merge_ids()[0]
    )
    thread.join(timeout=5)
    assert set(result["decision"].keys()) == {"allow"}


def test_stale_request_id_is_dropped(make_controller):
    """Security-critical: a resolve carrying a PRIOR round's id must not
    authorize a DIFFERENT, still-armed round -- mirrors
    `resolve_pending_skill_script`'s identical guard."""
    controller = make_controller()
    result: dict[str, Any] = {}

    def worker():
        result["decision"] = controller.request_worktree_merge_confirm(
            {"handle_id": "h1"}
        )

    thread = threading.Thread(target=worker)
    thread.start()
    _wait_until(lambda: bool(controller.pending_worktree_merge_ids()))
    controller.resolve_pending_worktree_merge(True, request_id="not-the-real-id")
    # Round is still armed -- the mismatched id was silently dropped.
    assert controller.pending_worktree_merge_ids()
    controller.resolve_pending_worktree_merge(
        True, request_id=controller.pending_worktree_merge_ids()[0]
    )
    thread.join(timeout=5)
    assert result["decision"] == {"allow": True}


# -- Bridge wiring: run_reply -> AgentService.run_turn ---------------------


def _capture_run_turn_kwargs(
    tmp_path,
    monkeypatch,
    *,
    request_worktree_merge_confirm: Callable[[dict], dict] | None,
) -> dict[str, Any]:
    """Build a real `ConsoleAgentBridge` and run one plain-text turn to
    capture the exact kwargs `run_reply` hands to `AgentService.run_turn`,
    by intercepting the method on a subclass -- exercises the REAL
    forwarding code, never a reimplementation of it. Mirrors
    `test_console_skill_script_confirm._capture_run_skill_script_tool`'s
    `AgentService(...)`-interception idea, applied to the per-call method
    instead of the constructor (this kwarg is a `run_turn` param, not one
    `AgentService.__init__` takes).
    """

    class _ChunkGateway:
        async def stream_chat(self, resolution, messages, tools=None, **kwargs):
            yield "ok"

    captured: dict[str, Any] = {}
    real_agent_service = console_agent_bridge_module.AgentService

    class _CapturingAgentService(real_agent_service):
        def run_turn(self, *args, **kwargs):
            captured.update(kwargs)
            return super().run_turn(*args, **kwargs)

    monkeypatch.setattr(
        console_agent_bridge_module, "AgentService", _CapturingAgentService
    )

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = persisted_console_store()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway()
    )
    kwargs: dict[str, Any] = dict(
        conversation_id="conv-worktree-confirm",
        session_id=session.id,
        resolution=object(),
        assistant_message_id=assistant.id,
        model="test-model",
        session_system_prompt="",
        agent_messages=[{"role": "user", "content": "hi"}],
        should_cancel=lambda: False,
    )
    if request_worktree_merge_confirm is not None:
        kwargs["request_worktree_merge_confirm"] = request_worktree_merge_confirm
    bridge.run_reply(**kwargs)

    assert real_agent_service is _RealAgentService  # sanity: patched the real class
    return captured


def test_bridge_forwards_the_confirm_kwarg_to_run_turn(tmp_path, monkeypatch):
    def confirm(payload: dict[str, Any]) -> dict[str, bool]:
        return {"allow": True}

    captured = _capture_run_turn_kwargs(
        tmp_path, monkeypatch, request_worktree_merge_confirm=confirm
    )
    assert captured.get("request_worktree_merge_confirm") is confirm


def test_bridge_forwards_none_when_omitted(tmp_path, monkeypatch):
    """`AgentService.run_turn` defaults this kwarg to None -- a caller that
    never passes it (e.g. no UI wired) must not accidentally forward some
    stale prior callable."""
    captured = _capture_run_turn_kwargs(
        tmp_path, monkeypatch, request_worktree_merge_confirm=None
    )
    assert captured.get("request_worktree_merge_confirm") is None
