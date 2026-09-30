# Tests/Guardian/test_dispatch_seam.py
"""Dispatch-seam integration: the Guardian checker rides prompt_queue.dispatch.

Ordering contract (ADR-204 contract 4): the checker runs at the very head of
``ConsolePromptQueueUIController.dispatch`` -- before the existing
blocked-reason refusal gate, and therefore before ADR-148 ``UserPromptSubmit``
hooks, which fire far downstream inside
``Chat.console_chat_controller._submit_draft_body`` (reached only via
``_stage_normal_chain`` -> ``_launch_chain`` -> the controller submit chain).
A Guardian ``block`` returned at the seam provably short-circuits the send
before any user hook can run; an ``allow`` still counts the typed intent
ahead of the hooks. ``launch_chain`` in these fakes stands in for that
hook-emitting chain.
"""
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_prompt_queue import ConsolePromptQueueRegistry
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian import settings as guardian_settings
from tldw_chatbook.Guardian.check_pipeline import GuardianChecker
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptDispatchStatus,
    ConsolePromptQueueUIController,
)

FIXED_NOW = "2026-09-29T12:00:00+00:00"


class _FakeChatController:
    def __init__(self) -> None:
        self.prompt_queue_registry = ConsolePromptQueueRegistry()
        self.store = SimpleNamespace(
            active_session_id="session-a",
            conversation_context_epoch=lambda _session_id: 1,
        )

    def activity_for(self, session_id: str) -> ConsoleControllerActivity:
        return ConsoleControllerActivity(
            session_id=session_id,
            occupies_slot=False,
            preparing_before_acceptance=False,
            accepted_live_turn=False,
            needs_approval=False,
            queued_count=0,
            queue_paused=False,
            terminal_notification_eligible=False,
        )

    def queue_prompt(self, *_args, **_kwargs):  # pragma: no cover - unused
        raise AssertionError("no queue admission happens in these cases")

    def send_refusal_copy(self, _session_id: str) -> str:
        return ""


class _SpyChecker:
    """Minimal checker double recording calls and returning a fixed result."""

    def __init__(self, result: dict | None = None, order: list | None = None,
                 calls: list | None = None) -> None:
        self.result = result or {"action": "allow", "redacted_draft": None,
                                 "notice": None, "recorded": False}
        self.order = order if order is not None else []
        self.calls = calls if calls is not None else []

    async def check(self, draft: str) -> dict:
        self.calls.append(draft)
        if self.order is not None:
            self.order.append("checker")
        return self.result


def _calls() -> dict[str, Any]:
    return {
        "system": [],
        "sync": [],
        "notified": [],
        "focused": [],
        "staged": [],
        "queued": [],
        "committed": [],
        "follow": [],
    }


def _seam_controller(
    checker_getter=None,
    calls=None,
    *,
    blocked_reason: str = "",
    order: list | None = None,
) -> ConsolePromptQueueUIController:
    calls = calls if calls is not None else _calls()

    def blocked() -> str:
        if order is not None:
            order.append("blocked_reason")
        return blocked_reason

    async def append_system(text: str) -> None:
        calls["system"].append(text)

    async def sync_ui() -> None:
        calls["sync"].append(True)

    return ConsolePromptQueueUIController(
        chat_controller_accessor=lambda: _FakeChatController(),
        capture_configuration=lambda _session_id: None,
        ensure_active_session=lambda: None,
        blocked_reason_accessor=blocked,
        setup_blocked_reason_accessor=lambda: "",
        append_system_message=append_system,
        notify=lambda text, severity: calls["notified"].append((text, severity)),
        focus_composer=lambda: calls["focused"].append(True),
        note_follow_intent=lambda: calls["follow"].append(True),
        launch_chain=lambda draft, session_id: (
            calls["staged"].append((draft, session_id)) or "turn-a"
        ),
        commit_captured_draft=lambda session_id, stash: calls["committed"].append(
            (session_id, stash)
        ),
        commit_queued_draft=lambda session_id, stash: calls["queued"].append(
            (session_id, stash)
        ),
        turn_recovery_ids=lambda _session_id: (),
        restore_turn_recovery=lambda _turn_id: None,
        discard_turn_recovery=lambda _turn_id: False,
        load_recovered_turn=lambda _session_id: None,
        edit_refusal=lambda _text: "",
        sync_ui=sync_ui,
        guardian_checker_getter=checker_getter,
    )


async def test_guardian_block_routes_through_refusal_plumbing(tmp_path,
                                                              monkeypatch):
    """A real checker + real store: block refuses via the existing plumbing."""
    monkeypatch.setattr(
        guardian_settings,
        "guardian_setting",
        lambda key, default=None: True if key == "enabled" else default,
    )
    db = GuardianDB(tmp_path / "guardian.sqlite", "test-client")
    try:
        for row in db.list_rules():
            db.delete_rule(row["id"])
        rule_id = db.upsert_rule(
            name="held topic", topic="held", pattern="forbidden tangent",
            action="block", notification_frequency="every_message",
        )
        rows: list[str] = []

        async def append_system_row(text: str) -> None:
            rows.append(text)

        def notify(text: str, severity: str) -> None:
            rows.append(f"notify:{severity}:{text}")

        checker = GuardianChecker(
            db_getter=lambda: db,
            session_id_getter=lambda: "session-a",
            notify=notify,
            append_system_row=append_system_row,
            now=lambda: FIXED_NOW,
        )
        calls = _calls()
        controller = _seam_controller(lambda: checker, calls)

        outcome = await controller.dispatch("a forbidden tangent appears")

        assert outcome.status is ConsolePromptDispatchStatus.REFUSED
        assert "held topic" in outcome.detail
        assert calls["focused"] == [True], "the composer is refocused"
        assert calls["staged"] == [], "the chain (and its hooks) never launch"
        assert any("held topic" in row for row in calls["system"]), (
            "the block notice rides the existing system-message refusal row"
        )
        with db.connection() as conn:
            hits = conn.execute(
                "SELECT COUNT(*) FROM guardian_alerts WHERE rule_id = ?",
                (rule_id,),
            ).fetchone()[0]
        assert hits == 1, "the blocked intent still records exactly one alert"
    finally:
        db.close()


async def run_block_short_circuit_case() -> None:
    """Shared case for test_check_pipeline: block never reaches the hooks."""
    calls = _calls()
    order: list[str] = []
    spy = _SpyChecker(
        result={
            "action": "block",
            "redacted_draft": None,
            "notice": {
                "text": "Guardian held this message (rule: held).",
                "rule_name": "held",
                "topic": "held",
                "severity": "warning",
                "is_crisis": False,
                "span_text": "held",
            },
            "recorded": True,
        },
        order=order,
    )
    controller = _seam_controller(lambda: spy, calls)

    outcome = await controller.dispatch("please hold this")

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["staged"] == [], (
        "launch_chain stands in for the hook-emitting send chain "
        "(console_chat_controller._submit_draft_body); a Guardian block "
        "must short-circuit before it"
    )
    assert order == ["checker"], "no blocked-reason read is needed after block"
    assert calls["focused"] == [True]


async def test_guardian_allow_passes_draft_through_unchanged():
    calls = _calls()
    spy = _SpyChecker()
    controller = _seam_controller(lambda: spy, calls)

    outcome = await controller.dispatch("an ordinary message")

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("an ordinary message", "session-a")]
    assert spy.calls == ["an ordinary message"]
    assert calls["focused"] == []


async def test_guardian_redact_rewrites_draft_before_the_chain():
    calls = _calls()
    spy = _SpyChecker(
        result={
            "action": "redact",
            "redacted_draft": "a scrubbed message",
            "notice": None,
            "recorded": True,
        }
    )
    controller = _seam_controller(lambda: spy, calls)

    outcome = await controller.dispatch("a message with a secret")

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("a scrubbed message", "session-a")], (
        "the redacted draft is what the send chain receives"
    )


async def test_disabled_getter_passthrough_adds_no_checker_call():
    calls = _calls()
    getter_calls = {"n": 0}

    def getter():
        getter_calls["n"] += 1
        return None  # ADR-204 contract 1: None while [guardian] is disabled

    controller = _seam_controller(getter, calls)

    outcome = await controller.dispatch("an ordinary message")

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("an ordinary message", "session-a")]
    assert getter_calls["n"] == 1
    assert calls["system"] == [] and calls["notified"] == []


async def test_no_getter_injected_leaves_dispatch_untouched():
    calls = _calls()
    controller = _seam_controller(None, calls)

    outcome = await controller.dispatch("an ordinary message")

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("an ordinary message", "session-a")]


async def test_raising_checker_fails_open_at_the_seam():
    """Even a checker whose own fail-open is broken cannot eat a send."""
    calls = _calls()

    class _ExplodingChecker:
        async def check(self, draft: str) -> dict:  # pragma: no cover
            raise RuntimeError("checker exploded")

    controller = _seam_controller(lambda: _ExplodingChecker(), calls)

    outcome = await controller.dispatch("an ordinary message")

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("an ordinary message", "session-a")]


async def test_checker_runs_before_blocked_reason_gate_and_counts_intent():
    """Seam ordering pin: checker first, blocked-reason gate second.

    The draft is refused by the blocked-reason gate, yet the checker still
    observed the typed intent -- and because dispatch refused, the send
    chain (where UserPromptSubmit hooks fire) never ran at all.
    """
    calls = _calls()
    order: list[str] = []
    spy = _SpyChecker(order=order)
    controller = _seam_controller(
        lambda: spy,
        calls,
        blocked_reason="Console send blocked: setup incomplete",
        order=order,
    )

    outcome = await controller.dispatch("a message while blocked")

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert "setup incomplete" in outcome.detail
    assert order == ["checker", "blocked_reason"], (
        "the Guardian check precedes the blocked-reason refusal gate"
    )
    assert spy.calls == ["a message while blocked"], (
        "typed intent counts even when the send is later refused"
    )
    assert calls["staged"] == []
