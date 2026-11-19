"""TASK-41643 delta review: a ConsoleChatController subclass never gets the pressed queue admission (ADR-356 rule 4)."""

from dataclasses import replace

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as wired

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = wired.catalog_store
local_root = wired.local_root
mcp_sources = wired.mcp_sources
snapshot_case = wired.snapshot_case
wired_case = wired.wired_case
wired_composer_case = wired.wired_composer_case


@pytest.mark.parametrize("mode", ["subclass", "instance_override"])
async def test_pressed_queue_admission_refuses_subclass_and_instance_override(wired_composer_case, monkeypatch, mode):
    from types import MethodType

    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    case = wired_composer_case
    screen = case.screen
    runtime = screen._console_runtime()
    queue = screen._prompt_queue
    controller = case.controller
    calls = []

    async def foreign_queue_prompt(self, session_id, **kwargs):
        calls.append(sorted(kwargs))
        raise RuntimeError("probe: foreign queue_prompt ran")

    def live_activity(self, session_id):
        return replace(
            ConsoleChatController.activity_for(self, session_id),
            accepted_live_turn=True,
            preparing_before_acceptance=False,
        )

    if mode == "subclass":
        sub = type(
            "SubController",
            (ConsoleChatController,),
            {"queue_prompt": foreign_queue_prompt, "activity_for": live_activity},
        )
        monkeypatch.setattr(controller, "__class__", sub)
    else:
        monkeypatch.setattr(controller, "activity_for", MethodType(live_activity, controller))
        monkeypatch.setattr(
            controller, "queue_prompt", MethodType(foreign_queue_prompt, controller)
        )

    async def capture(session_id, controller_arg=None, *a, **k):
        return object()

    monkeypatch.setattr(queue, "_capture_configuration_for_dispatch", capture)
    monkeypatch.setattr(queue, "_blocked_reason_accessor", lambda: "")
    monkeypatch.setattr(queue, "_notify", lambda *a, **k: None)
    draft = "probe draft"
    case.composer.load_draft(draft)
    case.store.set_session_draft(case.session.id, draft)
    stash = case.composer.capture_draft_for_send()
    captured = case.store.session_input_snapshot(case.session.id)
    try:
        result = await queue.dispatch(
            draft, session_id=case.session.id, stash=stash, _captured_inputs=captured
        )
        assert not calls, f"{mode}: foreign queue_prompt received the pressed admission {calls}"
    finally:
        monkeypatch.undo()
        await runtime.dispose(timeout_seconds=3)
