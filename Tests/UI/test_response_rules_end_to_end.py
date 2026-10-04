"""Integrated public journeys; scripted transport proves wiring, not model quality."""

import asyncio

import pytest
from textual.widgets import Button

from Tests.UI.response_rules_fixtures import mounted_rules_console
from tldw_chatbook.Widgets.Console.response_rules_modal import ResponseRulesModal

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]


async def wait_for(case, predicate):
    try:
        async with asyncio.timeout(30):
            while not predicate():
                await case.pilot.pause(0.02)
    except TimeoutError:
        pytest.fail(
            f"Console state: {case.controller.run_state}; rules: {case.rules.state(case.session_id).phase}; primary calls: {len(case.requests)}; helper calls: {len(case.transport.requests)}; learning: {case.learning_results}; draft: {case.composer.draft_text()!r}; dispatched: {[(r.accepted, r.visible_copy) for r in case.dispatches]}; failures: {case.failures}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
async def test_learn_repair_reopen_and_exclude_through_console(tmp_path, size):
    async with mounted_rules_console(tmp_path, size) as case:
        case.reply = "Missing proof"
        case.composer.load_draft("Explain result")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        await wait_for(
            case,
            lambda: bool(case.requests)
            and not case.controller.run_state.is_stop_allowed
            and not case.controller._submit_tasks_for_session(case.session_id),
        )
        original = case.chats.get_message(case.chats.active_leaf(case.session_id))
        assert original.content == "Missing proof" and original.status == "complete"
        calls = len(case.requests)
        case.reply = "Evidence: repaired existing answer"
        case.composer.load_draft("/omfg The answer omitted evidence")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        await wait_for(
            case,
            lambda: case.rules.state(case.session_id).learning is not None
            and case.rules.state(case.session_id).phase == "idle"
            and case.composer.draft_text() == "",
        )
        learned = case.rules.state(case.session_id).learning
        assert learned.state == "active", learned.reason
        assert (
            len(case.requests) == calls + 1
        ), f"Dispatch: {[(r.accepted, r.visible_copy) for r in case.dispatches]}; failures: {case.failures}"
        assert case.chats.get_message(original.id).content == original.content
        assert case.rules.state(case.session_id).assessment.outcome == "pass"
        assert case.composer.draft_text() == ""
        helper_calls = len(case.transport.requests)
        repairs = len(case.requests)
        saved = next(s for s in case.chats.sessions() if s.id == case.session_id)
        conversation_id = saved.persisted_conversation_id
        original_persisted_id = original.persisted_message_id
        case.chats.close_session(case.session_id)
        assert await case.console._workspace._resume_console_workspace_conversation(
            conversation_id, preserve_persisted_scope=True
        )
        case.session_id = case.chats.active_session_id
        await case.pilot.pause()
        assert len(case.transport.requests) == helper_calls
        assert len(case.requests) == repairs
        assert len(case.rules.effective_rules(case.session_id)) == 1
        originals = [
            m
            for m in case.chats.read_only_messages_for_session(case.session_id)
            if m.persisted_message_id == original_persisted_id
        ]
        assert len(originals) == 1 and originals[0].content == "Missing proof"
        case.composer.load_draft("/rules")
        await case.pilot.pause()
        assert await case.pilot.click("#console-send-message")
        await wait_for(case, lambda: isinstance(case.host.screen, ResponseRulesModal))
        modal = case.host.screen
        assert modal.scope == case.rules.scopes(case.session_id)[0]
        assert await case.pilot.click("#rr-exclude")
        await wait_for(
            case,
            lambda: not case.rules.effective_rules(case.session_id) and not modal._busy,
        )
        assert len(case.transport.requests) == helper_calls
        assert len(case.requests) == repairs
        assert originals[0].content == original.content
