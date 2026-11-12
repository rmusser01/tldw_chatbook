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
                await asyncio.sleep(0.02)
        await case.pilot.pause()
    except TimeoutError:
        pytest.fail(
            f"Console state: {case.controller.run_state}; rules: {case.rules.state(case.session_id).phase}; primary calls: {len(case.requests)}; helper calls: {len(case.transport.requests)}; learning: {case.learning_results}; draft: {case.composer.draft_text()!r}; dispatched: {[(r.accepted, r.visible_copy) for r in case.dispatches]}; failures: {case.failures}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 35)])
@pytest.mark.parametrize("review_check", ["history", "ancestry"])
async def test_learn_repair_reopen_and_exclude_through_console(
    tmp_path, size, review_check
):
    async with mounted_rules_console(tmp_path, size) as case:
        from tldw_chatbook.Chat.prompt_history import PromptHistory

        history = PromptHistory(tmp_path / "history.jsonl")
        case.controller.prompt_history = history
        case.reply = "Missing proof"
        case.composer.load_draft("Explain result")
        await case.pilot.pause()
        await wait_for(
            case,
            lambda: (
                not case.console.query_one("#console-send-message", Button).disabled
                and not case.console.query_one(
                    "#console-send-message", Button
                ).has_class("-active")
            ),
        )
        assert await case.pilot.click("#console-send-message")
        await wait_for(
            case,
            lambda: (
                bool(case.requests)
                and not case.controller.run_state.is_stop_allowed
                and not case.controller._submit_tasks_for_session(case.session_id)
            ),
        )
        original = case.chats.get_message(case.chats.active_leaf(case.session_id))
        assert original.content == "Missing proof" and original.status == "complete"
        calls = len(case.requests)
        case.reply = "Evidence: repaired existing answer"
        case.composer.load_draft("/omfg The answer omitted evidence")
        await case.pilot.pause()
        await wait_for(
            case,
            lambda: (
                not case.console.query_one("#console-send-message", Button).disabled
                and not case.console.query_one(
                    "#console-send-message", Button
                ).has_class("-active")
            ),
        )
        assert await case.pilot.click("#console-send-message")
        await wait_for(
            case,
            lambda: (
                case.rules.state(case.session_id).learning is not None
                and case.rules.state(case.session_id).phase == "idle"
                and case.composer.draft_text() == ""
            ),
        )
        learned = case.rules.state(case.session_id).learning
        assert learned.state == "active", learned.reason
        assert len(case.requests) == calls + 1, (
            f"Dispatch: {[(r.accepted, r.visible_copy) for r in case.dispatches]}; failures: {case.failures}; repairs: {case.repairs}; learning: {case.learning_results}; state: {case.rules.state(case.session_id)}"
        )
        assert case.chats.get_message(original.id).content == original.content
        assert case.rules.state(case.session_id).assessment.outcome == "pass", (
            case.repairs,
            case.learning_results,
            case.rules.state(case.session_id),
        )
        assert case.composer.draft_text() == ""
        if review_check == "history":
            assert history.size == 1
            assert (await history.get_entry(-1))["input"] == "Explain result"
        helper_calls = len(case.transport.requests)
        repairs = len(case.requests)
        saved = next(s for s in case.chats.sessions() if s.id == case.session_id)
        conversation_id = saved.persisted_conversation_id
        original_persisted_id = original.persisted_message_id
        # Retire the live composer and runtime through the same close owner as the UI.
        await case.console._session._close_console_session_tab(case.session_id)
        assert all(s.id != case.session_id for s in case.chats.sessions())
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
        if review_check == "ancestry":
            from Tests.Chat.response_rules_fixtures import source
            from tldw_chatbook.Chat.response_rules.evidence import capture_rule_input

            answer = case.chats.get_message(case.chats.active_leaf(case.session_id))
            origin = source(
                profile_id=case.rules.profile_id,
                session_id=case.session_id,
                conversation_id=conversation_id,
                message_id=answer.persisted_message_id,
                branch_id=answer.id,
                message_version=case.chats.response_rule_source_version(answer.id),
            )
            captured = capture_rule_input(case.chats, origin)
            assert captured.request_text == "Explain result"
        case.composer.load_draft("/rules")
        await case.pilot.pause()
        await wait_for(
            case,
            lambda: (
                not case.console.query_one("#console-send-message", Button).disabled
                and not case.console.query_one(
                    "#console-send-message", Button
                ).has_class("-active")
            ),
        )
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


@pytest.mark.asyncio
async def test_close_during_resume_retires_the_old_visible_draft(tmp_path, monkeypatch):
    from Tests.UI.response_rules_fixtures import seed_answer
    from tldw_chatbook.UI.Console_Modules import composer_run_controls, workspace

    async with mounted_rules_console(tmp_path, (120, 35)) as case:
        await wait_for(
            case,
            lambda: (
                case.console._console_attach_sync_complete
                and case.console._console_attach_reconciled
                and not case.console._console_attach_reconcile_running
                and not case.console._console_sync_in_progress
                and not case.console._console_sync_requested
                and composer_run_controls.poll_source_is_current(
                    case.console, case.console._console_completed_full_source
                )
            ),
        )
        answer = seed_answer(case)
        old_session_id = case.session_id
        conversation_id = next(
            session.persisted_conversation_id
            for session in case.chats.sessions()
            if session.id == old_session_id
        )
        case.composer.load_draft("authored before closing")
        case.composer.load_draft("")
        await case.pilot.pause()
        entered, release = asyncio.Event(), asyncio.Event()
        load_tree = workspace.load_console_conversation_tree

        async def held_original_tree(*args, **kwargs):
            entered.set()
            await release.wait()
            return await load_tree(*args, **kwargs)

        monkeypatch.setattr(
            workspace, "load_console_conversation_tree", held_original_tree
        )
        resume = asyncio.create_task(
            case.console._workspace._resume_console_workspace_conversation(
                conversation_id,
                reuse_existing=False,
                preserve_persisted_scope=True,
            )
        )
        try:
            await wait_for(case, entered.is_set)
            snapshot = case.console._session._console_draft_switch_snapshot
            assert snapshot is not None and snapshot[0] == old_session_id
            assert case.console._session._console_visible_draft_revision is not None
            await case.console._session._close_console_session_tab(old_session_id)
            assert all(s.id != old_session_id for s in case.chats.sessions())
        finally:
            release.set()
            resumed = await resume
        assert resumed is True
        case.session_id = case.chats.active_session_id
        assert case.session_id != old_session_id
        assert case.console._console_visible_draft_session_id == case.session_id
        assert case.composer.draft_text() == ""
        assert any(
            message.persisted_message_id == answer.persisted_message_id
            and message.content == "Missing proof"
            for message in case.chats.read_only_messages_for_session(case.session_id)
        )
        case.composer.load_draft("/rules")
        await case.pilot.pause()
        await wait_for(
            case,
            lambda: (
                not case.console.query_one("#console-send-message", Button).disabled
                and not case.console.query_one(
                    "#console-send-message", Button
                ).has_class("-active")
            ),
        )
        assert await case.pilot.click("#console-send-message")
        await wait_for(case, lambda: isinstance(case.host.screen, ResponseRulesModal))
        assert case.host.screen.scope == case.rules.scopes(case.session_id)[0]
        assert case.requests == [] and case.transport.requests == []
        assert case.host._exception is None
