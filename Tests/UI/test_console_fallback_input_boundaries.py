"""Fallback pressed inputs retain original runtime and wired-entry fences."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as wired
from Tests.Chat import test_console_received_custody as custody
from Tests.Chat.test_console_first_send_atomicity import _until
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
catalog_store = wired.catalog_store
local_root = wired.local_root
mcp_sources = wired.mcp_sources
snapshot_case = wired.snapshot_case
wired_case = wired.wired_case
wired_composer_case = wired.wired_composer_case


@pytest.mark.parametrize("image_only", [False, True], ids=["text", "image_only"])
async def test_fallback_request_preserves_pressed_revision_through_cancel(image_only):
    runtime, store, session, request = custody._case()
    try:
        draft = "" if image_only else request.draft
        store.set_session_draft(session.id, draft)
        attachment = custody._attachment("fallback-image") if image_only else None
        if attachment is not None:
            assert store.add_pending_attachment(session.id, attachment)
        captured = store.session_input_snapshot(session.id)
        request = replace(
            request,
            draft=draft,
            attachment_ids=(attachment.attachment_id,) if attachment else (),
            _pressed_inputs=captured,
            _pressed_stash=None,
        )
        store.set_session_draft(session.id, "newer draft", authored_token=(12, 2))
        assert not store.session_inputs_are_current(captured)
        assert store.session_inputs_are_current(captured, include_draft=False)

        turn_id = runtime.accept_turn(request)
        record = runtime._turn_custody[turn_id]
        claim = store.received_turn_for_session(session.id)
        assert record.request is request
        assert record.received_claim is claim
        assert claim.draft_revision == captured.draft_revision
        assert record.task is not None and not record.task.done()
        if attachment is not None:
            assert record.inputs.attachments == (attachment,)
            assert record.inputs.attachments[0] is attachment
            assert store.pending_attachments(session.id) == []

        # The original lazy driver has not stepped: cancellation must release
        # that exact claim without accepting text or consuming the newer draft.
        await custody._cancel_custody(runtime)
        assert store.received_turn_for_session(session.id) is None
        assert not runtime.has_custodied_turns(session.id)
        assert not store.messages_for_session(session.id)
        assert store.session_draft(session.id) == "newer draft"
        recovery = runtime.recoveries_for_session(session.id)
        assert len(recovery) == 1
        assert recovery[0].draft == draft
        if attachment is not None:
            assert recovery[0].attachments == (attachment,)
            assert recovery[0].attachments[0] is attachment
    finally:
        await custody._retire(runtime)


async def test_same_id_stale_fallback_snapshot_refuses_before_composer_mirror(
    wired_composer_case,
):
    case = wired_composer_case
    runtime = case.screen._console_runtime()
    try:
        case.composer.load_draft("captured draft")
        case.store.set_session_draft(case.session.id, "captured draft")
        stash = case.composer.capture_draft_for_send()
        captured = case.store.session_input_snapshot(case.session.id)
        assert case.store.session_inputs_are_current(captured)
        queue = case.screen._prompt_queue
        adapters = [
            row
            for row in case.screen._console_received_live_adapters
            if row[0] is queue
        ]
        assert adapters and all(
            getattr(owner, name) is original for owner, name, original in adapters
        )
        case.store.replace_session_settings(
            case.session.id,
            ConsoleSessionSettings(provider="deepseek", model="changed-model"),
        )
        assert captured.session_id == case.session.id
        assert not case.store.session_inputs_are_current(captured, include_draft=False)
        case.composer.load_draft("unmirrored newer draft")
        before = case.store.session_input_snapshot(case.session.id)
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await queue._launch_chain_async(
                stash.text,
                case.session.id,
                stash,
                case.controller,
                _captured_inputs=captured,
            )
        after = case.store.session_input_snapshot(case.session.id)
        assert after.draft == before.draft == "captured draft"
        assert after.draft_revision == before.draft_revision
        assert case.composer.draft_text() == "unmirrored newer draft"
        assert not runtime.has_custodied_turns()
        assert not case.store.messages_for_session(case.session.id)
    finally:
        await runtime.dispose(timeout_seconds=3)


@pytest.mark.parametrize("callback", ["_launch_chain_async", "_commit_queued_draft"])
async def test_late_fallback_adapter_drift_refuses_before_admission(
    wired_composer_case, monkeypatch, callback
):
    case = wired_composer_case
    runtime = case.screen._console_runtime()
    wired._loop_projection(case)
    draft = "captured fallback input"
    case.composer.load_draft(draft)
    case.store.set_session_draft(case.session.id, draft)
    stash = case.composer.capture_draft_for_send()
    captured = case.store.session_input_snapshot(case.session.id)
    queue = case.screen._prompt_queue
    entered, release = asyncio.Event(), asyncio.Event()
    original_capture = case.controller.capture_turn_configuration_snapshot
    replaced_calls = []

    async def held_capture(*args, **kwargs):
        result = await original_capture(*args, **kwargs)
        entered.set()
        await release.wait()
        return result

    def replacement(*args, **kwargs):
        replaced_calls.append((args, kwargs))
        pytest.fail("Changed fallback adapter must not be invoked")

    monkeypatch.setattr(
        case.controller, "capture_turn_configuration_snapshot", held_capture
    )
    task = asyncio.create_task(
        queue._launch_chain_async(
            draft, case.session.id, stash, case.controller, _captured_inputs=captured
        )
    )
    try:
        assert await _until(lambda: entered.is_set() or task.done(), timeout=5)
        if task.done():
            task.result()
        assert entered.is_set() and not task.done()
        assert not runtime.has_custodied_turns()
        assert case.store.session_inputs_are_current(captured)
        monkeypatch.setattr(queue, callback, replacement)
        release.set()
        with pytest.raises(RecoveryRequired, match="console_snapshot_owner_changed"):
            await task
        assert replaced_calls == []
        assert not runtime.has_custodied_turns()
        assert not case.store.messages_for_session(case.session.id)
        assert case.store.session_draft(case.session.id) == draft
        assert case.composer.draft_text() == draft
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.dispose(timeout_seconds=3)


async def test_screen_fallback_preserves_replaced_class_dispatch_abi(
    wired_composer_case, monkeypatch
):
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptDispatchResult,
        ConsolePromptDispatchStatus,
        ConsolePromptQueueUIController,
    )

    case = wired_composer_case
    screen = case.screen
    runtime = screen._console_runtime()
    queue = screen._prompt_queue
    draft = "legacy class dispatch"
    case.composer.load_draft(draft)
    case.store.set_session_draft(case.session.id, draft)
    stash = case.composer.capture_draft_for_send()
    captured = case.store.session_input_snapshot(case.session.id)
    calls = []

    async def legacy_dispatch(self, draft, *, session_id=None, stash=None):
        calls.append((self, draft, session_id, stash))
        return ConsolePromptDispatchResult(
            ConsolePromptDispatchStatus.REFUSED, session_id=session_id
        )

    async def pass_hook_review(draft, *, session_id, stash, dispatch):
        return await dispatch()

    # A custom hook-session callback selects the documented legacy route.
    # The transparent review adapter isolates the real Screen dispatch closure:
    # this is an ABI control, not evidence about hook review or native custody.
    monkeypatch.setattr(screen._hooks, "_session", lambda: case.session.id)
    monkeypatch.setattr(screen._hooks, "dispatch", pass_hook_review)
    monkeypatch.setattr(ConsolePromptQueueUIController, "dispatch", legacy_dispatch)
    queue_adapters = [
        row for row in screen._console_received_live_adapters if row[0] is queue
    ]
    try:
        assert type(queue) is ConsolePromptQueueUIController
        assert queue_adapters and all(
            getattr(owner, name) is original for owner, name, original in queue_adapters
        )
        assert case.store.session_inputs_are_current(captured)
        assert not await screen._dispatch_console_draft_send(
            draft, session_id=case.session.id, stash=stash, _captured_inputs=captured
        )
        assert len(calls) == 1
        assert calls[0][:3] == (queue, draft, case.session.id)
        assert calls[0][3] is stash
        assert case.store.session_inputs_are_current(captured)
        assert case.composer.draft_text() == draft
        assert not runtime.has_custodied_turns()
        assert not case.store.messages_for_session(case.session.id)
    finally:
        await runtime.dispose(timeout_seconds=3)
