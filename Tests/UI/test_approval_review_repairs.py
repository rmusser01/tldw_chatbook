"""Real mount, teardown and navigation regressions from the approval review."""

import asyncio
import threading
from dataclasses import replace
from typing import ClassVar

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.Agents import test_mcp_tool_provider as mcp_helpers
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_approval_interaction import view
from Tests.UI.test_console_mcp_approval import _CardHarnessApp, _sample_calls
from tldw_chatbook.UI.Console_Modules import approval_details
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

running_loop = mcp_helpers.running_loop

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("commit", ["fast", "bulk", "staged"])
async def test_commits_wait_for_actual_request_mount(monkeypatch, commit):
    app = _CardHarnessApp()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        rows = card.query_one("#approval-batch-rows")
        mount = rows.mount
        pending = []
        ready = asyncio.Event()

        class HeldMount:
            def __await__(self):
                return ready.wait().__await__()

        def hold(*widgets, **kwargs):
            pending.append((widgets, kwargs))
            return HeldMount()

        monkeypatch.setattr(rows, "mount", hold)
        calls = _sample_calls()[:1] if commit == "fast" else _sample_calls()
        card.set_batch(calls, timeout_seconds=0, round_id="held")
        try:
            await pilot.pause()
            assert not rows.query(".approval-row-header")
            button = card.query_one(
                ".approval-row-fast-approve"
                if commit == "fast"
                else "#approval-submit"
                if commit == "staged"
                else "#approval-deny-all",
                Button,
            )
            assert button.disabled, "decision active before request facts mounted"
            if commit == "fast":
                card._submit_fast_decision("approve_once")
            elif commit == "bulk":
                card._commit_bulk("deny")
            else:
                card._submit_batch_decisions()
            await pilot.pause()
            assert app.decided == []
        finally:
            for widgets, kwargs in pending:
                await mount(*widgets, **kwargs)
            ready.set()
            for _ in range(10):
                await pilot.pause()
                if card._information_ready:
                    break
        assert card._information_ready
        assert rows.query_one(".approval-row-header", Static).region.width > 0
        assert not button.disabled
        if commit == "staged":
            card._set_options_open(True)
        button.press()
        await pilot.pause()
        assert len(app.decided) == 1


@pytest.mark.asyncio
async def test_actual_unmount_cancels_held_details_and_rejects_queued_page(monkeypatch):
    app = _CardHarnessApp()
    started = threading.Event()
    release = threading.Event()
    original = approval_details.iter_redacted_details

    def held(*args, **kwargs):
        started.set()
        assert release.wait(5)
        yield from original(*args, **kwargs)

    monkeypatch.setattr(approval_details, "iter_redacted_details", held)
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        call = _sample_calls()[0]
        captured = replace(
            view(),
            rows=(
                replace(
                    view().rows[0],
                    verdict_key=call["llm_name"],
                    call_count=1,
                    argument_sets=({"content": "x" * 10000},),
                ),
            ),
            call_count=1,
        )
        card.set_batch(
            [call],
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        card._open_details(card.query_one(".approval-details-open", Button))
        assert await asyncio.to_thread(started.wait, 2)
        cancel = card._details._cancel
        identity = card._details_identity
        try:
            await card.remove()
            assert cancel.is_set(), (
                "Textual worker cancellation leaves preparation alive"
            )
            assert card._current_details_identity() is None
            assert not card._details._arguments and not card._details._targets
            assert not card._details._pages and card._details_opener is None
            assert not card._details.deliver_page(
                identity, approval_details.ApprovalDetailsPage(0, "late", False)
            )
        finally:
            release.set()
            await pilot.pause()


@pytest.mark.asyncio
async def test_advisory_reason_names_its_untrusted_author():
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        call = dict(_sample_calls()[0], rationale="Check this file")
        card.set_batch([call], timeout_seconds=0, round_id="reason")
        await pilot.pause()
        assert "Agent stated reason" in str(
            next(
                widget
                for widget in card.query(Static)
                if (widget.id or "").startswith("approval-context-")
            ).content
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed_raw", [False, True])
async def test_shipping_cards_restore_parked_round_after_actual_recreation(
    tmp_path, running_loop, mixed_raw
):
    from Tests.Chat.test_console_approval_scope_journeys import owner_journey
    from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
    from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards

    class TaskApp(_CardHarnessApp):
        CSS_PATH: ClassVar = [str(path) for path in APP_STYLESHEETS]

        def compose(self):
            yield ChatTaskCards()

    controller, session, provider, _service, transport = await asyncio.to_thread(
        owner_journey, tmp_path, running_loop
    )
    tool = provider.list_catalog()[0].id
    pending = [provider.pending_gate_for(tool, {"query": "x" * 18000}, "first")]
    raw_provider = None
    if mixed_raw:
        from Tests.Chat.test_console_raw_shell_approval import _provider
        from tldw_chatbook.Agents.agent_models import ToolCall

        raw_provider = _provider(tmp_path)
        raw = raw_provider.pending_gate_for(
            ToolCall("shell_exec", {"command": "printf first\nprintf second"}, "raw")
        )
        assert raw is not None
        pending.append(raw)
    received = threading.Event()
    controller.set_pending_approval = lambda payload: (
        received.set() if payload else None
    )
    controller.mcp_approval_timeout_seconds = lambda: 60
    waiting = asyncio.create_task(
        asyncio.to_thread(
            controller.request_mcp_approvals, pending, session_id=session.id
        )
    )
    assert await asyncio.to_thread(received.wait, 3)
    payload = controller._head_round_payload(
        controller._parked_approval_payloads, session.id
    )
    assert payload and payload["view"].session_id == session.id
    app = TaskApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            cards = app.query_one(ChatTaskCards)
            state = TaskResumeState(pending_approval=payload)
            cards.sync_state(state)
            await pilot.pause()
            card = cards.query_one(ChatApprovalCard)
            await pilot.click(card.query_one(".approval-more-options", Button))
            choice = card._batch_selects[0]
            choice.focus()
            await pilot.press("enter", "down", "enter")
            await pilot.pause()
            assert choice.value == "approve_session"
            if mixed_raw:
                raw_choice = card._batch_selects[1]
                raw_choice.focus()
                await pilot.press("enter", "enter")
                await pilot.pause()
                assert raw_choice.value == "deny"
                assert card._draft.can_apply()
                assert (
                    "\n" in card.query_one(".approval-row-full-command", TextArea).text
                )
            card._open_details(card.query_one(".approval-details-open", Button))
            await pilot.pause()
            card._details.request_page(1)
            await pilot.pause()
            assert card._details_page_index == 1
            page = card.query_one("#approval-details-page", TextArea)
            page.scroll_to(y=3, animate=False)
            await pilot.pause()
            scroll = (page.scroll_x, page.scroll_y)
            rows = card.query_one("#approval-batch-rows")
            rows.scroll_to(y=2, animate=False)
            await pilot.pause()
            rows_scroll = (rows.scroll_x, rows.scroll_y)
            if mixed_raw:
                assert rows_scroll[1] > 0
            old_button = card.query_one("#approval-submit", Button)
            assert card._presentation_state is payload["presentation_state"]
            # Navigation clears the mounted view while the real round remains parked.
            cards.sync_state(TaskResumeState())
            await pilot.pause()
            await cards.remove()
            assert (
                payload["presentation_state"]
                .get("draft", {})
                .get("choices", {})
                .get("first")
                == "approve_session"
            ), payload["presentation_state"]

            def primitive(value):
                return (
                    value is None
                    or isinstance(value, (str, bool, int, float))
                    or isinstance(value, (tuple, list))
                    and all(primitive(item) for item in value)
                    or isinstance(value, dict)
                    and all(
                        isinstance(key, str) and primitive(item)
                        for key, item in value.items()
                    )
                )

            assert primitive(payload["presentation_state"])
            assert "x" * 100 not in repr(payload["presentation_state"])
            new_cards = ChatTaskCards()
            await app.mount(new_cards)
            restored_payload = controller._head_round_payload(
                controller._parked_approval_payloads, session.id
            )
            new_cards.sync_state(TaskResumeState(pending_approval=restored_payload))
            await pilot.pause()
            restored = new_cards.query_one(ChatApprovalCard)
            for _ in range(10):
                if restored._information_ready and restored._details_page_index == 1:
                    break
                await pilot.pause()
            assert restored is not card
            assert restored._information_ready, "layout readiness callback has not run"
            assert restored._batch_selects[0].value == "approve_session"
            assert restored._options_open
            assert (
                restored._details_identity is not None
                and restored._details_page_index == 1
            )
            restored_page = restored.query_one("#approval-details-page", TextArea)
            await pilot.pause()
            assert (restored_page.scroll_x, restored_page.scroll_y) == scroll
            restored_rows = restored.query_one("#approval-batch-rows")
            assert (restored_rows.scroll_x, restored_rows.scroll_y) == rows_scroll
            if mixed_raw:
                assert restored._batch_selects[1].value == "deny"
                assert restored._draft.can_apply() and 1 in restored._raw_reviewed
                assert raw_provider._runtime._executor.execute.call_count == 0
            old_button.press()
            await pilot.pause()
            assert app.decided == [] and not transport.execute_calls
    finally:
        denied = {row.call_id: "deny" for row in pending}
        controller.resolve_pending_approval(denied, round_id=payload["round_id"])
        assert dict(await waiting) == denied
        assert payload["presentation_state"] == {"retired": True}
        assert (
            controller._head_round_payload(
                controller._parked_approval_payloads, session.id
            )
            is None
        )
        controller.store.persistence.db.close()


@pytest.mark.asyncio
async def test_old_layout_callback_cannot_unlock_reused_row(monkeypatch):
    app = _CardHarnessApp()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        queued = []
        monkeypatch.setattr(
            card,
            "call_after_refresh",
            lambda callback, *args, **kwargs: queued.append(
                lambda: callback(*args, **kwargs)
            ),
        )
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="old")
        await pilot.pause()
        assert queued
        old_ready = queued.pop()
        card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id="new")
        await pilot.pause()
        old_ready()
        assert not card._information_ready
        assert all(button.disabled for button in card._batch_fast_buttons)
        assert queued
        queued.pop()()
        assert card._information_ready
        assert not card.query_one(".approval-row-fast-approve", Button).disabled
        assert app.decided == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "replacement", ["session_id", "run_id", "round_id", "revision"]
)
async def test_saved_draft_never_crosses_captured_identity(replacement):
    app = _CardHarnessApp()
    captured = replace(
        view(),
        rows=(
            replace(
                view().rows[0], verdict_key=_sample_calls()[0]["llm_name"], call_count=1
            ),
        ),
        call_count=1,
    )
    saved = {
        "identity": (
            captured.session_id,
            captured.run_id,
            captured.round_id,
            captured.revision,
        ),
        "draft": {
            "choices": {captured.rows[0].verdict_key: "approve_session"},
            "reviewed": [],
        },
        "options_open": True,
    }
    replacement_view = replace(
        captured, **{replacement: 2 if replacement == "revision" else "different"}
    )
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            _sample_calls()[:1],
            timeout_seconds=0,
            round_id=replacement_view.round_id,
            view=replacement_view,
            presentation_revision=replacement_view.revision,
            presentation_state=saved,
        )
        for _ in range(10):
            await pilot.pause()
            if card._information_ready:
                break
        assert card._information_ready
        assert card._batch_selects[0].value == "approve_once"
        assert not card._options_open
        assert saved == {} and not app.decided


@pytest.mark.asyncio
async def test_rapid_replacement_does_not_orphan_layout_coroutines():
    import inspect

    app = _CardHarnessApp()
    owned = set()
    async with app.run_test() as pilot:
        card = app.query_one(ChatApprovalCard)
        for index in range(3):
            card.set_batch(_sample_calls()[:1], timeout_seconds=0, round_id=str(index))
            owned.update(
                worker for worker in card.workers if worker.group == "approval-layout"
            )
        await pilot.pause()
    unclosed = []
    for worker in owned:
        work = worker._work
        if inspect.iscoroutine(work):
            state = inspect.getcoroutinestate(work)
            if state != inspect.CORO_CLOSED:
                unclosed.append(state)
                work.close()
    assert not unclosed, (
        f"Retired layout workers retained unclosed coroutines: {unclosed}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["\n", "\u202e"])
async def test_admitted_raw_directory_metadata_is_literal_but_command_multiline(
    tmp_path, control
):
    from copy import deepcopy

    from Tests.Chat.test_console_raw_shell_approval import _provider
    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.tool_catalog import ToolExecutionPolicy
    from tldw_chatbook.Chat.approval_presentation import _build_approval_payload

    directory = tmp_path / ("directory" + control + "Timeout: FORGED")
    directory.mkdir()
    provider = _provider(tmp_path)
    command = "printf first\nprintf second"
    arguments = {
        "command": command,
        "initial_directory": str(directory),
        "shell": "bash",
    }
    original = deepcopy(arguments)
    pending = provider.pending_gate_for(ToolCall("shell_exec", arguments, "raw"))
    assert pending is not None and pending.full_command == command
    assert pending.arguments["initial_directory"] == str(directory)
    assert pending.options == ("approve_once", "approve_session", "deny")
    payload = _build_approval_payload(
        "round",
        "chat",
        "run",
        [pending],
        0,
        None,
        read_global_ToolExecutionPolicy=lambda: ToolExecutionPolicy,
    )
    app = _CardHarnessApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            card = app.query_one(ChatApprovalCard)
            card.set_batch(
                payload["calls"],
                timeout_seconds=0,
                round_id="round",
                view=payload["view"],
                presentation_revision=1,
            )
            await pilot.pause()
            assert (
                card.query_one(".approval-row-full-command", TextArea).text == command
            )
            metadata = next(
                widget
                for widget in card.query(Static)
                if str(widget.content).startswith("Shell:")
            )
            text = str(metadata.content)
            assert arguments == original and pending.arguments["command"] == command
            assert provider._runtime._executor.execute.call_count == 0
            assert len(text.splitlines()) == 3
            assert (
                control not in text.split("Directory: ", 1)[1].split("\nTimeout:", 1)[0]
            )
            assert control.encode("unicode_escape").decode("ascii") in text
            assert arguments == original and pending.arguments["command"] == command
            assert provider._runtime._executor.execute.call_count == 0
    finally:
        provider._runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transition",
    [
        "current",
        "reopen",
        "row",
        "page",
        "round",
        "revision",
        "settled",
        "retired",
        "unmount",
    ],
)
async def test_queued_details_scroll_stays_with_exact_live_page(
    monkeypatch, transition
):
    from Tests.UI.test_approval_details import _DetailsCardHarness

    app = _DetailsCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        calls = _sample_calls()
        captured = replace(
            view(),
            rows=tuple(
                replace(
                    view().rows[0],
                    verdict_key=call["llm_name"],
                    call_count=1,
                    argument_sets=(
                        {f"field-{i}": "reading line" for i in range(1000)},
                    ),
                )
                for call in calls
            ),
            call_count=len(calls),
        )
        state = {}

        def set_batch(round_id="round", revision=1):
            card.set_batch(
                calls,
                timeout_seconds=0,
                round_id=round_id,
                view=captured,
                presentation_revision=revision,
                presentation_state=state,
            )

        set_batch()
        await pilot.pause()
        card._open_details(card.query(".approval-details-open").first(Button))
        await pilot.pause()
        area = card.query_one("#approval-details-page", TextArea)
        assert "field-0" in area.text and area.max_scroll_y > 5
        assert card._current_details_identity() is not None
        held = []
        scroll_attempts = []
        real_scroll_to = area.scroll_to
        if transition == "settled":

            def scroll_to(*args, **kwargs):
                scroll_attempts.append((args, kwargs))
                return real_scroll_to(*args, **kwargs)

            monkeypatch.setattr(area, "scroll_to", scroll_to)
        after_refresh = area.call_after_refresh
        monkeypatch.setattr(
            area,
            "call_after_refresh",
            lambda *args, **kwargs: held.append((args, kwargs)),
        )
        card._restore_details_scroll = (0, 5)
        # Real current cached delivery queues the restore.
        card._details.request_page(0)
        callback_args, callback_kwargs = held[-1]
        monkeypatch.setattr(area, "call_after_refresh", after_refresh)

        if transition == "reopen":
            card._close_details()
            card._open_details(card.query(".approval-details-open").first(Button))
        elif transition == "row":
            card._open_details(list(card.query(".approval-details-open"))[1])
        elif transition == "page":
            card._details.request_page(1)
        elif transition in {"round", "revision"}:
            set_batch("replacement" if transition == "round" else "round", 2)
            await pilot.pause()
            card._open_details(card.query(".approval-details-open").first(Button))
        elif transition == "settled":
            card._commit_bulk("deny")
            assert card._batch_submitted
        elif transition == "retired":
            state["retired"] = True
        elif transition == "unmount":
            await card.remove()
        await pilot.pause()
        if transition != "unmount":
            assert area is card.query_one("#approval-details-page", TextArea)
            if transition != "settled":
                assert area.text and area.max_scroll_y > 5
        real_scroll_to(y=1, animate=False, immediate=True)
        before = area.scroll_y
        assert before == (0 if transition == "settled" else 1)
        scroll_attempts.clear()
        callback_args[0](*callback_args[1:], **callback_kwargs)
        assert area.scroll_y == (5 if transition == "current" else before), (
            "queued restore crossed its Details page lifetime"
        )
        if transition == "settled":
            assert app.decided and not area.text
            assert not scroll_attempts, "settled disclosure received a stale scroll"
