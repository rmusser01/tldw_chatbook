"""Complete redacted pages and stale delivery must not change another request."""

import asyncio
import json
from dataclasses import replace

import pytest

from Tests.UI.test_approval_interaction import view
from tldw_chatbook.UI.Console_Modules import approval_details as details

from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_mcp_approval import _CardHarnessApp

pytestmark = pytest.mark.bootstrap_profile


class _DetailsCardHarness(_CardHarnessApp):
    CSS_PATH = [str(path) for path in APP_STYLESHEETS]


def test_paged_arguments_reconstruct_complete_redacted_capture():
    originals = [
        {
            "path": "資料/" + "長" * 300 + "😀.md",
            "content": "x" * 1048576,
            "password": "synthetic-secret",
            "paths": [f"target-{index}.md" for index in range(1000)],
        }
    ]
    pages = list(details.iter_redacted_details(originals))
    assert all(len(page.text) <= 4096 for page in pages)
    assert [page.index for page in pages] == list(range(len(pages)))
    assert all(page.has_more for page in pages[:-1])
    assert not pages[-1].has_more
    reconstructed = json.loads("".join(page.text for page in pages))
    assert reconstructed[0]["password"] == "***"
    assert reconstructed[0]["content"] == "x" * 1048576
    assert reconstructed[0]["path"] == originals[0]["path"]
    assert reconstructed[0]["paths"] == [f"target-{index}.md" for index in range(1000)]
    assert originals[0]["password"] == "synthetic-secret"


def harness(arguments=({"path": "first.md"},)):
    identity = [None]
    workers = []
    posted = []
    painted = []
    loading = []
    controller = details.ApprovalDetailsController(
        current_identity=lambda: identity[0],
        spawn_worker=workers.append,
        post_page=lambda current, page: posted.append((current, page)),
        paint_page=painted.append,
        paint_loading=lambda: loading.append(True),
    )
    row = replace(view().rows[0], argument_sets=arguments)
    return controller, row, identity, workers, posted, painted, loading


def test_collapsed_details_does_not_serialize_large_body():
    controller, row, identity, workers, posted, painted, loading = harness(
        ({"content": "x" * 1048576},)
    )
    assert not workers and not loading and not painted
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    assert len(workers) == 1 and loading == [True]
    assert not posted and not painted
    workers.pop()()
    assert controller.deliver_page(*posted.pop())
    assert len(painted[-1].text) <= 4096


def test_page_from_old_revision_is_ignored():
    controller, row, identity, workers, posted, painted, _ = harness()
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    workers.pop()()
    identity[0] = ("round", 2, 3, row.verdict_key)
    assert not controller.deliver_page(*posted.pop())
    assert not painted


def test_switching_rows_rejects_late_page():
    controller, row, identity, workers, posted, painted, _ = harness()
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    workers.pop()()
    old = posted.pop()
    second = replace(row, verdict_key="second", argument_sets=({"path": "second.md"},))
    identity[0] = ("round", 1, 3, "second")
    controller.open(second, round_id="round", revision=1, generation=3)
    assert not controller.deliver_page(*old)
    assert not painted


def test_previous_page_result_cannot_replace_newer_request():
    controller, row, identity, workers, posted, painted, _ = harness(
        ({"content": "x" * 20000},)
    )
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    workers.pop()()
    old = posted.pop()
    controller.request_page(1)
    assert not controller.deliver_page(*old)
    workers.pop()()
    assert controller.deliver_page(*posted.pop())
    assert painted[-1].index == 1


def test_close_cooperatively_cancels_work_and_rejects_delivery():
    controller, row, identity, workers, posted, painted, _ = harness()
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    controller.close()
    workers.pop()()
    assert not posted and not painted


def test_worker_never_reads_ui_identity():
    controller, row, identity, workers, posted, painted, _ = harness()
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    controller._current_identity = lambda: (_ for _ in ()).throw(
        AssertionError("UI identity on worker")
    )
    workers.pop()()
    assert posted and not painted


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 40), (170, 48)])
async def test_escape_does_not_resolve_round(size):
    from Tests.UI.test_console_mcp_approval import _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
    from textual.widgets import Button, TextArea

    app = _DetailsCardHarness()
    async with app.run_test(size=size) as pilot:
        card = app.query_one(ChatApprovalCard)
        call = _sample_calls()[0]
        row = replace(
            view().rows[0],
            verdict_key=call["llm_name"],
            call_count=1,
            argument_sets=(
                {"content": "synthetic-body-" * 1000, "password": "secret"},
            ),
        )
        captured = replace(view(), rows=(row,), call_count=1)
        card.set_batch(
            [call],
            timeout_seconds=120,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        deadline = card._deadline_at
        card.focus_first_decision()
        await pilot.pause()
        assert app.focused.has_class("approval-details-open")
        await pilot.press("enter")
        await pilot.pause()
        page = card.query_one("#approval-details-page", TextArea)
        assert "synthetic-body-" in page.text and len(page.text) <= 4096
        assert page.read_only
        body = card.query_one("#approval-batch-rows")
        assert body.content_region.contains_region(page.region)
        hit, _ = app.screen.get_widget_at(*page.region.center)
        assert hit is page
        assert "synthetic-body-" in app.export_screenshot()
        for button in card._batch_fast_buttons:
            assert card.content_region.contains_region(button.region)
            hit, _ = app.screen.get_widget_at(*button.region.center)
            assert hit is button
        import os
        from pathlib import Path

        if os.environ.get("TLDW_APPROVAL_DETAILS_FRAMES"):
            destination = Path(os.environ["TLDW_APPROVAL_DETAILS_FRAMES"])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"details-{size[0]}x{size[1]}.svg").write_text(
                app.export_screenshot(), encoding="utf-8"
            )
        assert not card.query_one("#approval-details-next", Button).disabled
        card.query_one("#approval-details-next", Button).press()
        await pilot.pause()
        assert "Page 2" in str(card.query_one("#approval-details-status").render())
        await pilot.press("escape")
        await pilot.pause()
        assert not card.query_one("#approval-details-panel").display
        assert app.focused.has_class("approval-details-open")
        assert card._deadline_at == deadline
        assert app.decided == []
        card.query_one(".approval-row-fast-approve", Button).press()
        await pilot.pause()
        assert app.decided == [{call["llm_name"]: "approve_once"}]


@pytest.mark.asyncio
async def test_raw_command_remains_primary():
    from Tests.UI.test_console_mcp_approval import _raw_shell_call
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
    from textual.widgets import TextArea

    from tldw_chatbook.Tools.raw_cli_executor import MAX_RAW_COMMAND_BYTES

    command = "echo " + "c" * (MAX_RAW_COMMAND_BYTES - 5)
    call = _raw_shell_call(command)
    row = replace(
        view(True).rows[0],
        verdict_key=call["call_id"],
        call_count=1,
        argument_sets=(call["arguments"],),
    )
    captured = replace(view(True), rows=(row,), call_count=1)
    app = _DetailsCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            [call],
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        primary = card.query_one(".approval-row-full-command", TextArea)
        assert primary.text == command and primary.display
        card.focus_first_decision()
        await pilot.pause()
        assert app.focused.has_class("approval-details-open")
        await pilot.press("enter")
        await pilot.pause()
        assert primary.text == command and primary.display
        assert card.query(".approval-row-raw-warning")
        assert app.decided == []


@pytest.mark.asyncio
async def test_replaced_card_rejects_old_details_gesture_and_page():
    from Tests.UI.test_console_mcp_approval import _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
    from textual.widgets import Button

    app = _DetailsCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        call = _sample_calls()[0]
        row = replace(
            view().rows[0],
            verdict_key=call["llm_name"],
            call_count=1,
            argument_sets=({"path": "old.md"},),
        )
        captured = replace(view(), rows=(row,), call_count=1)
        card.set_batch(
            [call],
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        opener = card.query_one(".approval-details-open", Button)
        opener.press()
        await pilot.pause()
        identity = card._details_identity
        opener.press()
        replacement = replace(
            captured,
            revision=2,
            rows=(replace(row, argument_sets=({"path": "new.md"},)),),
        )
        card.set_batch(
            [call],
            timeout_seconds=0,
            round_id="round",
            view=replacement,
            presentation_revision=2,
        )
        await pilot.pause()
        assert not card.query_one("#approval-details-panel").display
        assert not card._details.deliver_page(
            identity, details.ApprovalDetailsPage(0, "stale", False)
        )
        assert app.decided == []
        current = card.query_one(".approval-details-open", Button)
        assert current is not opener
        current.press()
        await pilot.pause()
        assert "new.md" in card.query_one("#approval-details-page").text


def test_prepared_page_cache_is_bounded_and_close_releases_it():
    controller, row, identity, workers, posted, painted, _ = harness(
        ({"content": "x" * 20000},)
    )
    identity[0] = ("round", 1, 3, row.verdict_key)
    controller.open(row, round_id="round", revision=1, generation=3)
    for index in range(3):
        if index:
            controller.request_page(index)
        workers.pop()()
        assert controller.deliver_page(*posted.pop())
    controller.request_page(1)
    assert not workers and painted[-1].index == 1
    controller.request_page(0)
    assert len(workers) == 1  # The evicted page is prepared again.
    controller.close()
    workers.pop()()
    assert not posted


def test_unserializable_capture_uses_fixed_error_without_exception_content():
    pages = list(details.iter_redacted_details(({"input": object()},)))
    assert pages[-1].text == "Arguments unavailable"
    assert not pages[-1].has_more


@pytest.mark.asyncio
async def test_details_binds_to_verdict_key_when_view_rows_are_reordered():
    from Tests.UI.test_console_mcp_approval import _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    calls = [_sample_calls()[0], _sample_calls()[2]]
    first = replace(
        view().rows[0],
        verdict_key=calls[0]["llm_name"],
        call_count=1,
        argument_sets=({"path": "first.md"},),
    )
    second = replace(
        first, verdict_key=calls[1]["llm_name"], argument_sets=({"path": "second.md"},)
    )
    captured = replace(view(), rows=(second, first), call_count=2)
    app = _DetailsCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            calls,
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        card.query(".approval-details-open").first().press()
        await pilot.pause()
        assert "first.md" in card.query_one("#approval-details-page").text
        assert "second.md" not in card.query_one("#approval-details-page").text


@pytest.mark.asyncio
async def test_actual_captured_card_skips_full_argument_formatter_on_mount_and_reuse(
    monkeypatch,
):
    import tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card as card_module
    from Tests.UI.test_approval_interaction import _owner_payload, _set_payload

    payload = _owner_payload(args={"path": "large.md", "content": "x" * 1048576})

    def unexpected_summary(_entry):
        raise AssertionError("Captured body formatted eagerly")

    monkeypatch.setattr(card_module, "_summarize_row_arguments", unexpected_summary)
    app = _DetailsCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(card_module.ChatApprovalCard)
        for round_id in ("review-round", "replacement"):
            _set_payload(card, payload, round_id=round_id)
            await pilot.pause()
            assert "large.md" in str(card.query_one(".approval-row-header").render())
            assert "x" * 400 not in app.export_screenshot()
            assert payload["view"].rows[0].argument_sets[0]["content"] == "x" * 1048576
            assert app.decided == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["next", "close"])
async def test_queued_details_navigation_cannot_change_another_row(action):
    from Tests.UI.test_console_mcp_approval import _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        ChatApprovalCard,
        ApprovalActionButton,
    )
    from textual.widgets import Button

    calls = [_sample_calls()[0], _sample_calls()[2]]
    first = replace(
        view().rows[0],
        verdict_key=calls[0]["llm_name"],
        call_count=1,
        argument_sets=({"content": "A" * 20000},),
    )
    second = replace(
        first, verdict_key=calls[1]["llm_name"], argument_sets=({"path": "B.md"},)
    )
    captured = replace(view(), rows=(first, second), call_count=2)
    app = _DetailsCardHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            calls,
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        openers = list(card.query(".approval-details-open"))
        card._open_details(openers[0])
        await pilot.pause()
        card._details.request_page(2)
        await pilot.pause()
        assert "Page 3" in str(card.query_one("#approval-details-status").render())
        event = ApprovalActionButton.Pressed(
            card.query_one(f"#approval-details-{action}", Button)
        )
        card._open_details(openers[1])
        card.on_button_pressed(event)
        await pilot.pause()
        assert card.query_one("#approval-details-panel").display
        assert "B.md" in card.query_one("#approval-details-page").text
        assert "Page 1" in str(card.query_one("#approval-details-status").render())
        assert app.decided == []


@pytest.mark.asyncio
async def test_mounted_grouped_targets_have_bounded_preview_and_complete_details():
    from textual.widgets import Button, Static
    from Tests.UI.test_console_mcp_approval import _sample_calls
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard

    targets = tuple(
        f"folder/{'long-segment/' * 8}distinct-{index:03d}.md" for index in range(300)
    )
    row = replace(
        view().rows[0],
        verdict_key="mcp__srv_a__search",
        call_count=len(targets),
        targets=targets,
        argument_sets=({"password": "synthetic-secret"},),
    )
    captured = replace(view(), rows=(row,), call_count=len(targets))
    app = _DetailsCardHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch(
            _sample_calls()[:1] * len(targets),
            timeout_seconds=0,
            round_id="round",
            view=captured,
            presentation_revision=1,
        )
        await pilot.pause()
        header = str(card.query_one(".approval-row-header", Static).content)
        assert len(header) < 700 and targets[-1] not in header
        assert targets[0] in header and "Details" in header and "more targets" in header
        await pilot.click(card.query_one(".approval-details-open", Button))
        await pilot.pause()
        pages = []
        for index in range(30):
            await asyncio.wait_for(app.workers.wait_for_complete(), timeout=5)
            assert card._details_page_index == index
            page = card._details._pages[index]
            assert card.query_one("#approval-details-page").text == page.text
            pages.append(page.text)
            assert len(page.text) <= 4096
            if not page.has_more:
                break
            card._details.request_page(index + 1)
        else:
            pytest.fail("Captured targets did not finish in bounded pages")
        reconstructed = "".join(pages)
        assert all(target in reconstructed for target in targets)
        assert "synthetic-secret" not in reconstructed and "***" in reconstructed
        assert not app.decided
