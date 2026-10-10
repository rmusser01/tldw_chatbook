"""TASK-33625.2: denial controls remain painted with Inspect open."""

from __future__ import annotations

import gc
import time
import warnings

import pytest
from textual.widgets import Button, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import (
    _build_test_app,
    attach_chachanotes_db,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_console_native_chat_flow import (
    _ASYNC_SETTLE_TIMEOUT,
    _POLL_INTERVAL_SECONDS,
    _configure_native_ready_console,
)
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


class ProductionConsoleHarness(ConsoleHarness):
    """Use shipping Console startup sheets and inherited widget/modal defaults."""

    CSS_PATH = TldwCli.CSS_PATH


async def _wait_for_reconciled_console(console, pilot):
    """Finish original startup before publishing a fixture-only decision card."""
    runtime = console._console_runtime()
    generation = console._console_runtime_attachment_generation

    def ready():
        return (
            console._console_attach_reconciled
            and not console._console_attach_reconcile_running
            and runtime.view is console
            and runtime._attached_generation == generation
            and runtime._reconciled_view is console
            and runtime.has_answerable_view()
        )

    deadline = time.monotonic() + _ASYNC_SETTLE_TIMEOUT
    while not ready() and time.monotonic() < deadline:
        await pilot.pause(_POLL_INTERVAL_SECONDS)
    assert ready(), "Console initial reconciliation did not finish"
    await pilot.pause()
    assert ready(), "Console initial attachment changed before the approval fixture"


def _pending_card(console, round_id="compact-round"):
    console.set_task_resume_state(
        TaskResumeState(
            pending_approval={
                "round_id": round_id,
                "timeout_seconds": 0,
                "calls": [
                    {
                        "llm_name": "fs_read",
                        "call_id": round_id,
                        "server_label": "Local workspace, web, and Watchlists",
                        "effects": ["private_read"],
                        "tool_name": "fs_read",
                        "arguments": {"path": "notes.md"},
                        "reason": "ask",
                    }
                ],
            }
        )
    )


def _assert_painted(host, card, button):
    region = button.region
    assert region.width >= len(button.label.plain), (button.id, region)
    assert region.height > 0, button.id
    assert card.content_region.contains_region(region), (
        button.id,
        region,
        card.content_region,
    )
    assert host.screen.region.contains_region(region), (button.id, region)
    hit, _ = host.screen.get_widget_at(*region.center)
    assert hit is button, (
        button.id,
        region,
        card.content_region,
        button.parent.region,
        hit,
    )


async def _verify_every_approval_action_is_painted_and_focusable(request):
    """Check every geometry on the real screen without repeat app startup.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_reconciled_console(console, pilot)
        for size, inspect_open in (
            ((80, 24), True),
            ((90, 30), True),
            ((100, 30), True),
            ((80, 24), False),
            ((235, 52), True),
        ):
            await pilot.resize_terminal(*size)
            console._set_console_rail_preference(
                left_open=False, right_open=inspect_open
            )
            _pending_card(console, f"geometry-{size[0]}-{size[1]}-{inspect_open}")
            await pilot.pause(0.3)
            card = console.query_one(ChatApprovalCard)
            select = card.query_one(".approval-row-decision", Select)
            assert card.content_region.contains_region(select.region), select.region
            buttons = list(card.query(Button))
            assert len(buttons) == 5
            for button in buttons:
                _assert_painted(host, card, button)
            deny = card.query_one(".approval-row-fast-deny", Button)
            approve = card.query_one(".approval-row-fast-approve", Button)
            assert deny.region.y <= approve.region.y
            card.focus_first_decision()
            focused = set()
            for _ in range(8):
                await pilot.press("tab")
                if host.focused in buttons:
                    _assert_painted(host, card, host.focused)
                    focused.add(host.focused.id)
            assert {button.id for button in buttons} <= focused


async def _verify_reflow_preserves_decision_and_reused_round_controls(request):
    """Preserve choices, focus and control identity across resize and round reuse.

    Args:
        request: Pytest request used by the private-profile test wrapper.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=(80, 24)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_reconciled_console(console, pilot)
        console._set_console_rail_preference(left_open=False, right_open=True)
        _pending_card(console)
        await pilot.pause(0.3)
        card = console.query_one(ChatApprovalCard)
        select = card.query_one(".approval-row-decision", Select)
        select.value = "deny"
        await pilot.pause(0.3)
        for button in card.query(Button):
            _assert_painted(host, card, button)
        select.focus()
        await pilot.resize_terminal(235, 52)
        await pilot.pause(0.3)
        assert not card.has_class("approval-compact")
        assert card.query_one(Select) is select
        assert select.has_focus and select.value == "deny"
        assert [
            child.id for child in card.query_one("#approval-batch-actions").children
        ] == [
            "approval-approve-all",
            "approval-submit",
            "approval-deny-all",
        ]
        for button in card.query(Button):
            _assert_painted(host, card, button)
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.3)
        assert select.has_focus and select.value == "deny"
        _pending_card(console, "next-round")
        await pilot.pause(0.3)
        assert card._batch_round_id == "next-round"
        assert card.query_one(Select) is not select
        for button in card.query(Button):
            _assert_painted(host, card, button)


async def _verify_height_only_resize_reflows_existing_controls(request):
    """Short terminals must retain painted actions even with Inspect closed.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ProductionConsoleHarness(app)
    async with host.run_test(size=(80, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await _wait_for_reconciled_console(console, pilot)
        console._set_console_rail_preference(left_open=False, right_open=False)
        _pending_card(console)
        await pilot.pause(0.3)
        card = console.query_one(ChatApprovalCard)
        select = card.query_one(Select)
        select.value = "deny"
        select.focus()
        assert not card.has_class("approval-compact")
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.3)
        assert card.has_class("approval-compact")
        assert card.query_one(Select) is select and select.has_focus
        for button in card.query(Button):
            _assert_painted(host, card, button)
        await pilot.resize_terminal(80, 40)
        await pilot.pause(0.3)
        assert not card.has_class("approval-compact")
        assert card.query_one(Select) is select and select.value == "deny"


@pytest.mark.asyncio
@private_profile_test
async def test_compact_approval_layout_journeys(request: pytest.FixtureRequest) -> None:
    """Run each compact approval journey with independent app ownership.

    Args:
        request: Pytest request selecting the isolated private-profile child.
    """
    for verify in (
        _verify_every_approval_action_is_painted_and_focusable,
        _verify_reflow_preserves_decision_and_reused_round_controls,
        _verify_height_only_resize_reflows_existing_controls,
    ):
        try:
            await verify(request)
        finally:
            drain_active_service_patches()
            drain_created_dirs()
            gc.unfreeze()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                gc.collect()
