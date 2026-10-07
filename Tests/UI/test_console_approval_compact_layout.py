"""TASK-33625.2: denial controls remain painted with Inspect open."""

from __future__ import annotations

import gc
import warnings
import os
from itertools import product
from pathlib import Path

import pytest
from textual.widgets import Button, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import (
    _build_test_app,
    attach_chachanotes_db,
    drain_active_service_patches,
    drain_created_dirs,
)
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
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


def _pending_card(
    console, round_id="compact-round", *, captured=False, captured_variant="warning"
):
    if captured:
        from tldw_chatbook.Agents.mcp_tool_provider import MCPPendingCall
        from tldw_chatbook.Chat.approval_presentation import profile_authority
        from tldw_chatbook.Chat.console_chat_controller import _build_approval_payload

        secret = "sk-test-" + "A" * 32
        prefix = "/" + "directory/" * 40
        arguments = (
            {
                "api_key": "private-key-value",
                "note": secret,
                "nested": {"password": "nested-secret"},
            }
            if captured_variant == "secrets"
            else {
                "path": prefix
                + ("one.md" if captured_variant == "long-one" else "two.md")
            }
            if captured_variant.startswith("long-")
            else {"path": "notes/owner-target.md"}
        )
        kind = "local" if captured_variant == "warning" else "mcp"
        name = "fs_read" if kind == "local" else "read_file"
        payload = _build_approval_payload(
            round_id,
            "paint-session",
            "paint-run",
            [
                MCPPendingCall(
                    name,
                    "local:__local__" if kind == "local" else "external:review",
                    name,
                    "Legacy server label",
                    arguments,
                    "ask",
                    options=("approve_once", "approve_session", "deny"),
                    call_id=round_id,
                    presentation_authority=profile_authority(
                        kind, "Writer", "Project scratch", "call"
                    ),
                    path_precheck_failed=captured_variant == "warning",
                    captured_arguments={
                        "path": "notes/owner-target.md",
                        "body": "original body stays private",
                    },
                )
            ],
            0,
            None,
            revision=7,
        )
        console.set_task_resume_state(TaskResumeState(pending_approval=payload))
        return
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


async def _verify_every_approval_action_is_painted_and_focusable(
    request, geometries=None, *, captured=False
):
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
        for geometry in geometries or product(
            ((80, 24), (120, 40), (170, 48)),
            (True, False),
            ("textual-dark", "textual-light"),
        ):
            size, inspect_open, theme, *variant = geometry
            captured_variant = variant[0] if variant else "warning"
            host.theme = theme
            await pilot.resize_terminal(*size)
            console._set_console_rail_preference(
                left_open=False, right_open=inspect_open
            )
            _pending_card(
                console,
                f"geometry-{size[0]}-{size[1]}-{inspect_open}-{theme}-{captured_variant}",
                captured=captured,
                captured_variant=captured_variant,
            )
            await pilot.pause(0.3)
            card = console.query_one(ChatApprovalCard)
            select = card.query_one(".approval-row-decision", Select)
            if captured:
                header_widget = card.query_one(".approval-row-header", Static)
                header = str(header_widget.content)
                body = card.query_one("#approval-batch-rows")
                # Inspect each new request from its top; previous Tab gestures can scroll the reused viewport.
                body.scroll_home(animate=False)
                await pilot.pause()
                assert card._batch_round_id.endswith(captured_variant)
                if captured_variant.startswith("long-"):
                    suffix = "one.md" if captured_variant == "long-one" else "two.md"
                    assert "/" + "directory/" * 40 + suffix in header
                    body.scroll_to(
                        y=max(0, header_widget.region.height - 3), animate=False
                    )
                    await pilot.pause()
                    assert suffix in host.export_screenshot()
                else:
                    assert body.content_region.contains_region(header_widget.region)
                    hit, _ = host.screen.get_widget_at(*header_widget.region.center)
                    assert hit is header_widget
                for fact in ("Writer", "Project scratch"):
                    assert fact in header
                if captured_variant == "warning":
                    for fact in (
                        "Read file",
                        "notes/owner-target.md",
                        "will fail even if approved",
                    ):
                        assert fact in header
                elif captured_variant == "secrets":
                    assert "Parameters preview" in header
                    for secret in (
                        "private-key-value",
                        "nested-secret",
                        "sk-test-" + "A" * 32,
                    ):
                        assert secret not in host.export_screenshot()
                    assert "***" in header
                assert "original body stays private" not in host.export_screenshot()
                assert "Allow this call once." in str(
                    card.query_one(".approval-row-scope", Static).content
                )
                assert card._presentation_revision == 7
            assert not select.display
            assert (
                card.query_one("#approval-request-summary").region
                in card.content_region
            )
            buttons = [button for button in card.query(Button) if button.display]
            assert len(buttons) == (4 if captured else 3)
            for button in buttons:
                if button.has_class("approval-details-open"):
                    button.scroll_visible(animate=False)
                    await pilot.pause()
                _assert_painted(host, card, button)
            deny = card.query_one(".approval-row-fast-deny", Button)
            approve = card.query_one(".approval-row-fast-approve", Button)
            assert approve.region.y <= deny.region.y
            frame_root = os.environ.get("TLDW_APPROVAL_TASK3_FRAMES")
            if frame_root:
                destination = Path(frame_root)
                destination.mkdir(parents=True, exist_ok=True)
                frame = (
                    destination
                    / f"{'captured-' + captured_variant + '-' if captured else ''}{size[0]}x{size[1]}-inspect-{inspect_open}-{theme}.svg"
                )
                frame.write_text(host.export_screenshot(), encoding="utf-8")
            composer = console.query_one("#console-native-composer")
            composer.load_draft("Approval focus draft stays unsent")
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            messages_before = list(store.messages_for_session(session_id))
            await pilot.press("alt+a", "enter")
            await pilot.pause()
            assert composer.draft_text() == "Approval focus draft stays unsent"
            assert list(store.messages_for_session(session_id)) == messages_before
            if captured:
                assert host.focused.has_class("approval-details-open")
                assert card.query_one("#approval-details-panel").display
                await pilot.press("escape")
                await pilot.pause()
                assert not card.query_one("#approval-details-panel").display
            else:
                assert host.focused.id == "approval-request-summary"
            assert not card._batch_submitted
            focused = {host.focused.id} if host.focused in buttons else set()
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
        console._set_console_rail_preference(left_open=False, right_open=True)
        _pending_card(console)
        await pilot.pause(0.3)
        card = console.query_one(ChatApprovalCard)
        card.query_one(".approval-more-options", Button).press()
        await pilot.pause()
        select = card.query_one(".approval-row-decision", Select)
        select.value = "deny"
        await pilot.pause(0.3)
        for button in card.query(Button):
            if button.display:
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
            "approval-more-options-batch",
            "approval-approve-all",
            "approval-submit",
            "approval-deny-all",
        ]
        for button in card.query(Button):
            if button.display:
                _assert_painted(host, card, button)
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.3)
        assert select.has_focus and select.value == "deny"
        _pending_card(console, "next-round")
        await pilot.pause(0.3)
        assert card._batch_round_id == "next-round"
        assert card.query_one(Select) is not select
        for button in card.query(Button):
            if button.display:
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
        console._set_console_rail_preference(left_open=False, right_open=False)
        _pending_card(console)
        await pilot.pause(0.3)
        card = console.query_one(ChatApprovalCard)
        card.query_one(".approval-more-options", Button).press()
        await pilot.pause()
        select = card.query_one(Select)
        select.value = "deny"
        select.focus()
        assert not card.has_class("approval-compact")
        await pilot.resize_terminal(80, 24)
        await pilot.pause(0.3)
        assert card.has_class("approval-compact")
        assert card.query_one(Select) is select and select.has_focus
        for button in card.query(Button):
            if button.display:
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


@pytest.mark.asyncio
@private_profile_test
async def test_approval_focus_enter_keeps_real_composer_draft(
    request: pytest.FixtureRequest,
) -> None:
    """A real draft exposes Enter bubbling on compact and wider Console layouts."""
    try:
        await _verify_every_approval_action_is_painted_and_focusable(
            request,
            geometries=(
                ((80, 24), True, "textual-dark", "warning"),
                ((80, 24), True, "textual-dark", "secrets"),
                ((80, 24), True, "textual-dark", "long-one"),
                ((120, 40), False, "textual-light", "long-two"),
            ),
            captured=True,
        )
    finally:
        drain_active_service_patches()
        drain_created_dirs()
        gc.unfreeze()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()


@pytest.mark.asyncio
@private_profile_test
async def test_batch_disclosure_actions_fit_real_compact_console(request):
    from dataclasses import replace
    from Tests.UI.test_approval_interaction import _owner_payload

    app = _build_test_app()
    attach_chachanotes_db(app)
    _configure_native_ready_console(app)
    host = ProductionConsoleHarness(app)
    try:
        async with host.run_test(size=(80, 24)) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            for theme, eligible in (("textual-dark", True), ("textual-light", False)):
                host.theme = theme
                console._set_console_rail_preference(left_open=False, right_open=True)
                payload = _owner_payload(count=2)
                payload["round_id"] = theme
                payload["view"] = replace(
                    payload["view"], round_id=theme, bulk_once=eligible
                )
                console.set_task_resume_state(TaskResumeState(pending_approval=payload))
                await pilot.pause()
                card = console.query_one(ChatApprovalCard)
                more = card.query_one("#approval-more-options-batch", Button)
                review = card.query_one("#approval-approve-all", Button)
                for button in (
                    more,
                    review,
                    card.query_one("#approval-deny-all", Button),
                ):
                    _assert_painted(host, card, button)
                await pilot.click(more if eligible else review)
                await pilot.pause()
                select = card.query_one(Select)
                select.focus()
                await pilot.press("enter", "down", "enter")
                await pilot.pause()
                assert select.value == "approve_session"
                for name in (
                    "#approval-more-options-batch",
                    "#approval-submit",
                    "#approval-deny-all",
                ):
                    _assert_painted(host, card, card.query_one(name, Button))
                summary = card.query_one("#approval-count-scope-summary", Static)
                assert "Allow 2" in str(
                    summary.content
                ) and "Until Chatbook exits" in str(summary.content)
                assert host.screen.region.contains_region(summary.region)
                await pilot.press("escape")
                assert select.value == "approve_session" and not select.display
    finally:
        drain_active_service_patches()
        drain_created_dirs()
        gc.unfreeze()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            gc.collect()
