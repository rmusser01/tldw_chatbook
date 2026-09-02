"""Reproducible compositor captures for TASK-21351.

Run from the repository root with:

    pytest -p Tests.conftest \\
      Docs/superpowers/qa/task-21351-console-switcher-activity/capture_evidence.py \\
      -q -s

Loading the shared test plugin is required: it redirects app configuration and
profile writes into a disposable pytest sandbox before this module imports the
production app.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Tests.UI.test_console_activity_switcher import (
    _ActivitySwitcherApp,
    _active_entry,
    _history_entry,
)
from Tests.UI.test_console_native_chat_flow import (
    _build_test_app,
    _configure_native_ready_console,
    _wait_for_selector,
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_activity_receipts import ConsoleActivityReceipt
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    CapturedReceipt,
    ConsoleSwitcherHistoryPage,
    UnavailableSessionNotice,
)
from tldw_chatbook.Widgets.Console.console_activity_outcome_notice import (
    ConsoleActivityOutcomeNotice,
)
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
)

HERE = Path(__file__).resolve().parent
CAPTURES = HERE / "captures"


def _write_frame(app, filename: str, *, title: str) -> None:  # type: ignore[no-untyped-def]
    CAPTURES.mkdir(parents=True, exist_ok=True)
    svg = app.export_screenshot(title=title, simplify=True)
    normalized_svg = "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"
    (CAPTURES / filename).write_text(
        normalized_svg,
        encoding="utf-8",
    )


class _EvidenceReceiptService:
    def __init__(self) -> None:
        self.rows: tuple[ConsoleActivityReceipt, ...] = ()
        self.calls: list[tuple[str, ...]] = []
        self.projection_generation = 1
        self.degraded = False

    def hydration_state(self) -> str:
        return "ready"

    def unseen_snapshot(self) -> tuple[ConsoleActivityReceipt, ...]:
        return self.rows

    def set_receipt(self, *, activity_id: str, status: str, session_id: str) -> None:
        self.projection_generation += 1
        self.rows = (
            ConsoleActivityReceipt(
                activity_id=activity_id,
                origin="ordinary",
                logical_outcome_id=f"turn:{activity_id}",
                transition_revision=1,
                session_id=session_id,
                conversation_id=None,
                run_id=None,
                assistant_message_id=f"message:{activity_id}",
                status=status,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def acknowledge(self, activity_ids) -> int:  # type: ignore[no-untyped-def]
        captured = tuple(activity_ids)
        self.calls.append(captured)
        selected = set(captured)
        before = len(self.rows)
        self.rows = tuple(
            receipt for receipt in self.rows if receipt.activity_id not in selected
        )
        self.projection_generation += 1
        return before - len(self.rows)


def _representative_active_results():
    waiting = replace(
        _active_entry(
            "session:approval",
            "Release approval",
            session_id="approval",
            group=ActivityGroup.WAITING_FOR_YOU,
        ),
        subtitle="APPROVAL · CONSOLE TAB · Shipping · now",
        state_label="APPROVAL",
        activity_state="approval",
    )
    working = replace(
        _active_entry(
            "session:working",
            "Index research corpus",
            session_id="working",
            group=ActivityGroup.WORKING,
        ),
        subtitle="RUNNING · CONSOLE TAB · Research · 1m",
        state_label="RUNNING",
        activity_state="running",
    )
    finished = replace(
        _active_entry(
            "session:finished",
            "Regression triage",
            session_id="finished",
            group=ActivityGroup.NEW_RESULTS,
        ),
        subtitle="FINISHED · UNSEEN · CONSOLE TAB · QA · 2m",
        state_label="FINISHED · UNSEEN",
        activity_state="done",
        target=replace(
            _active_entry(
                "session:finished",
                "Regression triage",
                session_id="finished",
                group=ActivityGroup.NEW_RESULTS,
            ).target,
            receipts=(CapturedReceipt("receipt-finished", "done"),),
        ),
    )
    current = replace(
        _active_entry(
            "session:current",
            "Current design review",
            session_id="current",
            group=ActivityGroup.CURRENT,
        ),
        subtitle="CURRENT · CONSOLE TAB · Design · now",
        state_label="CURRENT",
        activity_state="current",
        is_active=True,
    )
    other = _active_entry(
        "session:other",
        "API documentation",
        session_id="other",
        group=ActivityGroup.OTHER_OPEN,
    )
    unavailable = UnavailableSessionNotice(
        stable_result_key="unavailable-session:profile-a:closed-temp",
        profile_authority="profile-a",
        session_id="closed-temp",
        group=ActivityGroup.WAITING_FOR_YOU,
        latest_at=None,
        receipts=(CapturedReceipt("receipt-closed-temp", "failed"),),
        primary_status="failed",
        all_statuses=("failed",),
    )
    return waiting, unavailable, working, finished, current, other


@pytest.mark.asyncio
async def test_capture_production_stylesheet_switchboard_frames():
    async def load_history(*, query: str, offset: int, limit: int):
        entries = tuple(
            _history_entry(
                f"conversation:{offset + index}",
                f"Saved conversation {offset + index}",
            )
            for index in range(min(limit, 50))
        )
        return ConsoleSwitcherHistoryPage(entries, offset, limit, 73)

    app = _ActivitySwitcherApp(
        active_results=_representative_active_results(),
        history_loader=load_history,
        preferred_native_session_id="other",
    )
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleSessionSwitcherModal)
        assert modal.region.height <= 35
        assert len(app.screen.query(".console-switcher-result")) <= 50
        _write_frame(
            app,
            "active-switchboard-120x35.svg",
            title="TASK-21351 Active switchboard · 120×35",
        )

        await pilot.press("f3")
        await pilot.pause()
        assert modal.region.height <= 35
        assert len(app.screen.query(".console-switcher-result")) <= 50
        _write_frame(
            app,
            "history-switchboard-120x35.svg",
            title="TASK-21351 History switchboard · 120×35",
        )

    narrow = _ActivitySwitcherApp(
        active_results=_representative_active_results(),
        history_loader=load_history,
        preferred_native_session_id="other",
    )
    async with narrow.run_test(size=(72, 35)) as pilot:
        await pilot.pause()
        modal = narrow.screen
        assert isinstance(modal, ConsoleSessionSwitcherModal)
        assert modal.region.height <= 35
        assert modal.region.right <= narrow.size.width
        _write_frame(
            narrow,
            "active-switchboard-72x35.svg",
            title="TASK-21351 Active switchboard · 72×35",
        )


@pytest.mark.asyncio
async def test_capture_real_ctrl_k_destination_acknowledgement_path():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 45)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        await pilot.click("#console-new-chat-tab")
        await pilot.pause()
        store = console._console_chat_store
        first_id, second_id = (session.id for session in store.sessions())
        assert store.active_session_id == second_id
        receipts = _EvidenceReceiptService()
        console._console_runtime()._activity_receipts = receipts

        receipts.set_receipt(
            activity_id="success-background",
            status="done",
            session_id=first_id,
        )
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsoleSessionSwitcherModal)
        _write_frame(
            host,
            "real-ctrl-k-success-selection-160x45.svg",
            title="TASK-21351 real Ctrl+K success selection · 160×45",
        )
        await pilot.press("enter")
        console.query_one("#console-native-composer").insert_text(
            "typed immediately after switch"
        )
        await pilot.pause()
        notice = console.query_one(ConsoleActivityOutcomeNotice)
        assert store.active_session_id == first_id
        assert notice.display
        assert receipts.calls == [("success-background",)]
        assert (
            console.query_one("#console-native-composer").draft_text()
            == "typed immediately after switch"
        )
        _write_frame(
            host,
            "real-success-outcome-notice-160x45.svg",
            title="TASK-21351 visible success notice · 160×45",
        )

        receipts.set_receipt(
            activity_id="failure-background",
            status="failed",
            session_id=second_id,
        )
        await pilot.press("ctrl+k")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert store.active_session_id == second_id
        assert notice.display
        assert receipts.calls == [("success-background",)]
        _write_frame(
            host,
            "real-failure-mark-seen-160x45.svg",
            title="TASK-21351 failure requires Mark seen · 160×45",
        )
        await pilot.click("#console-activity-outcome-mark-seen")
        await pilot.pause()
        assert receipts.calls == [
            ("success-background",),
            ("failure-background",),
        ]
