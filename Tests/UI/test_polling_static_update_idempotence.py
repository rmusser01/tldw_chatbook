"""Unchanged polling must not request Static layout while source checks continue."""

from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp

pytestmark = pytest.mark.bootstrap_profile


def observe_updates(monkeypatch, ids):
    calls = []
    original = Static.update

    def observed(widget, *args, **kwargs):
        if widget.id in ids:
            calls.append(widget.id)
        return original(widget, *args, **kwargs)

    monkeypatch.setattr(Static, "update", observed)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("unavailable", [False, True])
async def test_progress_poll_keeps_unchanged_count_body_and_refusal(
    monkeypatch, unavailable
):
    from Tests.UI.test_console_agent_progress import (
        ProgressHarness,
        _modal_type,
        _queue,
    )

    _store, inbox, sender = _queue()
    sender.send("First report")
    modal = _modal_type()(
        conversation_id="conv-A", load=inbox.snapshot, discard=inbox.discard
    )
    if unavailable:
        modal._load_error = "queue_full"
    host = ProgressHarness(modal)
    async with host.run_test(size=(100, 36)) as pilot:
        await pilot.pause()
        calls = observe_updates(
            monkeypatch,
            {"agent-progress-count", "agent-progress-body", "agent-progress-status"},
        )
        for _ in range(8):
            modal._refresh_snapshot()
        assert calls == [], f"Unchanged progress repaints: {calls}"
        modal._load_error = None
        sender.send("Second report")
        modal._refresh_snapshot()
        await pilot.pause()
        assert "2 queued" in str(
            modal.query_one("#agent-progress-count", Static).renderable
        )
        assert "agent-progress-count" in calls


@pytest.mark.asyncio
async def test_stale_skill_review_still_checks_currentness_without_repainting(
    monkeypatch, tmp_path
):
    from tldw_chatbook.Skills_Interop.recovery_activation import RecoveryReview
    from tldw_chatbook.UI.Screens.skills_screen import SkillRecoveryReviewModal

    checks = []

    def current():
        checks.append(True)
        return False

    review = RecoveryReview("fixture", (), (str(tmp_path), 0, 0), (), (), (), "fixture")
    modal = SkillRecoveryReviewModal(review, tmp_path, current)
    host = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    async with host.run_test() as pilot:
        await host.push_screen(modal)
        await pilot.pause()
        calls = observe_updates(monkeypatch, {"skills-recovery-status"})
        before = len(checks)
        for _ in range(8):
            assert modal._check_selection() is False
        assert len(checks) >= before + 8
        assert modal.query_one("#skills-recovery-continue", Button).disabled
        assert "Selection changed" in str(
            modal.query_one("#skills-recovery-status", Static).renderable
        )
        assert calls == [], f"Unchanged stale-selection repaints: {calls}"


@pytest.mark.asyncio
async def test_buddy_speech_poll_keeps_same_notice_and_paints_changes(monkeypatch):
    from tldw_chatbook.Persona_Buddy.speech import BuddySpeechQueue
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_speech_controls import (
        BuddySpeechControls,
    )

    async def play(*_args):
        pytest.fail("A status refresh must not start playback")

    queue = BuddySpeechQueue(play)
    coordinator = SimpleNamespace(
        queue=queue, notice="Ready", enabled=True, needs_consent=False
    )
    widget = BuddySpeechControls(coordinator)
    host = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    async with host.run_test() as pilot:
        await host.mount(widget)
        await pilot.pause()
        calls = observe_updates(monkeypatch, {"buddy-speech-status"})
        for _ in range(8):
            widget.refresh_state()
        assert calls == [], f"Unchanged speech repaints: {calls}"
        prior_height = widget.query_one("#buddy-speech-status", Static).region.height
        coordinator.notice = "Changed notice\nSecond line"
        widget.refresh_state()
        await pilot.pause()
        assert (
            str(widget.query_one("#buddy-speech-status", Static).renderable)
            == coordinator.notice
        )
        assert calls == ["buddy-speech-status"]
        assert (
            widget.query_one("#buddy-speech-status", Static).region.height
            > prior_height
        )


@pytest.mark.asyncio
async def test_buddy_inbox_poll_keeps_title_error_and_observes_recovery(monkeypatch):
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    title, failed, reads = "Research", False, []

    async def snapshot():
        reads.append(True)
        if failed:
            raise ValueError("Unavailable workspace")
        return title, ()

    modal = BuddyWorkspaceModal(
        snapshot=snapshot, open_entry=lambda _: None, acknowledge=lambda _: None
    )
    host = ConsolidatedCSSApp(css_path=list(APP_STYLESHEETS))
    async with host.run_test() as pilot:
        await host.push_screen(modal)
        await pilot.pause()
        await modal.refresh_inbox()
        assert (
            str(modal.query_one("#buddy-inbox-title", Static).renderable)
            == "Buddy · Research"
        )
        calls = observe_updates(monkeypatch, {"buddy-inbox-title", "buddy-inbox-error"})
        before = len(reads)
        for _ in range(8):
            await modal.refresh_inbox()
        assert len(reads) >= before + 8
        assert calls == [], f"Unchanged inbox repaints: {calls}"
        failed = True
        await modal.refresh_inbox()
        assert modal.query_one("#buddy-inbox-open", Button).disabled
        assert (
            str(modal.query_one("#buddy-inbox-error", Static).renderable)
            == "Unavailable workspace"
        )
        calls.clear()
        for _ in range(8):
            await modal.refresh_inbox()
        assert calls == [], f"Unchanged failure repaints: {calls}"
        failed, title = False, "Renamed workspace"
        await modal.refresh_inbox()
        assert (
            str(modal.query_one("#buddy-inbox-title", Static).renderable)
            == "Buddy · Renamed workspace"
        )
        assert str(modal.query_one("#buddy-inbox-error", Static).renderable) == ""
