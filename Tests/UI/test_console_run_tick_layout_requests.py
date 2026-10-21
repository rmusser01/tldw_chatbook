"""TASK-33620.15.7: a Console run's syncs leave unchanged widgets un-laid-out.

``Static.update`` ends in ``refresh(layout=True)``, and clearing a scalar style
(``styles.width = None``) requests a layout even when no rule was set. Textual
answers any layout request with a whole-screen ``Screen._refresh_layout`` over
every widget on the screen. Measured live on dev 2d071d6d71 (2026-10-20,
160x45, a 36-turn chat, Anthropic claude-haiku-4-5): 22-38 full reflows per send
window (Enter to reply end) over ~500 widgets. Of ~260 ``Static.update`` calls
per window, ~245 re-applied copy that had not changed: the 0.2 s run poll
re-synced the hidden setup-card steps, the hidden trace-recovery callout, the
Model section values, the idle voice chip, the empty attachment indicator and
the hidden compat mode bar on every tick.

Each widget pin drives one widget the run touches through its own sync, twice
with the same state, and counts the layout requests the second sync makes; a
positive control per widget shows the counter sees a real change. The last test
holds a real Console run open and counts layout requests across real 0.2 s poll
ticks.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

# The UI fixture imports the app after selecting a per-test profile. Import it
# under the collection bootstrap first, before config freezes its source owner.
import tldw_chatbook.app  # noqa: F401
from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    ConsoleDisplayRow,
    ConsoleInspectorState,
)
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    ConsolePromptQueuePresentation,
    ConsolePromptQueueRegion,
)
from tldw_chatbook.UI.Console_Modules.trace_call_recovery import (
    TraceCallRecoveryCallout,
    TraceCallRecoveryState,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_control_bar import ConsoleControlBar
from tldw_chatbook.Widgets.Console.console_run_inspector import ConsoleRunInspector
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    ConsoleSendAuthoritySummary,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleStatusText,
    StatusSegment,
)

# Real config readers in CompactModelBar keep their source-bound sandbox for
# this interpreter. No case edits config or touches the real user profile.
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _LayoutRequests:
    """Every ``refresh(layout=True)`` or recompose request while counting.

    Wraps ``Widget.refresh`` itself, so a request made through
    ``Static.update``, a cleared or changed layout style, a ``display`` flip, a
    layout reactive or an explicit call is counted alike, synchronously and
    without depending on the screen's update timer.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.widgets: list[Widget] = []
        self._active = False
        original = Widget.refresh

        def recording(widget, *regions, repaint=True, layout=False, recompose=False):
            if self._active and (layout or recompose):
                self.widgets.append(widget)
            return original(
                widget, *regions, repaint=repaint, layout=layout, recompose=recompose
            )

        monkeypatch.setattr(Widget, "refresh", recording)

    @contextmanager
    def counting(self) -> Iterator[list[Widget]]:
        """Collect the requesting widgets made inside the block."""
        self.widgets.clear()
        self._active = True
        try:
            yield self.widgets
        finally:
            self._active = False


def _names(widgets: list[Widget]) -> list[str]:
    return [f"{type(w).__name__}#{w.id}" if w.id else type(w).__name__ for w in widgets]


class _Host(ConsolidatedCSSApp):
    """Mount one production widget under the app stylesheet."""

    def __init__(self, widget: Widget) -> None:
        super().__init__()
        self.app_config = {}
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


# -- ConsoleControlBar: the auto-speak sync and the hidden compat labels ----


class _ControlBarHost(ConsolidatedCSSApp):
    """The production control bar, as ``test_console_control_bar_recovery_height``."""

    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self.app_config = {}
        self.bar = ConsoleControlBar(state, self)

    def compose(self) -> ComposeResult:
        yield self.bar


async def test_unchanged_auto_speak_sync_requests_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    host = _ControlBarHost(ConsoleControlState.from_values())
    bar = host.bar
    async with host.run_test(size=(120, 10)) as pilot:
        bar.sync_auto_speak(enabled=True, paused=False)
        await pilot.pause()
        with requests.counting() as calls:
            bar.sync_auto_speak(enabled=True, paused=False)
        assert _names(calls) == []
        # Positive control: the recovery row really changes the bar's height.
        with requests.counting() as calls:
            bar.sync_auto_speak(enabled=True, paused=True)
        assert bar in calls


async def test_hidden_control_labels_take_new_copy_without_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    state = ConsoleControlState.from_values()
    host = _ControlBarHost(state)
    bar = host.bar
    async with host.run_test(size=(120, 10)) as pilot:
        await pilot.pause()
        changed = replace(state, approvals_label="Approvals: 2 pending")
        with requests.counting() as calls:
            bar.sync_state(changed)
        approvals = bar.query_one("#console-approvals-label", Static)
        assert str(approvals.render()) == "Approvals: 2 pending"
        assert approvals.display is False
        assert _names(calls) == []


# -- ConsoleSetupModal: the hidden card is re-synced every run tick ---------


def _quiet_card() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="quiet",
        steps=(
            ConsoleSetupStep(state="done", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(state="active", label="Send your first message"),
        ),
    )


async def test_unchanged_setup_card_resync_requests_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    modal = ConsoleSetupModal()
    async with _Host(modal).run_test(size=(120, 30)) as pilot:
        modal.sync_card_state(_quiet_card())
        await pilot.pause()
        with requests.counting() as calls:
            modal.sync_card_state(_quiet_card())
        assert _names(calls) == []
        # Positive control: a changed step label is new copy to lay out.
        changed = replace(
            _quiet_card(),
            steps=(
                *_quiet_card().steps[:2],
                ConsoleSetupStep(state="active", label="Send a message"),
            ),
        )
        with requests.counting() as calls:
            modal.sync_card_state(changed)
        assert "Static#console-setup-step-3" in _names(calls)


# -- TraceCallRecoveryCallout: hidden, yet re-synced on every transcript sync


async def test_hidden_trace_recovery_resync_requests_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    callout = TraceCallRecoveryCallout(state=None, on_action=lambda *_: None)
    async with _Host(callout).run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        with requests.counting() as calls:
            callout.sync_recovery(None)
        assert _names(calls) == []
        # Positive control: a held send's recovery state shows the callout.
        with requests.counting() as calls:
            callout.sync_recovery(TraceCallRecoveryState("preparation-1"))
        assert callout in calls


# -- ConsoleComposerBar: idle voice chip, attachment indicator, Send reason --


async def test_idle_voice_and_attachment_resyncs_request_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    composer = ConsoleComposerBar()
    async with _Host(composer).run_test(size=(120, 10)) as pilot:
        composer.set_voice_status("idle")
        composer.set_pending_attachment_label(None)
        await pilot.pause()
        with requests.counting() as calls:
            composer.set_voice_status("idle")
            composer.set_pending_attachment_label(None)
        assert _names(calls) == []
        # Positive control: a staged attachment shows the indicator.
        with requests.counting() as calls:
            composer.set_pending_attachment_label("notes.txt · 12 B", count=1)
        assert "Static#console-attachment-indicator" in _names(calls)
        # Positive control: a chip that was showing copy is emptied by idle.
        chip = composer.query_one("#console-voice-status", Static)
        composer.set_voice_status("listening", elapsed_seconds=3)
        await pilot.pause()
        assert chip.display and str(chip.content)
        with requests.counting() as calls:
            composer.set_voice_status("idle")
        assert "Static#console-voice-status" in _names(calls)
        assert not chip.display and str(chip.content) == ""


async def test_unchanged_send_reason_resync_requests_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    composer = ConsoleComposerBar()
    running = dict(
        has_draft=False,
        run_active=True,
        can_save_chatbook=False,
        send_blocked=True,
        queue_blocked_reason="Type to queue",
    )
    async with _Host(composer).run_test(size=(160, 10)) as pilot:
        composer.sync_action_state(**running)
        await pilot.pause()
        with requests.counting() as calls:
            composer.sync_action_state(**running)
        assert _names(calls) == []
        # Positive control: a changed reason is new copy for the strip.
        with requests.counting() as calls:
            composer.sync_action_state(**{**running, "queue_blocked_reason": "Full"})
        assert "Static#console-send-disabled-reason" in _names(calls)
        assert str(composer.query_one("#console-send-disabled-reason").render()) == (
            "Full"
        )


# -- ConsoleStatusText: the run/cost line --------------------------------


def _segments(cost: str) -> list[StatusSegment]:
    return [
        StatusSegment(name="run", label="Run: streaming"),
        StatusSegment(name="provider", label="Provider: Anthropic"),
        StatusSegment(name="cost", label=cost),
    ]


async def test_same_width_status_change_repaints_without_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    line = ConsoleStatusText(_segments("Current $0.01"), id="console-status-text")
    async with _Host(line).run_test(size=(120, 5)) as pilot:
        line.set_segments(_segments("Current $0.01"))
        await pilot.pause()
        region = line.region
        with requests.counting() as calls:
            line.set_segments(_segments("Current $0.02"))
        assert _names(calls) == []
        await pilot.pause()
        # Same geometry, new copy: the repaint without layout lands in place.
        assert line.region == region
        assert "Current $0.02" in str(line.render())
        # Positive control: a wider line changes the strip's scroll range.
        with requests.counting() as calls:
            line.set_segments(_segments("Current $10.02"))
        assert line in calls


# -- ConsoleSendAuthoritySummary: one Inspector fact changes at a time -----


def _inspector_state(**overrides) -> ConsoleInspectorState:
    values = {
        "rows": (
            ConsoleDisplayRow("Workspace", "Research Lab"),
            ConsoleDisplayRow("Selected conversation", "Research conversation 3"),
            ConsoleDisplayRow("Provider", "ready", status="ready"),
        ),
    }
    values.update(overrides)
    return ConsoleInspectorState(**values)


async def test_inspector_change_lays_out_only_changed_fact_rows(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    summary = ConsoleSendAuthoritySummary(_inspector_state())
    async with _Host(summary).run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        unrelated = _inspector_state(
            rows=(
                *_inspector_state().rows,
                ConsoleDisplayRow("Run recipe", "Chat with provider"),
            )
        )
        with requests.counting() as calls:
            summary.sync_state(unrelated)
        assert _names(calls) == []
        # Positive control: a blocked provider changes the Run fact alone.
        blocked = _inspector_state(
            rows=(
                *_inspector_state().rows[:2],
                ConsoleDisplayRow("Provider", "blocked", status="blocked"),
            )
        )
        with requests.counting() as calls:
            summary.sync_state(blocked)
        assert _names(calls) == ["Static#console-send-authority-run"]
        assert str(summary.query_one("#console-send-authority-run").render()) == (
            "Run: Blocked"
        )


# -- ConsolePromptQueueRegion: Send-button fields the shelf does not draw --


def _queue(**overrides) -> ConsolePromptQueuePresentation:
    values = dict(
        revision=1,
        count=0,
        send_label="Send",
        send_enabled=True,
        send_tooltip="Send the draft.",
        shelf_visible=False,
        state_label="Draining",
        paused=False,
        next_preview="",
        pause_label="Pause",
        primary_action="",
        pause_enabled=False,
    )
    values.update(overrides)
    return ConsolePromptQueuePresentation(**values)


async def test_undrawn_queue_projection_change_requests_no_layout(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    region = ConsolePromptQueueRegion(id="console-prompt-queue")
    async with _Host(region).run_test(size=(120, 10)) as pilot:
        region.sync_presentation("session-1", _queue())
        await pilot.pause()
        with requests.counting() as calls:
            region.sync_presentation(
                "session-1",
                _queue(revision=2, send_label="Queue", send_enabled=False),
            )
        assert _names(calls) == []
        # Positive control: a relabelled button re-lays itself out (TASK-33625.4).
        with requests.counting() as calls:
            region.sync_presentation(
                "session-1", _queue(revision=3, pause_label="Resume")
            )
        assert "Button#console-prompt-queue-pause" in _names(calls)


# -- ConsoleRunInspector: an in-place update inside the collapsed rail ------


def _rows(provider: str) -> ConsoleInspectorState:
    return ConsoleInspectorState(
        rows=(
            ConsoleDisplayRow("Run recipe", "Chat with provider"),
            ConsoleDisplayRow("Provider", provider, status="ready"),
        ),
        actions=(),
        dictionary_rows=(),
        dictionary_actions=(),
    )


class _CollapsedRailHost(ConsolidatedCSSApp):
    def __init__(self) -> None:
        super().__init__()
        self.inspector = ConsoleRunInspector(_rows("OpenAI / gpt-4o"), id="inspector")

    def compose(self) -> ComposeResult:
        with Vertical(id="rail"):
            yield self.inspector


async def test_collapsed_rail_inspector_update_keeps_rows_hidden(monkeypatch) -> None:
    requests = _LayoutRequests(monkeypatch)
    host = _CollapsedRailHost()
    async with host.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        rail = host.query_one("#rail")
        shown = [child for child in host.inspector.query("*") if child.display]
        assert shown
        ChatScreen._sync_console_rail_descendant_visibility(rail, False)
        await pilot.pause()
        with requests.counting() as calls:
            host.inspector.sync_state(_rows("OpenAI / gpt-4.1"))
            await pilot.pause()
            await pilot.pause()
        assert [child for child in shown if child.display] == []
        assert [w for w in calls if w.display] == []
        # Reopening the rail restores every row it hid.
        ChatScreen._sync_console_rail_descendant_visibility(rail, True)
        assert all(child.display for child in shown)


# -- A scripted run: the real Console, a held stream, real poll ticks -------


@private_profile_test
async def test_held_run_poll_ticks_request_no_layout_outside_their_owners(
    monkeypatch, request
) -> None:
    from Tests.UI.test_console_composer_run_controls import (
        _held_run_host,
        _mounted,
        _start_held_run,
    )

    requests = _LayoutRequests(monkeypatch)
    gateway, host = _held_run_host()
    async with host.run_test(size=(160, 45)) as pilot:
        console, composer = await _mounted(host, pilot)
        try:
            await _start_held_run(console, composer, pilot)
            assert console._console_transcript_sync_timer is not None
            ticks: list[float] = []
            original = console._sync_native_console_chat_ui

            async def counted(*args, **kwargs):
                ticks.append(time.monotonic())
                return await original(*args, **kwargs)

            monkeypatch.setattr(console, "_sync_native_console_chat_ui", counted)
            # Let the run's own start-up transitions land first.
            deadline = time.monotonic() + 5
            while len(ticks) < 3 and time.monotonic() < deadline:
                await pilot.pause(0.05)
            # The transcript streams; its rows may legitimately re-lay out.
            # The Character section recomposes on its own refresh cycle
            # (Refreshing... then the result), collapsed or not -- a separate
            # cost, left to TASK-35970.
            exempt = (
                console.query_one("#console-native-transcript"),
                console.query_one("#console-character-context"),
            )
            with requests.counting() as calls:
                start = len(ticks)
                deadline = time.monotonic() + 10
                while len(ticks) - start < 6 and time.monotonic() < deadline:
                    await pilot.pause(0.05)
                observed = len(ticks) - start
            assert observed >= 6, f"only {observed} poll ticks ran"
            outside = [
                widget
                for widget in calls
                if not any(
                    widget is owner or owner in widget.ancestors for owner in exempt
                )
            ]
            assert _names(outside) == [], (
                f"{len(outside)} layout requests over {observed} unchanged poll "
                f"ticks: {_names(outside)}"
            )
        finally:
            gateway.release.set()
            await pilot.pause(0.2)
