"""Equal Console copy must not invalidate native visuals on every sync."""

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.content import Content
from textual.widgets import Button, Static

from Tests.UI.console_fixture_ownership import owned_console_apps  # noqa: F401
from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from Tests.UI.test_console_left_rail import make_console_pilot
from Tests.UI.test_console_voice_chip import ProductionCssComposerApp
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsSummaryState
from tldw_chatbook.Chat.console_turn_preparation import ContextCompactionHold
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
    TraceCallRecoveryState,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

pytestmark = pytest.mark.bootstrap_profile


class ProjectionHarness(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal()
        yield ChatApprovalCard(id="chat-approval-card")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["card", "quiet"])
async def test_setup_sync_retains_unchanged_native_visuals_and_updates_changed_copy(
    mode,
):
    """Catch repeated Static.update without skipping hidden-state or class repair."""
    state = ConsoleSetupCardState(
        mode=mode,
        steps=(ConsoleSetupStep(state="active", label="Add an API key"),),
    )
    app = ProjectionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        modal = app.query_one(ConsoleSetupModal)
        modal.sync_card_state(state, staged_evidence_notice="Evidence staged")
        await pilot.pause()
        rows = [
            modal.query_one(f"#console-setup-step-{index}", Static)
            for index in (1, 2, 3)
        ]
        notice = modal.query_one("#console-setup-modal-staged-notice", Static)
        original_visuals = [row.visual for row in (*rows, notice)]
        for _ in range(5):
            modal.sync_card_state(state, staged_evidence_notice="Evidence staged")
        assert all(
            row.visual is visual
            for row, visual in zip((*rows, notice), original_visuals)
        )
        assert modal.display is (mode == "card")
        rows[0].set_classes("damaged")
        modal.sync_card_state(state, staged_evidence_notice="Evidence staged")
        assert rows[0].has_class("console-setup-step-active")
        updated = replace(
            state, steps=(ConsoleSetupStep(state="done", label="API key ready"),)
        )
        modal.sync_card_state(updated, staged_evidence_notice="New evidence staged")
        assert rows[0].content == "1. ✓ API key ready"
        assert notice.content == "New evidence staged"
        assert rows[0].visual is not original_visuals[0]
        assert notice.visual is not original_visuals[-1]
        assert rows[1].visual is original_visuals[1]
        assert rows[0].display is (mode == "card")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "size"),
    [
        ("idle", (120, 12)),
        ("unavailable", (120, 12)),
        ("listening", (120, 12)),
        ("listening", (40, 12)),
    ],
)
async def test_voice_projection_keeps_native_layout_and_visual_when_unchanged(
    state, size
):
    """Real sizing/class and Content invalidations must stop, not just calls."""
    app = ProductionCssComposerApp()
    async with app.run_test(size=size) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        chip = composer.query_one("#console-voice-status", Static)
        reason = composer.query_one("#console-send-disabled-reason", Static)
        composer.set_voice_status(state, partial="steady", elapsed_seconds=1)
        await pilot.pause()
        visual = chip.visual
        layout_generation = chip._layout_updates
        reason_generation = reason._layout_updates
        focused = app.focused
        for _ in range(3):
            composer.set_voice_status(state, partial="steady", elapsed_seconds=1)
        assert chip.visual is visual
        assert chip._layout_updates == layout_generation
        assert reason._layout_updates == reason_generation
        assert app.focused is focused

        chip.add_class("w-7", "external-state")
        chip.styles.width = 9
        reason.styles.width = 7
        reason.styles.height = 7
        composer.set_voice_status(state, partial="steady", elapsed_seconds=1)
        assert not chip.has_class("w-7")
        assert chip.has_class("external-state")
        assert not reason.styles.inline.has_rule("width")
        assert not reason.styles.inline.has_rule("height")
        if state in ("idle", "unavailable") or size[0] == 40:
            assert chip.display is False
            assert chip.styles.width.value == 0
            assert composer._voice_chip_last_width == 0
        else:
            # Content equality alone ignores spans; styling drift must be repaired.
            chip.update(Content.from_markup("[bold]$text[/]", text=chip.content.plain))
            composer.set_voice_status(state, partial="steady", elapsed_seconds=1)
            assert not chip.content.spans
            composer.set_voice_status("error", message="Fresh failure")
            assert chip.content.plain == "Fresh failure"
            assert chip.has_class("console-voice-status-error")


@pytest.mark.asyncio
@pytest.mark.parametrize("summary", [None, "Advisory context"])
async def test_cleared_approval_refresh_retains_unchanged_copy_but_repairs_a_new_phase(
    summary,
):
    """Catch repeated title/deadline/summary invalidation during empty syncs."""
    app = ProjectionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        card = app.query_one(ChatApprovalCard)
        card.set_batch([], timeout_seconds=45, summary=summary)
        await pilot.pause()
        rows = [
            card.query_one(selector, Static)
            for selector in (
                "#approval-title",
                "#approval-deadline",
                "#approval-summary",
            )
        ]
        original_visuals = [row.visual for row in rows]
        generation = card._batch_generation
        for _ in range(5):
            card.set_batch([], timeout_seconds=45, summary=summary)
        assert all(row.visual is visual for row, visual in zip(rows, original_visuals))
        assert card.display is False
        assert card._batch_generation == generation + 5
        assert card._deadline_at is None
        assert card._deadline_timer is None
        card.set_batch(
            [], timeout_seconds=45, phase="finishing", summary="Updated context"
        )
        assert rows[0].content == "Finishing — Stop will not cancel"
        assert "Updated context" in rows[2].content
        assert rows[0].visual is not original_visuals[0]
        assert rows[1].visual is original_visuals[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("label", [None, "2 files"])
@pytest.mark.parametrize("preparing", [False, True])
async def test_attachment_refresh_retains_native_visual_and_final_row_budget(
    label, preparing
):
    """Unchanged metadata must not churn classes, copy or transient action width."""
    app = ProductionCssComposerApp()
    async with app.run_test(size=(160, 20)) as pilot:
        composer = app.query_one(ConsoleComposerBar)
        if preparing:
            composer.set_voice_status(
                "preparing",
                message="Local transcription busy — dictation will run next.",
            )
        composer.set_pending_attachment_label(label, count=2, total=5)
        await pilot.pause()
        indicator = composer.query_one("#console-attachment-indicator", Static)
        clear = composer.query_one("#console-clear-attachment")
        actions = composer.query_one("#console-composer-actions")
        visual = indicator.visual
        generations = (indicator._layout_updates, actions._layout_updates)
        focused = app.focused
        for _ in range(3):
            composer.set_pending_attachment_label(label, count=2, total=5)
        await pilot.pause()
        assert indicator.visual is visual
        assert (indicator._layout_updates, actions._layout_updates) == generations
        assert app.focused is focused
        assert indicator.display is (bool(label) and not preparing)
        assert clear.display is indicator.display

        indicator.add_class("w-7", "external-state")
        indicator.styles.width = 9
        composer.set_pending_attachment_label(label, count=3, total=5)
        assert not indicator.has_class("w-7")
        assert indicator.has_class("external-state")
        assert not indicator.styles.inline.has_rule("width")
        if label:
            assert str(clear.tooltip) == "Remove all 3 pending attachments."
        if preparing:
            composer.set_voice_status("idle")
            assert indicator.display is bool(label)
        composer.set_pending_attachment_label("new.txt", count=1, total=5)
        assert "new.txt" in indicator.content
        assert str(clear.tooltip) == "Remove the pending attachment."


class SettingsProjectionHarness(ConsolidatedCSSApp):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield ConsoleSettingsSummary(ConsoleSettingsSummaryState("", "", "", ""))


@pytest.mark.asyncio
async def test_settings_button_sizing_retains_native_layout_and_repairs_overrides():
    """Stable sizing must not clear missing rules or remove/re-add its height."""
    app = SettingsProjectionHarness()
    async with app.run_test(size=(80, 20)) as pilot:
        summary = app.query_one(ConsoleSettingsSummary)
        button = summary.query_one("#console-settings-open", Button)
        await pilot.pause()
        generation = button._layout_updates
        focused = app.focused
        for _ in range(3):
            summary._apply_button_sizing(button)
        await pilot.pause()
        assert button._layout_updates == generation
        assert app.focused is focused
        assert button.region.height == 1
        assert button.region.width == 11

        button.add_class("h-7", "w-7", "external-state")
        button.styles.height = 9
        summary._apply_button_sizing(button)
        await pilot.pause()
        assert not {"h-7", "w-7"} & button.classes
        assert button.has_class("h-1", "external-state")
        assert not button.styles.inline.has_rule("height")
        assert button.region.height == 1
        for label, width in (("Go", 9), ("A longer configuration action", 14)):
            summary.sync_state(replace(summary.state, action_label=label))
            await pilot.pause()
            assert button.region.width == width
            assert str(button.label) == label


@pytest.mark.asyncio
async def test_settings_child_updates_changed_rows_without_replacing_equal_visuals():
    app = SettingsProjectionHarness()
    async with app.run_test(size=(80, 20)) as pilot:
        summary = app.query_one(ConsoleSettingsSummary)
        state = ConsoleSettingsSummaryState("Model: local", "Context: 1", "", "")
        summary.sync_state(state)
        await pilot.pause()
        model = summary.query_one("#console-settings-model-row", Static)
        context = summary.query_one("#console-settings-context-row", Static)
        visual = model.visual
        old_context = context.visual
        changed = replace(state, context_row="Context: 2", action_tooltip="New action")
        summary.sync_state(changed)
        assert model.visual is visual
        assert context.visual is not old_context
        assert context.content == "Context: 2"
        assert summary.query_one("#console-settings-open").tooltip == "New action"
        model.update(Content.from_markup("[bold]Model: local[/]"))
        summary.sync_state(replace(changed, context_row="Context: 3"))
        assert type(model.content) is str
        assert model.content == "Model: local"


class TraceProjectionHarness(ConsolidatedCSSApp):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield TraceCallRecoveryCallout(state=None, on_action=lambda *_: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize(
    "state",
    [
        None,
        TraceCallRecoveryState("preparation"),
        TraceCallRecoveryState("preparation", temporary_capture=True),
        TraceCallRecoveryState("preparation", provenance=True),
        TraceCallRecoveryState(
            "preparation",
            context_hold=ContextCompactionHold("session", 80, 70, 100, False),
        ),
    ],
)
async def test_trace_copy_retains_visuals_without_skipping_live_actions(state, busy):
    app = TraceProjectionHarness()
    async with app.run_test(size=(120, 30)) as pilot:
        callout = app.query_one(TraceCallRecoveryCallout)
        callout._busy = busy
        callout.sync_recovery(state)
        await pilot.pause()
        title = callout.query_one("#console-trace-title", Static)
        status = callout.query_one("#console-trace-status", Static)
        rows = [title, *list(callout.query(".console-trace-detail"))[1:3], status]
        visuals = [row.visual for row in rows]
        focused = app.focused
        cancel = callout.query_one("#console-trace-cancel", Button)
        cancel.disabled = not (busy or state is None)
        for _ in range(3):
            callout.sync_recovery(state)
        assert all(row.visual is visual for row, visual in zip(rows, visuals))
        assert cancel.disabled is (busy or state is None)
        assert cancel.display is (state is not None)
        assert app.focused is focused
        assert callout.display is (state is not None)
        title.update(Content.from_markup("[bold]$copy[/]", copy=str(title.content)))
        callout.sync_recovery(state)
        assert type(title.content) is str
        callout._busy = False
        callout._status_copy = "Recovery needs another choice."
        callout.sync_recovery(state)
        assert status.content == "Recovery needs another choice."
        callout.sync_recovery(
            TraceCallRecoveryState("replacement", temporary_capture=True)
        )
        assert title.content == "Save chat to capture this send"
        assert status.content == "Choose one action."
        assert callout.query_one("#console-trace-save-send", Button).display
        assert not callout.query_one("#console-trace-retry", Button).display


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["character", "model"])
async def test_remaining_rail_copy_keeps_visuals_while_live_state_changes(kind):
    from Tests.UI.test_console_mcp_approval import _wait_for_production_console_ready
    from tldw_chatbook.Character_Chat.character_conversation_navigation import (
        CharacterConversationGroup,
        ResolvedLocalCharacterKey,
    )
    from tldw_chatbook.Chat.console_settings_defaults import (
        ConsoleDefaultDurabilityState,
    )
    from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
    from tldw_chatbook.Widgets.Console.console_character_context import (
        ConsoleCharacterContext,
    )

    # The existing opt-in owner captures this helper's actual constructor stores.
    async with make_console_pilot(production_styles=True) as pilot:
        console = await _wait_for_production_console_ready(pilot.app, pilot)
        rail = console.query_one(ConsoleLeftRail)
        state = rail.query_one(ConsoleCharacterContext).state
        generation = 1

        def sync():
            if kind == "character":
                rail.sync_character_context(state)
            else:
                rail.sync_model_recovery(
                    session_id="current-session",
                    failures={},
                    default_state=ConsoleDefaultDurabilityState(
                        newest_intent_generation=generation
                    ),
                )

        sync()
        await pilot.pause()
        ids = [f"console-rail-section-title-{kind}"]
        if kind == "model":
            ids.append("console-default-recovery-copy")
        rows = [rail.query_one(f"#{widget_id}", Static) for widget_id in ids]
        visuals = [row.visual for row in rows]
        focused = pilot.app.focused
        for _ in range(3):
            sync()
        assert all(row.visual is visual for row, visual in zip(rows, visuals))
        assert pilot.app.focused is focused
        rows[0].update(Content.from_markup("[bold]$copy[/]", copy=str(rows[0].content)))
        sync()
        assert type(rows[0].content) is str
        if kind == "character":
            rows[0].tooltip = "stale tooltip"
            state = replace(
                state,
                groups=(
                    CharacterConversationGroup(
                        ResolvedLocalCharacterKey("authority", 1),
                        "New character",
                        (),
                        3,
                        True,
                    ),
                ),
            )
            sync()
            assert rows[0].content == "Character · New character · 3 chats"
            assert rows[0].tooltip == "Character · New character · 3 chats"
        else:
            retry = rail.query_one("#console-retry-default-save", Button)
            generation_retry = rail.query_one(
                "#console-retry-generation-settings", Button
            )
            generation_retry.disabled = False
            generation = 2
            sync()
            assert generation_retry.disabled
            assert retry.console_default_intent_generation == 2
            assert rows[0].content == "Model"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_name", ["ACTIVE", "HISTORY", "CHARACTER_CHATS"])
@pytest.mark.parametrize("receipt", ["ready", "degraded"])
async def test_remaining_switcher_copy_retains_visuals_and_current_status(
    mode_name, receipt
):
    from Tests.UI.test_console_character_switcher import _CharacterSwitcherApp
    from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
        ConsoleSessionSwitcherModal,
        SwitcherMode,
    )

    app = _CharacterSwitcherApp()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, ConsoleSessionSwitcherModal)
        modal._mode = SwitcherMode[mode_name]
        modal._activity_receipt_state = receipt
        modal._set_status("Ready")
        modal._update_receipt_status()
        rows = [
            modal.query_one(f"#console-switcher-{name}", Static)
            for name in ("status", "scope", "divider", "receipt-state")
        ]
        visuals = [row.visual for row in rows]
        focused = app.focused
        for _ in range(3):
            modal._set_status("Ready")
            modal._update_receipt_status()
        assert all(row.visual is visual for row, visual in zip(rows, visuals))
        assert app.focused is focused
        modal._query_pending = True
        modal._set_status("Searching")
        assert rows[0].content == "Searching"
        assert rows[1].content == (
            "This profile · Local chats"
            if mode_name == "CHARACTER_CHATS"
            else "Searching"
        )
        assert rows[2].content == (
            "Searching" if mode_name == "CHARACTER_CHATS" else "─" * 48
        )
        assert rows[3].display is (receipt == "degraded")
        rows[1].update(Content.from_markup("[bold]$copy[/]", copy=str(rows[1].content)))
        modal._set_status("Searching")
        assert type(rows[1].content) is str
        await pilot.resize_terminal(52, 20)
        modal._update_receipt_status()
        assert not rows[3].display


@pytest.mark.asyncio
async def test_remaining_queue_copy_retains_visuals_across_revision_and_session():
    from Tests.UI.test_console_prompt_queue import (
        _activity,
        _RegionApp,
        _registry_with_chain,
    )
    from tldw_chatbook.UI.Console_Modules.prompt_queue import (
        ConsolePromptQueueRegion,
        derive_prompt_queue_presentation,
    )

    registry = _registry_with_chain()
    snapshot = registry.admit(
        "session-a",
        text="safe preview",
        expected_revision=registry.snapshot("session-a").revision,
    ).snapshot
    state = derive_prompt_queue_presentation(
        snapshot, _activity(accepted=True, count=1)
    )
    app = _RegionApp()
    async with app.run_test(size=(120, 24)) as pilot:
        region = app.query_one(ConsolePromptQueueRegion)
        region.sync_presentation("session-a", state)
        await pilot.pause()
        rows = [
            region.query_one(f"#console-prompt-queue-{name}", Static)
            for name in ("summary", "preview")
        ]
        visuals = [row.visual for row in rows]
        focused = app.focused
        for revision in (20, 21, 22):
            assert region.sync_presentation(
                "session-b", replace(state, revision=revision)
            )
        assert all(row.visual is visual for row, visual in zip(rows, visuals))
        assert region._session_id == "session-b"
        assert region._presentation.revision == 22
        assert app.focused is focused
        changed = replace(
            state, revision=23, next_preview="new preview", pause_enabled=False
        )
        region.sync_presentation("session-b", changed)
        assert rows[1].content == ' · Next: "new preview"'
        assert region.query_one("#console-prompt-queue-pause", Button).disabled
        assert not region.query_one("#console-prompt-queue-manage", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 120])
async def test_remaining_cost_label_keeps_visual_and_layout_until_copy_changes(width):
    from Tests.Chat.test_console_status_chips_cost import (
        _ChipsApp,
        _control_state,
        _cost_state,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

    state = _cost_state()
    app = _ChipsApp(_control_state(), cost_state=state)
    async with app.run_test(size=(width, 8)) as pilot:
        await pilot.pause()
        chips = app.query_one(ConsoleStatusChips)
        line = chips.query_one("#console-status-text", Static)
        visual, layout = line.visual, line._layout_updates
        for _ in range(3):
            chips.sync_cost_state(replace(state))
        assert line.visual is visual
        assert line._layout_updates == layout
        chips.sync_cost_state(
            replace(
                state,
                label="$0.99",
                compact_label="$0.99",
                tooltip="New total",
                alert=True,
            )
        )
        assert chips.segment_text("cost") == "$0.99"
        cost = next(segment for segment in chips._build_segments() if segment.name == "cost")
        assert cost.alert
