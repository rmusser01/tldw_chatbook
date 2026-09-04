"""TASK-361: live-resize reflow convergence + stale-overlay dismissal.

The review saw a live browser-viewport resize (900x620 -> 700x480) leave the
rail full-width with the transcript/inspector gone and a nav tooltip stuck over
the header, whereas a cold start at the same size was fine. On a native resize
the pane reflow converges to the cold-start layout (locked here); the resize now
also dismisses any visible tooltip so a mounted overlay can't survive the repaint.
"""

from dataclasses import replace

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, Static, Tooltip

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
)
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
)
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultDurabilityState,
    ConsoleDefaultMutationIntent,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultSavePhase,
    ConsoleEndpointPatch,
)
from tldw_chatbook.UI.Console_Modules.left_rail import (
    CONTEXT_SECTION_DESCRIPTORS,
    ConsoleLeftRail,
)
from tldw_chatbook.UI.Console_Modules.right_rail import ConsoleInspectorRail
from tldw_chatbook.Widgets.Console.console_bounded_section import ConsoleBoundedSection
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal

_PANES = (
    "#console-left-rail",
    "#console-transcript-surface",
    "#console-native-composer",
)


def _ready_console_host() -> ConsoleHarness:
    """Build a Console whose setup modal cannot steal resize-test focus."""

    app = _build_test_app()
    _configure_native_ready_console(app)
    return ConsoleHarness(app)


class _ProductionResizeConsoleHarness(ConsoleHarness):
    """Real ChatScreen hierarchy with the exact application stylesheet stack."""

    CSS_PATH = TldwCli.CSS_PATH


class _ProductionResizeModalHarness(ConsolidatedCSSApp):
    """Standalone modal harness with the complete production app CSS bundle."""

    CSS_PATH = TldwCli.CSS_PATH


def _ready_production_console_host() -> _ProductionResizeConsoleHarness:
    app = _build_test_app()
    _configure_native_ready_console(app)
    return _ProductionResizeConsoleHarness(app)


def _resize_popover() -> ConsoleModelPopover:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    origin = ConsoleSettingsOrigin("session-a", None, 0)
    draft = ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=origin.session_id,
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )

    return ConsoleModelPopover(
        origin=origin,
        app_config={"api_settings": {"llama_cpp": {}}},
        initial_draft=draft,
        providers_models={"llama_cpp": ["model-a"]},
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=lambda state, **_kwargs: state,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: ConsoleSettingsReadiness(
            "Ready", "Ready.", True
        ),
    )


def _resize_full_settings(
    *,
    focus_model: bool = False,
    focus_context: bool = False,
    transfer: ConsoleSettingsTransfer | None = None,
) -> ConsoleSettingsModal:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    return ConsoleSettingsModal(
        settings=settings,
        transfer=transfer,
        app_config={
            "chat_defaults": {"provider": "llama_cpp", "model": "model-a"},
            "api_settings": {"llama_cpp": {}},
        },
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        focus_model=focus_model,
        focus_context=focus_context,
        default_readiness_resolver=lambda _provider, _model: ConsoleSettingsReadiness(
            "Ready", "Ready.", True
        ),
    )


def _failed_default_state(
    phase: ConsoleDefaultSavePhase,
) -> ConsoleDefaultDurabilityState:
    intent = ConsoleDefaultMutationIntent(
        generation=7,
        action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        provider_config_key="llama_cpp",
        literal_model_id="vendor/private:model",
        field_mask=FULL_MODEL_DEFAULT_FIELDS,
        values={name: None for name in FULL_MODEL_DEFAULT_FIELDS},
        endpoint_patch=None,
    )
    return ConsoleDefaultDurabilityState(
        newest_intent_generation=7,
        recovery_intent=intent,
        failure_phase=phase,
    )


def _long_failed_default_state(
    phase: ConsoleDefaultSavePhase,
) -> ConsoleDefaultDurabilityState:
    """Return a valid recovery whose safe summary must scroll at 60/72."""

    state = _failed_default_state(phase)
    assert state.recovery_intent is not None
    hostname = ".".join(("a" * 63, "b" * 63, "c" * 63, "d" * 61))
    model_id = f"vendor/{'m' * 249}"
    assert len(hostname) == 253
    assert len(model_id) == 256
    return replace(
        state,
        recovery_intent=replace(
            state.recovery_intent,
            literal_model_id=model_id,
            endpoint_patch=ConsoleEndpointPatch(
                value=f"https://{hostname}:8443/v1?token=not-rendered",
                bound_provider_config_key="llama_cpp",
                dirty=True,
                checked=True,
            ),
        ),
    )


def _assert_real_mouse_target(modal, button: Button) -> None:
    x = button.region.x + button.region.width // 2
    for y in (
        button.region.y,
        button.region.y + button.region.height // 2,
    ):
        assert modal.get_widget_at(x, y)[0] is button


def _assert_non_overlapping_regions(buttons: list[Button]) -> None:
    assert all(
        button.region.width > 0 and button.region.height > 0 for button in buttons
    )
    for index, button in enumerate(buttons):
        assert all(
            not button.region.overlaps(other.region) for other in buttons[index + 1 :]
        )


def _pane_layout(console) -> dict:
    """Return the display state of the required Console panes plus compact.

    Queries every pane directly (no swallowing): a missing selector raises and
    fails the test loudly rather than degrading to ``None`` and passing.
    """
    layout = {
        selector: bool(console.query_one(selector).display) for selector in _PANES
    }
    layout["compact"] = console.query_one("#console-shell").has_class(
        "-console-compact"
    )
    return layout


async def _wait_for_context_condition(
    pilot,
    condition,
    *,
    attempts: int = 20,
) -> None:
    """Wait through bounded refresh turns until one Context condition is stable."""

    stable_passes = 0
    for _ in range(attempts):
        await pilot.pause()
        if condition():
            stable_passes += 1
            if stable_passes == 2:
                return
        else:
            stable_passes = 0
    pytest.fail("Context condition did not stabilize within the refresh bound")


def _context_allocation_idle(rail: ConsoleLeftRail) -> bool:
    return not rail._allocation_reconcile_scheduled and all(
        not section._reconcile_scheduled
        for section in rail.query(ConsoleBoundedSection)
    )


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.asyncio
async def test_popover_actions_remain_reachable_and_ordered_at_narrow_width(
    width: int,
) -> None:
    app = ConsolidatedCSSApp()
    modal = _resize_popover()

    async with app.run_test(size=(width, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        panel = modal.query_one("#console-model-popover")
        main = list(modal.query("#console-popover-main-actions Button"))
        assert [str(button.label) for button in main] == [
            "Cancel",
            "Full settings…",
            "Defaults…",
            "Apply to this chat",
        ]
        assert panel.region.x >= 0
        assert panel.region.right <= width
        assert panel.region.bottom <= 24
        assert all(panel.region.contains_region(button.region) for button in main)
        assert all(button.can_focus and not button.disabled for button in main)
        _assert_non_overlapping_regions(main)
        main[0].focus()
        await pilot.pause()
        main_focus_order: list[str] = []
        for _ in main:
            focused = app.focused
            main_focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert main_focus_order == [
            "console-popover-cancel",
            "console-popover-full-settings",
            "console-popover-defaults",
            "console-popover-apply",
        ]

        await pilot.click("#console-popover-defaults")
        await pilot.pause()
        await pilot.pause()
        defaults = list(modal.query("#console-popover-default-actions Button"))
        assert [str(button.label) for button in defaults] == [
            "Save as model default",
            "Make default for new chats",
            "Back",
        ]
        assert all(panel.region.contains_region(button.region) for button in defaults)
        assert all(button.can_focus and not button.disabled for button in defaults)
        _assert_non_overlapping_regions(defaults)
        defaults_focus_order: list[str] = []
        for _ in defaults:
            focused = app.focused
            defaults_focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert defaults_focus_order == [
            "console-popover-save-model-default",
            "console-popover-make-new-chat-default",
            "console-popover-defaults-back",
        ]
        defaults[0].focus()
        await pilot.pause()
        assert app.focused is defaults[0]


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.asyncio
async def test_full_settings_actions_remain_mouse_reachable_at_narrow_width(
    width: int,
) -> None:
    """Every full-Settings action stays painted inside the production modal."""

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings()
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        panel = modal.query_one("#console-settings-modal")
        actions = list(modal.query("#console-settings-actions Button"))
        assert [str(button.label) for button in actions] == [
            "Cancel",
            "Save as provider defaults",
            "Default for new chats",
            "Use for this conversation",
        ]
        assert panel.region.x >= 0
        assert panel.region.right <= width
        assert panel.region.bottom <= 24
        assert all(button.display for button in actions)
        assert all(button.can_focus and not button.disabled for button in actions)
        assert all(
            panel.content_region.contains_region(button.region) for button in actions
        ), (
            panel.content_region,
            [(button.id, button.region) for button in actions],
        )
        _assert_non_overlapping_regions(actions)

        actions[1].focus()
        await pilot.pause()
        focus_order: list[str] = []
        for _ in actions:
            focused = app.focused
            focus_order.append(getattr(focused, "id", "") or "")
            assert focused is not None
            assert panel.region.contains_region(focused.region)
            await pilot.press("tab")
            await pilot.pause()
        assert focus_order == [
            "console-settings-save-default",
            "console-settings-make-default",
            "console-settings-save",
            "console-settings-cancel",
        ]
        apply_button = actions[-1]
        top_hit = modal.get_widget_at(
            apply_button.region.x + apply_button.region.width // 2,
            apply_button.region.y,
        )[0]
        center_hit = modal.get_widget_at(
            apply_button.region.x + apply_button.region.width // 2,
            apply_button.region.y + apply_button.region.height // 2,
        )[0]
        assert top_hit is apply_button
        assert center_hit is apply_button
        assert await pilot.click("#console-settings-save") is True
        await pilot.pause()
        assert modal not in app.screen_stack


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.parametrize(
    ("phase", "button_id", "expected_action"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-retry",
            ConsoleDefaultRecoveryAction.RETRY_SAVE,
        ),
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-discard",
            ConsoleDefaultRecoveryAction.DISCARD_RETRY,
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-refresh",
            ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-dismiss",
            ConsoleDefaultRecoveryAction.DISMISS_REFRESH,
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_recovery_actions_are_mouse_reachable_at_narrow_width(
    width: int,
    phase: ConsoleDefaultSavePhase,
    button_id: str,
    expected_action: ConsoleDefaultRecoveryAction,
) -> None:
    """Every phase-appropriate recovery action is a real narrow mouse target."""

    requests: list[ConsoleDefaultRecoveryRequest] = []

    async def recover(
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        requests.append(request)
        return ConsoleDefaultDurabilityState(newest_intent_generation=7)

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings()
    modal._default_durability_state = _long_failed_default_state(phase)
    modal._default_recovery_handler = recover
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        body = modal.query_one("#console-settings-body")
        panel = modal.query_one("#console-settings-modal")
        recovery = modal.query_one("#console-settings-default-recovery")
        summary = modal.query_one("#console-settings-default-recovery-summary")
        fold = modal.query_one("#console-settings-fold-hint", Static)
        applicable = [
            button
            for button in modal.query(
                "#console-settings-default-recovery-actions Button"
            )
            if button.display
        ]
        assert recovery.display
        assert not body.content_region.contains_region(summary.region)
        assert body.max_scroll_y > 0
        assert fold.display
        assert "recovery summary" in str(fold.renderable)
        assert "token=not-rendered" not in str(summary.renderable)
        assert len(applicable) == 2
        assert all(
            panel.content_region.contains_region(button.region) for button in applicable
        ), (
            panel.content_region,
            [(button.id, button.region) for button in applicable],
        )
        for button in applicable:
            _assert_real_mouse_target(modal, button)

        assert await pilot.click(f"#{button_id}") is True
        await pilot.pause()

    assert requests == [ConsoleDefaultRecoveryRequest(expected_action, 7)]


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.parametrize(
    ("phase", "first_button_id", "second_button_id"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-retry",
            "console-settings-default-discard",
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-refresh",
            "console-settings-default-dismiss",
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_recovery_keyboard_scrolls_and_stays_in_phase_actions(
    width: int,
    phase: ConsoleDefaultSavePhase,
    first_button_id: str,
    second_button_id: str,
) -> None:
    """Page keys read the exact summary; Tab cannot reach hidden draft commits."""

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings()
    modal._default_durability_state = _long_failed_default_state(phase)
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        body = modal.query_one("#console-settings-body")
        first = modal.query_one(f"#{first_button_id}", Button)
        second = modal.query_one(f"#{second_button_id}", Button)
        cancel = modal.query_one("#console-settings-cancel", Button)
        assert app.focused is first
        assert body.scroll_y == 0
        assert body.max_scroll_y > 0

        await pilot.press("pagedown")
        await pilot.pause()
        assert body.scroll_y > 0
        assert app.focused is first

        await pilot.press("end")
        await pilot.pause()
        assert body.scroll_y == body.max_scroll_y
        assert app.focused is first

        assert not modal.query_one("#console-settings-save-default", Button).display
        assert not modal.query_one("#console-settings-make-default", Button).display
        assert not modal.query_one("#console-settings-save", Button).display
        assert cancel.display
        _assert_real_mouse_target(modal, cancel)

        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is second
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is cancel
        await pilot.press("tab")
        await pilot.pause()
        assert app.focused is first


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.parametrize(
    ("phase", "first_button_id", "expected_action"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-retry",
            ConsoleDefaultRecoveryAction.RETRY_SAVE,
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-refresh",
            ConsoleDefaultRecoveryAction.REFRESH_RUNNING_APP,
        ),
    ),
)
@pytest.mark.parametrize(
    ("focus_model", "focus_context", "restored_focus_id"),
    (
        (True, False, "model-search-picker-input"),
        (False, True, "console-context-budget-mode"),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_recovery_restores_visible_focus_after_keyboard_success(
    width: int,
    phase: ConsoleDefaultSavePhase,
    first_button_id: str,
    expected_action: ConsoleDefaultRecoveryAction,
    focus_model: bool,
    focus_context: bool,
    restored_focus_id: str,
) -> None:
    """Recovery owns initial focus, then returns it to the requested editor."""

    requests: list[ConsoleDefaultRecoveryRequest] = []

    async def recover(
        request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        requests.append(request)
        return ConsoleDefaultDurabilityState(newest_intent_generation=7)

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings(
        focus_model=focus_model,
        focus_context=focus_context,
    )
    modal._default_durability_state = _long_failed_default_state(phase)
    modal._default_recovery_handler = recover
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        focused = app.focused
        assert focused is modal.query_one(f"#{first_button_id}", Button)
        _assert_real_mouse_target(modal, focused)

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert requests == [ConsoleDefaultRecoveryRequest(expected_action, 7)]
        assert not modal.query_one("#console-settings-default-recovery").display
        restored = app.focused
        assert restored is not None
        assert restored.id == restored_focus_id
        assert restored.region.width > 0 and restored.region.height > 0
        body = modal.query_one("#console-settings-body")
        assert body.content_region.contains_region(restored.region)
        hit = modal.get_widget_at(
            restored.region.x + restored.region.width // 2,
            restored.region.y + restored.region.height // 2,
        )[0]
        assert hit is restored or restored in hit.ancestors


@pytest.mark.parametrize(
    ("phase", "first_button_id"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-retry",
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-refresh",
        ),
    ),
)
@pytest.mark.parametrize("outcome", ("exception", "invalid", "same-phase"))
@pytest.mark.asyncio
async def test_full_settings_failed_recovery_refocuses_enabled_phase_action(
    phase: ConsoleDefaultSavePhase,
    first_button_id: str,
    outcome: str,
) -> None:
    """A failed recovery keeps one visible keyboard action ready to retry."""

    failed_state = _failed_default_state(phase)

    async def recover(_request: ConsoleDefaultRecoveryRequest):
        if outcome == "exception":
            raise RuntimeError("injected recovery failure")
        if outcome == "invalid":
            return object()
        return failed_state

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings()
    modal._default_durability_state = failed_state
    modal._default_recovery_handler = recover
    async with app.run_test(size=(60, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        first = modal.query_one(f"#{first_button_id}", Button)
        assert app.focused is first
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert modal.query_one("#console-settings-default-recovery").display
        assert first.display and not first.disabled
        assert app.focused is first
        _assert_real_mouse_target(modal, first)


@pytest.mark.parametrize(
    ("phase", "first_button_id"),
    (
        (
            ConsoleDefaultSavePhase.BEFORE_REPLACE,
            "console-settings-default-retry",
        ),
        (
            ConsoleDefaultSavePhase.CACHE_PUBLICATION,
            "console-settings-default-refresh",
        ),
    ),
)
@pytest.mark.asyncio
async def test_full_settings_successful_recovery_clears_prior_error_banner(
    phase: ConsoleDefaultSavePhase,
    first_button_id: str,
) -> None:
    """A successful retry removes the error from its previous failed attempt."""

    attempts = 0

    async def recover(
        _request: ConsoleDefaultRecoveryRequest,
    ) -> ConsoleDefaultDurabilityState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected recovery failure")
        return ConsoleDefaultDurabilityState(newest_intent_generation=7)

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings(focus_model=True)
    modal._default_durability_state = _failed_default_state(phase)
    modal._default_recovery_handler = recover
    async with app.run_test(size=(60, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.pause()

        first = modal.query_one(f"#{first_button_id}", Button)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        error = modal.query_one("#console-settings-error", Static)
        assert error.display
        assert "failed" in str(error.renderable).lower()

        await pilot.press("tab")
        await pilot.pause()
        await pilot.press("shift+tab")
        await pilot.pause()
        assert app.focused is first
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert not modal.query_one("#console-settings-default-recovery").display
        assert not error.display
        assert str(error.renderable) == ""


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.parametrize(
    "button_id",
    (
        "console-settings-cancel",
        "console-settings-save-default",
        "console-settings-save",
    ),
)
@pytest.mark.asyncio
async def test_blocked_full_settings_keeps_enabled_actions_mouse_reachable(
    width: int,
    button_id: str,
) -> None:
    """Blocked Make Default must not displace the other full-modal actions."""

    app = _ProductionResizeModalHarness()
    modal = _resize_full_settings()
    modal._default_readiness_resolver = lambda _provider, _model: (
        ConsoleSettingsReadiness(
            "Missing key",
            "Provider is not configured for native Console sending.",
            False,
        )
    )
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        panel = modal.query_one("#console-settings-modal")
        blocked = modal.query_one("#console-settings-new-chat-default-block")
        actions = list(modal.query("#console-settings-actions Button"))
        assert blocked.display
        assert "not configured" in str(blocked.renderable)
        assert modal.query_one("#console-settings-make-default", Button).disabled
        assert all(
            panel.content_region.contains_region(button.region) for button in actions
        )
        for button in actions:
            _assert_real_mouse_target(modal, button)

        assert await pilot.click(f"#{button_id}") is True
        await pilot.pause()
        assert modal not in app.screen_stack


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.asyncio
async def test_blocked_quick_transfer_keeps_initial_model_focus_visible(
    width: int,
) -> None:
    """Blocked explanation stays discoverable without scrolling transfer focus away."""

    app = _ProductionResizeModalHarness()
    transfers: list[ConsoleSettingsTransfer | None] = []
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(_resize_popover(), callback=transfers.append)
        await pilot.pause()
        assert await pilot.click("#console-popover-full-settings") is True
        await pilot.pause()
        assert len(transfers) == 1
        assert isinstance(transfers[0], ConsoleSettingsTransfer)

        modal = _resize_full_settings(focus_model=True, transfer=transfers[0])
        modal._default_readiness_resolver = lambda _provider, _model: (
            ConsoleSettingsReadiness(
                "Missing key",
                "Provider is not configured for native Console sending.",
                False,
            )
        )
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        await pilot.pause()

        body = modal.query_one("#console-settings-body")
        blocked = modal.query_one("#console-settings-new-chat-default-block", Static)
        fold = modal.query_one("#console-settings-fold-hint", Static)
        focused = app.focused
        assert focused is not None
        assert focused.id == "model-search-picker-input"
        assert body.content_region.contains_region(focused.region)
        assert blocked.display
        assert "not configured" in str(blocked.renderable)
        assert body.max_scroll_y > 0
        assert fold.display
        assert "scroll" in str(fold.renderable).lower()


@pytest.mark.parametrize("width", (60, 72))
@pytest.mark.parametrize(
    ("blocked", "commit_button_id"),
    (
        (False, "console-popover-make-new-chat-default"),
        (True, "console-popover-save-model-default"),
    ),
)
@pytest.mark.asyncio
async def test_quick_defaults_reveals_intent_before_narrow_commit(
    width: int,
    blocked: bool,
    commit_button_id: str,
) -> None:
    """Defaults intent and any block reason are visible above pinned actions."""

    app = _ProductionResizeModalHarness()
    modal = _resize_popover()
    if blocked:
        modal._default_readiness_resolver = lambda _provider, _model: (
            ConsoleSettingsReadiness(
                "Missing key",
                "Provider is not configured for native Console sending.",
                False,
            )
        )
    async with app.run_test(size=(120, 24)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        await pilot.resize_terminal(width, 24)
        await pilot.pause()
        assert await pilot.click("#console-popover-defaults") is True
        await pilot.pause()
        await pilot.pause()

        body = modal.query_one("#console-model-popover-body")
        panel = modal.query_one("#console-popover-defaults-panel")
        assert body.content_region.contains_region(panel.region)
        assert "Defaults target: llama_cpp/model-a" in str(
            modal.query_one("#console-popover-defaults-target", Static).renderable
        )
        assert "Compaction stays with this chat" in str(
            modal.query_one(
                "#console-popover-defaults-compaction-scope", Static
            ).renderable
        )
        block = modal.query_one("#console-popover-new-chat-default-block", Static)
        assert block.display is blocked
        if blocked:
            assert "not configured" in str(block.renderable)

        actions = [
            button
            for button in modal.query("#console-popover-default-actions Button")
            if button.display
        ]
        for button in actions:
            _assert_real_mouse_target(modal, button)
        assert await pilot.click(f"#{commit_button_id}") is True
        await pilot.pause()
        assert modal not in app.screen_stack


@pytest.mark.asyncio
async def test_resize_auto_open_does_not_evict_focused_context() -> None:
    """Crossing 117-to-118 keeps focused Context instead of auto-swapping rails."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()
        assert pilot.app.focused is collapse

        await pilot.resize_terminal(118, 40)
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is False
        assert pilot.app.focused is collapse


@pytest.mark.asyncio
async def test_consecutive_resize_preserves_focused_context_when_auto_open_is_suppressed() -> (
    None
):
    """The protected Context rail and its focus survive adjacent width bands."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()

        await pilot.resize_terminal(118, 40)
        await pilot.pause(0.2)
        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert pilot.app.focused is collapse

        await pilot.resize_terminal(129, 40)
        await pilot.pause(0.2)

        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert console.query_one("#console-context-rail-handle").display is False
        assert collapse.display is True
        assert pilot.app.focused is collapse


@pytest.mark.asyncio
async def test_resize_priority_hands_inspector_focus_to_reveal_button() -> None:
    """Crossing 128-to-129 hides focused Inspector and focuses its handle."""
    host = _ready_console_host()

    async with host.run_test(size=(128, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-inspector-rail-collapse")
        collapse.focus()
        await pilot.pause()
        assert pilot.app.focused is collapse

        await pilot.resize_terminal(129, 40)
        await pilot.pause(0.2)

        reveal = console.query_one("#console-inspector-rail-open")
        assert console.query_one("#console-left-rail").display is True
        assert console.query_one("#console-right-rail").display is False
        assert reveal.display is True
        assert pilot.app.focused is reveal


@pytest.mark.asyncio
async def test_resize_event_width_drives_priority_and_focus(monkeypatch) -> None:
    """The Resize width wins over a stale screen-width lookup."""
    host = _ready_console_host()

    async with host.run_test(size=(117, 40)) as pilot:
        console = host.screen_stack[-1]
        collapse = console.query_one("#console-context-rail-collapse")
        collapse.focus()
        await pilot.pause()
        monkeypatch.setattr(console, "_console_rail_available_columns", lambda: 160)

        await pilot.resize_terminal(120, 40)
        await pilot.pause(0.2)

        rail_state = console._last_console_rail_state
        assert rail_state is not None
        assert rail_state.left_open is True
        assert rail_state.right_open is False
        assert rail_state.right_compact_override is False
        assert rail_state.compact_override is False
        assert pilot.app.focused is collapse


@pytest.mark.asyncio
async def test_console_live_resize_converges_to_cold_start_layout() -> None:
    """A live resize converges to the cold-start layout at that size.

    TASK-361 AC#1: after resizing down, the panes are all present and the header
    is compacted -- the same layout a cold start produces -- not the review's
    rail-full-width / panes-gone divergence.

    TASK-2154.1 (LY-08) changed WHAT the cold-start layout is at 90 cols: the
    left rail now force-collapses below 100 columns (rendering override)
    instead of overflowing the grid. The convergence contract itself --
    ``live == cold`` -- is unchanged.
    """
    cold_host = _ready_console_host()
    async with cold_host.run_test(size=(90, 30)) as pilot:
        cold_console = cold_host.screen_stack[-1]
        await pilot.pause()
        await pilot.pause()
        cold = _pane_layout(cold_console)

    live_host = _ready_console_host()
    async with live_host.run_test(size=(160, 48)) as pilot:
        live_console = live_host.screen_stack[-1]
        await pilot.pause()
        await pilot.resize_terminal(90, 30)
        await pilot.pause()
        await pilot.pause()
        live = _pane_layout(live_console)

    assert live == cold
    # The transcript and composer stay present and the header is compacted at
    # 30 rows; at 90 cols the left rail is force-collapsed by the TASK-2154.1
    # narrow-width rule, not eaten by / eating the grid.
    assert cold["#console-left-rail"] is False
    assert cold["#console-transcript-surface"] is True
    assert cold["#console-native-composer"] is True
    assert cold["compact"] is True


@pytest.mark.asyncio
async def test_console_resize_dismisses_stale_tooltip() -> None:
    """A live resize dismisses a visible tooltip overlay.

    TASK-361 AC#2: the review saw a nav tooltip stick over the header across
    reflows. With a tooltip shown, a resize must hide the real overlay widget so
    it cannot survive the repaint.
    """
    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await pilot.pause()

        # _clear_tooltip() (Textual Screen) hides the screen's Tooltip child;
        # ensure one exists and is shown, exactly as a hover would leave it.
        try:
            tooltip = console.get_child_by_type(Tooltip)
        except NoMatches:
            tooltip = Tooltip(id="textual-tooltip")
            await console.mount(tooltip)
            await pilot.pause()
        tooltip.display = True
        assert tooltip.display is True

        await pilot.resize_terminal(120, 40)
        await pilot.pause()

        assert tooltip.display is False


@pytest.mark.asyncio
async def test_model_summary_sync_invalidates_mounted_context_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production Model mutation seam refreshes bounded demand afterward."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        readiness = {
            "value": ConsoleSettingsReadiness("Ready", "", True),
        }
        monkeypatch.setattr(
            console,
            "_build_console_settings_summary_state",
            lambda: ConsoleSettingsSummaryState(
                provider_row="Provider: test",
                model_row="Model: test",
                context_row="Context: 0",
                sampling_row="T 0.7 · max_tokens 100",
                identity_row="Identity: character",
                readiness_label=readiness["value"].label,
                readiness=readiness["value"],
            ),
        )
        console._sync_console_settings_summary()
        rail.apply_section_open("model", True)
        rail.activate_section("model")
        await _wait_for_context_condition(
            pilot,
            lambda: (
                rail._active_section_id == "model"
                and not rail.query_one("#console-model-section-recovery").display
                and _context_allocation_idle(rail)
            ),
        )
        before_demand = model.desired_content_lines
        before_allocation = model.allocation
        recovery = rail.query_one("#console-model-section-recovery")
        reconcile_runs = 0
        original_reconcile = rail._run_allocation_reconcile

        def reconcile_spy() -> None:
            nonlocal reconcile_runs
            reconcile_runs += 1
            original_reconcile()

        monkeypatch.setattr(rail, "_run_allocation_reconcile", reconcile_spy)
        readiness["value"] = ConsoleSettingsReadiness(
            "Provider recovery required",
            "Provider configuration needs attention.",
            False,
        )
        console._sync_console_settings_summary()
        await _wait_for_context_condition(
            pilot,
            lambda: recovery.display and _context_allocation_idle(rail),
        )

        assert recovery.display is True
        assert model.desired_content_lines >= before_demand
        assert model.allocation == before_allocation
        assert reconcile_runs >= 1
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_height_resize_requests_one_coalesced_context_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public height-adaptation lifecycle invalidates after compact mutation."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        for descriptor in CONTEXT_SECTION_DESCRIPTORS:
            rail.apply_section_open(descriptor.section_id, True)
        await _wait_for_context_condition(
            pilot,
            lambda: _context_allocation_idle(rail),
        )
        sections = list(rail.query(ConsoleBoundedSection))
        before_allocations = tuple(section.allocation for section in sections)
        before_viewport_height = rail.query_one(
            "#console-left-rail-body"
        ).content_region.height

        helper_calls = 0
        reconcile_runs = 0
        original_helper = console._request_console_context_allocation_reconcile
        original_reconcile = rail._run_allocation_reconcile

        def helper_spy() -> None:
            nonlocal helper_calls
            helper_calls += 1
            original_helper()

        def reconcile_spy() -> None:
            nonlocal reconcile_runs
            reconcile_runs += 1
            original_reconcile()

        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            helper_spy,
        )
        monkeypatch.setattr(rail, "_run_allocation_reconcile", reconcile_spy)
        await pilot.resize_terminal(160, 30)
        await _wait_for_context_condition(
            pilot,
            lambda: (
                console.query_one("#console-shell").has_class("-console-compact")
                and _context_allocation_idle(rail)
            ),
        )
        after_allocations = tuple(section.allocation for section in sections)
        outer = rail.query_one("#console-left-rail-body")
        cue = rail.query_one("#console-left-rail-outer-hint", Static)
        assert console.query_one("#console-shell").has_class("-console-compact")
        assert helper_calls >= 1
        assert reconcile_runs == 1
        assert outer.content_region.height < before_viewport_height
        assert (
            before_allocations
            == after_allocations
            == tuple(None for _section in sections)
        )
        assert [section.max_content_lines for section in sections] == [
            descriptor.max_content_lines for descriptor in CONTEXT_SECTION_DESCRIPTORS
        ]
        assert all(
            sum(
                child.virtual_region_with_margin.height
                for child in section.children
                if child not in {section.viewport, section.hint} and child.display
            )
            + section.viewport.content_region.height
            == min(section.desired_content_lines, section.max_content_lines)
            for section in sections
        )
        assert str(outer.styles.overflow_y) == "auto"
        assert outer.max_scroll_y > 0
        assert cue.display is True
        last_header = rail.query_one("#console-rail-section-header-character")
        outer.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert last_header.region.overlaps(outer.content_region)
        stable_runs = reconcile_runs
        await pilot.pause()
        assert reconcile_runs == stable_runs
        assert rail._allocation_reconcile_scheduled is False


@pytest.mark.asyncio
async def test_production_bounded_rail_resize_reconciles_geometry_and_focus() -> None:
    """Resize, recompose, and shrink preserve honest local geometry and focus."""

    host = _ready_production_console_host()
    async with host.run_test(size=(160, 52)) as pilot:
        screen = host.screen_stack[-1]
        if not screen.query_one("#console-right-rail").display:
            assert await pilot.click("#console-inspector-rail-open")
        inspector = screen.query_one("#console-right-rail", ConsoleInspectorRail)
        sources = inspector.query_one(
            "#console-bounded-section-sources", ConsoleBoundedSection
        )
        target = Button("source action", id="production-resize-source-action")
        content = Static("\n".join(f"resize source {row}" for row in range(29)))
        await sources.viewport.remove_children()
        await sources.viewport.mount(content, target)
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 30
                and sources.viewport.content_region.height == 20
                and sources.viewport.max_scroll_y == 10
                and sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )

        target.focus()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                pilot.app.focused is target
                and sources.viewport.scroll_y == sources.viewport.max_scroll_y
            ),
        )
        hit = screen.get_widget_at(target.region.x + 1, target.region.y)[0]
        assert hit is target or target in hit.ancestors

        await pilot.resize_terminal(160, 45)
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.viewport.content_region.height == 20
                and sources.viewport.scroll_y <= sources.viewport.max_scroll_y
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert pilot.app.focused is target
        assert sources.region.contains_region(sources.viewport.region)
        assert sources.region.contains_region(sources.hint.region)

        original_section = sources
        content.update("\n".join(f"recomposed source {row}" for row in range(24)))
        await sources.recompose()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 25
                and sources.viewport.content_region.height == 20
                and sources.viewport.max_scroll_y == 5
                and sources.viewport.scroll_y == 5
                and sources.hint.display
                and not sources._reconcile_scheduled
            ),
        )
        assert (
            inspector.query_one(
                "#console-bounded-section-sources", ConsoleBoundedSection
            )
            is original_section
        )

        await content.remove()
        replacement = Button(
            "replacement source action", id="production-resize-source-replacement"
        )
        await sources.viewport.mount(
            Static("\n".join(f"shrunk source {row}" for row in range(9))),
            replacement,
        )
        await target.remove()
        sources.request_reconcile()
        inspector.request_outer_reconcile()
        await _wait_for_context_condition(
            pilot,
            lambda: (
                sources.desired_content_lines == 10
                and sources.viewport.content_region.height == 10
                and sources.viewport.scroll_y == 0
                and not sources.hint.display
                and not inspector._outer_reconcile_scheduled
            ),
        )
        assert pilot.app.focused is replacement


@pytest.mark.asyncio
async def test_public_close_active_falls_back_and_rail_reopen_keeps_local_state() -> (
    None
):
    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        rail = console.query_one("#console-left-rail", ConsoleLeftRail)
        for _ in range(5):
            await pilot.pause()

        workspace_toggle = rail.query_one(
            "#console-rail-section-toggle-workspace", Button
        )
        console._toggle_console_rail_section("workspace", next_open=True)
        rail.activate_section("workspace")
        for _ in range(4):
            await pilot.pause()
        assert rail._active_section_id == "workspace"

        workspace_toggle.scroll_visible(animate=False)
        await pilot.pause()
        assert await pilot.click(workspace_toggle)
        for _ in range(4):
            await pilot.pause()
        assert rail._active_section_id == "conversations"

        model = rail.query_one("#console-bounded-section-model", ConsoleBoundedSection)
        model_body = rail.query_one("#console-rail-section-body-model")
        overflow = Static("\n".join(f"line {index}" for index in range(30)))
        await model_body.mount(overflow)
        console._toggle_console_rail_section("model", next_open=True)
        rail.activate_section("model")
        for _ in range(6):
            await pilot.pause()
        model.viewport.scroll_to(y=3, animate=False, immediate=True)
        await pilot.pause()
        retained_offset = model.viewport.scroll_y
        assert retained_offset > 0

        assert await pilot.click("#console-context-rail-collapse")
        for _ in range(4):
            await pilot.pause()
        assert rail.display is False
        assert await pilot.click("#console-context-rail-open")
        for _ in range(6):
            await pilot.pause()

        assert rail.display is True
        assert rail._active_section_id == "model"
        assert model.viewport.scroll_y == retained_offset


@pytest.mark.asyncio
async def test_all_named_context_mutation_seams_request_the_mounted_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every production Context mutation seam delegates to the rail helper."""

    host = _ready_console_host()
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        requests: list[str] = []
        current_seam = ""

        def request_spy() -> None:
            requests.append(current_seam)

        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            request_spy,
        )

        def assert_requested(label: str, mutation) -> None:
            nonlocal current_seam
            current_seam = label
            before = len(requests)
            mutation()
            assert requests[before:] == [label]

        console._console_rail_system_line_last = ("stale", False)
        monkeypatch.setattr(
            console,
            "_console_rail_system_line_state",
            lambda: ("System: changed", False),
        )
        assert_requested("session settings", console._sync_console_rail_system_line)

        monkeypatch.setattr(console, "_sync_console_rail_system_line", lambda: None)
        monkeypatch.setattr(console, "_sync_console_agent_section", lambda: None)
        assert_requested("model rows", console._sync_console_settings_summary)

        monkeypatch.undo()
        requests.clear()
        monkeypatch.setattr(
            console,
            "_request_console_context_allocation_reconcile",
            request_spy,
        )
        console._console_agent_section_last = object()
        assert_requested(
            "agent status steps actions steering fleet and pinned summary",
            console._sync_console_agent_section,
        )
        assert_requested(
            "workspace conversations and details",
            console._sync_console_workspace_context,
        )

        current_seam = "character remount and reaction"
        before = len(requests)
        await console._render_character_avatar_into_section(
            spec=None,
            name="Changed character",
            manual_label="happy",
            is_current=lambda: True,
        )
        assert requests[before:] == [current_seam]

        rail_state = console._current_console_rail_state()
        assert_requested(
            "rail collapse and reopen",
            lambda: console._sync_console_rail_visibility(rail_state),
        )

        monkeypatch.setattr(console, "_set_console_rail_preference", lambda **_: None)
        assert_requested(
            "section toggles and open state",
            lambda: console._toggle_console_rail_section(
                "details",
                next_open=not rail_state.details_open,
            ),
        )

        # Let the alias sync queued by the workspace mutation finish before
        # exercising the alias-mount seam directly.
        for _ in range(3):
            await pilot.pause()
        aliases = list(console.query("#console-new-workspace-conversation"))
        if aliases and isinstance(aliases[0], Button):
            await aliases[0].remove()
        current_seam = "conversation alias mount"
        before = len(requests)
        await console._sync_console_legacy_workspace_context_aliases()
        assert requests[before:] == [current_seam]
