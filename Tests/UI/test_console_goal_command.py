"""Composer goal commands reach real launch review without ordinary chat sends."""

from types import SimpleNamespace

import pytest
import pytest_asyncio
from textual.widgets import Button, TextArea

from Tests.Chat.test_console_goal_dispatch import build_goal_rig
from Tests.Chat.test_goal_conversation_provisioning import stores as _stores_fixture
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_scope_service import (
    ChatConversationScopeService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Widgets.Console import ConsoleCommandPopup, ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_goal_setup_modal import ConsoleGoalSetupModal

stores = _stores_fixture


async def wait_until(pilot, predicate):
    """Wait for the asserted product state, with a finite wall-clock deadline."""
    import time

    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        await pilot.pause(0.02)
    assert predicate()


@pytest_asyncio.fixture
async def goal_console(stores, monkeypatch):
    seed, _store, _session, controller, co, gateway, calls = build_goal_rig(
        stores, monkeypatch
    )
    backend = _build_test_app()
    _configure_native_ready_console(backend)
    backend.chachanotes_db = stores[1].db
    backend.workspace_registry_service = stores[2]
    backend.chat_conversation_scope_service = ChatConversationScopeService(
        local_service=ChatConversationService(stores[1].db), server_service=None
    )
    # Only provider transport/resolution is synthetic; command dispatch, setup,
    # permissions, launch persistence and the goal coordinator remain real.
    backend.console_provider_gateway_factory = lambda: gateway
    notices = []
    backend.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(backend)
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            console = host.screen
            await wait_until(pilot, lambda: bool(console.query(ConsoleComposerBar)))
            console._goals.get_controller = lambda: controller
            console._goals.get_coordinator = lambda: co
            composer = console.query_one(ConsoleComposerBar)
            yield SimpleNamespace(
                host=host,
                pilot=pilot,
                console=console,
                composer=composer,
                co=co,
                seed=seed,
                calls=calls,
                notices=notices,
            )
    finally:
        await co.shutdown()
        await gateway.aclose()


@pytest.mark.asyncio
async def test_goal_completion_and_enter_open_review_then_persist_objective(
    goal_console,
):
    """Catches missing popup/dispatch wiring, lost objective and premature work."""
    ui = goal_console
    ui.console._focus_console_composer_if_needed(force=True)
    await ui.pilot.press("/", "g", "o")
    popup = ui.console.query_one(ConsoleCommandPopup)
    await wait_until(ui.pilot, lambda: popup.is_open)
    assert [item.label for item in popup._suggestions] == ["/goal"]
    await ui.pilot.press("tab")
    await wait_until(ui.pilot, lambda: ui.composer.draft_text() == "/goal ")
    assert ui.host.screen is ui.console
    await ui.pilot.press(*"Repair fixture", "enter")
    await wait_until(
        ui.pilot, lambda: isinstance(ui.host.screen, ConsoleGoalSetupModal)
    )
    modal = ui.host.screen
    assert modal.query_one("#goal-objective", TextArea).text == "Repair fixture"
    assert modal.request.provider == ui.seed.request.provider
    assert modal.request.binding == ui.seed.request.binding
    assert modal.request.human_review_required
    assert modal.request.policy.iterations > 0
    assert not ui.calls
    assert len(ui.co.service.list_goals()) == 1

    modal.query_one("#goal-criteria", TextArea).load_text("The fixture is repaired")
    review = modal.query_one("#goal-start", Button)
    await wait_until(ui.pilot, lambda: not review.disabled)
    review.press()
    await wait_until(
        ui.pilot, lambda: modal._submitted is not None and not review.disabled
    )
    assert not ui.calls
    assert len(ui.co.service.list_goals()) == 1
    review.press()
    await wait_until(ui.pilot, lambda: len(ui.co.service.list_goals()) == 2)
    entry = next(goal for goal in ui.co.service.list_goals() if goal.id != ui.seed.id)
    saved = ui.co.service.get(entry.id)
    assert saved.request.objective == "Repair fixture"
    assert saved.request.criteria == "The fixture is repaired"
    assert saved.request.human_review_required
    assert saved.request.provider == ui.seed.request.provider
    assert saved.request.binding == ui.seed.request.binding
    assert ui.composer.draft_text() == "/goal Repair fixture"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("draft", "objective"),
    [
        (
            "/GoAl   Fix `fixture.txt`\n  Keep the second line.  ",
            "Fix `fixture.txt`\n  Keep the second line.",
        ),
        ("/goal", "Describe the result you want"),
    ],
)
async def test_goal_description_and_cancel_preserve_the_draft(
    goal_console, draft, objective
):
    """Catches command text truncation, bare-command fallthrough and draft loss."""
    ui = goal_console
    ui.composer.load_draft(draft)
    ui.console.query_one("#console-send-message", Button).press()
    await wait_until(
        ui.pilot, lambda: isinstance(ui.host.screen, ConsoleGoalSetupModal)
    )
    assert ui.host.screen.query_one("#goal-objective", TextArea).text == objective
    ui.host.screen.query_one("#goal-cancel", Button).press()
    await wait_until(ui.pilot, lambda: ui.host.screen is ui.console)
    assert ui.composer.draft_text() == draft
    assert not ui.calls
    assert len(ui.co.service.list_goals()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked", ["disabled", "missing_binding"])
async def test_goal_refusal_preserves_draft_and_never_arms_literal_send(
    goal_console, stores, monkeypatch, blocked
):
    """Catches a known /goal becoming an ordinary send after repeated refusal."""
    ui = goal_console
    if blocked == "disabled":
        monkeypatch.setenv("TLDW_AGENTS_GOAL_RUNS_ENABLED", "false")
    else:
        stores[2].remove_runtime_binding("binding")
    ui.composer.load_draft("/goal Repair fixture")
    send = ui.console.query_one("#console-send-message", Button)
    for attempt in range(2):
        send.press()
        await wait_until(ui.pilot, lambda attempt=attempt: len(ui.notices) > attempt)
        assert ui.host.screen is ui.console
        assert ui.composer.draft_text() == "/goal Repair fixture"
        assert ui.console._console_unknown_send_armed is None
        assert not ui.calls
        assert len(ui.co.service.list_goals()) == 1
    assert (
        "Enable Goal runs" in ui.notices[-1]
        if blocked == "disabled"
        else "ready local folder" in ui.notices[-1]
    )
