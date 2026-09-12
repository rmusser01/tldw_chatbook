"""Production-CSS geometry coverage for Console Library controls."""

from __future__ import annotations

import pytest
from textual.geometry import Region
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleLibraryPolicyDisplayState,
)
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Widgets.Console.console_library_access_modal import (
    ConsoleLibraryAccessModal,
    ConsoleLibraryPolicySaveOutcome,
)
from tldw_chatbook.Widgets.Console.console_library_search_modal import (
    ConsoleLibrarySearchModal,
)
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.UI.Workbench import WorkbenchActionRequested


def _snapshot() -> ConsoleLibraryPolicySnapshot:
    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=4,
        source="durable",
    )


async def _save(candidate) -> ConsoleLibraryPolicySaveOutcome:
    return ConsoleLibraryPolicySaveOutcome(
        status="saved",
        snapshot=ConsoleLibraryPolicySnapshot(
            auto_retrieve=candidate.auto_retrieve,
            assistant_access=candidate.assistant_access,
            policy_revision=5,
            source="durable",
        ),
        copy="Saved.",
    )


async def _reload() -> ConsoleLibraryPolicySnapshot:
    return _snapshot()


def _assert_inside_viewport(widget, viewport: Region) -> None:
    visible = widget.region.intersection(viewport)
    assert widget.region.width > 0 and widget.region.height > 0
    assert visible == widget.region, (
        f"#{widget.id} clipped: {widget.region!r} outside {viewport!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (48, 24)])
async def test_library_access_modal_keeps_body_and_conflict_actions_visible(
    size: tuple[int, int],
) -> None:
    snapshot = _snapshot()
    state = ConsoleLibraryPolicyDisplayState.from_snapshot(
        snapshot,
        provider_intent_label="Library tool mode: Direct 研究🙂 e\u0301 العربية",
        resolved_destination_label=(
            "Resolved destination: provider.example.invalid / " + "long-copy " * 8
        ),
        feedback="conflict",
        feedback_copy="Policy changed elsewhere. Compare the local and saved values.",
        dirty=True,
    )
    modal = ConsoleLibraryAccessModal(
        snapshot=snapshot,
        state=state,
        save_policy=_save,
        reload_policy=_reload,
    )
    app = ConsolidatedCSSApp()

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        viewport = Region(0, 0, *size)
        _assert_inside_viewport(modal.query_one("#console-library-access"), viewport)
        _assert_inside_viewport(
            modal.query_one(".console-library-access-actions"), viewport
        )
        for button in modal.query(".console-library-access-actions Button").results(
            Button
        ):
            if button.display:
                _assert_inside_viewport(button, viewport)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (48, 24)])
async def test_library_search_modal_and_actions_fit_the_viewport(
    size: tuple[int, int],
) -> None:
    modal = ConsoleLibrarySearchModal(
        query="研究🙂 e\u0301 العربية " + "draft " * 8,
        source_types=("notes", "media", "conversations", "prompts"),
        item_scope_summary="Scope: 10 items · 研究🙂 e\u0301 العربية",
    )
    app = ConsolidatedCSSApp()

    async with app.run_test(size=size) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        viewport = Region(0, 0, *size)
        box = modal.query_one("#console-rag-settings")
        _assert_inside_viewport(box, viewport)
        actions = modal.query_one(".console-rag-settings-actions")
        _assert_inside_viewport(actions, viewport)
        _assert_inside_viewport(actions, box.region)
        for button in actions.query(Button):
            _assert_inside_viewport(button, viewport)
            _assert_inside_viewport(button, box.region)
        item_scope = modal.query_one("#console-library-search-item-scope", Static)
        assert item_scope._render_markup is False
        _assert_inside_viewport(item_scope, viewport)
        for selector in (
            "#console-rag-settings-scope",
            ".console-rag-settings-hint",
        ):
            _assert_inside_viewport(modal.query_one(selector), viewport)
        source_toggles = list(
            modal.query(".console-rag-settings-source-toggle").results(Button)
        )
        assert len(source_toggles) == 4
        for button in source_toggles:
            _assert_inside_viewport(button, viewport)
            _assert_inside_viewport(button, box.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (48, 24)])
async def test_real_console_search_action_opens_the_complete_manual_surface(
    size: tuple[int, int],
) -> None:
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    draft = "  preserve 研究🙂 e\u0301 العربية spacing exactly  "

    async with host.run_test(size=size) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-command-input")
        composer = console.query_one(ConsoleComposerBar)
        composer.load_draft(draft)

        console.post_message(WorkbenchActionRequested("run-library-rag"))
        await pilot.pause()
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsoleLibrarySearchModal)
        assert modal._query == draft
        assert modal.query_one("#console-library-search-item-scope", Static)
        assert len(list(modal.query(".console-rag-settings-source-toggle"))) == 4
