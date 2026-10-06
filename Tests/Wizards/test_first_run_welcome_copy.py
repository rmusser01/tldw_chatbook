"""TASK-34100.13 (entry-exit-handoff-18, E7): Welcome no longer oversells.

The time line is the approved setup-shape spec's D1 copy, the Full choice
says what Full adds, the privacy line is scoped to local models, the empty
error band is gone, Tab reaches Next before "Restore a backup" (which says
who it is for), and a "Reduce motion" choice applies at once.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, Checkbox, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
)
from tldw_chatbook.UI.Wizards.first_run_welcome_step import WelcomeStep

pytestmark = pytest.mark.bootstrap_profile

_BUNDLE = str(
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)

D1_TIME_LINE = (
    "Quick takes about 2 minutes if you have an API key or a local AI server "
    "running. You can change any of this later by running setup again."
)


class _Host(ConsolidatedCSSApp):
    CSS_PATH = _BUNDLE

    def __init__(self, wizard) -> None:
        super().__init__()
        self._wizard = wizard

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        self.push_screen(self._wizard)


def _wizard(app_config=None):
    app_instance = MagicMock()
    app_instance.app_config = {} if app_config is None else app_config
    return FirstRunSetupWizard(app_instance), app_instance


def _welcome_text(screen) -> str:
    step = screen.query_one(WelcomeStep)
    return "\n".join(str(static.render()) for static in step.query(Static))


@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
@pytest.mark.asyncio
async def test_welcome_copy_is_honest(size) -> None:
    wizard, _ = _wizard()
    app = _Host(wizard)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        text = _welcome_text(app.screen)
        full = str(app.screen.query_one("#setup-track-full").label)
        step = app.screen.query_one(WelcomeStep)
        empty_bands = [
            widget
            for widget in step.query(".setup-step-error")
            if not str(widget.render()).strip()
        ]
    assert D1_TIME_LINE in text
    assert "Full about 10" not in text
    assert "configure everything" not in full
    assert full.startswith("Full setup — adds ")
    assert "With a model running on this computer" in text
    assert "nothing leaves" not in text.lower()
    assert empty_bands == []


@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
@pytest.mark.asyncio
async def test_tab_reaches_next_before_restore_a_backup(size) -> None:
    wizard, _ = _wizard()
    app = _Host(wizard)
    async with app.run_test(size=size) as pilot:
        await pilot.pause(0.3)
        chain = [widget.id for widget in app.screen.focus_chain]
        restore = app.screen.query_one("#setup-backup-restore", Button)
        label = str(
            restore.parent.query_one(".setup-welcome-restore-label", Static).render()
        )
        assert restore.region.height > 0, "Restore is not on screen"
    assert chain.index("wizard-next") < chain.index("setup-backup-restore")
    assert chain.index("setup-track-choice") < chain.index("wizard-next")
    assert label == "Moving from another computer?"


@pytest.mark.asyncio
async def test_welcome_rows_leave_with_welcome() -> None:
    wizard, _ = _wizard()
    app = _Host(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        container = app.screen.query_one(SetupWizardContainer)
        footer = app.screen.query_one("#setup-welcome-footer")
        assert footer.display
        container.show_step(1)
        await pilot.pause()
        assert not footer.display
        container.show_step(0)
        await pilot.pause()
        assert footer.display


@pytest.mark.asyncio
async def test_reduce_motion_applies_live_and_is_written_only_when_ticked() -> None:
    wizard, app_instance = _wizard({"appearance": {"reduce_motion": False}})
    app = _Host(wizard)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        step = app.screen.query_one(WelcomeStep)
        container = app.screen.query_one(SetupWizardContainer)
        container.commit_config = AsyncMock(return_value=True)
        step.wizard = container
        ok, _ = await step.commit()
        assert ok
        container.commit_config.assert_not_awaited()  # untouched: nothing written

        box = app.screen.query_one("#setup-welcome-reduce-motion", Checkbox)
        box.value = True
        await pilot.pause()
        assert app_instance.app_config["appearance"]["reduce_motion"] is True
        ok, _ = await step.commit()
    assert ok
    container.commit_config.assert_awaited_once_with(
        {"appearance": {"reduce_motion": True}}
    )


def test_the_welcome_rows_keep_their_handlers_on_a_message_pump() -> None:
    """The Restore handler moved with its button (TASK-34100.1's frozen-list rule)."""
    from tldw_chatbook.UI.Wizards.first_run_welcome_step import WelcomeFooter

    registered = {
        handler.__name__
        for klass in WelcomeFooter.__mro__
        for handlers in vars(klass).get("_decorated_handlers", {}).values()
        for handler, _selectors in handlers
    }
    assert {"open_backup_restore", "_on_reduce_motion"} <= registered
