"""TASK-33001.7 AC#5: focusing a field never hides its committed value.

The 2026-09-26 model-config review saw inputs "clear to their placeholder on
focus" at 211x44. Traced: both Conversation-settings comboboxes (the model
and provider pickers) blanked their Input on focus to open an unfiltered
list. These probes drive real Tab presses at the review's size and read the
text the compositor actually painted, so a field that paints its placeholder
(or paints its value in the background colour) fails.
"""

from typing import ClassVar

import pytest
from textual.widgets import Collapsible, Input

from Tests.private_profile import private_profile_test
from Tests.UI.consolidated_css import APP_STYLESHEETS
from Tests.UI.test_console_session_settings import StyledModalHarness
from Tests.UI.test_destination_shells import DestinationHarness, _build_test_app
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Widgets.Console.console_settings_field_row import (
    CONNECTION_DISCLOSURE_ID,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsContextEstimate,
    ConsoleSettingsModal,
)

SIZE = (211, 44)


class _StyledSettingsHarness(DestinationHarness):
    CSS_PATH: ClassVar[list[str]] = [str(path) for path in APP_STYLESHEETS]


def _painted_cells(screen, widget) -> list[tuple[str, object]]:
    """(text, style) segments the compositor painted inside ``widget``."""
    strips = screen._compositor.render_strips()
    region = widget.region
    cells = []
    for row in range(region.y, min(region.bottom, len(strips))):
        x = 0
        for segment in strips[row]:
            if region.x <= x < region.right:
                cells.append((segment.text, segment.style))
            x += len(segment.text)
    return cells


def _assert_committed_value_painted(screen, field: Input, committed: str) -> None:
    cells = _painted_cells(screen, field)
    painted = "".join(text for text, _style in cells)
    assert committed in painted, (
        f"#{field.id} hid {committed!r} on focus; painted {painted.strip()!r}"
    )
    for text, style in cells:
        if committed[0] in text and style is not None and style.color is not None:
            assert style.color != style.bgcolor, f"#{field.id} value is invisible"


async def _tab_through(app, pilot, screen, committed: dict[str, str], presses: int):
    """Press Tab ``presses`` times, probing each focused committed field."""
    probed: set[str] = set()
    for _ in range(presses):
        await pilot.press("tab")
        await pilot.pause()
        field = app.focused
        if isinstance(field, Input) and committed.get(field.id):
            _assert_committed_value_painted(screen, field, committed[field.id])
            probed.add(field.id)
    return probed


@pytest.mark.asyncio
async def test_conversation_settings_fields_keep_committed_value_on_focus():
    app = StyledModalHarness()
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", base_url="http://127.0.0.1:9099"
    )
    async with app.run_test(size=SIZE) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"llama_cpp": ["model-a", "model-b"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
                can_save=True,
            ),
            callback=app.capture_saved_settings,
        )
        await pilot.pause()
        screen = app.screen
        # Read committed values with nothing focused (the modal focuses a
        # field on open, which is exactly the state under test).
        screen.set_focus(None)
        await pilot.pause(0.1)
        committed = {field.id: field.value for field in screen.query(Input)}
        assert committed["console-settings-temperature"] == "0.7"
        # TASK-33006.4: the model is shown in the MODEL row, not an input.
        assert "model-a" in str(
            screen.query_one("#console-settings-model-summary").render()
        )
        # TASK-33006.1: Connection follows the core fields as a closed
        # disclosure; open it so Tab reaches its fields too.
        screen.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible).collapsed = False
        await pilot.pause()

        probed = await _tab_through(app, pilot, screen, committed, presses=30)

    assert {"console-settings-temperature", "console-settings-base-url"} <= probed


@pytest.mark.asyncio
@private_profile_test
async def test_settings_provider_fields_keep_committed_value_on_focus(request):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {
            "api_url": "http://127.0.0.1:9099",
            "model": "model-a",
            "api_key_env_var": "LLAMA_CPP_API_KEY",
        }
    }
    host = _StyledSettingsHarness(app, "settings")
    async with host.run_test(size=SIZE) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = host.screen_stack[-1]
        committed = {field.id: field.value for field in screen.query(Input)}
        # TASK-33007.3, rewritten on purpose: Model is the Default model
        # picker's field; #settings-model-value is its hidden adapter.
        assert committed["model-search-picker-input"] == "model-a"

        probed = await _tab_through(host, pilot, screen, committed, presses=45)

    assert {
        "model-search-picker-input",
        "settings-provider-endpoint-value",
        "settings-provider-credential-env-var",
    } <= probed
