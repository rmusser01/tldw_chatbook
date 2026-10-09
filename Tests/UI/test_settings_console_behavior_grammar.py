"""Settings ▸ Console Behavior reads with the Providers & Models card's grammar (TASK-33007).

The Phase 7 full-screen captures (qa/model-config-33007-captures, review notes
9 and 10) found Console Behavior still in the old layout: two nested frames and
a rounded box around the replay override, Selects that span the card beside
narrow fallback Selects, two rows both labelled "Replay", a cut "Default chat
display name" label, and config-key prose ("Fallback source", "Save targets")
left in the card. Mounted with the real application stylesheet at 211x44.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.widgets import Collapsible, Input, Select, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _static_text
from Tests.UI.test_settings_advanced_disclosures import _framed
from Tests.UI.test_settings_console_fallback_rows import _SIZE, _app, _open
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from tldw_chatbook.UI.Screens.settings_screen import CONSOLE_DEFAULT_FIELD_NAMES
from tldw_chatbook.UI.Settings_Modules.settings_field_rows import UNSET_FALLBACK_HELP


def _shown(widget) -> bool:
    return widget.display and all(node.display for node in widget.ancestors)


async def _open_card(host, pilot):
    screen = await _open(host, pilot)
    card = screen.query_one("#settings-console-behavior-card")
    for disclosure in card.query(Collapsible):
        disclosure.collapsed = False
    await pilot.pause()
    await pilot.pause()
    return screen, card


def _row_label(control) -> str:
    return _static_text(control.parent.query_one(".settings-input-label", Static))


@pytest.mark.asyncio
@private_profile_test
async def test_console_behavior_draws_one_frame_and_no_boxes(request):
    """Review note 9: the detail pane's border is the only frame -- the card,
    its wrapper, its disclosures and its instant-apply group draw none."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen, card = await _open_card(host, pilot)
        boxes = [
            screen.query_one("#settings-console-behavior-detail"),
            card,
            *card.query(Collapsible),
            *card.query(".settings-instant-apply-group"),
        ]
        assert len(boxes) >= 4, boxes
        assert [box.id or repr(box) for box in boxes if _framed(box)] == []
        override = card.query_one("#settings-console-reasoning-override")
        title = override.query_ancestor(Collapsible).query_one("CollapsibleTitle")
        assert title.region.height == 1


@pytest.mark.asyncio
@private_profile_test
async def test_every_console_behavior_control_shares_one_width_and_shows_whole_labels(
    request,
):
    """Review note 9: every one-row Input and Select sits in one control
    column -- none spans the card -- and that column is wide enough for the
    longest option of every Select; every row label is painted whole
    ("Default chat display name" was cut to "Default chat display")."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        controls = [
            control
            for control in card.query("Input, Select")
            if _shown(control) and control.parent.has_class("settings-input-row")
        ]
        assert len(controls) >= 30, len(controls)
        widths = {control.id: control.region.width for control in controls}
        assert len(set(widths.values())) == 1, widths
        for select in (c for c in controls if isinstance(c, Select)):
            longest = max(cell_len(str(prompt)) for prompt, _value in select._options)
            shown = select.query_one("SelectCurrent #label", Static)
            assert shown.region.width >= longest, (select.id, shown.region, longest)
        for control in controls:
            label = control.parent.query_one(".settings-input-label", Static)
            assert label.region.width >= cell_len(_static_text(label)), (
                control.id,
                _static_text(label),
            )
        # The wider column still leaves the built-in fallback's help whole.
        unset = card.query_one("#settings-console-default-streaming-help", Static)
        assert _static_text(unset) == UNSET_FALLBACK_HELP
        assert unset.region.width >= cell_len(UNSET_FALLBACK_HELP), unset.region


@pytest.mark.asyncio
@private_profile_test
async def test_the_replay_default_and_its_override_are_named_apart(request):
    """Review note 9: the global replay default and the per-model override
    were both labelled "Replay"; every row label on the card is now unique."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        default = _row_label(card.query_one("#settings-console-reasoning-history"))
        override = _row_label(card.query_one("#settings-console-reasoning-override"))
        assert default == "Default replay"
        assert override == "This model's replay"
        labels = [
            _static_text(label)
            for label in card.query(".settings-input-row > .settings-input-label")
            if _shown(label) and _static_text(label)
        ]
        assert len(labels) == len(set(labels)), sorted(labels)


@pytest.mark.asyncio
@private_profile_test
async def test_config_key_prose_lives_in_the_inspector_and_lists_every_fallback(
    request,
):
    """Review note 10: "Fallback source: [chat_defaults]..." and "Save
    targets: ..." left the card; the Inspector's closed "config key"
    disclosure names every fallback the card shows, not only four."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen, card = await _open_card(host, pilot)
        detail = screen.query_one("#settings-detail-pane-body")
        prose = [
            _static_text(widget)
            for widget in detail.query(Static)
            if "chat_defaults" in _static_text(widget)
            or "[console]" in _static_text(widget)
            or _static_text(widget).startswith(("Fallback source", "Save targets"))
        ]
        assert prose == []

        disclosure = screen.query_one(
            "#settings-console-behavior-config-key", Collapsible
        )
        assert disclosure.collapsed is True
        keys = " ".join(
            _static_text(widget) for widget in disclosure.query(Static)
        ).replace("\n", "")
        shown = [
            CONSOLE_DEFAULT_FIELD_NAMES[str(control.id)]
            for control in card.query("Input, Select")
            if str(control.id) in CONSOLE_DEFAULT_FIELD_NAMES
        ]
        assert len(shown) == len(CONSOLE_DEFAULT_FIELD_NAMES)
        assert screen.query_one("#settings-console-default-user-display-name", Input)
        missing = [
            name for name in ("user_display_name", *shown) if f" {name}" not in keys
        ]
        assert missing == [], keys
        assert "chat_defaults" in keys
