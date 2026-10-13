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
    """Review note 9: every one-row Select and number Input sits in one
    control column -- none spans the card -- and that column is wide enough
    for the longest option of every Select; every row label is painted whole
    ("Default chat display name" was cut to "Default chat display").

    Rewritten on purpose (#3057 review R3 #10): the two free-text Side chat
    fields leave the column (test_settings_console_side_chat_field_width.py);
    so does Permission summaries' Model, a model id too (capture finding 21).
    """
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        controls = [
            control
            for control in card.query("Input, Select")
            if _shown(control) and control.parent.has_class("settings-input-row")
        ]
        assert len(controls) >= 30, len(controls)
        free_text = {
            "settings-console-sidechat-model",
            "settings-console-sidechat-prompt-template",
            "settings-permission-summary-model",
        }
        assert free_text <= {control.id for control in controls}
        widths = {
            control.id: control.region.width
            for control in controls
            if control.id not in free_text
        }
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


@pytest.mark.asyncio
@private_profile_test
async def test_console_behavior_prose_starts_where_its_section_headers_do(request):
    """Self-review of note 9: with the frames gone, the help lines without a
    class of their own started one cell left of the section headers and the
    detail rows ("Global keeps one arrangement..." under "Rail presentation");
    every prose line on the card now starts in the headers' column."""
    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        header_x = {
            header.content_region.x
            for header in card.query(".destination-section")
            if _shown(header)
        }
        assert len(header_x) == 1, header_x
        holders = (card, *card.query(".settings-instant-apply-group, Contents"))
        prose = {
            str(static.id or _static_text(static)[:30]): static.content_region.x
            for holder in holders
            for static in holder.children
            if isinstance(static, Static)
            and _shown(static)
            and _static_text(static).strip()
            and not static.has_class("destination-section")
        }
        assert len(prose) >= 15, prose
        assert {name: x for name, x in prose.items() if x not in header_x} == {}


@pytest.mark.asyncio
@private_profile_test
async def test_status_row_chat_images_and_summary_prompt_read_as_controls(request):
    """#3057 capture finding 12 (06b, 06c): "Above composer", "Disabled" and
    "Edit summary prompt…" were Buttons standing alone at the prose indent,
    bold on the same fill as the section headers above them, so nothing said
    they could be pressed; two of them were values posing as labels. Status
    row placement is a one-row Select in a labelled row, as Rail layout scope
    is; Chat images is a Checkbox, as Collapse large pastes is; the summary
    prompt's button sits in a labelled row and says where it goes."""
    from textual.widgets import Button, Checkbox

    host = _SettingsCssHarness(_app(), "settings")

    async with host.run_test(size=_SIZE) as pilot:
        _screen, card = await _open_card(host, pilot)
        position = card.query_one("#settings-console-status-row-position-toggle")
        images = card.query_one("#settings-console-remote-images-toggle")
        prompt = card.query_one("#settings-console-context-edit-summary-prompt")

        assert isinstance(position, Select) and position.value == "above"
        assert _row_label(position) == "Status row"
        assert isinstance(images, Checkbox) and images.value is False
        assert str(images.label) == "Render linked images"
        assert isinstance(prompt, Button)
        assert str(prompt.label) == "Edit in Internal Prompts…"
        assert _row_label(prompt) == "Summary prompt"
        assert {position.parent, prompt.parent}.isdisjoint({card})
        rail = card.query_one("#settings-console-rail-layout-scope", Select)
        assert position.region.x == rail.region.x == prompt.region.x
