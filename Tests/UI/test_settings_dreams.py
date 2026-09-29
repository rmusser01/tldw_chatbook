"""TASK-33244: Settings > Domain Defaults > Dreams category.

The section is the canonical enable path for Dreams (no config.toml
hand-editing) plus the interest-profile topic editor. These tests pin:

- the enable toggle persists to ``[dreams] enabled`` and round-trips;
- provider/model can be set back to "follow chat defaults" (keys removed);
- region persists on submit;
- a topic added from the UI lands as ``source='user'`` (the protected
  source the cycle's signal refresh never clobbers) and removes cleanly;
- goals stay out of this editor (cross-link only);
- the section degrades to a notice when the Dreams DB is unavailable.

Every mounted-app case is a ``@private_profile_test``: each one needs real
config-file round-trips against the factory-built app, which requires the
config source to own one interpreter lifetime (a plain in-process factory
boot under the per-test env redirect trips the raw-participant admission
with ``raw_source_selection_changed``; see Tests/conftest.py's
``isolate_test_environment`` notes). The harness otherwise mirrors
test_settings_configuration_hub.py's Schedules-gate tests (the closest
immediate-apply precedent) and test_settings_workspaces_category.py's
registration test.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from textual.widgets import Button, Input, Select, Static

import tldw_chatbook.config as config_module

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import (
    _click_scrolled_settings_button,
    _select_settings_category,
    _settle_settings_mount_storm,
)
from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.cycle_service import _upsert_profile_signals
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


def _config_path() -> Path:
    """The config file the active profile pinned (child) or the autouse
    isolate fixture redirected to (in-process); seed and assert THIS file."""
    return Path(os.environ["TLDW_CONFIG_PATH"])


def _seed_config(text: str) -> Path:
    config_path = _config_path()
    config_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    # The config cache is already warm by test time (the conftest bootstrap
    # imported config consumers and default-created this very file), so the
    # seeded values only become visible to get_cli_setting after a reload.
    config_module.load_settings(force_reload=True)
    return config_path


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _static_text(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).renderable)


async def _open_dreams_category(pilot, host, *, selector: str) -> None:
    screen = _active_destination_screen(host)
    await _settle_settings_mount_storm(pilot)
    await _select_settings_category(
        screen, pilot, SettingsCategoryId.DREAMS, selector=selector
    )


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_category_registered_with_toggle_and_budget_rows(request):
    """The category exists, shows the gate, and renders read-only budgets."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = false\n'
        'region = "Seattle"\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        assert "Dreams is disabled" in _static_text(
            screen, "#settings-dreams-status"
        )
        assert (
            str(screen.query_one("#settings-dreams-toggle", Button).label)
            == "Enable Dreams"
        )
        visible = _visible_text(screen)
        # Region round-trips into the field; budgets show live config values
        # (region from config, cadence from the defaults).
        assert screen.query_one("#settings-dreams-region", Input).value == "Seattle"
        assert "cadence_hours" in visible
        assert "24" in visible
        assert "max_llm_calls_per_day" in visible
        assert "Goals: press g on any Dreams story" in visible


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_toggle_persists_to_config_and_round_trips(request):
    config_path = _seed_config('[general]\nusers_name = "t"\n')
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-toggle"
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["enabled"] is True
        assert "Dreams is enabled" in _static_text(screen, "#settings-dreams-status")
        assert (
            str(screen.query_one("#settings-dreams-toggle", Button).label)
            == "Disable Dreams"
        )

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-toggle"
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        assert "Dreams is disabled" in _static_text(screen, "#settings-dreams-status")


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_provider_and_model_unset_follow_chat_defaults(request):
    """"Follow chat defaults" is a real option; picking it removes the keys."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        'provider = "openai"\nmodel = "gpt-4o-mini"\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-provider")
        screen = _active_destination_screen(host)

        provider = screen.query_one("#settings-dreams-provider", Select)
        assert provider.value == "openai"
        option_pairs = {
            (str(prompt), value) for prompt, value in provider._options
        }
        assert ("Follow chat defaults", "") in option_pairs

        # Setting .value posts Select.Changed for real; the direct-call idiom
        # used elsewhere would double-dispatch and the second exclusive
        # worker would cancel the first mid-flight.
        provider.value = ""
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "provider" not in _read_toml(config_path)["dreams"]
        assert "chat defaults" in _static_text(screen, "#settings-dreams-result")

        model = screen.query_one("#settings-dreams-model", Input)
        assert model.value == "gpt-4o-mini"
        model.value = ""
        screen.handle_dreams_model_submitted(Input.Submitted(model, ""))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        dreams_section = _read_toml(config_path)["dreams"]
        assert "model" not in dreams_section
        assert dreams_section["enabled"] is True


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_region_persists_on_submit(request):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)

        region = screen.query_one("#settings-dreams-region", Input)
        assert region.value == ""
        region.value = "Seattle"
        screen.handle_dreams_region_submitted(Input.Submitted(region, "Seattle"))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["region"] == "Seattle"
        assert "Region saved" in _static_text(screen, "#settings-dreams-result")


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_topic_add_creates_user_source_row(request, tmp_path):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    app.dreams_db = db
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-topic-add")
        screen = _active_destination_screen(host)

        assert "Dreams storage unavailable" not in _visible_text(screen)
        text_input = screen.query_one("#settings-dreams-topic-text", Input)
        text_input.value = "Vintage synths"
        assert screen.query_one("#settings-dreams-topic-weight", Select).value == "0.5"

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-topic-add"
        )
        await pilot.pause()

        topics = [row for row in db.list_profile() if row.get("facet") == "topic"]
        assert len(topics) == 1
        assert topics[0]["text"] == "Vintage synths"
        assert topics[0]["source"] == "user"
        assert topics[0]["weight"] == pytest.approx(0.5)
        assert topics[0]["searchable"] == 1

        visible = _visible_text(screen)
        assert "Vintage synths" in visible
        assert "yours" in visible
        assert screen.query(
            f"#settings-dreams-topic-remove-{topics[0]['id']}"
        ), "added topic must offer a Remove button"


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_topic_remove_deletes_and_goals_stay(request, tmp_path):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    db.upsert_profile_entry(
        "topic", "Old topic", weight=0.4, searchable=1, source="user"
    )
    db.upsert_profile_entry(
        "goal", "Learn to sail", weight=0.9, searchable=0, source="user"
    )
    app.dreams_db = db
    topic_id = [
        row for row in db.list_profile() if row.get("facet") == "topic"
    ][0]["id"]
    goal_id = [
        row for row in db.list_profile() if row.get("facet") == "goal"
    ][0]["id"]
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(
            pilot, host, selector=f"#settings-dreams-topic-remove-{topic_id}"
        )
        screen = _active_destination_screen(host)

        # Goals are cross-linked, never editable here: no goal row may
        # carry a remove button even though one sits in the profile.
        assert not screen.query(f"#settings-dreams-topic-remove-{goal_id}")

        await _click_scrolled_settings_button(
            screen, pilot, f"#settings-dreams-topic-remove-{topic_id}"
        )
        await pilot.pause()

        remaining = {(row["facet"], row["text"]) for row in db.list_profile()}
        assert ("topic", "Old topic") not in remaining
        assert ("goal", "Learn to sail") in remaining


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_section_renders_gracefully_without_dreams_db(request):
    _seed_config('[general]\nusers_name = "t"\n\n[dreams]\nenabled = false\n')
    app = _build_test_app()
    assert getattr(app, "dreams_db", None) is None
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        # The gate stays usable; only the topic editor stands down.
        assert screen.query_one("#settings-dreams-toggle", Button)
        assert "Dreams storage unavailable" in _visible_text(screen)
        assert not screen.query("#settings-dreams-topic-add")
        assert "Goals: press g on any Dreams story" in _visible_text(screen)


def test_user_topic_weight_survives_signal_refresh(tmp_path):
    """Protected semantics: source='user' rows the UI writes are never
    clobbered by the cycle's signal refresh (ruling R18 upholds R1)."""
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-protected")
    db.upsert_profile_entry(
        "topic", "Vintage synths", weight=0.9, searchable=1, source="user"
    )
    db.upsert_profile_entry(
        "topic", "rust lang", weight=0.2, searchable=1, source="notes"
    )

    _upsert_profile_signals(
        db,
        [
            {"facet": "topic", "text": "Vintage synths", "weight": 0.1},
            {"facet": "topic", "text": "rust lang", "weight": 0.8},
        ],
        {("topic", "Vintage synths"): "notes", ("topic", "rust lang"): "notes"},
        now_iso="2026-09-28T00:00:00+00:00",
    )

    by_text = {row["text"]: row for row in db.list_profile()}
    assert by_text["Vintage synths"]["weight"] == pytest.approx(0.9)
    assert by_text["Vintage synths"]["source"] == "user"
    assert by_text["rust lang"]["weight"] == pytest.approx(0.8)
