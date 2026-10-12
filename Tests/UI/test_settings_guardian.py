"""TASK-33200 Task 3: Settings > Domain Defaults > Guardian category.

The section is Guardian's canonical enable path (ADR-204 contract 1: the
 Dreams-pattern build-on-enable toggle is the ONE caller of the app's
``get_guardian_db`` builder) plus the rules table/editor and the
anti-impulsive-disable cooldown refusals (contract 6). These tests pin:

- the category registers with the off-gate copy ("Nothing runs, nothing is
  recorded") and builds no storage while disabled;
- the toggle persists ``[guardian] enabled`` and builds storage on enable;
- the disable toggle and rule deactivation REFUSE while any rule's
  cooldown is future ("available in N min"), reading cooldowns via the DB;
- the cooldown-remaining display and the single ``guardian_cooldown_bypassed``
  log when a config-file disable is detected at the next settings render;
- the rule editor modal: CRUD happy path, the crisis write-boundary
  rejections surfaced as notices, and the crisis-fixed controls.

Mounted-app cases are ``@private_profile_test`` (real config round-trips
need one interpreter lifetime -- see test_settings_dreams.py's notes); the
editor modal runs on the bare-App harness mirroring
test_artifacts_dreams_goals_modal.py.
"""

from __future__ import annotations

import os
import time
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from loguru import logger
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, Static

import Tests.UI.app_factory  # noqa: F401 - config-participant admission binding
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
from tldw_chatbook.DB.Guardian_DB import GuardianDB
from tldw_chatbook.Guardian.settings import guardian_db_path
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


def _config_path() -> Path:
    return Path(os.environ["TLDW_CONFIG_PATH"])


def _seed_config(text: str) -> Path:
    config_path = _config_path()
    config_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    config_module.load_settings(force_reload=True)
    return config_path


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _static_text(screen, selector: str) -> str:
    return str(screen.query_one(selector, Static).renderable)


async def _open_guardian_category(pilot, host, *, selector: str):
    screen = _active_destination_screen(host)
    await _settle_settings_mount_storm(pilot)
    await _select_settings_category(
        screen, pilot, SettingsCategoryId.GUARDIAN, selector=selector
    )


async def _wait_for_guardian_status(
    screen, pilot, needle: str, *, timeout: float = 5.0
) -> str:
    """Poll the section status row until it renders ``needle``.

    The pane swap after ``_refresh_guardian_pane`` runs in a worker, so a
    single pause is not enough for the recomposed widgets to exist again.
    """
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            last = _static_text(screen, "#settings-guardian-status")
        except Exception:  # noqa: BLE001 - mid-swap, the row is gone briefly
            last = ""
        if needle in last:
            await pilot.pause()
            return last
        await pilot.pause(0.05)
    raise AssertionError(
        f"Timed out waiting for {needle!r} in guardian status; last: {last!r}"
    )


def _future_iso(minutes: int) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    ).isoformat()


def _past_iso(minutes: int = 1) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes)
    ).isoformat()


def _attach_enabled_db(tmp_path, *, cooldown_rule_index: int | None = None,
                       cooldown_minutes: int = 30) -> GuardianDB:
    """A store with the three v1 seeds, optionally one cooldown armed."""
    db = GuardianDB(tmp_path / "guardian.sqlite", "settings-test")
    if cooldown_rule_index is not None:
        rule = db.list_rules()[cooldown_rule_index]
        db.set_cooldown(int(rule["id"]), _future_iso(cooldown_minutes))
    return db


# ---------------------------------------------------------------------------
# Registration + the off gate (contract 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_guardian_category_registered_with_off_gate_copy(request, tmp_path):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[guardian]\nenabled = false\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_guardian_category(
            pilot, host, selector="#settings-guardian-toggle"
        )
        screen = _active_destination_screen(host)

        assert "Nothing runs, nothing is recorded" in _static_text(
            screen, "#settings-guardian-status"
        )
        assert (
            str(screen.query_one("#settings-guardian-toggle", Button).label)
            == "Enable Guardian"
        )
        # Contract 1: rendering the disabled section builds no storage.
        assert getattr(app, "guardian_db", None) is None
        assert not guardian_db_path().exists(), (
            "a disabled Guardian must never create its database file"
        )
        assert "Guardian storage unavailable" in _visible_text(screen)


@pytest.mark.asyncio
@private_profile_test
async def test_guardian_toggle_enables_builds_storage_and_lists_rules(request):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[guardian]\nenabled = false\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_guardian_category(
            pilot, host, selector="#settings-guardian-toggle"
        )
        screen = _active_destination_screen(host)

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-guardian-toggle"
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["guardian"]["enabled"] is True
        # Build-on-enable: the toggle is the ONE caller of the builder.
        assert getattr(app, "guardian_db", None) is not None
        assert "Guardian is enabled" in _static_text(
            screen, "#settings-guardian-status"
        )
        # Seed rules render with the table's columns.
        visible = _visible_text(screen)
        assert "Crisis awareness (self-harm)" in visible
        assert "crisis_awareness" in visible
        assert "critical" in visible
        # The rules table offers per-rule edit buttons (attribute-prefix
        # selectors are not Textual CSS; filter the Buttons in Python).
        edit_buttons = [
            button
            for button in screen.query(Button)
            if str(button.id or "").startswith("settings-guardian-rule-edit-")
        ]
        assert edit_buttons
        assert screen.query_one("#settings-guardian-rule-add", Button)


# ---------------------------------------------------------------------------
# Cooldown-aware refusals (contract 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_guardian_disable_refused_while_cooldown_active(
    request, tmp_path
):
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[guardian]\nenabled = true\n'
    )
    app = _build_test_app()
    db = _attach_enabled_db(tmp_path, cooldown_rule_index=2)  # late-night seed
    app.guardian_db = db
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_guardian_category(
            pilot, host, selector="#settings-guardian-toggle"
        )
        screen = _active_destination_screen(host)

        assert "Disable Guardian" in str(
            screen.query_one("#settings-guardian-toggle", Button).label
        )
        await _click_scrolled_settings_button(
            screen, pilot, "#settings-guardian-toggle"
        )
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["guardian"]["enabled"] is True, (
            "the disable must be refused during an active cooldown"
        )
        result = _static_text(screen, "#settings-guardian-result")
        assert "available in" in result, result
        assert "min" in result
        # The cooldown-remaining display is live on the section.
        assert "Cooldown" in _static_text(screen, "#settings-guardian-status")


@pytest.mark.asyncio
@private_profile_test
async def test_rule_deactivation_refused_while_its_cooldown_active(
    request, tmp_path
):
    _seed_config('[general]\nusers_name = "t"\n\n[guardian]\nenabled = true\n')
    app = _build_test_app()
    db = _attach_enabled_db(tmp_path, cooldown_rule_index=2)
    app.guardian_db = db
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_guardian_category(
            pilot, host, selector="#settings-guardian-rule-add"
        )
        screen = _active_destination_screen(host)
        rule = db.list_rules()[2]

        screen.open_guardian_rule_editor(rule_id=int(rule["id"]))
        await pilot.pause()
        modal = pilot.app.screen
        modal.query_one("#grm-enabled", Checkbox).value = False
        modal.action_save()
        await pilot.pause()

        status = str(modal.query_one("#grm-status", Static).renderable)
        assert "available in" in status, status
        assert db.get_rule(int(rule["id"]))["enabled"] == 1, (
            "deactivation must be refused while the rule's cooldown is active"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_config_disable_during_cooldown_logged_once(request, tmp_path):
    """The config-file escape hatch works (contract 6) but is logged."""
    _seed_config('[general]\nusers_name = "t"\n\n[guardian]\nenabled = true\n')
    app = _build_test_app()
    db = _attach_enabled_db(tmp_path, cooldown_rule_index=2)
    app.guardian_db = db
    host = DestinationHarness(app, "settings")

    records: list[str] = []

    def sink(message):
        records.append(str(message))

    handler_id = logger.add(sink, level="WARNING")
    try:
        async with host.run_test(size=(180, 50)) as pilot:
            # Open the section while still enabled (no bypass yet: the
            # detection only binds the disabled branch of the render).
            await _open_guardian_category(
                pilot, host, selector="#settings-guardian-status"
            )
            screen = _active_destination_screen(host)
            assert not [
                r for r in records if "guardian_cooldown_bypassed" in r
            ], "an enabled Guardian never logs a bypass"

            # The config-file escape hatch: flip enabled straight on disk.
            config_path = _config_path()
            text = config_path.read_text(encoding="utf-8")
            config_path.write_text(
                text.replace("enabled = true", "enabled = false"),
                encoding="utf-8",
            )
            config_module.load_settings(force_reload=True)

            # The NEXT settings render detects the disable-during-cooldown
            # (the pane swap runs in a worker; poll for the disabled copy).
            screen._refresh_guardian_pane()
            await _wait_for_guardian_status(
                screen, pilot, "Nothing runs, nothing is recorded"
            )
            bypasses = [r for r in records if "guardian_cooldown_bypassed" in r]
            assert len(bypasses) == 1, (
                "one bypass event detected at the next settings render"
            )
            # A re-render never duplicates the log for the same cooldown.
            screen._refresh_guardian_pane()
            await _wait_for_guardian_status(
                screen, pilot, "Nothing runs, nothing is recorded"
            )
            bypasses = [r for r in records if "guardian_cooldown_bypassed" in r]
            assert len(bypasses) == 1
    finally:
        logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Read-only trend-key rows (final-review T3-1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_guardian_renders_read_only_trend_key_rows(request):
    """T3-1: trend thresholds + retention display live config values.

    The Dreams ``_DREAMS_READ_ONLY_KEYS`` idiom: each key renders as a
    read-only detail row reading ``guardian_setting`` live, with the
    "editable in config.toml only" hint the section banner promises. One
    seeded override (doomloop 45 vs the default 20) proves the rows read
    config, not the hardcoded defaults.
    """
    _seed_config(
        '[general]\nusers_name = "t"\n\n[guardian]\nenabled = false\n'
        'doomloop_hits_per_day = 45\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_guardian_category(
            pilot, host, selector="#settings-guardian-toggle"
        )
        screen = _active_destination_screen(host)
        visible = _visible_text(screen)

        # The hint matches the banner copy: read-only, config.toml only.
        assert "read-only" in visible
        assert "config.toml" in visible

        for key, value in (
            ("fixation_share_threshold", "0.6"),  # default
            ("fixation_window_days", "7"),  # default
            ("fixation_min_hits", "30"),  # default
            ("doomloop_hits_per_day", "45"),  # the seeded override
            ("alert_retention_days", "180"),  # default
        ):
            assert f"{key}: {value}" in visible, (key, value)


# ---------------------------------------------------------------------------
# Rule editor modal (bare-App harness, the DreamsGoalsModal test pattern)
# ---------------------------------------------------------------------------


def _modal_app(db: GuardianDB):
    from tldw_chatbook.UI.Screens.settings_screen import GuardianRuleEditModal

    app = App()
    return app, GuardianRuleEditModal(
        db_getter=lambda: db,
        on_changed=lambda: None,
    )


@pytest.mark.asyncio
async def test_rule_editor_add_creates_rule(tmp_path):
    db = GuardianDB(tmp_path / "guardian.sqlite", "editor-test")
    app, modal = _modal_app(db)
    async with app.run_test(size=(100, 46)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#grm-name", Input).value = "news addiction"
        modal.query_one("#grm-topic", Input).value = "news_habit"
        modal.query_one("#grm-pattern", Input).value = "refresh the feed"
        modal.query_one("#grm-feeds-discovery", Checkbox).value = True
        modal.action_save()
        await pilot.pause()

        rules = [r for r in db.list_rules() if r["topic"] == "news_habit"]
        assert len(rules) == 1
        assert rules[0]["name"] == "news addiction"
        assert rules[0]["feeds_discovery"] == 1
        assert rules[0]["enabled"] == 1


@pytest.mark.asyncio
async def test_rule_editor_delete_removes_rule(tmp_path):
    db = GuardianDB(tmp_path / "guardian.sqlite", "editor-test")
    seed = db.list_rules()[1]  # the doomscrolling demo rule
    app, modal = _modal_app(db)
    modal.edit_rule(int(seed["id"]))
    async with app.run_test(size=(100, 46)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()
        modal.action_delete()
        await pilot.pause()
        assert db.get_rule(int(seed["id"])) is None


@pytest.mark.asyncio
async def test_rule_editor_delete_refused_while_cooldown_active(tmp_path):
    """T3-2: delete is gated exactly like deactivation (refusal-total).

    While the rule's own cooldown is future, delete refuses with the
    "available in N min" copy and the rule survives; once the cooldown has
    expired, the same action succeeds.
    """
    db = GuardianDB(tmp_path / "guardian.sqlite", "editor-test")
    seed = db.list_rules()[1]
    db.set_cooldown(int(seed["id"]), _future_iso(30))
    app, modal = _modal_app(db)
    modal.edit_rule(int(seed["id"]))
    async with app.run_test(size=(100, 46)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        modal.action_delete()
        await pilot.pause()

        status = str(modal.query_one("#grm-status", Static).renderable)
        assert "available in" in status, status
        assert "min" in status, status
        assert db.get_rule(int(seed["id"])) is not None, (
            "deletion must be refused while the rule's cooldown is active"
        )

        # After the cooldown expires, the same delete succeeds.
        db.set_cooldown(int(seed["id"]), _past_iso())
        modal.action_delete()
        await pilot.pause()
        assert db.get_rule(int(seed["id"])) is None


@pytest.mark.asyncio
async def test_rule_editor_surfaces_both_crisis_write_boundary_rejections(
    tmp_path,
):
    """The two GuardianRuleConflict rejections reach the user as notices."""
    db = GuardianDB(tmp_path / "guardian.sqlite", "editor-test")
    app, modal = _modal_app(db)
    async with app.run_test(size=(100, 46)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        modal.query_one("#grm-name", Input).value = "flag it"
        modal.query_one("#grm-topic", Input).value = "flagged"
        modal.query_one("#grm-pattern", Input).value = "pattern"
        # Rejection 1: is_crisis + action redact.
        modal.query_one("#grm-action", Select).value = "redact"
        modal.query_one("#grm-is-crisis", Checkbox).value = True
        modal.action_save()
        await pilot.pause()
        status = str(modal.query_one("#grm-status", Static).renderable)
        assert "may only surface notifications" in status, status

        # Rejection 2: is_crisis + feeds_discovery.
        modal.query_one("#grm-action", Select).value = "notify"
        modal.query_one("#grm-feeds-discovery", Checkbox).value = True
        modal.action_save()
        await pilot.pause()
        status = str(modal.query_one("#grm-status", Static).renderable)
        assert "may never feed Dreams discovery" in status, status

        assert not [
            r for r in db.list_rules() if r["topic"] == "flagged"
        ], "rejected writes must not land"


@pytest.mark.asyncio
async def test_rule_editor_pins_crisis_controls(tmp_path):
    """A crisis rule's action selector is fixed to notify and its
    feeds_discovery checkbox is disabled with explanatory copy."""
    db = GuardianDB(tmp_path / "guardian.sqlite", "editor-test")
    crisis = db.list_rules()[0]  # the seeded crisis rule
    app, modal = _modal_app(db)
    modal.edit_rule(int(crisis["id"]))
    async with app.run_test(size=(100, 46)) as pilot:
        await app.push_screen(modal)
        await pilot.pause()

        assert modal.query_one("#grm-action", Select).value == "notify"
        assert modal.query_one("#grm-action", Select).disabled is True
        feeds = modal.query_one("#grm-feeds-discovery", Checkbox)
        assert feeds.disabled is True
        assert feeds.value is False
        visible = "\n".join(
            str(w.renderable) for w in modal.query(Static) if w.display
        )
        assert "crisis" in visible.lower()
