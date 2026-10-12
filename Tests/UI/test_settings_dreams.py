"""TASK-34792: Settings > Domain Defaults > Dreams category.

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


def test_disabled_gate_copy_says_what_off_stops_and_what_it_does_not():
    """Review round 3, item 5. "Nothing runs and nothing is spent" was
    false right after switching off: a search or model call already in
    flight finishes and is counted."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    copy = SettingsScreen._dreams_gate_copy(False)

    assert copy.startswith("Dreams is disabled.")
    assert "starts no cycle, check, search or model call" in copy
    assert "work already under way finishes its current step" in copy
    assert "stay on disk" in copy
    assert "nothing is spent" not in copy.lower()


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


async def _toggle_dreams(screen, pilot, config_path, *, enabled: bool) -> None:
    """Press the Dreams toggle and wait until the switch has really landed.

    Review of #2921, round 4, C7. These tests used to click, wait for the
    workers plus ONE pause, and go on. Under load (the failures were seen
    at a load average around 45) that is not enough twice over: the wait
    can return before the click's own handler has started the worker, and
    the next click can land on the old, disabled button before the pane
    rebuild has replaced it. So the state is polled for, bounded: the
    saved flag, and a toggle button that is the NEW one -- relabelled and
    enabled again. The button is looked up afresh each time because the
    rebuild replaces the widget.
    """
    await _click_scrolled_settings_button(
        screen, pilot, "#settings-dreams-toggle")
    label = "Disable Dreams" if enabled else "Enable Dreams"
    for _ in range(400):
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        buttons = list(screen.query("#settings-dreams-toggle").results(Button))
        if (
            len(buttons) == 1
            and str(buttons[0].label) == label
            and not buttons[0].disabled
            and _read_toml(config_path).get("dreams", {}).get("enabled")
            is enabled
        ):
            # One more settle so the status line of the same rebuild is in.
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            return
    raise AssertionError(
        f"the Dreams toggle did not settle on enabled={enabled}")


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

        await _toggle_dreams(screen, pilot, config_path, enabled=True)

        assert _read_toml(config_path)["dreams"]["enabled"] is True
        assert "Dreams is enabled" in _static_text(screen, "#settings-dreams-status")
        assert (
            str(screen.query_one("#settings-dreams-toggle", Button).label)
            == "Disable Dreams"
        )

        await _toggle_dreams(screen, pilot, config_path, enabled=False)

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        status = _static_text(screen, "#settings-dreams-status")
        assert "Dreams is disabled" in status
        # Review round 3, item 5: the copy shown the moment Dreams goes off
        # must be true at that moment -- work under way is not recalled.
        assert "finishes its current step" in " ".join(status.split())
        assert "nothing is spent" not in status.lower()


def _resolve_hosts_publicly():
    """Point the egress policy's DNS seam at one public address.

    Returns the undo callable (these tests take no ``monkeypatch``: the
    private-profile wrapper owns their fixture list).
    """
    from tldw_chatbook.Utils import egress

    original = egress._resolve_async

    async def resolve(_host):
        return ["93.184.216.34"]

    egress._resolve_async = resolve

    def undo() -> None:
        egress._resolve_async = original

    return undo


class _RecordingWatchlistsService:
    """Stand-in for the app's watchlists service: sources that are running
    until told otherwise, with every ``update_source`` recorded."""

    def __init__(self) -> None:
        self.updates: list[tuple[int, dict]] = []
        self._active: dict[int, bool] = {}

    async def get_source(self, source_id):
        return {"id": source_id,
                "url": f"https://example.com/tracked/{source_id}",
                "active": self._active.get(int(source_id), True),
                "paused": False}

    async def update_source(self, source_id, payload):
        self.updates.append((int(source_id), dict(payload)))
        if "active" in payload:
            self._active[int(source_id)] = bool(payload["active"])
        return {"id": source_id, **dict(payload)}

    async def list_alert_rules(self, *, job_id=None, source_id=None):
        return []


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_toggle_suspends_and_resumes_dream_owned_watches(
    request, tmp_path
):
    """Qodo #5 (PR #2921): the master switch reaches what Dreams scheduled
    OUTSIDE its own loop. A page watch Dreams created is a watchlists
    subscription the watchlist queue checks on its own, so flipping only
    ``[dreams] enabled`` left it running and notifying; disabling now
    disables that subscription (a user-owned one Dreams merely attached to
    is never touched), re-enabling restores it, and the scheduler queue is
    asked to reload so already-loaded Dreams work drops out at once."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=43200,
        subscription_id=7, created_by_dreams=1,
    )
    db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=43200,
        subscription_id=9, created_by_dreams=0,
    )
    app.dreams_db = db
    service = _RecordingWatchlistsService()
    app.local_watchlists_service = service
    reloads: list[int] = []
    app.scheduler_loop.request_reload = lambda: reloads.append(1)
    # Resuming re-checks that a source Dreams stored is still a public
    # destination before it re-enables it; resolve without leaving the box.
    request.addfinalizer(_resolve_hosts_publicly())
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        await _toggle_dreams(screen, pilot, config_path, enabled=False)

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        assert service.updates == [(7, {
            "active": False,
            # Round 3, T1: a Dreams-created source that carries no egress
            # mark yet is stamped by the same write that stops it.
            "processing_options": {"url_provenance": "unknown"},
        })], (
            "disabling Dreams must stop the subscription Dreams created, and "
            "only that one"
        )
        assert reloads, "the scheduler queue reloads so the switch is immediate"
        assert [row["status"] for row in db.list_tracked_items()] == [
            "active", "active"
        ], "suspended watches stay tracked, ready to resume"

        await _toggle_dreams(screen, pilot, config_path, enabled=True)

        assert _read_toml(config_path)["dreams"]["enabled"] is True
        assert [
            (source_id, payload["active"])
            for source_id, payload in service.updates
        ] == [(7, False), (7, True)], (
            "re-enabling Dreams resumes the watch it suspended"
        )


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_switch_is_not_cut_short_by_another_settings_action(
    request, tmp_path
):
    """Review of #2921, item 3c: the suspension used to run inline in a
    worker sharing an exclusive group with the provider/model/region
    handlers, so saving the region while it was mid-way cancelled it and
    left the watch running under a disabled Dreams."""
    import asyncio

    from tldw_chatbook.Dreams import track_service

    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=43200,
        subscription_id=7, created_by_dreams=1,
    )
    app.dreams_db = db

    class _ParkedService(_RecordingWatchlistsService):
        def __init__(self) -> None:
            super().__init__()
            self.parked = asyncio.Event()
            self.release = asyncio.Event()

        async def get_source(self, source_id):
            self.parked.set()
            await self.release.wait()
            return await super().get_source(source_id)

    service = _ParkedService()
    app.local_watchlists_service = service
    app.scheduler_loop.request_reload = lambda: None
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-toggle"
        )
        await asyncio.wait_for(service.parked.wait(), timeout=10)

        # Another Dreams setting is saved while the switch is mid-way.
        region = screen.query_one("#settings-dreams-region", Input)
        region.value = "Seattle"
        screen.handle_dreams_region_submitted(Input.Submitted(region, "Seattle"))
        await pilot.pause()
        service.release.set()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        switch = getattr(track_service, "_RECONCILE_TASK", None)
        if switch is not None:
            await asyncio.wait_for(asyncio.shield(switch), timeout=10)

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        assert [
            (source_id, payload["active"])
            for source_id, payload in service.updates
        ] == [(7, False)], "the switch finished: the watch is stopped"
        assert db.get_tracked_item(1)["suspension"] is not None


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_switch_survives_leaving_settings_during_the_save(
    request, tmp_path
):
    """Review round 3, T6. The toggle's worker belonged to the Settings
    screen, and Textual cancels a screen's workers when it is unmounted.
    Leaving Settings while the config save was still on its thread hop
    left ``enabled = false`` on disk and nothing switched: no record, no
    reconcile task -- only the next launch put it right."""
    import asyncio
    import threading

    from tldw_chatbook.Dreams import track_service
    from tldw_chatbook.UI.Screens import settings_screen as settings_module

    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=43200,
        subscription_id=7, created_by_dreams=1,
    )
    app.dreams_db = db
    service = _RecordingWatchlistsService()
    app.local_watchlists_service = service
    app.scheduler_loop.request_reload = lambda: None
    saving, release = threading.Event(), threading.Event()
    real_apply = settings_module.apply_settings_mutation_to_cli_config

    def slow_apply(*args, **kwargs):
        saving.set()
        assert release.wait(timeout=20)
        return real_apply(*args, **kwargs)

    settings_module.apply_settings_mutation_to_cli_config = slow_apply
    request.addfinalizer(lambda: setattr(
        settings_module, "apply_settings_mutation_to_cli_config", real_apply))
    request.addfinalizer(release.set)
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-toggle"
        )
        for _ in range(1000):
            if saving.is_set():
                break
            await asyncio.sleep(0.01)
        assert saving.is_set(), "fixture: the save hop is running"

        # What unmounting the Settings screen does to its workers.
        pilot.app.workers.cancel_node(screen)
        await pilot.pause()
        release.set()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        switch = getattr(track_service, "_RECONCILE_TASK", None)
        if switch is not None:
            await asyncio.wait_for(asyncio.shield(switch), timeout=10)

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        assert [
            (source_id, payload["active"])
            for source_id, payload in service.updates
        ] == [(7, False)], "the switch ran although its screen was left"
        assert db.get_tracked_item(1)["suspension"] is not None


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_toggle_survives_failing_runtime_seams(request, tmp_path):
    """The toggle worker runs with Textual's default ``exit_on_error``, so
    anything escaping it would take the app down AFTER the gate was saved.
    A watchlists service that raises and a scheduler that refuses the
    reload request must both degrade: the switch still persists and the
    pane still reports it."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    db.create_tracked_item(
        mechanism="page", intent="topic", cadence_seconds=43200,
        subscription_id=7, created_by_dreams=1,
    )
    app.dreams_db = db

    class _BrokenService:
        async def get_source(self, source_id):
            return {"id": source_id, "active": True, "paused": False}

        async def update_source(self, source_id, payload):
            raise RuntimeError("watchlists unavailable")

        async def list_alert_rules(self, *, job_id=None, source_id=None):
            return []

    def _refuse_reload():
        raise RuntimeError("scheduler loop closed")

    app.local_watchlists_service = _BrokenService()
    app.scheduler_loop.request_reload = _refuse_reload
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-toggle")
        screen = _active_destination_screen(host)

        await _toggle_dreams(screen, pilot, config_path, enabled=False)

        assert _read_toml(config_path)["dreams"]["enabled"] is False
        assert "Dreams is disabled" in _static_text(
            screen, "#settings-dreams-status"
        )


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
        # Review round 3, item 6: the region is sent with every synthesis
        # and story prompt and fills tracked-question searches; the field
        # used to say "Used only when it appears in a source".
        assert "prompts and searches" in region.placeholder
        assert "only" not in region.placeholder.lower()
        help_line = " ".join(
            _static_text(screen, "#settings-dreams-field-help").split())
        assert "sent to the model" in help_line
        # Round 4, I7: it names the prompts that carry the region; "every
        # Dreams prompt" was not true (the judge prompt carries none).
        assert "writes its search queries" in help_line
        assert "writes a story" in help_line
        assert "every Dreams prompt" not in help_line
        region.value = "Seattle"
        screen.handle_dreams_region_submitted(Input.Submitted(region, "Seattle"))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["region"] == "Seattle"
        assert "Region saved" in _static_text(screen, "#settings-dreams-result")


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_region_that_fails_the_text_rule_is_not_saved(request):
    """Review of #2921, item 8: the region was the one free-text Dreams
    field saved after a bare ``strip()``. It is embedded in the synthesis
    prompt and in tracked-question searches, so it passes the same check
    as a topic or goal, and a refusal says so and writes nothing."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        'region = "Seattle"\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)

        region = screen.query_one("#settings-dreams-region", Input)
        hostile = "Seattle\u202e"
        region.value = hostile
        screen.handle_dreams_region_submitted(Input.Submitted(region, hostile))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["region"] == "Seattle"
        result = _static_text(screen, "#settings-dreams-result")
        assert "Region not saved" in result
        # The notice names what was wrong (round 4, C4).
        assert "hidden text-direction characters" in result
        assert region.value == hostile, "the typed text stays to be fixed"


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_region_ignored_by_dreams_is_flagged_and_enter_explains(
    request,
):
    """Review round 3, item 7. A region put into config.toml by hand that
    fails the text rule is ignored by Dreams, but Settings showed it in the
    field as if it were in use, and pressing Enter on it said nothing (the
    same-value shortcut ran before the validation)."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        'region = "Seattle\\u202e"\n'
    )
    stored = _read_toml(config_path)["dreams"]["region"]
    assert stored == "Seattle\u202e"
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)

        flag = " ".join(
            _static_text(screen, "#settings-dreams-region-status").split())
        assert "saved region is not being used" in flag

        region = screen.query_one("#settings-dreams-region", Input)
        assert region.value == stored, "the stored text stays editable"
        screen.handle_dreams_region_submitted(Input.Submitted(region, stored))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert "Region not saved" in _static_text(
            screen, "#settings-dreams-result")

        # Correcting it saves, and the flag goes away.
        region.value = "Seattle"
        screen.handle_dreams_region_submitted(Input.Submitted(region, "Seattle"))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["region"] == "Seattle"
        assert not screen.query("#settings-dreams-region-status")


@pytest.mark.asyncio
@private_profile_test
async def test_a_region_stored_as_a_number_is_rewritten_as_text_on_enter(
    request,
):
    """Review round 4, C5. ``region = 98101`` -- a ZIP code typed without
    quotes -- is a number, and Dreams ignores a region that is not text.
    Settings showed "98101" in the field with the "not being used" flag,
    and pressing Enter did nothing: the same-value shortcut compared the
    typed text with ``str()`` of the stored number."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        "region = 98101\n"
    )
    assert _read_toml(config_path)["dreams"]["region"] == 98101
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)
        assert "saved region is not being used" in " ".join(
            _static_text(screen, "#settings-dreams-region-status").split())
        region = screen.query_one("#settings-dreams-region", Input)
        assert region.value == "98101"

        screen.handle_dreams_region_submitted(Input.Submitted(region, "98101"))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert _read_toml(config_path)["dreams"]["region"] == "98101", (
            "Enter rewrites it as the text Dreams can use"
        )
        assert not screen.query("#settings-dreams-region-status")


@pytest.mark.asyncio
@private_profile_test
async def test_a_region_stored_as_a_number_can_be_cleared(request):
    """...and clearing the field removes it, although "" is also what the
    unusable number counts as."""
    config_path = _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        "region = 98101\n"
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)
        region = screen.query_one("#settings-dreams-region", Input)

        region.value = ""
        screen.handle_dreams_region_submitted(Input.Submitted(region, ""))
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert "region" not in _read_toml(config_path)["dreams"]


@pytest.mark.asyncio
@private_profile_test
async def test_dreams_region_in_use_carries_no_flag(request):
    _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
        'region = "Seattle"\n'
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-region")
        screen = _active_destination_screen(host)

        assert not screen.query("#settings-dreams-region-status")


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
async def test_dreams_topic_add_refuses_overlong_text(request, tmp_path):
    """Qodo #8 (PR #2921): a topic enters the interest profile -- and from
    there query synthesis -- so the add path validates it through the
    shared input validation; a refusal writes no row, says why in the
    result row, and keeps the typed text for correction."""
    _seed_config(
        '[general]\nusers_name = "t"\n\n[dreams]\nenabled = true\n'
    )
    app = _build_test_app()
    db = DreamsDB(tmp_path / "dreams.sqlite", "dreams-settings")
    app.dreams_db = db
    host = DestinationHarness(app, "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _open_dreams_category(pilot, host, selector="#settings-dreams-topic-add")
        screen = _active_destination_screen(host)

        typed = "t" * 201
        screen.query_one("#settings-dreams-topic-text", Input).value = typed
        await _click_scrolled_settings_button(
            screen, pilot, "#settings-dreams-topic-add"
        )
        await pilot.pause()

        assert db.list_profile() == [], "a refused topic must not be persisted"
        result = _static_text(screen, "#settings-dreams-result")
        assert "Topic not added" in result and "200" in result
        assert (
            screen.query_one("#settings-dreams-topic-text", Input).value == typed
        )


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
