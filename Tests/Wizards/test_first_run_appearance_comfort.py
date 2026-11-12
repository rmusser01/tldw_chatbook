"""TASK-34100.13: the Appearance step's startup and comfort settings.

coverage-15: setup offered a 78-card splash gallery but no way to turn the
splash off, shorten it, reduce motion or use ASCII marks. fulltrack-09: the
gallery is gone; one line names the current card (from the full TOML) and
points at Settings ▸ Splash Screen. fulltrack-13: themes are labelled by
display name and tone, and the curated shortlist hides no row.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import ComposeResult
from textual.widgets import Checkbox, RadioButton, RadioSet, Static

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Constants import DEFAULT_SPLASH_DURATION_SECONDS
from tldw_chatbook.UI.Wizards import first_run_appearance_state as look
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_appearance_step import AppearanceStep

_BUNDLE = str(
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)


class _StepHost(ConsolidatedCSSApp):
    CSS_PATH = _BUNDLE

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _step(app_config, *, environ=None, rerun=False):
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        commit_config=AsyncMock(return_value=True),
        rerun=rerun,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=8),
        environ=environ if environ is not None else {},
    )
    return wizard, step


def _raw(**sections):
    return {"COMPREHENSIVE_CONFIG_RAW": sections}


def _text(step) -> str:
    return "\n".join(str(widget.render()) for widget in step.query(Static))


# -- AC#13: the gallery is gone -------------------------------------------------


@pytest.mark.asyncio
async def test_no_splash_gallery_one_line_names_the_card_from_the_full_toml() -> None:
    _wizard, step = _step(_raw(splash_screen={"card_selection": "matrix"}))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert not step.query("#setup-splash-choice")
        assert not step.query("#setup-splash-show-all")
        line = str(step.query_one("#setup-appearance-splash-card", Static).render())
    assert line == (
        "Splash card: Matrix — preview and pick one in Settings ▸ Splash Screen."
    )


# -- AC#12: startup animation, reduce motion, ASCII marks ------------------------


@pytest.mark.asyncio
async def test_untouched_step_writes_nothing() -> None:
    wizard, step = _step(_raw(splash_screen={"enabled": True}))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        labels = [
            str(button.label)
            for button in step.query("#setup-startup-choice RadioButton")
        ]
        full = step.query_one("#setup-startup-full", RadioButton)
        assert full.value
        ok, error = await step.commit()
    assert (ok, error) == (True, "")
    assert labels == [
        "Off",
        "Short (2 s)",
        f"Full ({float(DEFAULT_SPLASH_DURATION_SECONDS):g} s)",
    ]
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_off_writes_only_splash_enabled_false() -> None:
    wizard, step = _step(_raw(splash_screen={"enabled": True}))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-startup-off", RadioButton).value = True
        await pilot.pause()
        await step.commit()
    assert wizard.commit_config.await_args.args[0] == {
        "splash_screen": {"enabled": False}
    }


@pytest.mark.asyncio
async def test_short_and_reduce_motion_and_ascii_are_delta_commits() -> None:
    wizard, step = _step(_raw(splash_screen={}, appearance={}))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.query_one("#setup-startup-short", RadioButton).value = True
        step.query_one("#setup-appearance-reduce-motion", Checkbox).value = True
        step.query_one("#setup-appearance-ascii", Checkbox).value = True
        await pilot.pause()
        await step.commit()
    committed = wizard.commit_config.await_args.args[0]
    assert committed == {
        "splash_screen": {"enabled": True, "duration": look.SHORT_SPLASH_SECONDS},
        "appearance": {"reduce_motion": True, "ascii_glyphs": True},
    }
    assert wizard_state.commit_sections_allowed(committed)


@pytest.mark.asyncio
async def test_rerun_prefills_saved_comfort_and_short_splash() -> None:
    _wizard, step = _step(
        _raw(
            splash_screen={"enabled": True, "duration": 2},
            appearance={"reduce_motion": True},
        )
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step.query_one("#setup-startup-short", RadioButton).value
        assert step.query_one("#setup-appearance-reduce-motion", Checkbox).value
        assert not step.query_one("#setup-appearance-ascii", Checkbox).value


@pytest.mark.parametrize(
    ("environ", "expected"),
    [({"NO_COLOR": "1"}, True), ({"TERM": "linux"}, True), ({"TERM": "xterm"}, False)],
)
@pytest.mark.asyncio
async def test_ascii_marks_are_preset_from_no_color_or_the_linux_console(
    environ, expected
) -> None:
    _wizard, step = _step(_raw(appearance={}), environ=environ)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert step.query_one("#setup-appearance-ascii", Checkbox).value is expected


@pytest.mark.parametrize("environ", [{"NO_COLOR": "1"}, {"TERM": "linux"}])
@pytest.mark.asyncio
async def test_a_saved_ascii_choice_beats_the_terminal_preset(environ) -> None:
    """Review F2 (TASK-34100.13): the preset only stands in for a choice never
    made. A deliberately saved ``ascii_glyphs = false`` showed ticked under
    NO_COLOR, and Next without touching it wrote ``true``."""
    wizard, step = _step(
        _raw(appearance={"ascii_glyphs": False}), environ=environ, rerun=True
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ticked = step.query_one("#setup-appearance-ascii", Checkbox).value
        ok, _ = await step.commit()
    assert ticked is False
    assert ok
    wizard.commit_config.assert_not_awaited()


def _template_appearance() -> dict:
    """The [appearance] table every fresh profile gets from the shipped template."""
    import tomllib

    from tldw_chatbook import config as config_module

    return dict(tomllib.loads(config_module.CONFIG_TOML_CONTENT)["appearance"])


@pytest.mark.parametrize("environ", [{"NO_COLOR": "1"}, {"TERM": "linux"}])
@pytest.mark.asyncio
async def test_a_fresh_profile_gets_the_terminal_preset_despite_the_template_default(
    environ,
) -> None:
    """Resume pass (TASK-34100.13): the shipped template writes
    ``ascii_glyphs = false`` into every new config.toml, so after the F2 fix
    ("a saved choice beats the preset") the preset never applied on a real
    first run. On a first run that value is the template's, not a choice."""
    appearance = _template_appearance()
    assert appearance.get("ascii_glyphs") is False  # the premise
    wizard, step = _step(_raw(appearance=appearance), environ=environ)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ticked = step.query_one("#setup-appearance-ascii", Checkbox).value
        ok, _ = await step.commit()
    assert ticked is True
    assert ok
    assert wizard.commit_config.await_args.args[0] == {
        "appearance": {"ascii_glyphs": True}
    }


@pytest.mark.asyncio
async def test_a_rerun_over_the_template_default_keeps_the_saved_value() -> None:
    """On a re-run the saved value stands, template-written or not: the user
    has run setup before and Settings ▸ Appearance shows it."""
    wizard, step = _step(
        _raw(appearance=_template_appearance()), environ={"NO_COLOR": "1"}, rerun=True
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ticked = step.query_one("#setup-appearance-ascii", Checkbox).value
        ok, _ = await step.commit()
    assert ticked is False and ok
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_never_chosen_ascii_preset_is_saved_on_next() -> None:
    """The preset is the step's proposal when nothing was ever chosen; Next
    accepts it (the app reads ASCII mode from config, not the terminal)."""
    wizard, step = _step(_raw(appearance={}), environ={"NO_COLOR": "1"})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        ok, _ = await step.commit()
    assert ok
    assert wizard.commit_config.await_args.args[0] == {
        "appearance": {"ascii_glyphs": True}
    }


def test_summary_shows_the_startup_and_motion_row() -> None:
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            {
                "splash_screen": {"enabled": True, "duration": 2.0},
                "appearance": {"reduce_motion": True},
            },
            {},
            rag_deps_installed=True,
        )
    }
    row = rows["Startup and motion"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == "short splash (2 s) · motion reduced"
    default = {
        r.label: r
        for r in wizard_state.build_summary_rows({}, {}, rag_deps_installed=True)
    }["Startup and motion"]
    assert default.state == wizard_state.ROW_DEFAULT
    assert default.detail == (
        f"full splash ({float(DEFAULT_SPLASH_DURATION_SECONDS):g} s)"
    )


@pytest.mark.parametrize(
    ("config", "detail"),
    [
        ({"splash_screen": {"enabled": False}}, "no splash"),
        (
            {"appearance": {"ascii_glyphs": True}},
            f"full splash ({float(DEFAULT_SPLASH_DURATION_SECONDS):g} s)"
            " · plain ASCII marks",
        ),
        (
            {
                "splash_screen": {"enabled": False},
                "appearance": {"reduce_motion": True, "ascii_glyphs": True},
            },
            "no splash · motion reduced · plain ASCII marks",
        ),
    ],
)
def test_summary_shows_splash_off_and_ascii_marks(config, detail) -> None:
    """T-G13-2: each Appearance choice reaches the Summary row -- "Off" as
    "no splash" and the ASCII choice as "plain ASCII marks"."""
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(config, {}, rag_deps_installed=True)
    }
    row = rows["Startup and motion"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == detail


# -- pre-merge review (2026-11-12): a custom splash length is named as saved ------


@pytest.mark.parametrize(
    ("duration", "choice", "detail"),
    [
        (4, look.STARTUP_FULL, "full splash (4 s)"),
        (15.0, look.STARTUP_FULL, "full splash (15 s)"),
        (1, look.STARTUP_SHORT, "short splash (1 s)"),
        (0.5, look.STARTUP_SHORT, "short splash (0.5 s)"),
    ],
)
def test_summary_names_a_custom_splash_length_in_its_saved_seconds(
    duration, choice, detail
) -> None:
    """Pre-merge review (2026-11-12): a length saved in Settings ▸ Splash
    Screen read as one of setup's two -- 4 s or 15 s as "full splash (7 s)",
    1 s as "short splash (2 s)" -- and a 15 s splash ticked as the default."""
    splash = {"enabled": True, "duration": duration}
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            {"splash_screen": splash}, {}, rag_deps_installed=True
        )
    }
    row = rows["Startup and motion"]
    assert look.startup_choice(splash) == choice
    assert row.detail == detail
    assert row.state == wizard_state.ROW_CONFIGURED


@pytest.mark.parametrize(
    ("duration", "labels"),
    [
        (
            15,
            ["Off", "Short (2 s)", "Full (15 s, as saved)"],
        ),
        (
            1,
            [
                "Off",
                "Short (1 s, as saved)",
                f"Full ({float(DEFAULT_SPLASH_DURATION_SECONDS):g} s)",
            ],
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_step_names_a_custom_splash_length_on_its_choice(
    duration, labels
) -> None:
    """The radio row a saved custom length selects says that length, and an
    untouched step keeps it (writes nothing)."""
    wizard, step = _step(_raw(splash_screen={"enabled": True, "duration": duration}))
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        shown = [
            str(button.label)
            for button in step.query("#setup-startup-choice RadioButton")
        ]
        ok, _ = await step.commit()
    assert shown == labels
    assert ok
    wizard.commit_config.assert_not_awaited()


# -- AC#14: theme names and the whole shortlist ---------------------------------


@pytest.mark.asyncio
async def test_theme_rows_use_display_names_with_a_tone_tag_and_hide_no_row() -> None:
    """T-G13-1: the step is shown (``active``) before the list is measured --
    hidden, every region is 0x0 and "nothing scrolls" held for any height."""
    _wizard, step = _step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        step.add_class("active")
        await pilot.pause()
        await pilot.pause()
        radio_set = step.query_one("#setup-theme-choice", RadioSet)
        buttons = list(radio_set.query(RadioButton))
        labels = {getattr(b, "_theme_name", ""): str(b.label) for b in buttons}
        hidden = radio_set.max_scroll_y
        list_height = radio_set.region.height
        rows = radio_set.content_region
        outside = [
            getattr(b, "_theme_name", "")
            for b in buttons
            if not b.region.height or not rows.contains_region(b.region)
        ]
    assert labels["textual-dark"] == "Default dark"
    assert labels["textual-light"] == "Default light"
    assert labels["nord"] == "Nord · dark"
    assert labels["tokyo-night"] == "Tokyo Night · dark"
    assert len(buttons) >= 6
    assert list_height > 0, "the step was not shown, so nothing was measured"
    assert hidden == 0, "a shortlist row sits below the list's visible rows"
    assert outside == [], f"shortlist rows outside the list's visible rows: {outside}"


def test_summary_theme_row_uses_the_display_name() -> None:
    rows = {
        row.label: row
        for row in wizard_state.build_summary_rows(
            {"general": {"default_theme": "nord"}}, {}, rag_deps_installed=True
        )
    }
    assert rows["Theme"].detail == "Nord · dark"
    default = {
        r.label: r
        for r in wizard_state.build_summary_rows(
            {"general": {"default_theme": "textual-dark"}}, {}, rag_deps_installed=True
        )
    }["Theme"]
    assert default.detail == "Default dark"


class _LiveWizard(SimpleNamespace):
    """Stands in for SetupWizardContainer: the app's own wizard reads live config."""

    reads_live_config = True


@pytest.mark.asyncio
async def test_a_rerun_shows_splash_settings_changed_since_launch(monkeypatch) -> None:
    """Review F8 (TASK-34100.13): Settings ▸ Splash Screen changes made after
    launch were invisible to a same-session re-run (launch-time snapshot)."""
    import tldw_chatbook.config as config_module

    stale = _raw(splash_screen={"enabled": True, "card_selection": "random"})
    live = {"splash_screen": {"enabled": False, "card_selection": "matrix"}}
    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", lambda **_: live
    )
    wizard = _LiveWizard(
        app_instance=MagicMock(app_config=stale),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=8),
        environ={},
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        off = step.query_one("#setup-startup-off", RadioButton).value
        line = str(step.query_one("#setup-appearance-splash-card", Static).render())
        ok, _ = await step.commit()
    assert off is True
    assert line.startswith("Splash card: Matrix")
    assert ok
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_untouched_step_writes_nothing_when_the_config_moves_after_compose(
    monkeypatch,
) -> None:
    """Review V2 (TASK-34100.13): Appearance re-read its baseline in commit(),
    so a config file that changed after the step was built (an external edit
    the warm cache picked up later) made the untouched step write the shown,
    stale values back. The delta is taken against what the controls show."""
    import tldw_chatbook.config as config_module

    live = {"splash_screen": {"enabled": True}, "appearance": {"ascii_glyphs": False}}
    monkeypatch.setattr(
        config_module, "load_cli_config_and_ensure_existence", lambda **_: live
    )
    wizard = _LiveWizard(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=True,
    )
    step = AppearanceStep(
        wizard=wizard,
        config=WizardStepConfig(id="appearance", title="Appearance", step_number=8),
        environ={},
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        live["splash_screen"] = {"enabled": False}
        live["appearance"] = {"ascii_glyphs": True, "reduce_motion": True}
        ok, _ = await step.commit()
        untouched_writes = wizard.commit_config.await_count
        step.query_one("#setup-startup-short", RadioButton).value = True
        await pilot.pause()
        await step.commit()
        first_write = wizard.commit_config.await_args.args[0]
        await step.commit()
        writes = wizard.commit_config.await_count
    assert ok
    assert untouched_writes == 0
    assert first_write == {
        "splash_screen": {"enabled": True, "duration": look.SHORT_SPLASH_SECONDS}
    }
    assert writes == 1


@pytest.mark.asyncio
async def test_an_untouched_step_writes_nothing_after_mirroring_welcomes_saved_reduce_motion() -> None:
    """Review TE-1 (TASK-34100.13): V2's Welcome-mirror half. Welcome's
    "Reduce motion" is saved on its Next and mirrored into app_config; when
    Appearance shows it, that saved value is the baseline too. Without that,
    an untouched Appearance wrote the same value a second time; a later
    untick writes only the change."""
    app_config = {
        "appearance": {"reduce_motion": False},
        "COMPREHENSIVE_CONFIG_RAW": {"appearance": {"reduce_motion": False}},
    }
    wizard, step = _step(app_config)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        box = step.query_one("#setup-appearance-reduce-motion", Checkbox)
        assert box.value is False
        # Welcome's Next saved reduce_motion = true and mirrored it.
        app_config["appearance"]["reduce_motion"] = True
        step.on_show()
        await pilot.pause()
        mirrored = box.value
        ok, _ = await step.commit()
        untouched_writes = wizard.commit_config.await_count
        box.value = False
        await pilot.pause()
        await step.commit()
        unticked = wizard.commit_config.await_args.args[0]
    assert mirrored is True
    assert ok
    assert untouched_writes == 0
    assert unticked == {"appearance": {"reduce_motion": False}}
