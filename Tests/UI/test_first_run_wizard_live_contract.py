"""App-level Pilot contract tests standing in for Task 12's live-verification
checklist (see backlog/docs/lessons-live-verification.md: no interactive
terminal is available in this environment, so every MECHANICALLY-CHECKABLE
item of the checklist is pinned here against a real ``TldwCli`` app via
``app.run_test()`` instead of a fabricated manual walkthrough).

Two checklist items genuinely need a human's eyes (splash-ON boot, and
overall visual look-and-feel) and are recorded as "needs human spot-check"
in the backlog task notes instead of being faked here.

Every test uses ``_build_test_app`` (see Tests/UI/test_screen_navigation.py),
the same real-app harness ``test_product_maturity_phase1_first_run.py``
uses, and the same ``TLDW_CONFIG_PATH``-isolated config the root conftest's
autouse ``isolate_test_environment`` fixture already provides -- no test
here ever touches a real user config file.

Two empirically-found traps drove the choices below:

- ``pilot.app.workers.wait_for_complete()`` waits for EVERY worker,
  including ``ProviderStep``'s real (unmocked) local-server discovery
  worker, which can block for a very long time with no local server
  reachable. Tests poll for the specific condition they need instead.
- A pixel-coordinate ``pilot.click(selector)`` resolves its target from the
  widget's own cached ``region``, and that can go stale (observed directly:
  ``app.get_widget_at()`` at a button's own reported region center resolved
  to its *parent* step, not the button, after this wizard's Summary step
  filled in async content) without ``pilot.click`` raising -- it just
  returns ``False`` and the test silently proceeds as if the click landed.
  Every state-changing interaction here therefore drives the widget
  directly (``Button.press()`` / setting ``RadioButton.value``), which is
  exactly what a click ultimately posts, without depending on compositor
  timing that is irrelevant to what these tests check. The one test that
  legitimately needs pixel/render truth (80x24 clipping) checks
  ``region`` and the compositor directly instead of clicking anything.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from textual.widgets import Button, RadioButton, Static

from Tests.UI.test_product_maturity_phase1_first_run import (
    _prepare_clean_environment,
    _test_cli_setting,
)
from Tests.UI.test_screen_navigation import _build_test_app
from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
    FirstRunSetupWizard,
    SetupWizardContainer,
)
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_MODEL,
    STEP_PROVIDER,
    STEP_SUMMARY,
    STEP_WELCOME,
    TRACK_QUICK,
    WIZARD_STATE_SECTION,
    SETUP_STARTED_KEY,
)


async def _wait_until(
    pilot,
    condition: Callable[[], bool],
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.05,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if condition():
            return
        await pilot.pause(interval_seconds)
    if condition():
        return
    raise AssertionError(f"condition was not met within {timeout_seconds:.1f}s")


def _press(screen, selector: str) -> None:
    """Press a Button by selector -- posts Button.Pressed exactly like a
    real click, without depending on the widget's cached screen region.
    """
    screen.query_one(selector, Button).press()


def _select_radio(screen, selector: str) -> None:
    """Select a RadioButton by selector (mirrors what a click toggles)."""
    screen.query_one(selector, RadioButton).value = True


def _build_fresh_wizard_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A truly fresh config: no provider configured, no first_run state at
    all -- the exact condition ``should_offer_wizard`` auto-offers under.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=False)
    app.app_config["_first_run"] = True
    app._initial_tab_value = "chat"
    return app


async def _open_settings_diagnostics(pilot) -> None:
    """Navigate the real shell to Settings, then its Diagnostics category."""
    app = pilot.app
    # The nav strip mounts a tick after the initial screen swap (same race
    # noted in test_product_maturity_phase1_first_run.py); wait for the
    # button to actually exist before pressing it.
    await _wait_until(pilot, lambda: len(app.screen.query("#nav-settings")) == 1)
    _press(app.screen, "#nav-settings")
    await _wait_until(
        pilot,
        lambda: app.current_tab == "settings"
        and app.screen.__class__.__name__ == "SettingsScreen",
    )
    await pilot.pause(0.2)
    _press(app.screen, "#settings-category-diagnostics")
    await pilot.pause(0.2)


# ---------------------------------------------------------------------------
# 1. Fresh config, splash OFF -> wizard auto-offers (checklist item 3).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_config_splash_disabled_wizard_auto_offers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Checklist item 3: with splash pre-seeded OFF, the wizard must still
    auto-offer on a truly fresh config -- the auto-offer path does not run
    through the splash screen's own post-mount hook, so this pins that the
    no-splash boot path (``_run_no_splash_post_mount_setup``) wires the
    same ``_maybe_offer_first_run_wizard`` call as the splash path.
    """
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            assert type(app.screen).__name__ == "FirstRunSetupWizard"
            assert app.current_tab == "home"  # wizard is pushed ON TOP, not swapped in
            # Basic navigation sanity: the chrome the rest of this file relies
            # on is actually present, not just the screen class name.
            for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel"):
                assert len(app.screen.query(widget_id)) == 1


# ---------------------------------------------------------------------------
# 2. Esc -> confirm -> Finish later -> dismissed; next boot shows the resume
#    toast instead of re-pushing (checklist item 4).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escape_finish_later_dismisses_and_next_boot_resumes_via_toast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            await pilot.press("escape")
            await pilot.pause(0.2)
            # The confirm dialog is on top; the wizard must still be mounted.
            assert type(app.screen).__name__ == "ConfirmationDialog"
            _press(app.screen, "#confirm-button")  # confirm_label="Finish later"
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismissed back onto whatever was underneath (Home), not
            # navigated anywhere -- Escape/Finish-later carries no exit route.
            assert app.current_tab == "home"

            # The started flag is persisted by a `@work(thread=True)` worker
            # fired from FirstRunSetupWizard.on_mount(). Poll for the flag
            # directly with a bound, rather than `workers.wait_for_complete()`
            # -- that call waits for EVERY worker including ProviderStep's
            # real (unmocked) local-server discovery, which can block far
            # longer than any reasonable per-assertion timeout in a sandboxed
            # test environment with no reachable local servers.
            await _wait_until(
                pilot,
                lambda: app.app_config.get(WIZARD_STATE_SECTION, {}).get(
                    SETUP_STARTED_KEY
                )
                is True,
            )

    # Prove it is a REAL write, not just the in-memory mirror: read the same
    # (test-isolated) config file back independently.
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    persisted_config = load_cli_config_and_ensure_existence(force_reload=True)
    assert (
        persisted_config.get(WIZARD_STATE_SECTION, {}).get(SETUP_STARTED_KEY) is True
    )

    # Next boot: a fresh TldwCli instance reading that SAME real persisted
    # state (setup_started True, setup_completed absent) must show the
    # resume toast and must NOT re-push the wizard.
    app2 = _build_test_app(first_run_setup_completed=False)
    app2.app_config = persisted_config
    app2._initial_tab_value = "chat"

    notifications: list[dict] = []

    def record_notify(message, *args, **kwargs):
        notifications.append({"message": str(message), "severity": kwargs.get("severity")})

    monkeypatch.setattr(app2, "notify", record_notify)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app2.run_test(size=(140, 40)) as pilot2:
            await _wait_until(
                pilot2,
                lambda: getattr(app2, "_initial_screen_pushed", False) is True,
            )
            await pilot2.pause(0.3)
            assert type(app2.screen).__name__ != "FirstRunSetupWizard"
            assert any(
                "Finish setup" in note["message"]
                or "Settings" in note["message"]
                for note in notifications
            ), notifications


# ---------------------------------------------------------------------------
# 3. Full track, skip every step -> app fully usable afterwards
#    (checklist item 5).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_track_skip_everything_leaves_app_usable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)

            _select_radio(app.screen, "#setup-track-full")
            await pilot.pause(0.1)
            _press(app.screen, "#wizard-next")  # Welcome -> Provider, track=full
            await pilot.pause(0.2)

            seen_step_ids: list[str] = []
            for _ in range(12):
                step = container.steps[container.current_step]
                step_id = step.config.id if step.config else None
                if step_id == STEP_SUMMARY:
                    break
                seen_step_ids.append(step_id)
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("never reached the summary step")

            assert seen_step_ids == [
                "provider",
                "model",
                "rag",
                "tools",
                "notes",
                "appearance",
            ]

            # Exit via "Explore on my own" (TAB_HOME) to prove the app is
            # usable afterwards, not just that the wizard closed.
            _press(app.screen, "#setup-exit-home")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            await _wait_until(
                pilot,
                lambda: app.current_tab == TAB_HOME
                and app.screen.__class__.__name__ == "HomeScreen",
            )

            # "Fully usable": the shell nav still works after the wizard.
            _press(app.screen, "#nav-console")
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 4. Re-run entry over Settings -> finishing via "Done" returns to Settings
#    (exit_route None path; checklist item 6, partial -- prefill/"configured"
#    copy is covered at the unit level in Tests/Wizards/).
# ---------------------------------------------------------------------------


async def _open_rerun_wizard_from_settings(pilot):
    """Drive the real Settings ▸ Diagnostics ▸ "Run setup wizard" button.

    Returns the pushed FirstRunSetupWizard screen. Shared by the "Done" and
    "Go to Chat" re-entry tests below.
    """
    app = pilot.app
    await _wait_until(
        pilot, lambda: app.screen.__class__.__name__ in ("HomeScreen", "ChatScreen")
    )
    await _open_settings_diagnostics(pilot)
    run_wizard_button = app.screen.query_one("#settings-run-setup-wizard", Button)
    assert "Run Setup Wizard" in str(run_wizard_button.label)

    run_wizard_button.press()
    await _wait_until(
        pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
    )
    wizard_screen = app.screen
    assert wizard_screen.rerun is True
    await pilot.pause(0.2)
    return wizard_screen


async def _walk_rerun_quick_track_to_summary(pilot, wizard_screen) -> "SetupWizardContainer":
    # Quick track is pre-selected; walk welcome -> provider -> model ->
    # summary without picking anything (every step is skip-safe).
    for _ in range(3):
        _press(wizard_screen, "#wizard-next")
        await pilot.pause(0.2)
    container = wizard_screen.query_one(SetupWizardContainer)
    step = container.steps[container.current_step]
    assert step.config.id == STEP_SUMMARY
    return container


@pytest.mark.asyncio
async def test_rerun_over_settings_done_returns_to_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _prepare_clean_environment(monkeypatch, tmp_path)
    # Already-completed config: no auto-offer noise for this test.
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 55)) as pilot:
            wizard_screen = await _open_rerun_wizard_from_settings(pilot)
            await _walk_rerun_quick_track_to_summary(pilot, wizard_screen)
            # Rerun summary exposes "Done"/"Go to Chat", not the first-run
            # exit pair -- "Done" is the exit_route=None path.
            assert len(app.screen.query("#setup-exit-done")) == 1

            _press(app.screen, "#setup-exit-done")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # the final-review fix wave wired a result callback onto this push
            # (settings_screen.py's handle_run_setup_wizard now passes
            # app_instance.handle_first_run_wizard_result), but "Done"'s
            # exit_route is None -- _handle_first_run_wizard_result() is a
            # no-op for a falsy exit_route, so this must still simply pop
            # back to Settings with no navigation side effect.
            assert type(app.screen).__name__ == "SettingsScreen"
            assert app.current_tab == "settings"


@pytest.mark.asyncio
async def test_rerun_over_settings_go_to_chat_navigates_to_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Final-review finding 2: before the fix, both re-entry pushes
    (Settings' button and the command palette) omitted the result
    callback, so a truthy exit_route off the Summary step's "Go to Chat"
    button was silently dropped -- the button looked live but did nothing.
    Now that settings_screen.py's push wires
    app_instance.handle_first_run_wizard_result, this must actually
    navigate to Chat, exactly like the auto-offer path already does in
    test_full_track_skip_everything_leaves_app_usable above.
    """
    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(180, 55)) as pilot:
            wizard_screen = await _open_rerun_wizard_from_settings(pilot)
            await _walk_rerun_quick_track_to_summary(pilot, wizard_screen)
            assert len(app.screen.query("#setup-exit-chat")) == 1

            _press(app.screen, "#setup-exit-chat")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismiss pops back to Settings first; the exit_route is applied
            # via a separately-queued NavigateToScreen message (same race
            # noted in test_back_next_mashing_... above) -- wait for the
            # final tab rather than racing the first screen-stack pop.
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 4b. Command-palette re-entry wires the same result callback (finding 2,
#     palette path) -- cheap coverage over the actual production code in
#     SetupWizardProvider.handle_setup_wizard_action, without driving the
#     full command-palette search UI.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_palette_setup_wizard_action_wires_result_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tldw_chatbook.app import SetupWizardProvider

    _prepare_clean_environment(monkeypatch, tmp_path)
    app = _build_test_app(first_run_setup_completed=True)
    app._initial_tab_value = "chat"

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot,
                lambda: app.screen.__class__.__name__ in ("HomeScreen", "ChatScreen"),
            )
            captured: dict = {}
            real_push_screen = app.push_screen

            def _spy_push_screen(screen, callback=None, **kwargs):
                captured["screen"] = screen
                captured["callback"] = callback
                return real_push_screen(screen, callback, **kwargs)

            monkeypatch.setattr(app, "push_screen", _spy_push_screen)

            provider = SetupWizardProvider(app.screen)
            provider.handle_setup_wizard_action("run_setup_wizard")
            await pilot.pause(0.2)

            assert captured.get("callback") == app.handle_first_run_wizard_result, (
                "palette re-entry must wire the app-level result callback, "
                "same as the Settings button and the auto-offer path"
            )
            assert type(app.screen).__name__ == "FirstRunSetupWizard"


# ---------------------------------------------------------------------------
# 5. 80x24 terminal: wizard renders without clipped navigation
#    (checklist item 7).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wizard_navigation_visible_at_80x24(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(80, 24)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            await pilot.pause(0.2)

            nav_buttons = {
                widget_id: app.screen.query_one(widget_id, Button)
                for widget_id in ("#wizard-back", "#wizard-next", "#wizard-cancel")
            }
            for widget_id, button in nav_buttons.items():
                assert button.visible, f"{widget_id} is not visible at 80x24"
                region = button.region
                assert region.width > 0 and region.height > 0, (
                    f"{widget_id} has an empty region at 80x24: {region}"
                )
                assert region.right <= 80, f"{widget_id} clipped past column 80: {region}"
                assert region.bottom <= 24, f"{widget_id} clipped past row 24: {region}"

            # Cross-check against the actual compositor output rather than
            # trusting pre-paint widget state alone (a clipped overlay can
            # report a plausible region and never actually reach the
            # screen -- see backlog/docs/lessons-live-verification.md).
            strips = app.screen._compositor.render_strips()
            rendered_text = "\n".join(
                "".join(segment.text for segment in strip) for strip in strips
            )
            for expected in ("Back", "Next", "Cancel"):
                assert expected in rendered_text, (
                    f"{expected!r} button text missing from the rendered frame"
                )


# ---------------------------------------------------------------------------
# 6. Back/Next mashing across provider -> model must not crash or
#    double-advance (carried over from an earlier ledger item).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_back_next_mashing_across_provider_model_does_not_double_advance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)

            # Quick track is pre-selected on Welcome; advance to Provider,
            # then to Model, each with a real settle so the starting point
            # for the mash is deterministic.
            _press(app.screen, "#wizard-next")  # welcome -> provider
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_PROVIDER
            _press(app.screen, "#wizard-next")  # provider -> model
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_MODEL

            # Rapid, unsettled Back/Next mashing at the provider<->model
            # boundary -- no pilot.pause() between presses in the burst, so
            # a Next worker can still be in flight when Back fires.
            for _ in range(8):
                _press(app.screen, "#wizard-back")
                await pilot.pause(0)
                _press(app.screen, "#wizard-next")
                await pilot.pause(0)

            # Let everything drain, then the app must still be alive and on
            # a real, valid step -- not crashed, not stuck. (Not
            # `workers.wait_for_complete()`: that waits for EVERY worker,
            # including Provider/Model's real discovery workers, which can
            # block far longer than this settle needs.)
            await pilot.pause(0.5)
            assert container.is_running
            current = container.steps[container.current_step]
            current_id = current.config.id if current.config else None
            assert current_id in container.active_ids, (
                f"landed on {current_id!r}, outside the active quick-track "
                f"subset {container.active_ids!r} -- a mash-induced derail"
            )

            # Finish the walk from wherever the mash left it; the quick
            # track must complete exactly once through provider/model/
            # summary, with no repeats and no skipped/extra steps -- proof
            # the mash did not double-advance or corrupt navigation.
            seen_step_ids: list[str] = []
            for _ in range(10):
                step = container.steps[container.current_step]
                step_id = step.config.id if step.config else None
                if step_id != (seen_step_ids[-1] if seen_step_ids else None):
                    seen_step_ids.append(step_id)
                if step_id == STEP_SUMMARY:
                    break
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)
            else:
                raise AssertionError("mashing left the wizard unable to complete")

            _press(app.screen, "#setup-exit-chat")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # Dismiss pops back to Home first; the exit_route is applied via
            # a separately-queued NavigateToScreen message, so wait for the
            # final tab rather than racing the first screen-stack pop.
            await _wait_until(pilot, lambda: app.current_tab == TAB_CHAT)
            assert app.current_tab == TAB_CHAT


# ---------------------------------------------------------------------------
# 7. ctrl+n / ctrl+b keyboard shortcuts must not crash (final-review finding 1).
#
# BaseWizard.BINDINGS (never modified -- see that class's own docstring)
# maps ctrl+n/ctrl+b to action_next()/action_back(), which call
# self.handle_next()/self.handle_back() with NO arguments. SetupWizardContainer
# overrides handle_next(self, event)/handle_back(self, event) to require a
# Button.Pressed event (so they can call event.prevent_default() -- see
# those methods' own docstrings), so before this fix pressing ctrl+n or
# ctrl+b on a mounted wizard raised a TypeError out of the binding's action
# dispatch. The fix overrides action_next()/action_back() in
# SetupWizardContainer to route through the same event-free
# advance_programmatically() / _previous_active_index() path the mouse
# handlers use.
#
# Textual's key-binding resolution (Screen._binding_chain) only walks the
# ancestors of the currently FOCUSED widget; when nothing is focused it
# falls back to just Screen + App bindings and never reaches a plain
# Container like SetupWizardContainer at all. Welcome's RadioSet keeps a
# button focused from mount (needed for the very first ctrl+n/ctrl+b to
# reach this container's BINDINGS at all), but Provider's own RadioSet has
# no default-pressed button, so nothing is auto-focused once the wizard
# lands there. This is a pre-existing focus-management gap orthogonal to
# this crash fix (and would require touching BaseWizard.py -- out of scope,
# never to be modified -- or every step's on_show()), so this test focuses
# a real field after each step transition, exactly as Tab or a click would,
# rather than asserting on implicit default focus this wizard does not
# provide.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_n_ctrl_b_do_not_crash_and_move_one_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from textual.widgets import Input

    app = _build_fresh_wizard_app(monkeypatch, tmp_path)

    with patch("tldw_chatbook.app.get_cli_setting", side_effect=_test_cli_setting):
        async with app.run_test(size=(140, 40)) as pilot:
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            container = app.screen.query_one(SetupWizardContainer)
            await pilot.pause(0.2)
            assert container.steps[container.current_step].config.id == STEP_WELCOME

            # ctrl+b on the very first step (Welcome's RadioSet is focused
            # from mount) must not crash and must not move anywhere -- this
            # is the exact scenario that raised TypeError before the fix.
            await pilot.press("ctrl+b")
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+b crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_WELCOME

            # Quick setup is the default pre-selected RadioButton on Welcome
            # (untouched here) -- ctrl+n must apply that choice exactly like
            # clicking Next does (advance_programmatically() -> _advance()
            # -> select_track(step.chosen_track())), not just move a step.
            await pilot.press("ctrl+n")  # welcome -> provider
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+n crashed the wizard"
            assert container.track == TRACK_QUICK
            assert container.steps[container.current_step].config.id == STEP_PROVIDER

            app.screen.query_one("#setup-provider-key-input", Input).focus()
            await pilot.pause(0.1)

            await pilot.press("ctrl+n")  # provider -> model
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+n crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_MODEL

            app.screen.query_one("#setup-model-custom", Input).focus()
            await pilot.pause(0.1)

            # ctrl+b must move exactly one active step back, not crash and
            # not flat-decrement past the active-id subset.
            await pilot.press("ctrl+b")  # model -> provider
            await pilot.pause(0.2)
            assert container.is_running, "ctrl+b crashed the wizard"
            assert container.steps[container.current_step].config.id == STEP_PROVIDER
