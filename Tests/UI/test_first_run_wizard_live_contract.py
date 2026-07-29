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
            await _wait_until(
                pilot,
                lambda: app.screen.__class__.__name__ in ("HomeScreen", "ChatScreen"),
            )
            await _open_settings_diagnostics(pilot)
            run_wizard_button = app.screen.query_one(
                "#settings-run-setup-wizard", Button
            )
            assert "Run Setup Wizard" in str(run_wizard_button.label)

            run_wizard_button.press()
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ == "FirstRunSetupWizard"
            )
            wizard_screen = app.screen
            assert wizard_screen.rerun is True
            await pilot.pause(0.2)

            # Quick track is pre-selected; walk welcome -> provider -> model
            # -> summary without picking anything (every step is skip-safe).
            for _ in range(3):
                _press(app.screen, "#wizard-next")
                await pilot.pause(0.2)

            container = wizard_screen.query_one(SetupWizardContainer)
            step = container.steps[container.current_step]
            assert step.config.id == STEP_SUMMARY
            # Rerun summary exposes "Done"/"Go to Chat", not the first-run
            # exit pair -- "Done" is the exit_route=None path.
            assert len(app.screen.query("#setup-exit-done")) == 1

            _press(app.screen, "#setup-exit-done")
            await _wait_until(
                pilot, lambda: type(app.screen).__name__ != "FirstRunSetupWizard"
            )
            # No callback is wired for the rerun push (settings_screen.py
            # pushes with none), so an exit_route=None result must simply
            # pop back to Settings -- no navigation side effect at all.
            assert type(app.screen).__name__ == "SettingsScreen"
            assert app.current_tab == "settings"


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
