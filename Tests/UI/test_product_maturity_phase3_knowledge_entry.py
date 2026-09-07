"""Product maturity Phase 3.1 Library knowledge entry contract."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)
from tldw_chatbook.Constants import TAB_STUDY
import tldw_chatbook.app as app_module
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    HandoffChannel,
    PendingHandoffStore,
)
from tldw_chatbook.UI.Screens.study_screen import StudyScreen


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKER = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
PHASE_3_1_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-1-library-study-entry.md"
)
TASK_10 = Path(
    "backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md"
)
TASK_10_1 = Path(
    "backlog/tasks/task-10.1 - Product-Maturity-Phase-3.1-Library-Study-Entry.md"
)


def _text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _disable_splash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable only the production splash setting."""
    real_get_cli_setting = app_module.get_cli_setting

    def get_cli_setting_without_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(app_module, "get_cli_setting", get_cli_setting_without_splash)


async def _close_production_app(app: TldwCli) -> None:
    """Release production-app resources even when the assertion fails."""
    try:
        if app._rich_log_handler:
            await app._rich_log_handler.stop_processor()
            logging.getLogger().removeHandler(app._rich_log_handler)
            app._rich_log_handler.close()
        await app.on_shutdown_request()
        await app.on_unmount()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_library_surfaces_study_workflow_entry_points() -> None:
    # The retired hub grouped these three entry points under a "Learning"
    # heading inside the now-gone #library-source-browser, with a single
    # static tooltip per button regardless of mode. The rail + canvas shell
    # groups the same three rows under the "Create" rail section instead,
    # and each row's canvas mounts exactly one live handoff button (the D2
    # fix), with tooltip copy naming the button's own action label.
    app = _build_test_app()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#library-row-create-study")

        assert "Create" in _visible_text(screen)
        for row_id in ("create-study", "create-flashcards", "create-quizzes"):
            assert screen.query_one(f"#library-row-{row_id}")

        # The button carries the verb-based handoff label (F1b L2) while the
        # tooltip keeps naming the concrete Study destination the snapshot
        # opens into (the mode's action_label).
        expected = {
            "study": ("library-open-study", "Study Dashboard"),
            "flashcards": ("library-open-flashcards", "Flashcards"),
            "quizzes": ("library-open-quizzes", "Quizzes"),
        }
        for mode, (button_id, action_label) in expected.items():
            screen.query_one(f"#library-row-create-{mode}", Button).press()
            await _wait_for_selector(screen, pilot, f"#{button_id}")
            button = screen.query_one(f"#{button_id}", Button)
            assert str(button.label) == "Continue in Study"
            assert str(button.tooltip) == (
                f"Open {action_label} with the current Library source snapshot, "
                "or globally when none is available."
            )


@pytest.mark.asyncio
async def test_library_study_entry_buttons_preserve_requested_section() -> None:
    app = _build_test_app()
    app.open_study_screen = Mock()
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 40)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_selector(screen, pilot, "#library-row-create-flashcards")

        # Reaching the Flashcards/Quizzes handoff buttons now requires
        # selecting their Create rail row first (they only mount inside
        # their own mode canvas).
        screen.query_one("#library-row-create-flashcards", Button).press()
        await _wait_for_selector(screen, pilot, "#library-open-flashcards")
        screen.query_one("#library-open-flashcards", Button).press()
        # A plain "was it called" check would already be true on the
        # quizzes press below (from this flashcards call) -- track the
        # count explicitly so each press's wait can't pass on a stale
        # earlier call.
        calls_before = app.open_study_screen.call_count
        for _ in range(150):
            if app.open_study_screen.call_count > calls_before:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("open_study_screen was never called for flashcards.")
        app.open_study_screen.assert_called_with(initial_section="flashcards")

        screen.query_one("#library-row-create-quizzes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-open-quizzes")
        calls_before = app.open_study_screen.call_count
        screen.query_one("#library-open-quizzes", Button).press()
        for _ in range(150):
            if app.open_study_screen.call_count > calls_before:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("open_study_screen was never called for quizzes.")
        app.open_study_screen.assert_called_with(initial_section="quizzes")


@pytest.mark.asyncio
async def test_study_screen_consumes_pending_initial_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_splash(monkeypatch)
    app = TldwCli()
    app.app_config["_first_run"] = False
    app.app_config.setdefault("first_run", {})["setup_completed"] = True
    app._initial_tab_value = "home"

    try:
        async with app.run_test() as pilot:
            for _ in range(150):
                if app.current_tab == "home":
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError(
                    f"initial route never became 'home' (was {app.current_tab!r})."
                )

            app.open_study_screen(initial_section="quizzes")
            for _ in range(150):
                if (
                    app.current_tab == TAB_STUDY
                    and isinstance(app.screen, StudyScreen)
                    and app.screen.current_section == "quizzes"
                ):
                    break
                await pilot.pause(0.02)
            else:
                raise AssertionError(
                    "current_section never became 'quizzes' (was "
                    f"{getattr(app.screen, 'current_section', None)!r})."
                )

            assert app.screen.current_section == "quizzes"
            assert not app.pending_handoffs.has_pending(
                HandoffChannel.STUDY_INITIAL_SECTION
            )
    finally:
        await _close_production_app(app)


def test_tldwcli_open_study_screen_accepts_initial_section() -> None:
    app = object.__new__(TldwCli)
    app.pending_handoffs = PendingHandoffStore()
    app.notify = Mock()
    app.post_message = Mock()

    TldwCli.open_study_screen(app, initial_section="flashcards")

    claim = app.pending_handoffs.claim(HandoffChannel.STUDY_INITIAL_SECTION)
    assert claim is not None
    assert claim.value == "flashcards"
    assert app.post_message.call_args.args[0].screen_name == TAB_STUDY


def test_pending_study_initial_section_overrides_restored_section() -> None:
    store = PendingHandoffStore()
    store.stage(HandoffChannel.STUDY_INITIAL_SECTION, "quizzes")
    applied_sections: list[str] = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(pending_handoffs=store),
        current_section="flashcards",
    )
    screen._apply_section_layout = lambda: applied_sections.append(
        screen.current_section
    )

    StudyScreen._apply_pending_section_handoff(screen)

    assert screen.current_section == "quizzes"
    assert applied_sections == ["quizzes"]
    assert not store.has_pending(HandoffChannel.STUDY_INITIAL_SECTION)
