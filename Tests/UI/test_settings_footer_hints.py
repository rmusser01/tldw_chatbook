"""TASK-1340: Settings footer hints must be honest (ADR-031).

Before this task the footer advertised ``s save category`` / ``r revert
category`` / ``t test category`` for ALL 19 Settings categories even though
``t`` is implemented for only five of them and ``s``/``r`` only apply to the
five guided-mutation categories; pressing those keys while a text input had
focus silently no-op'd; and revert discarded all staged edits instantly with
no confirmation.

These tests pin:
* per-category advertised hint sets (only working keys are advertised);
* REAL key behavior for s/r/t (task-1367): with a text field focused the
  printable keys type into the field (Textual Input consumes them -- the
  field IS the feedback); with focus outside text entry the screen bindings
  fire their actions;
* a confirmation dialog before a dirty category's staged edits are
  discarded (clean categories keep the "nothing to revert" path);
* narrow-width discoverability: task-2860/LIB-18 reordered the responsive
  ladder so the screen's OWN hints outrank the global cluster -- the
  globals compact first (``AppFooterStatus.GLOBAL_HINTS_COMPACT``) while
  the screen hints stay intact, and only once even that no longer fits
  does the screen context collapse to an ellipsis. F1 help must still list
  the ACTIVE category's working shortcuts once that collapse happens.
"""

import pytest
from textual.widgets import Input, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    ALL_CATEGORY_IDS,
    _click_settings_category,
    _settle_settings,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import (
    GUIDED_SETTINGS_MUTATION_CATEGORIES,
    SettingsScreen,
)
from tldw_chatbook.UI.Workbench.help import WorkbenchHelpPanel
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

_HINT_LABELS = ("save category", "revert category")
# task-1714: the `t` hint uses each category's real verb, so the honest-set
# checks must track every advertised verb, not a single generic label.
_TEST_VERB_LABELS = frozenset(SettingsScreen.TEST_ACTION_LABELS.values())
_ALL_KNOWN_LABELS = frozenset(_HINT_LABELS) | _TEST_VERB_LABELS | {"test category"}


def _expected_labels(category: SettingsCategoryId) -> tuple[str, ...]:
    return tuple(
        label for _key, label in SettingsScreen._category_footer_shortcuts(category)
    )


def test_category_footer_shortcuts_only_advertise_working_keys():
    """Pure mapping check across all 12 sidebar categories (no Pilot needed)."""
    assert len(ALL_CATEGORY_IDS) >= 12
    for category_value in ALL_CATEGORY_IDS:
        category = SettingsCategoryId(category_value)
        shortcuts = SettingsScreen._category_footer_shortcuts(category)
        keys = tuple(key for key, _label in shortcuts)
        # s/r are only advertised where the guided save/revert path exists.
        assert ("s" in keys) == (category in GUIDED_SETTINGS_MUTATION_CATEGORIES)
        assert ("r" in keys) == (category in GUIDED_SETTINGS_MUTATION_CATEGORIES)
        # t is only advertised where a test action is actually implemented.
        assert ("t" in keys) == (category in SettingsScreen.TESTABLE_SETTINGS_CATEGORIES)
        # Keys and labels stay in lockstep (ADR-031 rule 4: 1:1, no stubs).
        expected_labels = []
        if category in GUIDED_SETTINGS_MUTATION_CATEGORIES:
            expected_labels += ["save category", "revert category"]
        if category in SettingsScreen.TESTABLE_SETTINGS_CATEGORIES:
            expected_labels += [
                SettingsScreen.TEST_ACTION_LABELS.get(category, "test category")
            ]
        assert tuple(label for _key, label in shortcuts) == tuple(expected_labels)


@pytest.mark.asyncio
async def test_footer_hints_follow_the_active_category():
    """The live footer re-registers its hint set on every category switch."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        screen = _active_destination_screen(host)
        footer = screen.query_one(AppFooterStatus)

        for category_value in ALL_CATEGORY_IDS:
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)
            footer = screen.query_one(AppFooterStatus)
            expected = set(_expected_labels(SettingsCategoryId(category_value)))
            for label in expected:
                assert label in footer.shortcut_text, (
                    f"{category_value}: footer must advertise {label!r}, "
                    f"got {footer.shortcut_text!r}"
                )
            for label in _ALL_KNOWN_LABELS - expected:
                assert label not in footer.shortcut_text, (
                    f"{category_value}: footer must NOT advertise dead key "
                    f"{label!r}, got {footer.shortcut_text!r}"
                )


@pytest.mark.asyncio
async def test_printable_shortcut_keys_type_into_focused_text_fields():
    """Real key behavior (task-1367): Textual Input consumes printable keys,
    so s/r/t with a text field focused legitimately type into the field --
    the field IS the feedback. No action fires, no dialog, no crash."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "providers-models")
        screen = _active_destination_screen(host)
        model_input = screen.query_one("#settings-model-value", Input)
        model_input.focus()
        await pilot.pause()
        assert screen.app.focused is model_input
        before = model_input.value

        toasts = []
        host.notify = lambda message, **kwargs: toasts.append(message)

        await pilot.press("s", "r", "t")
        await pilot.pause()

        # The keys typed into the field; none of the s/r/t actions fired.
        # Not `before + "srt"`: Textual's `Input` ships `select_on_focus=True`,
        # so focusing a NON-EMPTY field selects its contents and the first
        # keystroke replaces them. That only became visible once the test
        # config stopped being empty (task-15270) -- the shipping app has
        # `[chat_defaults] model` set from the config template, so this is
        # the behaviour a real user gets, and the claim under test (printable
        # keys reach the field, not the screen bindings) is unchanged.
        assert model_input.value.endswith("srt")
        assert model_input.value != before
        assert not isinstance(host.screen, ConfirmationDialog)
        assert not toasts, f"no action toasts must fire, got {toasts}"


@pytest.mark.asyncio
async def test_shortcut_keys_fire_actions_outside_text_entry():
    """With focus outside text entry, s/r/t reach the screen bindings:
    s -> save notice, r -> revert confirmation dialog, t -> test stub toast."""
    app = _build_test_app()
    app.app_config["web_server"] = {"font_size": 12}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        toasts = []
        host.notify = lambda message, **kwargs: toasts.append(message)

        # t on a category without a test action -> honest stub toast.
        await _click_settings_category(pilot, "theme")
        await pilot.press("t")
        await pilot.pause()
        assert any("No test action is available" in toast for toast in toasts), (
            f"expected the test stub toast, got {toasts}"
        )

        # s on a clean guided category -> "no changes" notice, nothing saved.
        await _click_settings_category(pilot, "console-behavior")
        toasts.clear()
        await pilot.press("s")
        await pilot.pause()
        assert any("No Settings changes to save" in toast for toast in toasts), (
            f"expected the clean-save notice, got {toasts}"
        )

        # r on a dirty guided category -> the revert confirmation dialog.
        await _click_settings_category(pilot, "appearance")
        screen = _active_destination_screen(host)
        font_size = screen.query_one("#settings-appearance-font-size", Input)
        font_size.value = "16"
        screen.handle_appearance_font_size_changed(
            Input.Changed(font_size, font_size.value)
        )
        category = SettingsCategoryId.APPEARANCE
        assert screen._category_has_unsaved_changes(category)
        await pilot.press("r")
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        assert screen._category_has_unsaved_changes(category)  # nothing discarded
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert not isinstance(host.screen, ConfirmationDialog)
        assert font_size.value == "16"


@pytest.mark.asyncio
async def test_revert_with_unsaved_changes_requires_confirmation():
    """Dirty-category revert asks first; confirm discards, cancel keeps."""
    app = _build_test_app()
    app.app_config["web_server"] = {"font_size": 12}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "appearance")
        screen = _active_destination_screen(host)
        font_size = screen.query_one("#settings-appearance-font-size", Input)
        font_size.value = "16"
        screen.handle_appearance_font_size_changed(
            Input.Changed(font_size, font_size.value)
        )
        category = SettingsCategoryId.APPEARANCE
        assert screen._category_has_unsaved_changes(category)

        screen.action_settings_revert_category()
        await pilot.pause()

        # A confirmation dialog is up and NOTHING was discarded yet.
        dialog = host.screen
        assert isinstance(dialog, ConfirmationDialog)
        assert screen._category_has_unsaved_changes(category)
        assert font_size.value == "16"

        # Cancel keeps the staged edits.
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert not isinstance(host.screen, ConfirmationDialog)
        assert screen._category_has_unsaved_changes(category)
        assert font_size.value == "16"

        # Re-invoke and confirm: now the staged edits are discarded.
        screen.action_settings_revert_category()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(category)
        assert font_size.value == "12"


@pytest.mark.asyncio
async def test_revert_without_changes_keeps_the_nothing_to_revert_path():
    """A clean category never sees the confirmation dialog."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "console-behavior")
        screen = _active_destination_screen(host)
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.CONSOLE_BEHAVIOR
        )

        toasts = []
        host.notify = lambda message, **kwargs: toasts.append(message)
        screen.action_settings_revert_category()
        await pilot.pause()

        assert not isinstance(host.screen, ConfirmationDialog)
        assert any("No Settings changes to revert" in toast for toast in toasts)


@pytest.mark.asyncio
async def test_narrow_footer_collapses_but_f1_help_stays_truthful():
    """The screen's own hints outrank the global cluster (task-2860/LIB-18).

    The ladder compacts the globals first, then retains the highest-priority
    screen actions that fit. At 70 columns Storage keeps ``s`` visible while
    lower-priority ``r``/``t`` move to F1 help, where every active shortcut
    remains discoverable."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    # At 70 columns the responsive prefix keeps only the primary Settings
    # action alongside compact globals.
    async with host.run_test(size=(70, 28)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "storage")
        screen = _active_destination_screen(host)
        footer = screen.query_one(AppFooterStatus)

        collapsed_text = str(footer.query_one("#footer-key-quit", Static).renderable)
        assert collapsed_text.startswith("s save category"), collapsed_text
        assert "revert category" not in collapsed_text
        assert "check storage" not in collapsed_text

        screen.action_show_workbench_help()
        await pilot.pause()

        panel = host.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        help_text = panel.state.render_text()
        for label in _expected_labels(SettingsCategoryId.STORAGE):
            assert label in help_text, (
                f"F1 help must keep {label!r} discoverable, got {help_text!r}"
            )

        await pilot.press("escape")
        await pilot.pause()

        # On a read-only category the help panel is equally honest: no dead
        # keys are taught.
        await _click_settings_category(pilot, "theme")
        screen = _active_destination_screen(host)
        screen.action_show_workbench_help()
        await pilot.pause()
        panel = host.screen
        assert isinstance(panel, WorkbenchHelpPanel)
        help_text = panel.state.render_text()
        for label in _ALL_KNOWN_LABELS:
            assert label not in help_text


_TEST_STUB_TOAST = "No test action is available"
_SAVE_STUB_TOAST = "has no save action yet"


@pytest.mark.asyncio
async def test_advertised_capabilities_match_real_action_branches():
    """Drift guard (task-1340 review): advertised keys must track REAL code.

    Probes the actual actions per category instead of trusting the
    frozensets:

    * ``t``: the "No test action is available" stub toast must fire IFF the
      category is NOT in TESTABLE_SETTINGS_CATEGORIES. Adding a test
      branch (or deleting one) without updating the frozenset goes red.
    * ``s``: probed in CLEAN state only, so nothing is ever written -- every
      guided save branch early-returns "No Settings changes to save." when
      the category has no staged edits. The "has no save action yet"
      fallthrough stub must NEVER fire for a GUIDED_SETTINGS_MUTATION_
      CATEGORIES member (a removed save branch goes red). The reverse
      direction (a real save branch added for a non-guided category while
      the frozenset stays stale) is structurally unreachable: the action's
      own ``category not in GUIDED...`` guard returns first -- the frozenset
      IS the save capability gate, unlike the test side, where the stub is
      a fallthrough. Dirty-state save behavior per guided category is
      covered by the real save tests in test_settings_configuration_hub.py.
    """
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        toasts = []
        host.notify = lambda message, **kwargs: toasts.append(message)

        for category_value in ALL_CATEGORY_IDS:
            await _click_settings_category(pilot, category_value)
            screen = _active_destination_screen(host)
            category = SettingsCategoryId(category_value)

            toasts.clear()
            screen.action_settings_test_category()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            test_stubbed = any(_TEST_STUB_TOAST in toast for toast in toasts)
            testable = category in SettingsScreen.TESTABLE_SETTINGS_CATEGORIES
            assert test_stubbed == (not testable), (
                f"{category_value}: test stub fired={test_stubbed} but category "
                f"membership in TESTABLE_SETTINGS_CATEGORIES={testable} -- "
                "update the frozenset or the action branches (toasts: {toasts})"
            )

            toasts.clear()
            screen.action_settings_save_category()
            await pilot.app.workers.wait_for_complete()
            await pilot.pause()
            save_stubbed = any(_SAVE_STUB_TOAST in toast for toast in toasts)
            if category in GUIDED_SETTINGS_MUTATION_CATEGORIES:
                assert not save_stubbed, (
                    f"{category_value}: guided category hit the save stub -- a "
                    "save branch was removed without updating "
                    f"GUIDED_SETTINGS_MUTATION_CATEGORIES (toasts: {toasts})"
                )
            else:
                # Non-guided: guidance only, never a completed real save
                # (save-success toasts end in "saved."; guidance copy like
                # "Splash defaults are saved automatically." does not).
                assert not any("saved." in toast for toast in toasts), (
                    f"{category_value}: non-guided category performed a save "
                    f"(toasts: {toasts})"
                )


@pytest.mark.asyncio
async def test_stacked_revert_dialogs_revert_only_once():
    """Two queued confirms must not double the revert (double toast)."""
    app = _build_test_app()
    app.app_config["web_server"] = {"font_size": 12}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "appearance")
        screen = _active_destination_screen(host)
        font_size = screen.query_one("#settings-appearance-font-size", Input)
        font_size.value = "16"
        screen.handle_appearance_font_size_changed(
            Input.Changed(font_size, font_size.value)
        )
        category = SettingsCategoryId.APPEARANCE
        assert screen._category_has_unsaved_changes(category)

        toasts = []
        host.notify = lambda message, **kwargs: toasts.append(message)

        # Two stacked dialogs before either is confirmed.
        screen.action_settings_revert_category()
        await pilot.pause()
        screen.action_settings_revert_category()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)

        await pilot.click("#confirm-button")
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)  # second one underneath
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert not screen._category_has_unsaved_changes(category)
        assert font_size.value == "12"
        reverted = [toast for toast in toasts if "changes reverted" in toast]
        assert len(reverted) == 1, f"expected exactly one revert toast, got {reverted}"
