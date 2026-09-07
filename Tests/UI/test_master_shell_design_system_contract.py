import pytest
import re
from pathlib import Path


REQUIRED_DESIGN_SYSTEM_CLASSES = {
    "ds-destination-header",
    "ds-panel",
    "ds-inspector",
    "ds-status-badge",
    "ds-recovery-callout",
    "ds-source-role",
    "ds-approval-card",
    "ds-event-row",
    "ds-field-row",
    "ds-toolbar",
    "ds-shortcut-bar",
}

REQUIRED_STATE_CLASSES = {
    "is-active",
    "is-disabled",
    "is-blocked",
    "is-running",
    "is-paused",
    "is-unsaved",
    "is-stale",
    "is-conflict",
    "needs-approval",
    "source-local",
    "source-server",
    "source-workspace",
    "source-remote-only",
    "source-dry-run",
}

REQUIRED_SEMANTIC_TOKENS = {
    "ds-surface-panel",
    "ds-text-primary",
    "ds-action-focus",
    "ds-focus-fg",
    "ds-focus-bg",
    "ds-focus-accent",
    "ds-input-focus-border",
    "ds-input-focus-bg",
    "ds-input-focus-accent",
    "ds-status-ready",
    "ds-status-warning",
    "ds-status-error",
    "ds-authority-local",
    "ds-source-role-evidence",
}

READABLE_STATUS_LABELS = {
    "Ready",
    "Running",
    "Paused",
    "Blocked",
    "Unavailable",
    "Approval required",
    "Unsaved",
    "Recovered",
}

DESIGN_SYSTEM_SPEC = Path(
    "Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md"
)
DESIGN_SYSTEM_TCSS = Path("tldw_chatbook/css/components/_agentic_terminal.tcss")
CORE_VARIABLES_TCSS = Path("tldw_chatbook/css/core/_variables.tcss")
MAIN_TCSS = Path("tldw_chatbook/css/main.tcss")
LOADED_TCSS = Path("tldw_chatbook/css/tldw_cli_modular.tcss")
BUILD_CSS_PY = Path("tldw_chatbook/css/build_css.py")
APP_PY = Path("tldw_chatbook/app.py")
THEMES_PY = Path("tldw_chatbook/css/Themes/themes.py")
LIBRARY_SCREEN_PY = Path("tldw_chatbook/UI/Screens/library_screen.py")
CONTRACT_DOC = Path("Docs/Design/master-shell-design-system-contract.md")


def assert_no_dotted_design_tokens(text: str) -> None:
    assert re.search(r"\$ds\.", text) is None


def test_master_shell_design_system_class_contract_is_documented():
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in text


def test_master_shell_design_system_status_contract_is_documented():
    text = CONTRACT_DOC.read_text(encoding="utf-8")
    for label in READABLE_STATUS_LABELS:
        assert label in text


def test_agentic_terminal_design_system_spec_is_present():
    assert DESIGN_SYSTEM_SPEC.exists()


def test_agentic_terminal_tcss_module_is_implemented_and_imported():
    assert DESIGN_SYSTEM_TCSS.exists()
    class_text = DESIGN_SYSTEM_TCSS.read_text(encoding="utf-8")
    main_text = MAIN_TCSS.read_text(encoding="utf-8")
    build_text = BUILD_CSS_PY.read_text(encoding="utf-8")

    assert '@import "./components/_agentic_terminal.tcss";' in main_text
    assert '"components/_agentic_terminal.tcss"' in build_text
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in class_text
    assert ".density-compact" in class_text
    assert ".density-comfortable" in class_text


def test_loaded_stylesheet_contains_agentic_terminal_contract():
    loaded_text = LOADED_TCSS.read_text(encoding="utf-8")
    app_text = APP_PY.read_text(encoding="utf-8")

    assert "tldw_cli_modular.tcss" in app_text
    assert "components/_agentic_terminal.tcss" in loaded_text
    for class_name in REQUIRED_DESIGN_SYSTEM_CLASSES | REQUIRED_STATE_CLASSES:
        assert f".{class_name}" in loaded_text
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f"${token_name}" in loaded_text


def test_agentic_terminal_semantic_tokens_and_theme_exist():
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DESIGN_SYSTEM_TCSS, CORE_VARIABLES_TCSS)
        if path.exists()
    )
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f"${token_name}" in source_text

    themes_text = THEMES_PY.read_text(encoding="utf-8")
    assert "agentic_terminal" in themes_text


def test_design_system_tokens_use_textual_safe_names():
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (CORE_VARIABLES_TCSS, DESIGN_SYSTEM_TCSS)
    )

    assert_no_dotted_design_tokens(source_text)
    assert "$ds-" in source_text


def test_agentic_terminal_theme_variables_cover_required_tokens():
    themes_text = THEMES_PY.read_text(encoding="utf-8")
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f'"{token_name}"' in themes_text


def test_generated_stylesheet_preserves_textual_safe_tokens():
    loaded_text = LOADED_TCSS.read_text(encoding="utf-8")

    assert_no_dotted_design_tokens(loaded_text)
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert f"${token_name}" in loaded_text


def test_status_contract_requires_readable_labels():
    text = DESIGN_SYSTEM_TCSS.read_text(encoding="utf-8")
    for label in READABLE_STATUS_LABELS:
        assert label in text


def test_library_mode_chip_selector_is_retired():
    """``.library-mode-chip`` (the retired horizontal mode-strip's own
    selector, in every variant: base rule, ``:focus``, ``.is-active``, and
    ``.is-active:focus``) was deleted wholesale in L3b Task 9 along with
    ``LIBRARY_MODES`` and the rest of the mode-switch chrome the Library rail
    + canvas shell superseded. ``.notes-mode-chip`` later went the same way:
    the Notes screen rebuild retired its mode strip, and the orphaned
    selectors were pruned once they had zero Python users, leaving
    ``.personas-mode-chip`` as the only surviving mode-chip family (it keeps
    the base rule and the shared ``.is-active``/``.is-active:focus``
    variants). The ``$ds-library-mode-chip-*`` size token definitions
    survive in ``_variables.tcss``; only the Library-only
    ``$ds-library-mode-bar-height``/``$ds-library-mode-label-width``
    tokens, which had no other consumer, were deleted with the selector."""
    text = DESIGN_SYSTEM_TCSS.read_text(encoding="utf-8")
    variables = CORE_VARIABLES_TCSS.read_text(encoding="utf-8")
    library_screen = LIBRARY_SCREEN_PY.read_text(encoding="utf-8")

    assert ".library-mode-chip" not in text
    assert ".notes-mode-chip" not in text
    assert ".personas-mode-chip.is-active" in text

    assert "$ds-library-mode-bar-height" not in variables
    assert "$ds-library-mode-label-width" not in variables
    assert "$ds-library-mode-chip-height: 1;" in variables

    assert "library-mode-chip" not in library_screen
    assert "LIBRARY_MODES = {" not in library_screen
    assert "LIBRARY_MODE_BAR_HEIGHT" not in library_screen
    assert "LIBRARY_MODE_CHIP_WIDTH_PADDING" not in library_screen


@pytest.mark.unit
def test_disabled_menu_rows_survive_textuals_compounded_dim():
    """TASK-1801: disabled composer-menu rows must stay legible.

    Measured live, not read off token names -- which is why this survived
    review. Two dimmers compound and neither is visible in the stylesheet:
    the theme's ``text-disabled: auto 38%`` (~3.4:1 alone), and Textual's
    ``Button:disabled`` adding ``text-style: bold dim`` plus
    ``color: auto 50%``. All 58 shipped themes land below 3:1; the running
    app measured **1.05:1 and 1.25:1**, against enabled rows at 12.63:1.

    Two facts this test exists to pin, both established by measuring the
    running app after a fix that looked correct and was not:

    * the rule must live in the APP stylesheet. A screen's ``DEFAULT_CSS``
      and ``Button``'s sit in the same tier, where ``Button`` wins for a
      ``Button`` -- an identical rule inside the modal measured no change;
    * ``text-style: none`` does NOT clear Textual's ``dim``. The declared
      colour still renders at roughly half strength, so it has to be stated
      bright enough to survive that.
    """
    from pathlib import Path

    source = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text()

    match = re.search(
        r"\.console-composer-menu-item:disabled\s*\{([^}]*)\}", source, re.S
    )
    assert match, (
        "the disabled menu-row rule must live in the APP stylesheet -- inside "
        "the modal's DEFAULT_CSS it is outranked by Button's own rules and "
        "measurably does nothing"
    )
    body = match.group(1)

    colour = re.search(r"color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)", body)
    assert colour, "the disabled colour must be stated explicitly, not left to a token"
    channels = [int(c) for c in colour.groups()]
    # Textual halves it; the post-dim value must still clear 3:1 on the
    # modal's near-black background, which needs ~123+ after halving.
    assert min(channels) >= 240, (
        f"disabled colour {channels} is too dark to survive Textual's dim "
        f"halving -- it would render near {min(channels) // 2}, measured "
        "1.25:1 at rgb(150,150,150)"
    )


@pytest.mark.unit
def test_console_action_row_disabled_rules_still_clear_dim():
    """The action-row half of the same contract, so a future edit cannot
    quietly delete the `text-style: none` that already protects it."""
    from pathlib import Path

    source = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text()
    rules = re.findall(r"(\.console-action-disabled[^{]*)\{([^}]*)\}", source, re.S)
    assert rules, "no .console-action-disabled rules found"
    for selector, body in rules:
        assert not re.search(r"text-style:[^;]*\bdim\b", body), (
            f"{selector.strip()} compounds dim onto the theme alpha"
        )


@pytest.mark.unit
def test_workbench_disabled_actions_do_not_stack_three_dimmers():
    """TASK-1801: the Workbench action bar compounded THREE dimmers.

    Measured 1.45:1 in the running app. The layers, none of them visible
    from any single rule:

    1. ``.workbench-action.is-disabled`` sets ``color: $ds-text-disabled``
       -- an alpha token, ~38% over the panel;
    2. ``.is-disabled`` adds ``opacity: 0.55`` on top of that;
    3. Textual's ``Button:disabled`` adds ``text-style: bold dim`` and
       ``color: auto 50%``.

    This bar uses the class ``is-disabled``, NOT ``console-action-disabled``
    -- which is why an earlier edit to the latter measured no change at all
    and was reverted rather than shipped. Pinning the class here so the next
    person does not repeat that.

    Even with the stack broken, this bar's lighter background caps the
    achievable ratio below 4.5:1, so DESIGN.md states 3:1 as the floor.
    """
    from pathlib import Path

    workbench = Path("tldw_chatbook/css/components/_workbench.tcss").read_text()

    match = re.search(
        r"\.workbench-action\.is-disabled\s*\{([^}]*)\}", workbench, re.S
    )
    assert match, ".workbench-action.is-disabled rule is missing"
    body = match.group(1)

    colour = re.search(r"color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)", body)
    assert colour, (
        "the disabled colour must be stated explicitly: `$ds-text-disabled` "
        "is an alpha token that Textual's dim then halves again"
    )
    assert min(int(c) for c in colour.groups()) >= 240, (
        f"colour {colour.groups()} is too dark to survive dim halving"
    )
    assert re.search(r"opacity:\s*1", body), (
        "must neutralise the `.is-disabled { opacity: 0.55 }` layer, or a "
        "third multiplier stacks on and no colour can reach the floor"
    )
