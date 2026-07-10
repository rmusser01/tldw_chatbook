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

DESIGN_SYSTEM_SPEC = Path("Docs/superpowers/specs/2026-05-02-agentic-terminal-design-system-design.md")
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
    + canvas shell superseded. ``.notes-mode-chip``/``.personas-mode-chip``
    still render their own mode strips and keep both the base rule and the
    shared ``.is-active``/``.is-active:focus`` variants -- only the
    ``library-`` selector is gone. The ``$ds-library-mode-chip-*`` size
    tokens survive (still consumed by ``.notes-mode-chip``); only the
    Library-only ``$ds-library-mode-bar-height``/``$ds-library-mode-label-width``
    tokens, which had no other consumer, were deleted with the selector."""
    text = DESIGN_SYSTEM_TCSS.read_text(encoding="utf-8")
    variables = CORE_VARIABLES_TCSS.read_text(encoding="utf-8")
    library_screen = LIBRARY_SCREEN_PY.read_text(encoding="utf-8")

    assert ".library-mode-chip" not in text
    assert ".notes-mode-chip.is-active" in text
    assert ".notes-mode-chip.is-active:focus" in text
    assert ".personas-mode-chip.is-active" in text

    assert "$ds-library-mode-bar-height" not in variables
    assert "$ds-library-mode-label-width" not in variables
    assert "$ds-library-mode-chip-height: 1;" in variables

    assert "library-mode-chip" not in library_screen
    assert "LIBRARY_MODES = {" not in library_screen
    assert "LIBRARY_MODE_BAR_HEIGHT" not in library_screen
    assert "LIBRARY_MODE_CHIP_WIDTH_PADDING" not in library_screen
