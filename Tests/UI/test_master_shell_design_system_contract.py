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
MAIN_TCSS = Path("tldw_chatbook/css/main.tcss")
LOADED_TCSS = Path("tldw_chatbook/css/tldw_cli_modular.tcss")
BUILD_CSS_PY = Path("tldw_chatbook/css/build_css.py")
APP_PY = Path("tldw_chatbook/app.py")
THEMES_PY = Path("tldw_chatbook/css/Themes/themes.py")


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
        assert token_name in loaded_text


def test_agentic_terminal_semantic_tokens_and_theme_exist():
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (DESIGN_SYSTEM_TCSS, Path("tldw_chatbook/css/core/_variables.tcss"))
        if path.exists()
    )
    for token_name in REQUIRED_SEMANTIC_TOKENS:
        assert token_name in source_text

    themes_text = THEMES_PY.read_text(encoding="utf-8")
    assert "agentic_terminal" in themes_text


def test_status_contract_requires_readable_labels():
    text = DESIGN_SYSTEM_TCSS.read_text(encoding="utf-8")
    for label in READABLE_STATUS_LABELS:
        assert label in text
