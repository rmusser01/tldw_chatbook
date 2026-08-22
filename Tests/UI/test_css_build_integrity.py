"""Regression coverage for deterministic, complete modular CSS builds."""

import ast
from contextlib import suppress
from pathlib import Path
import re

import pytest

from tldw_chatbook.css import build_css as css_builder


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = _REPO_ROOT / "tldw_chatbook/css"
_AGENTIC_SOURCE = _CSS_ROOT / "components/_agentic_terminal.tcss"
_SETTINGS_SOURCE = _CSS_ROOT / "components/_settings_splash_theme.tcss"
_SHARED_SOURCE = _CSS_ROOT / "components/_shared_components.tcss"
_BUNDLED_STYLESHEET = _CSS_ROOT / "tldw_cli_modular.tcss"
_LIBRARY_SCREEN_SOURCE = _REPO_ROOT / "tldw_chatbook/UI/Screens/library_screen.py"

_LIBRARY_NOTES_COMPACT_GEOMETRY = {
    "#library-shell-grid.library-notes-compact": {
        "padding": "0",
        "margin": "0",
        "border": "none",
    },
    "#library-canvas.library-notes-compact": {
        "padding": "0",
        "margin": "0",
        "border": "none",
    },
    "#library-canvas.library-notes-compact #library-notes-canvas": {
        "height": "100%",
        "min-height": "0",
        "padding": "0",
        "margin": "0",
    },
    "#library-canvas.library-notes-compact #library-notes-authority": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
        "text-wrap": "wrap",
        "overflow": "hidden hidden",
    },
    "#library-canvas.library-notes-compact #library-notes-header": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-canvas.library-notes-compact #library-notes-filter-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-notes-browse-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-notes-sort-choices": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-notes-transfer-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-notes-selection-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-notes-status-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-notes-selection-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-notes-list": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-note-title-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-note-editor-region": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "hidden",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-body": {
        "height": "1fr",
        "min-height": "0",
        "max-height": "100%",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-preview-region": {
        "height": "1fr",
        "min-height": "0",
        "max-height": "100%",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-preview-body": {
        "height": "auto",
        "min-height": "0",
        "border": "none",
        "overflow-y": "hidden",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-canvas.library-notes-compact #library-notes-canvas.library-note-validation #library-note-status": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
        "text-wrap": "wrap",
    },
    "#library-canvas.library-notes-compact #library-note-primary-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-conflict-region": {
        "height": "3",
        "min-height": "3",
        "max-height": "3",
    },
    "#library-canvas.library-notes-compact #library-note-conflict-copy": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
    },
    "#library-canvas.library-notes-compact #library-note-conflict-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-delete-confirmation": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
    },
    "#library-canvas.library-notes-compact #library-note-delete-confirm-copy": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-canvas.library-notes-compact #library-note-delete-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-context-region": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-context-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-canvas.library-notes-compact #library-notes-create-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-notes-create-viewport": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-load-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-canvas.library-notes-compact #library-note-loading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-canvas.library-notes-compact #library-note-load-state": {
        "height": "1fr",
        "min-height": "0",
        "overflow": "hidden",
    },
    "#library-canvas.library-notes-compact #library-note-loading-viewport": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
}


def _rule_body(css: str, selector: str) -> str:
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    body = None
    for match in re.finditer(
        r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", without_comments
    ):
        selectors = tuple(
            candidate.strip() for candidate in match.group("selectors").split(",")
        )
        if selector in selectors:
            body = match.group("body")
    if body is None:
        raise AssertionError(f"Missing CSS rule for {selector}")
    return body


def _declarations(css: str, selector: str) -> dict[str, str]:
    """Return cascaded declarations for one literal selector."""
    without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    declarations = {}
    matched = False
    for match in re.finditer(
        r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", without_comments
    ):
        selectors = tuple(
            candidate.strip() for candidate in match.group("selectors").split(",")
        )
        if selector not in selectors:
            continue
        matched = True
        for raw in match.group("body").split(";"):
            if ":" not in raw:
                continue
            name, value = raw.split(":", 1)
            declarations[name.strip()] = value.strip()
    if not matched:
        raise AssertionError(f"Missing CSS rule for {selector}")
    return declarations


def _library_screen_default_css() -> str:
    """Read LibraryScreen's own CSS without importing the application.

    TASK-15450 renamed the attribute to ``BUNDLED_CSS``: the class no longer
    registers its own stylesheet source, it is lifted into the generated
    widget-defaults sheets at build time. Either name is accepted so this guard
    keeps working for classes that have not been consolidated.
    """
    wanted = {"DEFAULT_CSS", "BUNDLED_CSS"}
    module = ast.parse(_LIBRARY_SCREEN_SOURCE.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name != "LibraryScreen":
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in statement.targets
            ):
                value = ast.literal_eval(statement.value)
                assert isinstance(value, str)
                return value
    raise AssertionError("LibraryScreen widget CSS not found")


def _bundled_module(bundle: str, module_path: str) -> str:
    marker = f"/* ===== MODULE: {module_path} ===== */"
    assert marker in bundle
    return bundle.split(marker, 1)[1].split("/* ===== MODULE:", 1)[0].strip()


def test_library_notes_compact_source_module_is_exactly_bundled() -> None:
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8").strip()
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    assert _bundled_module(bundle, "components/_agentic_terminal.tcss") == source


def test_console_bounded_sections_have_no_legacy_fractional_css_owner() -> None:
    """Only the bounded viewport may own direct-section scrolling geometry."""

    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    stylesheets = (
        _AGENTIC_SOURCE.read_text(encoding="utf-8"),
        _bundled_module(bundle, "components/_agentic_terminal.tcss"),
    )

    for css in stylesheets:
        without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
        assert "#console-left-rail-body .console-rail-section-body" not in (
            without_comments
        )
        assert not re.search(
            r"(?:^|})[^{}]*\.console-rail-section-body[^{}]*\{"
            r"[^{}]*max-height\s*:\s*20%",
            without_comments,
            flags=re.DOTALL,
        )
        viewport = _declarations(css, ".console-bounded-section-viewport")
        assert viewport["overflow-y"] == "auto"
        assert viewport["overflow-x"] == "hidden"
        assert viewport["scrollbar-gutter"] == "stable"


def test_library_notes_compact_geometry_matches_fallback_source_and_bundle() -> None:
    stylesheets = (
        _library_screen_default_css(),
        _AGENTIC_SOURCE.read_text(encoding="utf-8"),
        _BUNDLED_STYLESHEET.read_text(encoding="utf-8"),
    )

    for selector, expected in _LIBRARY_NOTES_COMPACT_GEOMETRY.items():
        declarations = [_declarations(css, selector) for css in stylesheets]
        for name, value in expected.items():
            assert [declaration.get(name) for declaration in declarations] == [
                value,
                value,
                value,
            ], (selector, name, declarations)


def test_file_notes_error_ink_and_disabled_opacity_are_app_tier() -> None:
    stylesheets = (
        _AGENTIC_SOURCE.read_text(encoding="utf-8"),
        _BUNDLED_STYLESHEET.read_text(encoding="utf-8"),
    )

    for css in stylesheets:
        git_error = _declarations(css, ".file-notes-git-commit-error")
        save_error = _declarations(css, "#file-notes-save-status.-error")
        assert git_error["color"] == "$ds-status-error-readable"
        assert save_error["color"] == "$ds-status-error-readable"
        assert git_error["background"] == "$error-darken-3"
        assert save_error["background"] == "$error-darken-3"
        disabled = _declarations(css, "LibraryFileNotesWorkspace Button:disabled")
        assert disabled["opacity"] == "100%"
        assert disabled["text-opacity"] == "100%"


def _missing_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    css_root = tmp_path / "css"
    module_path = css_root / "components/present.tcss"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(".present { height: 1; }\n", encoding="utf-8")
    output_file = css_root / "bundle.tcss"
    output_file.write_text("known-good bundle\n", encoding="utf-8")
    monkeypatch.setattr(
        css_builder,
        "CSS_MODULES",
        ["components/present.tcss", "components/missing.tcss"],
    )
    return css_root, output_file


def test_build_css_rejects_a_missing_declared_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_root, output_file = _missing_fixture(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="components/missing.tcss"):
        css_builder.build_css(css_root, output_file)


def test_build_css_preserves_existing_output_when_a_module_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    css_root, output_file = _missing_fixture(tmp_path, monkeypatch)

    with suppress(FileNotFoundError):
        css_builder.build_css(css_root, output_file)

    assert output_file.read_text(encoding="utf-8") == "known-good bundle\n"


def test_css_manifest_declares_only_existing_settings_source() -> None:
    assert "components/splash_viewer.css" not in css_builder.CSS_MODULES
    assert "components/_settings_splash_theme.tcss" in css_builder.CSS_MODULES
    assert [
        module
        for module in css_builder.CSS_MODULES
        if not (_CSS_ROOT / module).is_file()
    ] == []


def test_settings_splash_theme_rules_have_source_and_bundle_integrity() -> None:
    assert _SETTINGS_SOURCE.is_file()
    settings_source = _SETTINGS_SOURCE.read_text(encoding="utf-8")
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    # The live, feature-scoped theme-editor + splash-viewer selectors remain.
    required_selectors = (
        "#settings-theme-tree",
        ".settings-preview-grid",
        ".settings-splash-gallery",
        "#settings-splash-card-list",
        "#settings-splash-preview-scroll",
        ".settings-splash-preview #splash-display",
    )

    for selector in required_selectors:
        assert selector in settings_source
        assert selector in bundle
    module_marker = "/* ===== MODULE: components/_settings_splash_theme.tcss ===== */"
    assert module_marker in bundle
    bundled_module = bundle.split(module_marker, 1)[1].split("/* ===== MODULE:", 1)[0]
    assert bundled_module.strip() == settings_source.strip()
    assert "(NOT FOUND)" not in bundle


def test_splash_theme_module_has_no_bare_or_generic_component_selectors() -> None:
    """The splash/theme module defines no app-wide component styles.

    TASK-394 regression lock: no bare type selectors and none of the relocated
    generic class names, in ANY selector position -- bare, compound (``.cls
    Button``), or grouped (``.cls, .x``) -- so a leak can't slip back as a
    compound selector.
    """
    source = re.sub(
        r"/\*.*?\*/", "", _SETTINGS_SOURCE.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    # No bare type selectors (a rule/group whose token is an uppercase widget
    # name, e.g. VerticalScroll), including as the head of a grouped selector.
    assert not re.search(r"(?m)^[A-Z][A-Za-z]+\s*[\{,]", source), (
        "splash/theme module reintroduced a bare type selector"
    )
    # None of the relocated generic component classes -- matched as a selector
    # token anywhere (not as the prefix of a longer name like `.preview-panel-demo`).
    for cls in (
        "setting-label",
        "section-header",
        "help-text",
        "action-buttons",
        "settings-section",
        "card-list",
        "preview-panel",
        "preview-container",
        "preview-content",
    ):
        assert not re.search(rf"\.{re.escape(cls)}(?![\w-])", source), (
            f".{cls} must live in the shared component module, not splash/theme"
        )


def test_relocated_shared_component_rules_are_present() -> None:
    """The moved generic rules live in _shared_components and reach the bundle."""
    shared = _SHARED_SOURCE.read_text(encoding="utf-8")
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    for selector in (
        ".setting-label",
        ".section-header",
        ".preview-panel",
        ".action-buttons",
    ):
        assert selector in shared
        assert selector in bundle
    # The app-wide scrollbar default now lives in core, not a feature module.
    core_base = (_CSS_ROOT / "core/_base.tcss").read_text(encoding="utf-8")
    assert re.search(r"(?m)^VerticalScroll\s*\{", core_base)


def test_console_inspector_handle_full_height_rule_reaches_generated_bundle() -> None:
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for css in (source, bundle):
        inspector_handle = _rule_body(css, ".console-inspector-rail-handle")
        assert "height: 100%;" in inspector_handle
        assert "min-height: 20;" in inspector_handle
        assert "max-height: 100%;" in inspector_handle
        assert "background: $ds-surface-panel;" in inspector_handle


def test_settings_category_rules_have_source_and_bundle_integrity() -> None:
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for css in (source, bundle):
        category_pane = _rule_body(css, "#settings-category-pane")
        assert "overflow-y: hidden;" in category_pane
        assert "overflow-x: hidden;" in category_pane

        category_list = _rule_body(css, "#settings-category-list")
        assert "height: 1fr;" in category_list
        assert "min-height: 0;" in category_list
        assert "overflow-y: auto;" in category_list

        group_title = _rule_body(css, ".settings-category-group-title")
        assert "margin: 0;" in group_title
