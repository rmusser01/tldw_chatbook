"""Regression coverage for deterministic, complete modular CSS builds."""

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


def _rule_body(css: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>[^{{}}]*)\}}", css)
    assert match is not None, f"Missing CSS rule for {selector}"
    return match.group("body")


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
    for selector in (".setting-label", ".section-header", ".preview-panel", ".action-buttons"):
        assert selector in shared
        assert selector in bundle
    # The app-wide scrollbar default now lives in core, not a feature module.
    core_base = (_CSS_ROOT / "core/_base.tcss").read_text(encoding="utf-8")
    assert re.search(r"(?m)^VerticalScroll\s*\{", core_base)


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
