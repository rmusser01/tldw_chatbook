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

# TASK-25812: the build now emits the bundle PLUS three per-screen sheets
# split from the agentic-terminal module. "Reaches the generated output"
# means the union -- a rule in a split sheet is exactly as live as one in
# the bundle (the app loads it via the owning screen's CSS_PATH).
_GENERATED_SHEETS = (
    _BUNDLED_STYLESHEET,
    _CSS_ROOT / "screen_agentic_console.tcss",
    _CSS_ROOT / "screen_agentic_library.tcss",
    _CSS_ROOT / "screen_agentic_settings.tcss",
)


def _generated_css_text() -> str:
    return "\n".join(sheet.read_text(encoding="utf-8") for sheet in _GENERATED_SHEETS)


_LIBRARY_SCREEN_SOURCE = _REPO_ROOT / "tldw_chatbook/UI/Screens/library_screen.py"

_LIBRARY_NOTES_COMPACT_GEOMETRY = {
    "#library-shell-grid.library-notes-compact #library-canvas": {
        "padding": "0",
        "margin": "0",
        "border": "none",
    },
    "#library-shell-grid.library-notes-compact": {
        "padding": "0",
        "margin": "0",
        "border": "none",
    },
    "#library-shell-grid.library-notes-compact #library-notes-canvas": {
        "height": "100%",
        "min-height": "0",
        "padding": "0",
        "margin": "0",
    },
    "#library-shell-grid.library-notes-compact #library-notes-authority": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
        "text-wrap": "wrap",
        "overflow": "hidden hidden",
    },
    "#library-shell-grid.library-notes-compact #library-notes-header": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-shell-grid.library-notes-compact #library-notes-filter-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-notes-browse-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-notes-sort-choices": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-notes-transfer-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-notes-selection-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-notes-status-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-notes-selection-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-notes-list": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-note-title-row": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-note-editor-region": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "hidden",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-body": {
        "height": "1fr",
        "min-height": "0",
        "max-height": "100%",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-preview-region": {
        "height": "1fr",
        "min-height": "0",
        "max-height": "100%",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-preview-body": {
        "height": "auto",
        "min-height": "0",
        "border": "none",
        "overflow-y": "hidden",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-shell-grid.library-notes-compact .library-note-validation #library-note-status": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
        "text-wrap": "wrap",
    },
    "#library-shell-grid.library-notes-compact #library-note-primary-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-conflict-region": {
        "height": "3",
        "min-height": "3",
        "max-height": "3",
    },
    "#library-shell-grid.library-notes-compact #library-note-conflict-copy": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
    },
    "#library-shell-grid.library-notes-compact #library-note-conflict-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-delete-confirmation": {
        "height": "2",
        "min-height": "2",
        "max-height": "2",
    },
    "#library-shell-grid.library-notes-compact #library-note-delete-confirm-copy": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-shell-grid.library-notes-compact #library-note-delete-actions": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-context-region": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-context-status": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-shell-grid.library-notes-compact #library-notes-create-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-notes-create-viewport": {
        "height": "1fr",
        "min-height": "0",
        "overflow-y": "auto",
        "overflow-x": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-load-heading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
    },
    "#library-shell-grid.library-notes-compact #library-note-loading": {
        "height": "1",
        "min-height": "1",
        "max-height": "1",
        "text-wrap": "nowrap",
        "text-overflow": "ellipsis",
    },
    "#library-shell-grid.library-notes-compact #library-note-load-state": {
        "height": "1fr",
        "min-height": "0",
        "overflow": "hidden",
    },
    "#library-shell-grid.library-notes-compact #library-note-loading-viewport": {
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


def test_css_bundle_build_is_independent_of_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two identical source trees produce byte-identical app bundles."""
    css_dir = tmp_path / "css"
    css_dir.mkdir()
    (css_dir / "module.tcss").write_text("Screen { color: white; }\n")
    output = css_dir / "bundle.tcss"
    monkeypatch.setattr(css_builder, "CSS_MODULES", ["module.tcss"])

    class FirstClock:
        @staticmethod
        def now():
            return FirstClock()

        def strftime(self, _format: str) -> str:
            return "2026-08-29 01:00:00"

    class SecondClock(FirstClock):
        @staticmethod
        def now():
            return SecondClock()

        def strftime(self, _format: str) -> str:
            return "2026-08-29 02:00:00"

    monkeypatch.setattr(css_builder, "datetime", FirstClock, raising=False)
    css_builder.build_css(css_dir, output)
    first = output.read_bytes()
    monkeypatch.setattr(css_builder, "datetime", SecondClock, raising=False)
    css_builder.build_css(css_dir, output)

    assert output.read_bytes() == first


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


def _generated_agentic_css() -> str:
    """The agentic module as the app actually loads it, post-split.

    TASK-25812 split the module across the bundle (multi-screen remainder)
    and three per-screen sheets. Contracts about "the generated form of this
    module" now run against the union of those outputs.
    """
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")
    parts = [_bundled_module(bundle, "components/_agentic_terminal.tcss")]
    for filename in css_builder.AGENTIC_SPLIT_SHEETS.values():
        parts.append((_CSS_ROOT / filename).read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_library_notes_compact_source_module_is_exactly_bundled() -> None:
    """Every byte of the source module reaches exactly one generated output.

    Pre-split this was `bundle section == source`. The split makes that
    false BY DESIGN, so the contract is now the splitter's own lossless
    partition, re-checked here against the COMMITTED outputs: the bundle's
    module section must equal the remainder, and each split sheet must end
    with its owner's moved blocks. A hand-edit to any generated file still
    fails loudly, which is this test's entire purpose.
    """
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    remainder, _ = css_builder.split_agentic_terminal(source, css_dir=_CSS_ROOT)
    bundle = _BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    assert (
        _bundled_module(bundle, "components/_agentic_terminal.tcss")
        == remainder.strip()
    )
    # EXACT equality against a fresh rebuild, not endswith: a suffix check
    # accepts stale or hand-inserted CSS ahead of the expected tail (Qodo
    # #2281), which is precisely the drift this test exists to refuse.
    import tempfile

    with tempfile.TemporaryDirectory() as rebuilt_dir:
        css_builder.build_agentic_split(_CSS_ROOT, Path(rebuilt_dir))
        for filename in css_builder.AGENTIC_SPLIT_SHEETS.values():
            committed = (_CSS_ROOT / filename).read_text(encoding="utf-8")
            rebuilt = (Path(rebuilt_dir) / filename).read_text(encoding="utf-8")
            assert committed == rebuilt, (
                f"{filename} differs from a fresh build_agentic_split -- "
                "regenerate with build_css.py and commit the result"
            )


def test_console_bounded_sections_have_no_legacy_fractional_css_owner() -> None:
    """Only the bounded viewport may own direct-section scrolling geometry."""

    stylesheets = (
        _AGENTIC_SOURCE.read_text(encoding="utf-8"),
        _generated_agentic_css(),
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
        _generated_css_text(),
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
        _generated_css_text(),
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
    bundle = _generated_css_text()
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
    bundle = _generated_css_text()
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
    bundle = _generated_css_text()

    for css in (source, bundle):
        inspector_handle = _rule_body(css, ".console-inspector-rail-handle")
        assert "height: 100%;" in inspector_handle
        assert "min-height: 12;" in inspector_handle
        assert "max-height: 100%;" in inspector_handle
        assert "background: $ds-surface-panel;" in inspector_handle


def test_library_modular_css_compact_shell_and_emergency_return_reach_bundle() -> None:
    """Production CSS owns the narrow box model and its visible return seam."""
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    bundle = _generated_css_text()

    for css in (source, bundle):
        compact = _declarations(css, "#library-shell-grid.library-notes-compact")
        assert compact["padding"] == "0"
        assert compact["margin"] == "0"
        assert compact["border"] == "none"

        emergency_return = _declarations(css, "#library-emergency-return")
        assert emergency_return["width"] == "100%"
        assert emergency_return["height"] == "1"
        assert emergency_return["min-height"] == "1"
        assert emergency_return["border"] == "none"


def test_console_edge_ownership_rules_reach_generated_bundle() -> None:
    """Source and bundle retain the edge-native Console shell contract."""
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    bundle = _generated_css_text()

    for css in (source, bundle):
        grid = _declarations(css, "#console-workspace-grid")
        assert grid["padding"] == "0"
        assert grid["border-top"] == "solid $ds-grid-line"
        assert grid["border-bottom"] == "solid $ds-grid-line"
        assert "border-left" not in grid
        assert "border-right" not in grid

        transcript = _declarations(css, "#console-transcript-region")
        assert transcript["border"] == "none"

        left = _declarations(css, "#console-left-rail:focus")
        assert "border" not in left
        assert left["outline"] == "none"

        transcript_focus = _declarations(
            css,
            "#console-transcript-region.console-transcript-region-focused #console-transcript-title",
        )
        assert transcript_focus["background"] == "$ds-focus-bg"
        assert transcript_focus["color"] == "$ds-focus-fg"
        assert transcript_focus["text-style"] == "bold underline"


def test_settings_category_rules_have_source_and_bundle_integrity() -> None:
    source = _AGENTIC_SOURCE.read_text(encoding="utf-8")
    bundle = _generated_css_text()

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


# --- TASK-25812: split_agentic_terminal unit contracts (Qodo #2281 #3) ------


def test_split_partition_is_lossless_and_ownership_is_conservative() -> None:
    """The splitter's classification rules, each on a minimal input.

    A block moves only when EVERY id/class token belongs to exactly one
    owner; anything ambiguous stays. These are the shapes that decide
    whether a rule silently vanishes from a surface, so each gets a named
    case rather than trusting the full-file run to cover them.
    """
    css = (
        "/* header comment { brace inside comment } */\n"
        ".console-thing { color: red; }\n"
        ".library-thing > Button { color: blue; }\n"
        ".console-thing .library-thing { color: green; }\n"
        "Button { color: white; }\n"
        ".ds-panel .console-thing { color: black; }\n"
        ".settings-input-label { color: grey; }\n"
        "#settings-only { padding: 1; }\n"
        "/* tail comment */\n"
    )
    remainder, moved = css_builder.split_agentic_terminal(css)

    # Lossless: every byte lands in exactly one output.
    reassembled = sorted(
        remainder.splitlines()
        + [line for text in moved.values() for line in text.splitlines()]
    )
    assert reassembled == sorted(css.splitlines())

    assert ".console-thing { color: red; }" in moved["console"]
    assert ".library-thing > Button" in moved["library"]
    assert "#settings-only" in moved["settings"]
    # Multi-owner selector stays.
    assert ".console-thing .library-thing" in remainder
    # Bare TYPE subject stays.
    assert "Button { color: white; }" in remainder
    # A non-owner token anywhere pins the block to the bundle.
    assert ".ds-panel .console-thing" in remainder
    # Pinned cross-surface vocabulary stays even though it prefix-matches.
    assert ".settings-input-label" in remainder
    # Comments (including braces inside them) travel with the next block.
    assert "brace inside comment" in remainder + moved["console"]


def test_split_demotes_moved_blocks_that_later_kept_blocks_tie_with() -> None:
    """Intra-module cascade-order safety, both directions.

    A moved block parses after the whole bundle. A KEPT block later in the
    module that shares its selector used to win the tie by source order and
    would now lose it -- so the moved block must be demoted. A kept block
    EARLIER keeps its relative order and must not cause demotion.
    """
    # A tie needs the SAME selector (equal specificity); the incident case
    # was a comma group carrying the moved rule's exact selector.
    css = (
        ".console-a { color: red; }\n"
        ".mixed-tok, .console-a { color: blue; }\n"  # kept, LATER, exact tie
        ".console-b { color: green; }\n"
    )
    remainder, moved = css_builder.split_agentic_terminal(css)
    assert ".console-a { color: red; }" in remainder, (
        "moved block sharing a selector with a LATER kept block must be "
        "demoted or it wins a cascade tie it used to lose"
    )
    assert ".console-b" in moved["console"]

    css2 = (
        ".mixed-tok, .console-c { color: blue; }\n"  # kept, EARLIER
        ".console-c { color: red; }\n"
    )
    remainder2, moved2 = css_builder.split_agentic_terminal(css2)
    assert ".console-c { color: red; }" in moved2["console"], (
        "a kept block EARLIER in the module preserves relative order and "
        "must not force a demotion"
    )


def test_split_demotion_sees_later_modules(tmp_path: Path) -> None:
    """Cross-module cascade-order safety (Qodo #2281 #8).

    A selector collision with a module AFTER the agentic one in
    CSS_MODULES (features, utilities) must demote the moved block: those
    modules used to win the tie by bundle order and a screen sheet would
    now beat them.
    """
    css_dir = tmp_path / "css"
    (css_dir / "utilities").mkdir(parents=True)
    (css_dir / "utilities" / "_overrides.tcss").write_text(
        ".console-x { color: white; }\n", encoding="utf-8"
    )
    css = ".console-x { color: red; }\n.console-y { color: green; }\n"

    remainder, moved = css_builder.split_agentic_terminal(css, css_dir=css_dir)
    assert ".console-x" in remainder, (
        "a moved block colliding with a LATER module's selector must be "
        "demoted -- utilities exist to override anything"
    )
    assert ".console-y" in moved["console"]

    # Without a tree, only the intra-module pass applies.
    remainder_none, moved_none = css_builder.split_agentic_terminal(css)
    assert ".console-x" in moved_none["console"]


def test_split_sheets_carry_only_their_own_owners_rules() -> None:
    """No sheet holds another surface's tokens (Qodo #2281 finding 8).

    The union harnesses deliberately model the steady-state app, so they
    cannot catch a Console rule generated into the Library sheet -- a
    misplacement that would strand the rule behind the wrong screen's
    first-visit load. This checks ownership at the GENERATION level
    instead: every ``{owner}-`` prefixed token in a sheet belongs to that
    sheet's owner (the pinned cross-surface vocabulary lives in the bundle
    and must not appear in any sheet at all).
    """
    owners = tuple(css_builder.AGENTIC_SPLIT_SHEETS)
    for owner, filename in css_builder.AGENTIC_SPLIT_SHEETS.items():
        text = re.sub(
            r"/\*.*?\*/",
            "",
            (_CSS_ROOT / filename).read_text(encoding="utf-8"),
            flags=re.DOTALL,
        )
        selector_text = " ".join(match for match in re.findall(r"([^{}]+)\{", text))
        tokens = set(re.findall(r"[#.]([A-Za-z0-9_-]+)", selector_text))
        for token in sorted(tokens):
            assert token not in css_builder.AGENTIC_SPLIT_PINNED_TOKENS, (
                f"{filename}: pinned cross-surface token .{token} must stay "
                "in the bundle -- regenerate with build_css.py"
            )
            for other in owners:
                if other == owner:
                    continue
                assert token != other and not token.startswith(other + "-"), (
                    f"{filename}: token .{token} belongs to the {other} "
                    f"surface but was generated into the {owner} sheet"
                )
