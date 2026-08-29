#!/usr/bin/env python3
"""
CSS Build Script for tldw_chatbook
Concatenates modular CSS files into a single file for Textual

Five files are generated (TASK-15450 -- see ``widget_css.py`` for the why):

* ``tldw_cli_modular.tcss`` -- the app bundle, concatenated from ``CSS_MODULES``.
* ``widget_defaults_{self,scoped}.tcss`` -- the class-level ``BUNDLED_CSS``
  blocks, loaded by the app as two widget-defaults stylesheet sources in place
  of one source per widget class.
* ``screen_css_{self,scoped}.tcss`` -- the class-level ``BUNDLED_SCREEN_CSS``
  blocks, loaded as app CSS either side of the bundle, in place of the source
  Textual used to add (with a full cold reparse) on a modal's first open.
"""

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

try:
    # Normal package import (tests, `python -m`) -- no global side effects.
    from . import widget_css
except ImportError:  # pragma: no cover - only when run as a bare script
    # Direct script execution has no package context; add the sibling dir.
    sys.path.insert(0, str(Path(__file__).parent))
    import widget_css  # type: ignore[no-redef]

#: Output filenames for the consolidated widget-defaults stylesheets.
WIDGET_DEFAULTS_SELF_FILENAME = "widget_defaults_self.tcss"
WIDGET_DEFAULTS_SCOPED_FILENAME = "widget_defaults_scoped.tcss"

#: Output filenames for the consolidated screen/modal stylesheets.  These stay
#: *separate stylesheet files* rather than being concatenated into the bundle:
#: Textual accumulates ``$variable`` definitions per source, and a screen's CSS
#: is its own source today, so several of these blocks carry local ``$ds-*``
#: fallbacks ("so this CSS parses without the app bundle").  Concatenated into
#: the bundle those fallbacks redefine the real design tokens for every rule
#: after them -- measured: ``$ds-focus-bg`` went from ``#51677E`` to ``$surface``
#: across the app.  As separate sources they stay local, exactly as today.
#:
#: TASK-15993: that same "per-source" scope means every block CONSOLIDATED
#: INTO one of these two files also shares it with every other block in the
#: same file -- a block-local ``$ds-*`` fallback used to stay defined for
#: every block emitted after it here too, just narrowed from app-wide to
#: sheet-wide rather than eliminated. ``widget_css.render_stylesheets`` now
#: runs ``isolate_local_variables`` per block before emission, which inlines
#: and drops each block's own declarations so they cannot reach a later
#: block's text at all.
SCREEN_CSS_SCOPED_FILENAME = "screen_css_scoped.tcss"
SCREEN_CSS_SELF_FILENAME = "screen_css_self.tcss"

#: Content manifest for the generated sheets (TASK-18910). The boot-time
#: staleness check used to treat ANY input whose mtime moved past the build
#: as stale, so a branch switch / ``git checkout`` / stash pop -- which
#: rewrite mtimes without changing content -- cost a full synchronous
#: rebuild (~0.7 s: interpreter spawn + package AST scan) on the next boot.
#: The builder now records the sha256 of every input; the check treats an
#: mtime move as a hint to hash-check, and identical content is not stale.
#: Untracked by git on purpose: it describes the local build, not the repo.
BUILD_MANIFEST_FILENAME = ".css-build-manifest.json"


#: Streaming read size shared by the builder's hashing and the boot-time
#: staleness check's hashing (Qodo finding on PR #1831: one named constant,
#: not two hard-coded 65536s that can drift apart).
HASH_CHUNK_SIZE_BYTES = 65536


def _atomic_write_text(path: Path, content: str) -> None:
    """Publish generated text atomically for parallel readers/builders."""

    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass


def _file_sha256(path: Path) -> str:
    """Return the hex sha256 of a file's bytes (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_build_manifest(css_dir: Path) -> None:
    """Record the content hash of every build input beside the sheets.

    Inputs are the ``CSS_MODULES`` plus every Python module carrying a
    ``BUNDLED_CSS``/``BUNDLED_SCREEN_CSS`` declaration (the same set
    ``iter_blocks`` walks). Keys are package-root-relative POSIX paths so the
    manifest stays valid across absolute checkout paths.

    Each entry records ``[sha256, mtime_at_build]``: the staleness check
    compares mtimes first and hashes only entries whose mtime moved, so the
    steady state (nothing edited) reads the manifest and stats the inputs
    without hashing anything.

    Args:
        css_dir: Root directory containing the modular stylesheets; the
            manifest is written beside the generated sheets it describes.

    Raises:
        OSError: If any input cannot be read or the manifest cannot be
            written.
    """
    package_root = css_dir.parent
    entries: dict[str, list] = {}
    for module in CSS_MODULES:
        source = css_dir / module
        if source.is_file():
            stat = source.stat()
            entries[f"css/{module}"] = [_file_sha256(source), stat.st_mtime]
    for attr in (widget_css.WIDGET_ATTR, widget_css.SCREEN_ATTR):
        for block in widget_css.iter_blocks(package_root, attr):
            key = block.module
            if key not in entries:
                source = package_root / key
                entries[key] = [_file_sha256(source), source.stat().st_mtime]
    manifest_path = css_dir / BUILD_MANIFEST_FILENAME
    _atomic_write_text(
        manifest_path, json.dumps(entries, indent=2, sort_keys=True) + "\n"
    )


def _manifest_inputs_changed_since(css_dir: Path) -> bool:
    """Return whether any manifest input's hash no longer matches the file.

    The just-written manifest records each input's hash as observed by
    ``write_build_manifest``; re-hashing immediately after the build and
    comparing detects an edit that landed between the build's reads of the
    sources and the manifest's reads. See ``main()`` for the publish-or-
    refuse contract.
    """
    manifest_path = css_dir / BUILD_MANIFEST_FILENAME
    try:
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True  # cannot verify: do not publish
    if not isinstance(entries, dict):
        return True
    if not entries:
        # Nothing was recorded (e.g. a test tree with no CSS inputs): an
        # empty manifest is vacuously consistent with the sheets, not a
        # race. Production manifests always carry the CSS_MODULES set.
        return False
    package_root = css_dir.parent
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, list) or len(value) != 2:
            return True
        source = package_root / key
        try:
            if not source.is_file() or _file_sha256(source) != value[0]:
                return True
        except OSError:
            return True
    return False


#: Tie-breaker for the scoped widget-defaults source.  Textual's own default-CSS
#: sources sit at ``-(MRO depth)``, so any value below that floor makes the
#: scoped sheet lose every specificity tie -- which is exactly the set of ties
#: it only entered by having its scope selector written out.
SCOPED_DEFAULTS_TIE_BREAKER = -1_000_000

# Define the order of imports (based on dependencies)
CSS_MODULES = [
    # 1. Core - Foundation (no dependencies)
    "core/_variables.tcss",
    "core/_reset.tcss",
    "core/_base.tcss",
    "core/_typography.tcss",
    # 2. Layout - Structure (depends on core)
    "layout/_windows.tcss",
    "layout/_tabs.tcss",
    "layout/_sidebars.tcss",
    "layout/_panes.tcss",
    "layout/_containers.tcss",
    # 3. Components - Reusable UI (depends on core + layout)
    "components/_buttons.tcss",
    "components/_forms.tcss",
    "components/_lists.tcss",
    "components/_navigation.tcss",
    "components/_change_review.tcss",
    "components/_messages.tcss",
    "components/_dialogs.tcss",
    "components/_status.tcss",
    "components/_agentic_terminal.tcss",
    "components/_workbench.tcss",
    "components/_widgets.tcss",
    "components/stats_screen.css",
    # NOTE: "components/splash_viewer.css" was deleted in 778f75813 when the
    # splash viewer moved into Settings, but that commit wrote the replacement
    # styles directly into the GENERATED bundle instead of a source module —
    # every rebuild (including the app's boot-time mtime rebuild) silently
    # stripped the live Settings splash/theme-editor styling. The bundle-only
    # rules now live in _settings_splash_theme.tcss at this same manifest
    # position so cascade order is unchanged.
    "components/_settings_splash_theme.tcss",
    # TASK-394: generic, app-wide component styles moved OUT of the splash/theme
    # module. Kept at this manifest position (immediately after it) so the
    # bundle cascade is byte-for-byte equivalent for the relocated rules.
    "components/_shared_components.tcss",
    # 4. Features - Application Specific (depends on all above)
    "features/_chat.tcss",
    # task-577 T4: "features/_chat_tabs.tcss" removed -- every selector in it
    # (chat-tab-bar, .chat-tab, .chat-session, .close-tab-button,
    # .new-tab-button, .chat-sessions-container, .no-sessions-placeholder,
    # and the .chat-session-scoped chat-empty-state/chat-log/chat-input-area/
    # image-attachment-indicator variants) styled the retired
    # ChatTabContainer/ChatSession tabs subsystem (task-577 T2), composed
    # nowhere live (grep -rn confirmed zero id=/classes= compose sites).
    "features/_conversations.tcss",
    "features/_notes.tcss",
    "features/_media.tcss",
    # RAG UX v2 PR-2 Task 2: "features/_search-rag.tcss" removed -- an audit
    # found only 5 of its 104 selectors had live users. Three of those
    # (.action-button, .settings-section, .status-bar) were already shadowed
    # by same-name rules defined later in this manifest (_evaluation_unified.tcss,
    # _wizards.tcss), so dropping this sheet's copies is a no-op. The other two
    # (.action-spacer, .param-group) were this sheet's SOLE definitions with
    # live users (CodeRepoCopyPasteWindow, MediaViewerPanel) and were moved
    # verbatim to components/_shared_components.tcss to preserve them.
    "features/_llm-management.tcss",
    "features/_tools-settings.tcss",
    "features/_ingest.tcss",
    # task-745: "features/_ingest_tldw_api_tabs.tcss" removed -- it styled the
    # standalone ingest window's tab strip and API form, deleted in task-684.4.
    # Every distinctive selector in that retired sheet has zero mount sites.
    # Its scoped ".hidden" rule had no surviving owner, and other sheets define
    # .hidden anyway; its unscoped ".window-title" rule was the only bundled
    # definition and moved to components/_shared_components.tcss.
    # NOTE: "features/_evaluation_v2.tcss" was removed from the tree back in
    # ac937dab ("f", Aug 2025) when the Evals dashboard was consolidated
    # into _evaluation_unified.tcss below; this manifest entry was left
    # dangling (build_css.py has warned "Missing module" on every build
    # since). No Python source references "_evaluation_v2" (grepped
    # tldw_chatbook/**/*.py) -- T169 dropped the stale entry rather than
    # restoring a file nothing needs.
    "features/_evaluation_unified.tcss",
    # PR3a Task 3: the new Console-styled three-pane Evals workbench's own
    # rail/pane rules. Distinct from _evaluation_unified.tcss above (the
    # retired card hub's legacy dashboard CSS, PR 3b's concern) -- no
    # selector overlap between the two files.
    "features/_evals.tcss",
    "features/_metrics.tcss",
    "features/_embeddings.tcss",
    "features/_splash.tcss",
    "features/_wizards.tcss",
    "features/_chatbooks.tcss",
    "features/_scheduling.tcss",
    "features/_code_repo.tcss",
    "features/_coding.tcss",
    "features/_tab_dropdown.tcss",
    "features/_watchlists.tcss",
    "features/_lab.tcss",
    "features/_research_workspace.tcss",
    "features/_logs.tcss",
    "features/_writing.tcss",
    "features/config_search.tcss",
    "features/feature_alerts.tcss",
    # 5. Utilities - Helpers and Overrides (can override anything)
    "utilities/_helpers.tcss",
    "utilities/_states.tcss",
    "utilities/_overrides.tcss",
]


def build_css(css_dir: Path, output_file: Path) -> None:
    """Concatenate all declared CSS modules into a single file.

    Args:
        css_dir: Root directory containing the modular stylesheets.
        output_file: Generated bundle path.

    Raises:
        FileNotFoundError: If any declared module is missing. The existing
            output is left unchanged.
    """
    missing_modules = [
        module for module in CSS_MODULES if not (css_dir / module).is_file()
    ]
    if missing_modules:
        missing = ", ".join(missing_modules)
        raise FileNotFoundError(f"Missing declared CSS module(s): {missing}")

    # Header for the generated file
    header = f"""/* ========================================
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 * ======================================== 
 * Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 * 
 * This file is automatically generated by build_css.py
 * Edit the individual module files in core/, layout/, 
 * components/, features/, and utilities/ directories
 * ======================================== */

"""

    # Collect all CSS content
    combined_css = [header]

    for module in CSS_MODULES:
        print(f"✓ Processing: {module}")
        content = (css_dir / module).read_text(encoding="utf-8")

        # Add module separator
        combined_css.append(f"\n/* ===== MODULE: {module} ===== */\n")
        combined_css.append(content)

        # Ensure there's a newline at the end
        if not content.endswith("\n"):
            combined_css.append("\n")

    # Write the combined CSS
    _atomic_write_text(output_file, "".join(combined_css))

    print(f"\n✅ CSS build complete: {output_file}")
    print(f"📏 Total size: {len(''.join(combined_css)):,} characters")


def build_widget_defaults(css_dir: Path, self_file: Path, scoped_file: Path) -> None:
    """Write the two consolidated widget-defaults stylesheets.

    Every class-level ``BUNDLED_CSS`` block in the package is lifted here so the
    app registers **two** widget-defaults stylesheet sources instead of one per
    widget class (TASK-15450).

    Args:
        css_dir: Root directory containing the modular stylesheets.
        self_file: Output path for the self-selector stream.
        scoped_file: Output path for the scope-prefixed stream.

    Raises:
        ValueError: If a ``BUNDLED_CSS`` declaration cannot be lifted.
    """
    blocks = widget_css.iter_blocks(css_dir.parent, widget_css.WIDGET_ATTR)
    own, scoped = widget_css.render_stylesheets(
        blocks,
        "Widget DEFAULT_CSS (widget-defaults tier), lifted from Python sources",
        # TASK-15998: scope EVERY selector of a comma list, exactly as the
        # screen sheets below already do. Textual's scoped-DEFAULT_CSS parser
        # prefixes only the LAST selector of a comma list, so `A, .b {…}`
        # leaves `A` matching app-wide; while each class registered its own
        # source that leak went live only at the class's first mount, but
        # consolidation made these sheets live from boot. At the time of this
        # change the quirk was leaking 56 selectors across 6 classes (24 at
        # the TASK-15450 review -- the set had already grown silently). The
        # de-quirk was proven cascade-neutral before shipping, not assumed:
        # a computed-style diff between the quirked and de-quirked builds over
        # a 22-stop destination tour -- 9,449 node-states, including forced
        # :hover/:focus/:disabled sweeps and the Library notes/media/compact,
        # note-editor/sync/select/sort and nav-clip-ghost states that mount
        # the leaked selectors' targets -- found ZERO differences, and every
        # leaked selector's anchor id/class composes only inside its declaring
        # widget's subtree, so the added scope prefix cannot un-style anything.
        # The +1 specificity each rewritten selector gains is absorbed by the
        # scoped stream's tie-breaker exactly as for the screen sheets (see
        # `widget_defaults_sources`). Pinned at zero leaks by
        # Tests/UI/test_widget_css_consolidation.py::
        # test_generated_sheets_scope_every_selector.
        scope_every_selector=True,
    )
    # Filtering self versus scope-prefixed selectors can leave placeholder
    # blank lines after the final surviving rule.  Generated files must end in
    # exactly one newline so they remain compatible with `git diff --check`.
    own = own.rstrip() + "\n"
    scoped = scoped.rstrip() + "\n"
    _atomic_write_text(self_file, own)
    _atomic_write_text(scoped_file, scoped)
    print(f"\n✅ Widget defaults build complete: {self_file}, {scoped_file}")
    print(f"📏 {len(blocks)} widget classes, {len(own):,} + {len(scoped):,} characters")


def widget_defaults_sources(
    css_dir: Path,
) -> list[tuple[tuple[str, str], str, int, str]]:
    """The consolidated widget-defaults sources, as ``_get_default_css`` wants them.

    Shared by the real app and by test harnesses that mount a consolidated
    widget, so both put these rules in the same cascade position. Each entry is
    ``(location, css, tie_breaker, scope)``; prepend them to the stack returned
    by ``super()._get_default_css()``.

    The self stream keeps tie-breaker 0 -- the position each class's own
    ``DEFAULT_CSS`` had. The scoped stream takes a tie-breaker below every other
    default-CSS source, because writing its scope selector out costs it one
    specificity point Textual's injected one did not, and it must therefore lose
    the ties that shift created. See ``widget_css.py``.

    Args:
        css_dir: The package's ``css`` directory.

    Returns:
        The sources, self stream first. A sheet that cannot be read is skipped
        rather than raising: an unstyled widget beats an app that will not boot.
    """
    sources: list[tuple[tuple[str, str], str, int, str]] = []
    for filename, tie_breaker in (
        (WIDGET_DEFAULTS_SELF_FILENAME, 0),
        (WIDGET_DEFAULTS_SCOPED_FILENAME, SCOPED_DEFAULTS_TIE_BREAKER),
    ):
        path = css_dir / filename
        try:
            css = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sources.append(((str(path), filename), css, tie_breaker, ""))
    return sources


def screen_css_paths(css_dir: Path) -> tuple[Path, Path]:
    """The two screen/modal stylesheets, in cascade order.

    Order is the whole point, so it lives here rather than at each call site.
    The scope-prefixed sheet goes first (it must *lose* the specificity ties
    that writing its scope selector out created) and the self sheet last (where
    Textual appended a screen's class-level ``CSS`` on first open). The app
    slots the bundle between them; a test harness that has no bundle just uses
    the pair.

    Args:
        css_dir: The package's ``css`` directory.

    Returns:
        ``(scoped_first, self_last)``.
    """
    return (
        css_dir / SCREEN_CSS_SCOPED_FILENAME,
        css_dir / SCREEN_CSS_SELF_FILENAME,
    )


def build_screen_css(css_dir: Path, self_file: Path, scoped_file: Path) -> None:
    """Write the two consolidated screen/modal stylesheets.

    Every class-level ``BUNDLED_SCREEN_CSS`` block in the package is lifted here
    so a modal's first open no longer adds a stylesheet source -- which forced a
    full cold ``Stylesheet.reparse()`` and an app-wide restyle (TASK-15450).

    Unlike the widget defaults, these keep Textual's *app-CSS* origin tier, so
    the app loads them as ``CSS_PATH`` entries either side of the bundle.

    Args:
        css_dir: Root directory containing the modular stylesheets.
        self_file: Output path for the self-selector stream.
        scoped_file: Output path for the scope-prefixed stream.

    Raises:
        ValueError: If a ``BUNDLED_SCREEN_CSS`` declaration cannot be lifted.
    """
    blocks = widget_css.iter_blocks(css_dir.parent, widget_css.SCREEN_ATTR)
    own, scoped = widget_css.render_stylesheets(
        blocks,
        "Screen/modal CSS (app-CSS tier), lifted from Python sources",
        # These sheets are live from boot, where a screen's `CSS` only became
        # live once that screen was first opened. Textual's last-selector-only
        # scoping would therefore leak a modal's rules app-wide from startup, so
        # every selector is scoped here rather than reproducing that quirk.
        scope_every_selector=True,
    )
    own = own.rstrip() + "\n"
    scoped = scoped.rstrip() + "\n"
    _atomic_write_text(self_file, own)
    _atomic_write_text(scoped_file, scoped)
    print(f"\n✅ Screen CSS build complete: {self_file}, {scoped_file}")
    print(f"📏 {len(blocks)} screen classes, {len(own):,} + {len(scoped):,} characters")


def main():
    """Main entry point."""
    # Get the CSS directory (where this script is located)
    css_dir = Path(__file__).parent

    # Output file
    output_file = css_dir / "tldw_cli_modular.tcss"

    # Build the CSS
    build_css(css_dir, output_file)
    build_widget_defaults(
        css_dir,
        css_dir / WIDGET_DEFAULTS_SELF_FILENAME,
        css_dir / WIDGET_DEFAULTS_SCOPED_FILENAME,
    )
    build_screen_css(
        css_dir,
        css_dir / SCREEN_CSS_SELF_FILENAME,
        css_dir / SCREEN_CSS_SCOPED_FILENAME,
    )
    # Qodo finding on PR #1831 (build race): the sheets above were built
    # from one read of the sources and write_build_manifest re-reads them;
    # an edit between those reads would record NEW content in the manifest
    # while the sheets carry the OLD content -- and every later boot would
    # accept the stale sheets as current. Write the manifest, then re-hash
    # every input and refuse to keep it if any changed while the build ran:
    # the manifest then never describes content absent from the sheets.
    # (Deleting the just-written manifest on refusal keeps the NEXT boot on
    # the safe legacy mtime rule instead of a stale blessing.)
    write_build_manifest(css_dir)
    if _manifest_inputs_changed_since(css_dir):
        manifest_path = css_dir / BUILD_MANIFEST_FILENAME
        try:
            manifest_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "CSS inputs changed while building (build raced an edit); "
            "the manifest was not published. Re-run build_css."
        )

    print("\nTo use the modular CSS:")
    print("1. Update app.py to use 'tldw_cli_modular.tcss'")
    print("2. Run this script whenever you modify any module files")


if __name__ == "__main__":
    main()
