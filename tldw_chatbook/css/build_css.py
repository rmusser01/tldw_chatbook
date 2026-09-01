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
import re
import sys
import tempfile
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
    "components/_settings_personal_context.tcss",
    "components/_profile_interview.tcss",
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

#: TASK-25812 (owner decision 2026-08-31: "split-by-screen"): the module whose
#: single-screen rules are split OUT of the boot bundle and onto the owning
#: screens' ``CSS_PATH``. Measured 2026-08-30: this one file was 283 KB (32%
#: of boot CSS bytes), 1,251 rules, and ~54 ms of the ~191 ms pre-first-paint
#: parse -- while ChatScreen, the first real screen, is constructed ~1.5 s
#: AFTER first paint. Textual loads a screen's ``CSS_PATH`` lazily on first
#: visit (``App._load_screen_css``), so the moved rules parse then instead.
AGENTIC_SPLIT_MODULE = "components/_agentic_terminal.tcss"

#: Owner -> (token prefix, generated sheet). A rule block moves to a sheet
#: only when EVERY ``#id``/``.class`` token in its selector belongs to that
#: one owner; blocks naming several owners, none, or bare widget types stay
#: in the boot bundle. Split by SCREEN, never per-component: Textual's parse
#: cache is an ``LRUCache(64)`` per stylesheet and TASK-15450 measured 94
#: sources making every parse run fully cold (125-380 ms).
AGENTIC_SPLIT_SHEETS = {
    "console": "screen_agentic_console.tcss",
    "library": "screen_agentic_library.tcss",
    "settings": "screen_agentic_settings.tcss",
}

#: Tokens that PREFIX-match an owner but are composed by widgets on OTHER
#: surfaces, so their rules must stay in the boot bundle. Found by auditing
#: every moved token against Python compose sites (2026-08-31):
#: `.settings-input-label` is yielded by
#: ``Widgets/Persona_Widgets/personas_policy_rules_editor.py`` -- moving it
#: to the Settings sheet would leave Personas' policy-rule labels unstyled
#: until Settings is first visited. Add a token here (with the compose site
#: that pins it) rather than weakening the classifier.
AGENTIC_SPLIT_PINNED_TOKENS = {
    "settings-input-label",
    # The `console-*` DESIGN VOCABULARY (Qodo review of PR #2281, finding 1):
    # these carry a legacy Console prefix but are composed app-wide -- Evals,
    # MCP, Lab, Personas, Library and the shared destination rail all yield
    # them. The first cross-surface audit missed every one of these because
    # it filtered ABSOLUTE paths for the substring "console" and the
    # worktree directory was named console-inspect-burndown, so every path
    # matched "home" and the console audit was vacuous. Re-audited with
    # repo-relative paths; compose sites per token are in the PR record.
    "console-action-primary",
    "console-action-secondary",
    "console-action-subdued",
    "console-modal-header",
    "console-rail-collapse-button",
    "console-rail-handle",
    "console-rail-handle-badge",
    "console-rail-handle-button",
    "console-rail-handle-button-vertical",
    "console-rail-handle-vertical",
    "console-rail-header",
    "console-rail-section-header",
    "console-rail-section-title",
    "console-rail-section-toggle",
    "console-rail-title",
    "console-workspace-action",
}

_SPLIT_HEADER = """/* ========================================
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 * ========================================
 * {owner}-owned rules split out of components/_agentic_terminal.tcss
 * by build_css.py (TASK-25812). Loaded via the owning screen's CSS_PATH,
 * so these bytes are parsed on first visit instead of before first paint.
 * Edit components/_agentic_terminal.tcss and re-run build_css.py.
 * ======================================== */

"""

_VARIABLE_DEF_RE = re.compile(r"^\$[\w-]+\s*:[^;{}]*;\s*$", re.M)
_SELECTOR_TOKEN_RE = re.compile(r"[#.]([A-Za-z0-9_-]+)")
_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)


def _split_top_level_units(text: str) -> list[str]:
    """Partition ``text`` into top-level units, losslessly.

    A unit is one top-level ``{...}`` block together with everything since
    the previous unit ended (comments and blank lines travel with the block
    they precede); text after the final block is its own tail unit. Brace
    counting skips ``/* ... */`` spans so a brace inside a comment cannot
    corrupt the partition.

    ``"".join(result) == text`` always -- the caller asserts it, because a
    splitter that silently drops CSS is exactly the incident class the
    ``_settings_splash_theme.tcss`` manifest note records.
    """
    units: list[str] = []
    depth = 0
    unit_start = 0
    i = 0
    length = len(text)
    while i < length:
        ch = text[i]
        if ch == "/" and text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = length if end == -1 else end + 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # Include the trailing newline so the remainder file keeps
                # the exact spacing of the original.
                end = i + 1
                if end < length and text[end] == "\n":
                    end += 1
                units.append(text[unit_start:end])
                unit_start = end
                i = end
                continue
        i += 1
    if unit_start < length:
        units.append(text[unit_start:])
    return units


def _unit_owner(unit: str) -> str | None:
    """Which single owner a unit's selectors belong to, or ``None``.

    Only the text before the first ``{`` is inspected (the selector), with
    comments stripped first so prose cannot misclassify a block. A unit with
    no block (a tail comment) or whose tokens span owners stays in the
    bundle.
    """
    stripped = _COMMENT_RE.sub("", unit)
    brace = stripped.find("{")
    if brace == -1:
        return None
    selector = stripped[:brace]
    owners = set()
    for token in _SELECTOR_TOKEN_RE.findall(selector):
        if token in AGENTIC_SPLIT_PINNED_TOKENS:
            return None
        for owner in AGENTIC_SPLIT_SHEETS:
            if token == owner or token.startswith(owner + "-"):
                owners.add(owner)
                break
        else:
            # A token belonging to no owner (e.g. `.ds-panel`) pins the
            # block to the bundle: it may match widgets on any screen.
            return None
    if len(owners) == 1:
        return owners.pop()
    return None


def _unit_selector_set(unit: str) -> set[str]:
    """Whitespace-normalised selectors of a unit, comma members separately.

    Comments are stripped BEFORE locating the block's brace: a comment
    containing ``{`` ahead of the rule otherwise wins ``find("{")`` and the
    "selector" becomes comment prose (caught by the splitter's unit tests --
    the affected block was silently pinned to the bundle with garbage
    selector bookkeeping).
    """
    stripped = _COMMENT_RE.sub("", unit)
    brace = stripped.find("{")
    if brace == -1:
        return set()
    selector = stripped[:brace]
    return {
        " ".join(part.split())
        for part in selector.split(",")
        if part.strip()
    }


def _later_module_selectors(css_dir: Path | None) -> set[str]:
    """Selectors of every ``CSS_MODULES`` entry AFTER the agentic module.

    A moved block parses after the whole bundle, so an equal-specificity
    selector in a LATER module (features, utilities -- the latter documented
    as able to "override anything") would lose a tie it used to win. These
    selectors seed the demotion pass below so the protection is enforced at
    every build rather than by a one-time review audit.

    Args:
        css_dir: Root of the modular stylesheets, or ``None`` when the
            caller has no tree (pure-text unit tests) -- then no cross-module
            selectors are known and only the intra-module pass applies.

    Returns:
        Whitespace-normalised selectors, comma members separately.
    """
    if css_dir is None:
        return set()
    selectors: set[str] = set()
    seen_agentic = False
    for module in CSS_MODULES:
        if module == AGENTIC_SPLIT_MODULE:
            seen_agentic = True
            continue
        if not seen_agentic:
            continue
        source = css_dir / module
        if not source.is_file():
            continue
        for unit in _split_top_level_units(source.read_text(encoding="utf-8")):
            selectors |= _unit_selector_set(unit)
    return selectors


def split_agentic_terminal(
    text: str, css_dir: Path | None = None
) -> tuple[str, dict[str, str]]:
    """Split the agentic-terminal module into a bundle remainder + sheets.

    Args:
        text: The full source text of ``AGENTIC_SPLIT_MODULE``.
        css_dir: Root of the modular stylesheets, used to seed the
            cascade-order demotion with LATER modules' selectors. ``None``
            limits demotion to intra-module ordering (unit tests).

    Returns:
        ``(remainder, {owner: moved_css})``. Concatenating the remainder and
        every moved block in original order reproduces ``text`` exactly.

    Raises:
        AssertionError: If the partition is not lossless.
    """
    units = _split_top_level_units(text)
    assert "".join(units) == text, (
        "agentic split partition is not lossless -- refusing to build, "
        "because a lossy split silently drops live CSS"
    )
    owners: list[str | None] = [_unit_owner(unit) for unit in units]

    # Cascade-order safety (found live: `#settings-category-pane`). A moved
    # block parses AFTER the whole bundle, so a kept block LATER in this
    # module that shares a selector -- previously winning the tie by source
    # order -- would now lose to it. Demote any moved block whose selector
    # set intersects a later kept block's, iterating to a fixpoint because a
    # demotion makes that block "kept" for the ones before it. Different
    # owners cannot collide (each owner's selectors carry only its own
    # tokens), and kept-before-moved pairs keep their relative order, so
    # this is the only inversion the split can create within the module.
    unit_selectors = [_unit_selector_set(unit) for unit in units]
    later_modules = _later_module_selectors(css_dir)
    changed = True
    while changed:
        changed = False
        # Seeded with LATER modules' selectors (Qodo #2281 finding: the
        # intra-module pass alone was blind to features/utilities ties).
        kept_later: set[str] = set(later_modules)
        for index in range(len(units) - 1, -1, -1):
            if owners[index] is not None and unit_selectors[index] & kept_later:
                owners[index] = None
                changed = True
            if owners[index] is None:
                kept_later |= unit_selectors[index]

    remainder: list[str] = []
    moved: dict[str, list[str]] = {owner: [] for owner in AGENTIC_SPLIT_SHEETS}
    for unit, owner in zip(units, owners):
        if owner is None:
            remainder.append(unit)
        else:
            moved[owner].append(unit)
    return "".join(remainder), {
        owner: "".join(parts) for owner, parts in moved.items()
    }


def _agentic_variables_preamble(css_dir: Path) -> str:
    """Every top-level ``$var: value;`` visible to the agentic module.

    Variable definitions inside one stylesheet source do not carry into
    another, so each split sheet must restate the definitions the module saw
    in the bundle: those of every module at or before it in ``CSS_MODULES``
    order (later definitions win, which re-ordering preserves).
    """
    defs: list[str] = []
    for module in CSS_MODULES:
        source = css_dir / module
        if source.is_file():
            defs.extend(_VARIABLE_DEF_RE.findall(source.read_text(encoding="utf-8")))
        if module == AGENTIC_SPLIT_MODULE:
            break
    if not defs:
        return ""
    return (
        "\n".join(defs)
        + "\n\n"
        # Terminate the preamble with an empty rule: several test contracts
        # scan `selector { body }` pairs with regexes that treat everything
        # since the last `}` as the selector, and with no closing brace
        # before the first real rule they swallow the whole preamble into
        # its selector and stop matching anything. Textual parses an empty
        # rule without complaint; the sentinel name matches nothing.
        + ".tldw-agentic-split-preamble-end { }\n\n"
    )


def build_agentic_split(css_dir: Path, output_dir: Path) -> None:
    """Write the three per-screen sheets split from the agentic module.

    Args:
        css_dir: Root directory containing the modular stylesheets.
        output_dir: Directory the sheets are written into (``css_dir`` in a
            real build; a temp dir when ``check_bundle_sync`` verifies).
    """
    source = css_dir / AGENTIC_SPLIT_MODULE
    if AGENTIC_SPLIT_MODULE not in CSS_MODULES or not source.is_file():
        # The manifest is the authority on what this build contains: a tree
        # built from a patched/partial CSS_MODULES (the staleness test's
        # scratch checkout, an embedder vendoring a subset) simply has no
        # agentic module to split. Skipping is correct there; a REAL missing
        # module is caught loudly by build_css()'s own missing-modules check.
        print("Agentic split skipped: module not in this build")
        return
    text = source.read_text(encoding="utf-8")
    _, moved = split_agentic_terminal(text, css_dir=css_dir)
    preamble = _agentic_variables_preamble(css_dir)
    for owner, filename in AGENTIC_SPLIT_SHEETS.items():
        content = _SPLIT_HEADER.format(owner=owner) + preamble + moved[owner]
        _atomic_write_text(output_dir / filename, content)
    print(
        "Agentic split complete: "
        + ", ".join(
            f"{owner}={len(moved[owner]):,}B" for owner in AGENTIC_SPLIT_SHEETS
        )
    )


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
    header = """/* ========================================
 * GENERATED FILE - DO NOT EDIT DIRECTLY
 * ======================================== 
 * Generated: deterministic
 * 
 * This file is automatically generated by build_css.py
 * Edit the individual module files in core/, layout/, 
 * components/, features/, and utilities/ directories
 * ======================================== */

"""

    # Collect all CSS content
    combined_css = [header]

    for index, module in enumerate(CSS_MODULES, start=1):
        print(f"Processing CSS module {index} of {len(CSS_MODULES)}")
        content = (css_dir / module).read_text(encoding="utf-8")
        if module == AGENTIC_SPLIT_MODULE:
            # TASK-25812: only the multi-screen remainder rides the boot
            # bundle; the console/library/settings rules ship as the
            # per-screen sheets `build_agentic_split` writes, parsed on
            # first visit to the owning screen instead of before first
            # paint.
            content, _ = split_agentic_terminal(content, css_dir=css_dir)

        # Add module separator
        combined_css.append(f"\n/* ===== MODULE: {module} ===== */\n")
        combined_css.append(content)

        # Ensure there's a newline at the end
        if not content.endswith("\n"):
            combined_css.append("\n")

    # Write the combined CSS
    _atomic_write_text(output_file, "".join(combined_css))

    print("CSS build complete")
    print(f"Total size: {len(''.join(combined_css)):,} characters")


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
    print("Widget defaults build complete")
    print(
        f"Widget defaults: {len(blocks)} classes, "
        f"{len(own):,} + {len(scoped):,} characters"
    )


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
    print("Screen CSS build complete")
    print(
        f"Screen CSS: {len(blocks)} classes, {len(own):,} + {len(scoped):,} characters"
    )


def main():
    """Main entry point."""
    # Get the CSS directory (where this script is located)
    css_dir = Path(__file__).parent

    # Output file
    output_file = css_dir / "tldw_cli_modular.tcss"

    # Qodo #2281 (build atomicity): every output is generated into a staging
    # directory first and swapped into place only after ALL builders
    # succeed. A mid-run failure -- widget extraction raising, a screen
    # block that cannot be lifted -- previously left a mixed generation on
    # disk (new bundle beside old sheets drops the moved agentic rules),
    # and both production entry paths log a failed build and continue, so
    # the app would boot on it. The stage lives inside css_dir so each
    # os.replace stays a same-filesystem atomic rename.
    #
    # Within the swap the order still encodes the loss-direction: split
    # sheets land before the bundle, so a crash mid-swap duplicates moved
    # rules in the old fat bundle rather than dropping them from the new
    # remainder bundle.
    import shutil

    stage = css_dir / ".css-build-stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        build_agentic_split(css_dir, stage)
        build_css(css_dir, stage / output_file.name)
        build_widget_defaults(
            css_dir,
            stage / WIDGET_DEFAULTS_SELF_FILENAME,
            stage / WIDGET_DEFAULTS_SCOPED_FILENAME,
        )
        build_screen_css(
            css_dir,
            stage / SCREEN_CSS_SELF_FILENAME,
            stage / SCREEN_CSS_SCOPED_FILENAME,
        )
        publish_order = [
            *AGENTIC_SPLIT_SHEETS.values(),
            output_file.name,
            WIDGET_DEFAULTS_SELF_FILENAME,
            WIDGET_DEFAULTS_SCOPED_FILENAME,
            SCREEN_CSS_SELF_FILENAME,
            SCREEN_CSS_SCOPED_FILENAME,
        ]
        for name in publish_order:
            staged = stage / name
            if staged.is_file():  # the agentic split may legitimately skip
                os.replace(staged, css_dir / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
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
