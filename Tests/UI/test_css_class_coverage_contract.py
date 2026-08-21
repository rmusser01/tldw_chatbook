"""Every literal classes= token in the RAG/Console/Library/MCP surface must
be styled, or explicitly registered as known-unstyled.

Bug class this guards (RAG-55): a widget composed with classes that no
stylesheet rule ever touches ships invisible or default-styled — PR-2
found a zero-height toggle with green behavioral tests; PR-4's own
chat_screen compose calls carried two inert tokens
(console-library-rag-scope, console-library-rag-run), fixed by deleting
them (chat_screen.py:14314,14325) since both widgets are already reached
by their `id=`.

Scope: literal double-quoted classes="..." only; f-strings and
add_class(variable) are out of scope by design. Single-quoted
classes='...' is also out of scope by construction (the regex only
matches double quotes) but is a non-issue in practice: grepped
2026-08-04, zero occurrences across all scoped paths. `#token` id
rules count as styled (console-staged-context-empty). There is no
style-free `is-*` convention here (test_master_shell_design_system_
contract.py's REQUIRED_STATE_CLASSES, lines 20-35, pins those as
styled).

Scope addition 2026-08-04: `UI/Console_Modules/` (frame.py, left_rail.py,
right_rail.py, dictation.py) added to SCOPES -- the decomposed Console
rails were composing classes (including console-agent-section, already
registered below) with zero guardrail coverage. Widening surfaced no new
unstyled tokens: every class composed there is either bundle-styled
(.class rule) or covered by a same-named #id rule on the same widget
(e.g. #console-left-rail-body, #console-model-section-recovery,
#console-inspector-rail-body) -- no new KNOWN_UNSTYLED entries were
needed.

Correction to the original scout note: it is NOT true that no scoped
unstyled token is a query-selector marker. Verified 2026-08-04 by
grepping every scoped file for `query(".x")` / `query_one(".x")` /
`has_class("x")` / `@on(..., ".x")` against each unstyled token: several
of the registered tokens below ARE selector handles, not styling
oversights -- `console-send-button`, `console-settings-error-summary`,
`console-transcript-summary-banner`, `destination-purpose`,
`destination-status-row` and `mcp-perm-server-profile-row` are pinned by
has_class()/query() assertions in tests, and `console-markdown-header`/
`console-markdown-footer` are read by ConsoleMarkdownMessage's own
in-place `sync_message` update. Each is documented individually below
rather than folded into a blanket claim. (A seventh, `copy-command`, was
a production `@on(Button.Pressed, ...)` route until IngestGuardrailModal
was deleted; see the note where its entry used to be.)
"""
import re
from pathlib import Path

from Tests.UI.test_non_obscuring_focus_contract import (
    BUNDLE, css_selectors, css_selectors_contain_class,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tldw_chatbook"
SCOPES = [
    PACKAGE / "UI" / "MCP_Modules",
    PACKAGE / "UI" / "Console_Modules",
    PACKAGE / "Widgets" / "Console",
    PACKAGE / "UI" / "Screens" / "chat_screen.py",
    PACKAGE / "UI" / "Screens" / "library_screen.py",
]
CLASSES_ATTR = re.compile(r'classes="([^"{}]+)"')
# Final-review fix: `[fF]?` accepts an f-string opener (e.g.
# console_settings_modal.py:179's `DEFAULT_CSS = f"""`) so an f-string-only
# DEFAULT_CSS block isn't invisible to `_styled_tokens()` below -- before
# this, such a block contributed ZERO selectors, f-string braces or not.
# A rule with no `{NAME}` interpolation inside its own body (e.g. that
# file's `.console-settings-error`) is then parsed exactly like any other
# rule; a rule WITH one (`.console-settings-modal-row`/`-label`) can still
# lose its own selector text to the interpolation's braces (pinned/
# explained in test_f_string_default_css_block_is_visible_to_styled_tokens
# below) -- harmless only because KNOWN_UNSTYLED tokens are cross-checked
# against the bundle too, never this regex alone.
DEFAULT_CSS_BLOCK = re.compile(
    r'(?:DEFAULT_CSS|CSS)\s*(?::\s*\w+\s*)?=\s*[fF]?(?:"""|\'\'\')(.*?)(?:"""|\'\'\')',
    re.DOTALL)

KNOWN_UNSTYLED: dict[str, str] = {
    # token: one-line reason it is allowed to have no rule.
    # Frozen registry — additions require an explicit edit here, which is
    # the point. All entries below predate this program (RAG UX v2
    # PR-4/PR-5) per git blame -- none are tokens tasks 1-7 of this plan
    # composed, so none are fixed here (that would be scope creep).
    "console-agent-section": (
        "duplicates the agent rail body's own id "
        "(#console-rail-section-body-agent); toggled via query_one(#id) plus "
        "the shared, styled .console-rail-section-body class, never "
        "selected via .console-agent-section itself."
    ),
    "console-attachment-indicator": (
        "duplicates the widget's own id (#console-attachment-indicator); "
        "text/visibility are driven by query_one(#id) + styles.display, "
        "never selected via the class."
    ),
    "console-background-effect": (
        "ConsoleBackgroundEffect is positioned/sized by the type selector "
        "`ConsoleTranscriptSurface > ConsoleBackgroundEffect` in its own "
        "DEFAULT_CSS and paints itself via render_line(); the class token "
        "has no selector anywhere."
    ),
    "console-composer-menu-button": (
        "per-button identifier stacked on the styled `destination-action-"
        "button` base class; width/tooltip come from _bounded_button, not "
        "CSS -- carries no rule of its own and isn't queried."
    ),
    "console-composer-presentation": (
        "shared marker on both the expanded and collapsed composer "
        "Horizontals; visibility is toggled entirely via `.styles.display` "
        "in Python (set_collapsed), not this class."
    ),
    "console-composer-toggle": (
        "per-button identifier stacked on the styled `destination-action-"
        "button`; same pattern as console-composer-menu-button -- no "
        "distinct rule, not queried."
    ),
    "console-dictation-button": (
        "per-button identifier stacked on the styled `destination-action-"
        "button`; no distinct rule, not queried."
    ),
    "console-fleet-coachmark": (
        "duplicates the widget's own id (#console-fleet-coachmark); shown/"
        "hidden entirely via query_one(#id) + styles.display/height, never "
        "selected via the class."
    ),
    "console-markdown-footer": (
        "query-selector handle: ConsoleMarkdownMessage.sync_message reads "
        "query_one('.console-markdown-footer', Static) to update it in "
        "place. Sizing comes from that widget's own DEFAULT_CSS type rule "
        "(`ConsoleMarkdownMessage > Static`), so the class carries no "
        "style of its own by design."
    ),
    "console-markdown-selection-strip": (
        "query-selector handle used to update the reusable selection strip; "
        "visibility is driven directly by styles.display."
    ),
    "console-save-as-context": (
        "plain descriptive Static (role/excerpt text) with no visual "
        "treatment of its own; not queried by class anywhere."
    ),
    "console-send-button": (
        "query-selector handle pinned by "
        "test_console_internals_decomposition.py:427 "
        "(button.has_class('console-send-button')), not a style hook -- "
        "visuals come from the styled `destination-action-button` base."
    ),
    "console-settings-context-view": (
        "duplicates the section's own id (#console-settings-context-view), "
        "which is what both the widget (query_one) and "
        "Tests/UI/test_console_context_controls.py select on; the class "
        "token itself is never used as a selector."
    ),
    "console-settings-model-view": (
        "behavioural grouping marker, not a style hook: "
        "query('.console-settings-model-view') collects the model-mode "
        "sections to toggle their display in Python; styling comes from the "
        "styled console-settings-modal-section class stacked alongside it."
    ),
    "console-settings-error-summary": (
        "presence pinned by test_console_session_settings.py:1764 "
        "('console-settings-error-summary' in error.classes) for test "
        "identification; no distinct CSS rule."
    ),
    "console-setup-modal-detected-action": (
        "per-button identifier stacked on the styled "
        "`console-setup-modal-action`; no distinct rule, not queried."
    ),
    "console-stop-button": (
        "per-button identifier stacked on the styled `destination-action-"
        "button` (companion to console-send-button, but not itself test-"
        "pinned); no distinct rule, not queried."
    ),
    "console-tool-diff-selection-strip": (
        "query-selector handle used to update the reusable diff selection "
        "strip; visibility is driven directly by styles.display."
    ),
    "console-transcript-original-attempt": (
        "plain descriptive Static for a repaired citation's original text; "
        "no distinct rule, not queried."
    ),
    "console-transcript-summary-banner": (
        "query-selector handle pinned by test_console_native_transcript.py"
        ":1853,1860 (transcript.query('.console-transcript-summary-"
        "banner')), not a style hook."
    ),
    "console-turn-file-note-delete": (
        "event-routing and test-query handle on an otherwise standard "
        "compact Button; it carries no distinct visual rule."
    ),
    "console-turn-file-rows": (
        "query-selector handle for the file-card rows container; child rows "
        "carry their own styles."
    ),
    "console-workspace-recovery": (
        "plain descriptive recovery-copy Static shared across four compose "
        "sites in console_workspace_context.py; no distinct rule, not "
        "queried."
    ),
    "console-workspace-status-row": (
        "plain descriptive handoff-status-row Static; no distinct rule, "
        "not queried."
    ),
    # (xhigh review round) "copy-command" WAS here, as the event-routing
    # selector for @on(Button.Pressed, '.copy-command') in
    # library_screen.py. Deleting IngestGuardrailModal removed the only
    # composition of that class and the handler with it -- verified by
    # grep across the whole package: no classes="…copy-command…" and no
    # `.copy-command` selector survive anywhere. The inline warnings'
    # replacement copy buttons carry a DIFFERENT token
    # (`ingest-preflight-copy-command`, Widgets/Library/
    # library_ingest_canvas.py), which is not composed in any scoped file,
    # so widening SCOPES was not the answer either. The one remaining
    # "copy-command" in the tree is an unrelated `id=` in
    # Utils/widget_helpers.py, which this registry does not track.
    "destination-purpose": (
        "query-selector handle pinned by test_destination_shells.py:1094 "
        "(screen.query_one('.destination-purpose', Static)), not a style "
        "hook."
    ),
    "destination-status-row": (
        "query-selector handle pinned by test_home_screen.py:135 "
        "(has_class('destination-status-row')), not a style hook."
    ),
    "mcp-audit-subview-btn": (
        "shared base marker on the Executions/Findings toggle buttons; "
        "the active-state visual comes from the separate, styled "
        "`is-active` class (set_class in _apply_subview_display) -- this "
        "base class carries no rule itself."
    ),
    "mcp-optin": (
        "redundant marker alongside the styled `mcp-callout`/`console-"
        "action-subdued` siblings on the built-in-disabled callout Button "
        "(F-051); not queried anywhere -- the opt-in variant isn't "
        "visually differentiated from other callouts yet."
    ),
    "mcp-perm-server-profile-row": (
        "query-selector handle pinned by test_mcp_permissions_mode.py and "
        "test_mcp_workbench.py (multiple app.query('.mcp-perm-server-"
        "profile-row') assertions), not a style hook."
    ),
}

def _scoped_files():
    for scope in SCOPES:
        if scope.is_file():
            yield scope
        else:
            yield from sorted(scope.rglob("*.py"))

def _composed_tokens():
    tokens = {}
    for path in _scoped_files():
        text = path.read_text(encoding="utf-8")
        for match in CLASSES_ATTR.finditer(text):
            for token in match.group(1).split():
                tokens.setdefault(token, path.relative_to(ROOT))
    return tokens

def _styled_tokens():
    bundle_text = BUNDLE.read_text(encoding="utf-8")
    selectors = css_selectors(bundle_text)
    for path in _scoped_files():
        for block in DEFAULT_CSS_BLOCK.finditer(path.read_text(encoding="utf-8")):
            selectors.extend(css_selectors(block.group(1)))
    return selectors

def test_every_composed_class_is_styled_or_registered():
    selectors = _styled_tokens()
    missing = []
    for token, path in sorted(_composed_tokens().items()):
        if token in KNOWN_UNSTYLED:
            continue
        if css_selectors_contain_class(selectors, f".{token}"):
            continue
        if any(re.search(rf"#{re.escape(token)}(?![\w-])", s) for s in selectors):
            continue
        missing.append(f"{token}  (first composed in {path})")
    assert not missing, (
        "Composed class tokens with no .class or #id rule in the bundle or "
        "any DEFAULT_CSS, and not on KNOWN_UNSTYLED:\n  " + "\n  ".join(missing))

def test_registry_entries_are_still_unstyled():
    """A registry entry whose token gained a rule is stale — remove it."""
    selectors = _styled_tokens()
    stale = [t for t in KNOWN_UNSTYLED
             if css_selectors_contain_class(selectors, f".{t}")]
    assert not stale, f"KNOWN_UNSTYLED entries now styled — delete them: {stale}"

def test_registry_entries_are_still_composed():
    """A registry entry no one composes anymore is dead weight — remove it."""
    composed = _composed_tokens()
    dead = [t for t in KNOWN_UNSTYLED if t not in composed]
    assert not dead, f"KNOWN_UNSTYLED entries no longer composed — delete them: {dead}"

def test_f_string_default_css_block_is_visible_to_styled_tokens():
    """Pins the `[fF]?` widening of DEFAULT_CSS_BLOCK: an f-string-opened
    DEFAULT_CSS block (console_settings_modal.py:179) must still be scanned
    for .class rules, not silently skipped the way a bare triple-quote-only
    opener regex would skip it (before the fix, `blocks` below was empty --
    the whole file contributed zero selectors). Harmless today (all four
    .console-settings-* rules in this block are also covered by the bundle
    -- test_every_composed_class_is_styled_or_registered passes either way),
    but a future token styled ONLY in an f-string block must be detected
    here, or it would wrongly need a KNOWN_UNSTYLED entry that
    test_registry_entries_are_still_unstyled could never flag as stale.

    Only .console-settings-error is pinned here, not all four siblings:
    css_selectors() reads a rule's selector text from before the first
    brace IT matches, which is reliable only when nothing brace-bearing
    (like this file's `{MODAL_CONTROL_HEIGHT}` interpolations) precedes the
    rule's own opening brace pair inside the SAME body -- true for
    .console-settings-error (no interpolation in its body) but not for
    .console-settings-modal-row/-label (interpolated declarations before/
    inside them steal the match), which is exactly why f-string DEFAULT_CSS
    blocks stay unreliable enough that KNOWN_UNSTYLED tokens must keep
    verifying styling against the bundle too, not this regex alone."""
    path = PACKAGE / "Widgets" / "Console" / "console_settings_modal.py"
    text = path.read_text(encoding="utf-8")
    blocks = list(DEFAULT_CSS_BLOCK.finditer(text))
    assert blocks, "DEFAULT_CSS_BLOCK must match ConsoleSettingsModal's f-string DEFAULT_CSS"
    selectors = []
    for block in blocks:
        selectors.extend(css_selectors(block.group(1)))
    assert css_selectors_contain_class(selectors, ".console-settings-error"), (
        "f-string DEFAULT_CSS selector .console-settings-error not visible to css_selectors()")
