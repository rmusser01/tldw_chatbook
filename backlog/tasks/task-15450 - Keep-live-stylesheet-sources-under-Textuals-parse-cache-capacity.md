---
id: TASK-15450
title: Keep live stylesheet sources under Textual's parse-cache capacity
status: In Progress
assignee: []
created_date: '2026-08-11 12:05'
updated_date: '2026-08-14 00:20'
labels:
  - perf
  - ui-platform
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-11 input-latency audit: a live headless tour of all 13 hotkey destinations ends at 93 stylesheet sources, past Textual 8.2.8's parse cache (`LRUCache(64)` in `css/stylesheet.py`). Past the cliff, every `stylesheet.parse()` runs fully cold — measured 125-127 ms per call on fast hardware, repeated back-to-back with zero cache benefit — and Textual re-runs that parse whenever a widget class not seen this session first mounts (screen switches, modals, deferred mounts). The cliff is crossed at the 8th destination; Personas alone adds 30 sources. The repo carries 183 widget `DEFAULT_CSS` declarations; each distinct mounted class adds one source. Additionally six modal classes declare class-level `CSS` (ConversationSelectionDialog, EmojiPickerScreen, VoiceBlendDialog, FileExtractionDialog, DeleteConfirmationModal, NoteSelectionDialog, plus ScraperBuilderWindow), each triggering a full cold reparse plus a whole-app restyle on first open.

Stability constraint (owner preference): fix by consolidating widget `DEFAULT_CSS` into the built bundle (`css/build_css.py` is the seam) rather than patching/monkeypatching Textual's cache size — a cache-size override is fragile across Textual upgrades. Screens must NOT gain `CSS_PATH` (the task-262 no-split verdict, Docs/Design/2026-07-17-css-split-investigation.md: per-screen CSS files re-trigger the first-push reparse this task exists to avoid). Related open umbrella: task-2902. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a full 13-destination tour the live stylesheet source count is under 64, measured by a repeatable Pilot probe (probe method recorded in the task)
- [x] #2 Repeated stylesheet.parse() after a full tour is cache-warm (single-digit ms), measured before/after
- [x] #3 The six modal class-level CSS declarations no longer trigger an app-wide cold reparse on first open
- [x] #4 No visual regressions: representative screens compared before/after (pixel A/B or rendered-CSS diff), including specificity-sensitive widgets whose DEFAULT_CSS moved
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the tour probe at HEAD and record the source list per class.
2. Read Textual 8.2.8's cascade: origin tier, tie-breaker, SCOPED_CSS injection, LRUCache(64).
3. Add a build-time lift: BUNDLED_CSS / BUNDLED_SCREEN_CSS -> generated sheets, with Textual's scope prefixing baked into the selectors.
4. Move the classes that mount during the tour; regenerate; never hand-edit the bundle.
5. Prove cascade-exactness two ways: parser-level equivalence for every block, and a computed-style diff of every node on every screen against a same-tree control.
6. Pin it: a Pilot tour bound, the rewrite equivalence, and the sheet-reproduces-from-source guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Widget CSS is lifted out of Python at build time instead of being registered per class. A widget declares BUNDLED_CSS (a screen/modal, BUNDLED_SCREEN_CSS); css/build_css.py AST-scans the package, bakes Textual's SCOPED_CSS selector prefixing into the text, and writes widget_defaults_{self,scoped}.tcss (registered by TldwCli._get_default_css as two widget-defaults sources) and screen_css_{self,scoped}.tcss (CSS_PATH entries bracketing the bundle). 50 widget classes and 7 screen/modal classes moved -- the ones that mount during the tour. tldw_cli_modular.tcss is byte-identical to dev apart from its timestamp; check_bundle_sync guards all five generated files.

MEASURED (live probe, isolated scratch HOME, 235x52): sources after the 13-destination tour 94 -> 48 (cliff 64); 3x stylesheet.parse() after the tour 127/378/134 ms -> 0.63/0.13/0.12 ms; computed styles over 2,528 nodes x 14 screens differ at 4 nodes, all 4 of which a same-tree control run also reproduces (run-to-run flake, zero cascade regressions).

AC#3, stated precisely: after a full tour, pushing each of the six importable consolidated modals gives reparse=0 and cold_parse=0, against dev's 1 reparse + 64 cold source parses on the first one. The earlier "all six push with 0/0" phrasing was true only of that fully-warmed run: the independent review, which pushed modals BEFORE touring, measured reparse=0 but 0-10 residual cold parses per modal. Those are the modals' own child widget classes that were never consolidated (the deliberate "50 of ~184" decision), not an app-wide reparse. The AC -- no app-wide cold reparse on first open -- holds in both runs; only the "0 cold parses" wording was over-stated.

Two streams per tier, not one, and this is the load-bearing decision. Textual's INJECTED scope selector carries specificity (0,0,0) where the same type name WRITTEN OUT carries (0,0,1), so every rewritten selector gains +1 in the lowest-order component -- which flipped 179 nodes in the first cut (MainNavigationBar .nav-button (0,1,0) used to lose to Textual's Button.-style-default (0,1,1) and started tying it). A rewritten rule ties exactly what it used to lose to, so the shifted rules only have to lose every tie they now enter: the scoped widget sheet takes a tie-breaker below every other default-CSS source, the scoped screen sheet loads before the bundle, and selectors Textual would not have prefixed sit in a separate stream where nothing changes.

That compensation is ONE-DIRECTIONAL and the limit is now stated in widget_css.py: it handles LOSE->TIE, not TIE->WIN (a rule shifted S -> S+1 strictly outranks anything formerly level with it, and tie_breaker is last in the comparison key so it cannot undo a strict win). Mitigations are measured, not assumed: the app bundle sits in a strictly higher origin tier, and the fix round closed the review's matching evidence gap -- the resting-state diff never exercised pseudo-classes, so :hover, :focus and :disabled were each forced on every node of the Console, Personas and MCP screens and compared dev vs branch: 3,135 node-states, ZERO differences, 117 of them nav buttons (the family whose shifted :hover/:focus rules are most exposed). No TIE->WIN flip demonstrated; none claimed impossible.

Three real dev defects surfaced. (a) note_selection_dialog and conversation_selection_dialog carry 'font-size: 10' -- no such Textual property, and an invalid property fails the WHOLE sheet, so opening either dialog on dev raises StylesheetParseError out of _load_screen_css and poisons reparsing for the rest of the session (voice_profile_dialog had the same latent bug). Removed; the property never applied. (b) Textual scopes only the LAST selector of a comma list, so 'A, .b {}' in scoped DEFAULT_CSS leaves A matching app-wide; the widget rewrite reproduces that (measured: preserving it beats fixing it, 3 diffs vs 5), the screen sheets do not, because they are live from boot rather than from first open. (c) HuggingFaceModelBrowser .header and DownloadManager .header were an exact tie on the same node, decided by mount order; the child's rule now names its container type.

Avoided trap: concatenating screen CSS into the bundle leaked EmojiPickerScreen's local $ds-* fallbacks over the real design tokens ($ds-focus-bg #51677E -> $surface app-wide). Hence separate CSS_PATH files.

FIX ROUND (independent review, verdict FIX-FIRST, one blocker):
- M1 BLOCKER: .github/workflows/css-bundle-guard.yml only triggered on paths tldw_chatbook/css/**, but four of the five generated sheets now derive from literals in 57 modules under Widgets/** and UI/**, so a widget-CSS edit without regeneration ran no guard at all -- and this workflow is standalone precisely so it survives the intentionally-cancelled CI state. Added 'tldw_chatbook/**.py' (the documented any-depth form; '**/*.py' is ambiguous for app.py directly under the package) plus the workflow file itself, to both paths lists.
- M2: the boot rebuild only inspected .tcss mtimes, so a BUNDLED_CSS edit had no effect until someone ran build_css.py by hand. New _generated_css_is_stale() also treats a Python module as an input when it is newer than the build AND mentions the marker. Both halves are needed: keying on every .py (tried first) re-runs the build subprocess on every dev boot, because editing app.py would trip it; keying on the modules the sheets currently NAME misses a module that has just gained a declaration. Cost 6.1 ms for ~1,640 files, source-tree boots only, never on a wheel install or an input path. Verified: touching config.py does not trigger, touching mcp_rail.py does, touching _buttons.tcss does, a deleted sheet does. Known gap (documented): deleting a BUNDLED_CSS module leaves no newer file, so CI's guard is the authority there. The three duplicated rebuild sites now share the helper.
- Pinned dev defect (a) two ways. test_every_class_level_css_block_parses_as_a_stylesheet runs EVERY class-level DEFAULT_CSS/CSS/BUNDLED_* block through Stylesheet.parse(), which is what raises -- the existing rewrite test calls textual.css.parse.parse, which only collects errors onto rule.errors. That guards the defect CLASS, covers VoiceProfileDialog and every unconsolidated screen, and immediately found TWO MORE live instances nobody had spotted: audio_troubleshooting_dialog (3x 'border-color', not a Textual property, plus 'margin: 0.5 0') and dictation_performance_widget ('margin-bottom: 0.5'). AudioTroubleshootingDialog is reachable from Dictation_Window_Improved, i.e. opening it crashed the stylesheet on dev. Fixed by removal, not translation -- no rule in a sheet that never parsed has ever applied, so removal is the only provably inert repair. test_selection_dialog_opens_without_a_stylesheet_error additionally PUSHES each dialog on a mounted app and asserts no StylesheetParseError and that the stylesheet still parses afterwards. Both born-red: re-adding font-size fails both with the right messages.
- m8: ConsolidatedCSSApp now also carries the two screen sheets via CSS_PATH (through a new build_css.screen_css_paths helper that owns the ordering), so a harness pushing a consolidated modal no longer gets it with no CSS. Subclasses declaring their own CSS_PATH override it, exactly as before.
- M3 evidence gap closed empirically (above). Left as filed follow-ups, not done here: m2 (screen sheets share $variable scope sheet-wide -- currently inert, verified), m3 wording, m5 (ScraperBuilderWindow is unopenable on dev via a missing FormBuilder.create_switch -- pre-existing), m6 (base-before-subclass test is weaker than it reads across the two streams), m9 (stale DEFAULT_CSS doc references incl. Constants.py's KEEP-IN-SYNC note).

Tests: Tests/UI/test_widget_css_consolidation.py now 14 tests, all passing; it pins the rewrite against Textual's own parser for all 190 class-level blocks, each generated sheet parsing, base-before-subclass ordering, the tour bound (born-red: renaming the 50 declarations back gives '98 live stylesheet sources after the tour'), and now the parse-error defect class. Tests/UI/consolidated_css.py gives harness Apps the same CSS through the same helpers TldwCli uses; 176 harness classes across 122 modules repointed, which also revealed that test_master_shell_navigation's nav geometry assertions were measuring an unstyled widget. Fix round re-ran: consolidation + both CSS guards + modal/dictation suites (65), harnesses inheriting ConsolidatedCSSApp (119), navigation + smoke (146) -- all green except two failures reproduced identically on dev.

Scope: 50 of ~184 widget classes, deliberately -- the rest would be live from boot for widgets that may never mount, and Stylesheet.apply iterates the full rule list per node. Third_Party/ and examples/ excluded (own standalone apps). Three non-literal DEFAULT_CSS declarations cannot be lifted. Sharp edge: SomeWidget.DEFAULT_CSS on a consolidated class no longer raises, it silently returns the base class's CSS (review confirmed 0 live attribute reads remain).

Files: css/widget_css.py (new), css/build_css.py, css/check_bundle_sync.py, app.py, .github/workflows/css-bundle-guard.yml, 59 widget/screen modules, 4 generated .tcss (new), Tests/UI/test_widget_css_consolidation.py + consolidated_css.py (new), 122 test modules repointed.
<!-- SECTION:NOTES:END -->
