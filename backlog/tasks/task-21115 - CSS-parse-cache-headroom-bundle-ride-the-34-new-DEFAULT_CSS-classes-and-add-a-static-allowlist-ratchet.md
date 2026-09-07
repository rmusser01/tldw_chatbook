---
id: TASK-21115
title: >-
  CSS parse-cache headroom - bundle-ride the 34 new DEFAULT_CSS classes and add
  a static allowlist ratchet
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
updated_date: '2026-08-23 09:08'
labels:
  - performance
  - css
  - console
  - library
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21115).

34 new `DEFAULT_CSS` declarations across 29 files landed since the task-15450 consolidation
(Console modals/inspector rail/turn-file card, Library dialogs, trajectory, speech; full list in
the evidence doc). Live tour on the pin measured 47 sources (empty transcript, no modals)
against the LRUCache(64) cliff and the 56 soft guard limit; adding conversation-row classes and
~10 distinct modal opens crosses 64 today, at which point every later first-mount of any unseen
widget class re-pays a full cold parse (~150-450 ms fast HW, x3-5 constrained) for the rest of
the session. Accretion is ~+8 classes/3 days while the tour guard is red (TASK-21106) and CI
does not run. All 34 are plain string blocks that can ride the sanctioned
BUNDLED_CSS/BUNDLED_SCREEN_CSS + build_css.py mechanism; `UI/SiteConfigSettings.py:41` is also
the last class-level `CSS` remaining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The new-since-consolidation DEFAULT_CSS blocks (measured from git: 25 declarations across 21 files at both the review pin 35d4bf3a1 and base 41a240ccd; the review's "34/29" figure does not reproduce from any git baseline) and SiteConfigSettings' class CSS ride the bundle; harness parse-standalone requirements still hold
- [x] #2 A STATIC allowlist ratchet test (AST walk, no app boot) fails on any DEFAULT_CSS/CSS declaration outside the allowlist, so the invariant no longer depends on the slow integration tour running
- [x] #3 A post-change tour + 12-modal probe stays comfortably under the 56 soft limit; measured count recorded in the task
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline (teed): test_widget_css_consolidation.py (incl. integration tour with source-count instrumentation), test_ui_latency_guardrails.py, CSS bundle guard (test_css_bundle_sync_guard/build_integrity/staleness_manifest/consolidated_css_harness), focus-contract tests; probe tour+12-modal source count on base.\n2. Identify the new-since-15450 DEFAULT_CSS set from git (measured: 25 declarations across 21 files at both the review pin 35d4bf3a1 and base 41a240ccd; the doc's 34/29 does not reproduce from any git baseline -- record discrepancy).\n3. Convert in batches (Console cluster, Library cluster, misc): rename DEFAULT_CSS -> BUNDLED_CSS (none override SCOPED_CSS; all are plain string literals); convert SiteConfigSettings.CSS -> BUNDLED_SCREEN_CSS (orphan widget; attr is dead today). Re-run parse-standalone + affected widget tests per batch.\n4. Regenerate the bundle via build_css.py; commit generated sheets with sources.\n5. Add static allowlist ratchet test (AST walk, no app boot) seeded with exactly the post-conversion declarations; fails naming the offender + the two sanctioned options; also fails on stale entries.\n6. Re-run full verification incl. tour + 12-modal probe; record before/after counts; --collect-only sweep.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converted every new-since-consolidation class-level CSS block to the bundle and added a static allowlist ratchet, plus one mechanism fix the conversion surfaced.

MEASUREMENTS (tour = the 13-destination consolidation tour; probe = simulated first-mount source registration, faithful to DOMNode._get_default_css):
- BEFORE (base 41a240ccd): tour=47 sources; tour + 12 user-openable modal opens = 60 (over the 56 soft limit); tour + all 25 accreted classes = 70 (PAST the LRUCache(64) cliff -- confirms the review's arithmetic).
- AFTER: tour=44; tour+12modals=45; tour+all=45. Comfortably under the 56 soft limit; ~25 sources of headroom to the cliff.

CONVERSIONS (26 blocks / 22 files): Console cluster 16 (right_rail ConsoleInspectorRail; auto_speak_consent; changed_files_section; conversation_inspector; feedback_comment_modal; project_instructions x4; reaction_picker_modal; review_notes_modal; selection_menu; side_chat_modal; transcript ConsoleMessageHeader; turn_file_card). Library 3 (note_folder_dialog x2, note_import_canvas). Misc 6 (model_catalog_consent, trajectory_screen, trajectory_timeline, first_run_recovery_dialog SetupRecoveryDialog, modal_dismissal _BackdropClickShield, project_skills_import_modal, workspace_create_modal). Plus UI/SiteConfigSettings.py CSS -> BUNDLED_SCREEN_CSS (the last class-level CSS; it was DEAD as written -- Textual never reads CSS on a plain Container -- and the widget is an orphan, so zero production visual delta). Bundle regenerated via build_css.py (76 widget blocks, 6 screen blocks); generated sheets committed with sources.

SET-SIZE NOTE: the review's '34 across 29 files' does not reproduce -- git diff of class-level DEFAULT_CSS/CSS declarations vs BOTH the 15450 merge (c3ed2854a) and its branch point, and at BOTH the review pin (35d4bf3a1) and base, yields exactly 25 declarations / 21 files. All 25 converted; nothing needed the allowlist escape hatch (zero stragglers).

RATCHET: test_class_level_css_stays_within_the_allowlist in Tests/UI/test_widget_css_consolidation.py -- AST walk (broader than _class_css_blocks: catches AnnAssign and non-literal values, pinned by a seeded-tmp_path proof), explicit 129-entry (module, class, attr) allowlist seeded with exactly the post-conversion inventory, fails naming the offender with the two sanctioned options, and fails on STALE entries so the list only shrinks. Both failure directions mutation-tested (seeded offender file; ghost allowlist entry).

DISCOVERY + FIX (TieAwareStylesheet): converting a class OFF per-class DEFAULT_CSS exposed a Textual 8.2.8 staleness -- Stylesheet.add_source keeps the lowest tie-breaker offered for a source but does not arm _require_parse when lowering it; a class's own DEFAULT_CSS masked this by being a new source at first mount (arming the reparse itself). A consolidated class first-mounted DYNAMICALLY resolved against a stale parse where a bare Vertical's width/height:1fr defaults still held tie 0, tying and beating the sheet on source order (measured: ConsoleSelectionMenu 80x40 instead of 24x6; 19 UI tests red). Fix: css/tie_aware_stylesheet.py, wired into TldwCli and Tests/UI/consolidated_css.py's ConsolidatedCSSApp; pinned end-to-end + born-red-vs-plain-Stylesheet unit in test_consolidated_css_harness.py. Lesson added to backlog/docs/lessons-testing-evidence.md.

HARNESS UPDATES: 10 test files' bare-App harnesses switched to ConsolidatedCSSApp (the sanctioned way for a harness to get the widget-defaults geometry the classes' DEFAULT_CSS used to provide): console selection_menu (7 harnesses), reaction_picker, feedback_comment_modal, turn_file_card, conversation_inspector, modal_dismissal (2), library modal_dismissal (_FileNotesModalHarness), library note_folder_dialog, note_import_canvas; one source-text pin updated to read BUNDLED_CSS. New durable guard: Tests/UI/test_css_parse_cache_modal_probe.py asserts tour + every formerly-DEFAULT_CSS modal < 56.

REVIEW FIX ROUND (adversarial review, confirmed Major): TieAwareStylesheet's arming was a HALF-fix -- it set _require_parse on a tie-breaker lowering but left _rules_map non-None. Stylesheet.apply() reads the rules_map property BEFORE self.rules; rules_map short-circuits on a non-None _rules_map without honoring the armed reparse, and the reparse self.rules then performs replaces the re-tied source's RuleSet objects (parse cache keyed on tie_breaker), so limit_rules -- built from the STALE map -- filtered the fresh base-class default rules out of that one apply entirely. Exposed shape: a Vertical subclass that does NOT restate width/height (live: ConsoleInspectorRail, whose block styles only descendant text) first-mounted dynamically resolved width/height None/None instead of inheriting 1fr/1fr; the original e2e test missed it because ConsoleSelectionMenu restates geometry. Fix mirrors upstream's own new-source path (both flags): added self._rules_map = None in the arming branch. Red-first evidence: the new e2e test (test_dynamic_first_mount_keeps_inherited_base_defaults) and the extended unit test both FAILED on the shipped arm-only code (e2e reproduced the reviewer's None/None) and pass with the fix. Also added the reviewer's Minor: source-count-is-a-lower-bound-on-parse-cache-entries note at _PARSE_CACHE_CAPACITY in the modal probe (the hot-reload gap was already documented in the module docstring -- verified). Re-verified: consolidated-css harness file 5/5, consolidation suite 33/33 incl. tour, modal-probe guard 1/1 (counts unchanged 44/45/45), Console cluster sample 251 passed + only the 2 documented pre-existing inventory reds.

VERIFICATION (worktree venv, teed): consolidation suite 33/33 incl. the integration tour; CSS guards (bundle-sync/build-integrity/staleness-manifest/harness) 35/35; latency guardrails 2/2; focus contracts 106/106; console cluster 378 passed; library cluster 295 passed; full --collect-only sweep 55,381 collected with the SAME 29 collection errors as base (diff of ERROR sets empty; the RAG_Search config_profiles circular import class is pre-existing -- A/B'd at base for test_console_right_rail.py, test_library_note_import_flow.py, test_console_new_workspace.py). Pre-existing reds A/B'd at base and left alone: 2 modal-dismissal launch-inventory tests (documented in-file as a dev gap: ProjectInstruction modals undeclared in the contract tables), 2 note-import-canvas focus tests, tiktoken network-guard flake in turn_file_card_factory.
<!-- SECTION:NOTES:END -->
