---
id: TASK-36386
title: 'Recover boot CSS headroom: generated widget sheets copy Python indentation, and one mid-session modal rides the boot parse'
status: Done
assignee:
- '@claude'
created_date: 2026-10-22 23:58
updated_date: 2026-10-23 16:44
labels:
- follow-up
- css
- performance
priority: medium
references:
- tldw_chatbook/css/build_css.py
- tldw_chatbook/css/widget_defaults_self.tcss
- tldw_chatbook/css/widget_defaults_scoped.tcss
- Tests/Performance/test_boot_css_byte_budget.py
dependencies:
- TASK-35649
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of TASK-35649 (PR #3217, 2026-10-22). Once TASK-35649 lands, `MAX_BOOT_PARSED_CSS_BYTES` has 775 B of headroom (607,315 of 608,090 B). The PR's review measured 392 B of boot CSS added on dev in the week before, so the next ordinary CSS PR is close to a red build, the failure mode ADR-097's Context names. Two ways to shed bytes with no rendering change:
- **Indentation.** `widget_css.render_stylesheets` copies each Python block's indentation verbatim into the generated `widget_defaults_self.tcss` and `widget_defaults_scoped.tcss`. At #3217's head, the two sheets carry 30,848 B of leading whitespace and 1,212 blank lines. About 19.9 KB of that is copied per-block Python indent; the rest is the CSS's own declaration indent.
- **FileNotesHistoryDialog.** After TASK-35649 it is a mid-session ModalScreen on the boot parse (992 B). The repo's route for deferring a rarely opened mid-session modal is `MODAL_OWNED_SPLIT_SHEETS` (build_css.py around :386-399; TASK-33628.2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The generated widget-defaults sheets carry no copied per-block source indentation, and the CSS bundle still reproduces (preflight)
- [x] #2 FileNotesHistoryDialog's rules leave the boot parse through MODAL_OWNED_SPLIT_SHEETS, or the task records why they stay
- [x] #3 Every affected widget renders the same at 211x44 and 235x52 before and after
- [x] #4 The boot CSS budget's headroom is re-measured and recorded
<!-- AC:END -->

## Implementation Plan

1. Measure the base: the boot CSS census and headroom, and copies of the four generated sheets built from Python blocks.
2. (a) Dedent each block in `widget_css.render_stylesheets` with `textwrap.dedent` before isolation and splitting, so the CSS keeps its own relative indentation. The function also renders the two screen sheets, which get the same treatment. Rebuild with `build_css.py`, then parse base and head sheets with Textual's `Stylesheet` and require identical rules.
3. (b) Follow TASK-33628.2: move FileNotesHistoryDialog's `BUNDLED_CSS` into a source module as tokenized rules, add a modal-owned `ScreenOwnedSplit`, register it in `MODAL_OWNED_SPLIT_SHEETS`, point the dialog's `CSS_PATH` at the sheet, and audit that the moved ids are composed only by the dialog.
4. Render parity: a scratchpad harness through run.sh mounts the six TASK-35649 widgets and at least ten more consolidated widgets with production sheets at 211x44 and 235x52 on the base and head trees, and compares the SVGs.
5. Re-measure the boot census. Apply ADR-097's tightening convention and record the numbers in the budget comment and these notes.
6. Run the CSS build, bundle-sync, governance, boot-budget, consolidation and FileNotesHistoryDialog tests with `-n 4`, then preflight.

## Implementation Notes

Recovered 22,670 B of boot-parsed CSS with no rendering change. Boot CSS went from 607,315 B to 584,645 B against the unchanged 608,090 B limit, so headroom rose from 775 B to 23,445 B.

**(a) Dedent.** `widget_css.render_stylesheets` now applies `textwrap.dedent` to each block before isolating its variables and splitting it, so the CSS keeps only its own relative indentation. The same function also renders the two boot-parsed screen sheets, which get the same treatment.
- Bytes: `widget_defaults_self` 90,763 -> 80,459 B, `widget_defaults_scoped` 81,460 -> 71,864 B, `screen_css_self` 2,458 -> 2,318 B, `screen_css_scoped` 13,852 -> 12,176 B, measured before (b) was applied. Line counts are unchanged.
- Leading whitespace in the two widget-defaults sheets went from 30,848 B to 10,948 B; what remains is the CSS's own declaration indent. Before the change, 139 of 140 self-stream blocks and all 58 scoped-stream blocks carried a common indent; after it, none do in any of the four sheets.
- Parse parity: base and head sheets were parsed with Textual's `Stylesheet` (app theme variables, the right default/app tier). Rules matched one for one, comparing selectors, specificity, declarations, tier and tie-breaker: 648/538/13/102 rules in the four sheets. Negative control: changing one declaration in a copy of the sheet made the comparison report the rule.
- New test `Tests/UI/test_css_generated_comments.py::test_widget_streams_drop_the_class_body_indentation`. It is red with the dedent reverted (`assert '\nSample { height: auto; }\n' in ... '\n        Sample { height: auto; }\n'`) and green with it. That file's tests hit the known local `RecoveryRequired: raw_source_selection_changed` trap, so both runs used a temporary `@private_profile_test` marker, removed afterwards; CI runs the file plain, as a whole-file census entry.

**(b) FileNotesHistoryDialog.** TASK-33628.2's pattern:
- The dialog's `BUNDLED_CSS` became tokenized rules in `css/features/_notes.tcss`. That module held only comments before, and no other split owns it.
- A new `ScreenOwnedSplit` with the prefix `file-notes-history` emits `screen_modal_file_notes_history.tcss`.
- The sheet is registered in `MODAL_OWNED_SPLIT_SHEETS`, and the dialog's `CSS_PATH` loads it on first open.
- The screen is centered through its existing id `#file-notes-history-dialog-screen`, because a bare type selector has no owner token and would stay on the boot bundle.
- Owner audit (repo-relative): every moved id is composed only in `Widgets/Library/library_file_notes_workspace.py` by `FileNotesHistoryDialog`. The workspace's own History button, `#file-notes-history`, also matches the prefix, so it is pinned (`pinned={"file-notes-history"}`) and a rule naming it stays on the bundle.
- `test_modal_owned_sheets_are_loaded_by_their_modal` already iterates the registry, so the new sheet is covered without a new test.
- The moved rules are now app tier at (1,0,0) instead of widget-default tier at (1,0,1). Their declarations are identical, and nothing else in the app targets those ids, so no cascade outcome changed (see AC#3).
- Trade-off: the first open of the dialog adds one stylesheet source and pays one reparse and one app-wide restyle. Measured on the Library notes screen through the production CSS harness (139 nodes): read 1.2 ms, reparse 4.0-4.4 ms, restyle 50-54 ms, once per session. A restyle with nothing new costs the same 47-51 ms, so the cost is the restyle walk, not the new sheet. No `preload_sheet` was added; the receipt needed one only because the Console keeps 3,000 rows mounted, and Library unmounts the Console.

**AC#3 render parity.** A scratchpad harness (copied from TASK-35649's f-parity harness, never committed) mounted each surface under `TldwCli.CSS_PATH` plus the consolidated widget-defaults sources at 211x44 and 235x52, on base (dev `52b3207ff1`, two runs) and head. All 122 artifacts are byte-identical: SVG screenshots, text frames, and a computed-style JSON. Two base runs were also identical to each other.
- The six TASK-35649 widgets: SkillEvalDetail, SkillEvalPanel, LlamaCppSetupView (collapsibles expanded), VoiceOptionPicker, FileNotesHistoryDialog, SessionSummaryDialog.
- Thirteen sampled classes from the four sheets: MainNavigationBar, AppFooterStatus, LibraryEmergencyReturn, LibraryNoteFolderTargetDialog, ConsoleSaveMarkdownModal, ModelCatalogConsentModal, SkillImportChoiceModal, ConsoleFeedbackCommentModal, WorkspaceArchiveReceiptModal, AutoSpeakConsentModal, NewTaskChoiceModal, and from the screen sheets EmojiPickerScreen and VoiceBlendDialog.
- Five LibraryScreen states (home, notes browse, note editor, note create, sync roots).
- For the history dialog: a computed-style dump of every node (all `RulesMap` properties), at rest and with focus and hover on each of its six focusables, plus a screenshot per focus state.
- Negative control: with the dialog's `CSS_PATH` emptied, all 30 history artifacts differed from base, so the harness sees the lazily loaded rules.

**AC#4.** Re-measured at head: 584,645 / 608,090 B (screen_css_scoped 12,176; tldw_cli_modular 418,776; screen_css_self 2,318; widget_defaults_self 80,309; widget_defaults_scoped 71,066). ADR-097's tightening convention does not fire. The 22,670 B shed is under the 25,000 B standard slack, and 584,645 + 25,000 = 609,645 is above the limit, so "lowering" the constant would raise it. The limit stays at 608,090. This follows the ADR's TASK-23112 precedent, and the reasoning is recorded beside the constant. The CSS snapshot was refreshed with `scripts/update_boot_budget_snapshots.py --only css`, so the next breach diff names real growth rather than every dedented segment.

**Tests** (`run.sh`, `-n 4`): css build integrity, widget-css consolidation, generated comments, design-token governance, the boot CSS budget and its ratchet messages, File Notes history and workspace, modal wide tier, CSS staleness manifest, adaptive pane shell, roleplay stylesheet: 649 passed, 18 failed.
- Re-running just those 18 nodes with one command gave 15 failed on plain dev `52b3207ff1` and 14 failed at head. Every head failure also fails on dev.
- Three Folder-files tests that failed only in the long head run passed on the rerun; on dev, `test_folder_files_capability_inventory_and_modes` failed and passed in the same way, so these tests are load-sensitive on both trees.
- The consistent failures are on dev too:
  - 3 in `test_modal_wide_tier.py`: a missing `#console-hooks-review` anchor, and skip entries naming removed classes;
  - `test_main_end_to_end_manifest_and_staleness`: its scratch checkout lacks `core/_variables.tcss`;
  - 10 in `test_library_file_notes_workspace.py`: focus, contrast and timeouts.
- Component-pattern governance and the Python style inventory: 55 passed.
- The architecture size ratchets (serial): 9 failed, the same 9 on dev; none names a file this task touched.
- Run on its own, `test_css_generated_comments.py` errors on all 226 tests with the known `RecoveryRequired` trap, on dev as well as at head; in the combined run above the file passed.
- Preflight passed, including bundle reproduction for every generated sheet.

Also corrected the TASK-35649 entry in `backlog/docs/lessons-textual.md`, which said the generator copies class-body indentation.

Files: `tldw_chatbook/css/widget_css.py`, `tldw_chatbook/css/build_css.py`, `tldw_chatbook/css/features/_notes.tcss`, `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`, the regenerated sheets (`widget_defaults_{self,scoped}.tcss`, `screen_css_{self,scoped}.tcss`, `tldw_cli_modular.tcss`, new `screen_modal_file_notes_history.tcss`), `Tests/UI/test_css_generated_comments.py`, `Tests/Performance/test_boot_css_byte_budget.py`, `Tests/Performance/boot_budget_snapshots/boot_css_bytes.json`, `backlog/docs/lessons-textual.md`.
