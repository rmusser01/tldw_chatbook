# Library workflow audit — 2026-09-14

The reviewed Library paths preserve their main reading, editing and recovery flows, but three issues need follow-up: the rail can hide keyboard focus, Workspace handoff UI can retain its initial no-sources state, and focus/resize paths exceed their query budgets. This is a bounded audit, not a declaration that every Library feature is verified.

Baseline: `2939afda63` on `feat/component-pattern-library`, including the earlier component-pattern repairs and merged dev `fd30614dcdc1e6cbd39b1532769d3e10be9b12b6`. No production code or style was changed during this audit. Findings are present at this baseline; this audit does not attribute their introduction to the design-system branch.

## Scope and method

Reviewed Workspaces, Notes, Media, Conversations and Search/RAG against the existing design language. The Impeccable audit workflow was adapted to a Textual TUI: keyboard operation, compositor paint, terminal geometry, truthful status, retention and recovery. Browser-specific checks and an overall accessibility score would not be justified here.

ADR required: no

ADR path: existing [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md), [ADR-161](../../../backlog/decisions/161-component-pattern-library.md), [ADR-086](../../../backlog/decisions/086-library-adaptive-reader-shell.md), [pagination contract](../../../backlog/decisions/067-library-top-level-pagination-contracts.md) and [destructive-action reversibility](../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md).

Reason: this records evidence and follow-up work under existing contracts; it introduces no architecture or interaction design.

The capture matrix covered five destinations × empty/populated × dark/light, first at 120×45, then 80×24 and back to 120×45 on the same screen. Harnesses used the production Library screen and `TldwCli.CSS_PATH`, with deterministic service fixtures. Readers were opened through keyboard Enter; populated Search ran a fixture query through Enter. These captures establish rendering and UI behavior, not real-provider retrieval quality.

A separate native app used a private configuration with all ten configured database paths, `database.USER_DB_BASE_DIR` and `paths.data_dir` redirected into the audit scratch directory. It confirmed the hidden-focus defect and a real local Notes save → resize → close → reopen journey. No external model request was made. Existing targeted tests provided real SQLite conversation recovery and Notes import coverage.

## Product findings

| Priority | Finding | Evidence and impact | Follow-up |
| --- | --- | --- | --- |
| High | The rail's fold cue covers a focused section toggle at 80×24. | Create is focused at `(22,21,3,1)` while the docked cue occupies `(2,21,24,1)`; the toggle's painted crop is three spaces. Enter still collapses Create. This is reproduced in both themes and the native app. Users cannot see the control they will activate. | TASK-32598 |
| Medium | Workspace handoff retains the initial no-sources UI after records load. | The cached and freshly computed projection both report **2 eligible, 2 blocked**, with four source records, while the mounted row paints **unavailable until sources exist**. Two existing failing tests remain red with production CSS; this is more than a missing fixture or stylesheet. | Existing TASK-32462, updated with the differential evidence |
| Medium | Library focus and resize paths exceed their established query budgets. | Three resize frames within one layout band issue **23 Library queries** against an expected zero; each Tab issues **5** against a ceiling of one. This is reproducible query work, not a measurement of visible sluggishness. | TASK-32599 |

### Hidden rail focus reproduction

At 80×24 with the full Library rail, focus Search/RAG and press Tab once. `console-rail-section-toggle-library-create` receives focus, but the final visible row is the opaque “scroll for more” cue. The Create glyph and focus indicator are absent. Enter toggles it; enlarging to 120×45 reveals Create collapsed. Longer traversals also found obscured Import/Export and Diagnostics toggles. Reader-mode traversals did not find the same blank focus stops in Notes, Media or Conversations.

Relevant ownership: `tldw_chatbook/css/features/_library.tcss:615` (`#library-rail-fold-cue`) and `tldw_chatbook/Widgets/Library/library_rail.py`. Existing task-32219 introduced the cue; its approved discoverability purpose should remain. A fix must verify the focused control's actual paint, not just that its region is inside the screen.

Evidence: [dark](../qa/2026-09-14-library-workflow-audit/rail-fold-focus-textual-dark.svg), [light](../qa/2026-09-14-library-workflow-audit/rail-fold-focus-textual-light.svg), [geometry](../qa/2026-09-14-library-workflow-audit/rail-fold-focus-textual-dark-detail.json), and the three `rail-*.ansi` native captures in the evidence directory.

### Workspace handoff mismatch

The original two failures are `test_library_workspaces_mode_preserves_global_visibility_and_blocks_cross_workspace_handoff` and `test_library_details_section_renders_grouped_headers_and_drops_policy_prose`. The first retains the no-sources tooltip instead of the blocked cross-workspace remedy; the second retains the no-sources Handoff label.

After loading and settling the production-styled screen, `_workspace_source_records()` contains the two notes, one media item and one conversation. Both `_library_workspace_depth_state()` and its `refresh=True` result are correct. The existing task's description that sources “never register” is therefore too broad: in this reproduction the projection updates, while the mounted receipt does not. The exact missing refresh path remains for the repair task to establish.

Relevant code: `library_screen.py:12529` applies local snapshots and invalidates the projection cache; `:14016` builds/caches workspace depth; `:14330` constructs the Details rows; `:14389` constructs the actions. No permission-boundary bypass was observed or inferred.

Evidence: [painted mismatch](../qa/2026-09-14-library-workflow-audit/workspace-projection-mismatch.svg), [state comparison](../qa/2026-09-14-library-workflow-audit/workspace-projection-mismatch-detail.json).

### Query cost

The failing gates are in `Tests/UI/test_library_resize_focus_gates_t23025.py`. Nearest production call-site instrumentation attributes the resize sample to `_active_library_rail:4906` (9), `_library_focusable:7173` (9), `on_resize:8071` (1), and `_sync_library_ordinary_rail_width_contract` (4 across lines 6431, 6440–6442). The Tab sample is `_active_library_rail` (4) plus `_library_focusable` (1). These are call-site counts, not CPU profiles. Any repair must retain actual layout/focus behavior and justify any changed budget rather than merely raising the assertions.

## Verification gaps, separate from product defects

| Gap | Differential result | Follow-up |
| --- | --- | --- |
| Workspace mouse-create and rail-scroll tests omit production CSS. | Changing only `DestinationHarness.CSS_PATH` to `TldwCli.CSS_PATH` makes both pass. The original OutOfBounds click and zero scroll range are harness geometry failures. | Existing TASK-32462; include these two cases in its workspace file repair. |
| The 60×20 Notes import journey expects an authority prefix intentionally removed below 64 columns. | `library_notes_canvas.py:1060–1090` omits the redundant prefix because the source strip directly above names Library notes / Folder files. The 120×36 case passes; the compact case stops before the rest of its import journey. | TASK-32600 |
| The modal inventory aborts on `SkillImportChoiceModal(snapshot.candidates)`. | Gesture cases pass, but the bidirectional AST inventory cannot establish complete ownership coverage. This is already tracked with the moved presenter rows. | Existing TASK-31815; no duplicate task. |

Two empty-Search capture cases initially tried to run a query without source records and timed out waiting for Run to become ready. The disposable probe was corrected to capture the legitimate blocked state; both reruns passed. This was audit setup, not an additional product defect.

## What passed, and the limits of that evidence

- Notes title/body survived 120→80→120 resizing. The native app saved “Library audit evidence,” updated the list title on return, and reopened its full body. The bracketed and accented fixture title also painted literally.
- Media's loaded-row return, detail-error recovery, pending selected-versus-loaded banner and zero-result placeholder checks passed. The long transcript remained available in the reader.
- Conversation archive/restore/resume/Undo and snapshot failure tests passed, including real SQLite recovery. No live model turn was requested.
- Notes import review/cancel and retained-receipt checks passed in the selected tests, except the compact journey stopped at the stale assertion described above. This does not certify that entire 60×20 journey.
- Search/RAG's query entry, keyboard Enter, fixture handoff, empty results, missing-service recovery and uncited-answer recovery checks passed. Keystroke checks preserve mounted results/answers. Retrieval quality and provider integration were outside this audit.
- Five-cell reader grips and their vertical labels match ADR-086 and task-32355; their unusual width is intentional, not filed as a defect.

The primary workflow selection returned **198 passed, 6 failed**. The Search/RAG and resize selection returned **21 passed, 2 failed**. Failures are classified above; neither run was a full suite. The capture matrix initially returned 18 passed / 2 probe failures; both corrected cases subsequently passed. Differential checks returned 4 passes / 6 expected reproductions of the diagnosed failures, followed by the one failing projection-versus-paint assertion. Do not sum these as unique tests: reruns overlap.

Native shutdown returned to the shell with exit status **0**; its exit status and startup/exit log review are recorded with the artifacts. Startup logged a missing `#app-log-display` RichLog handler widget, plus optional-dependency/profile warnings. No clean-startup claim is made from the successful Library render. The RichLog startup issue is outside this bounded Library review.

## Next work

The initial product findings are repaired locally: [rail focus](../qa/2026-09-14-library-rail-focus/README.md), [Workspace refresh](../qa/2026-09-14-workspace-handoff/README.md), and [focus/resize query budgets](../qa/2026-09-14-library-query-budget/README.md). The compact Notes and modal inventory gaps are closed below. The broader feature review still includes Library Prompts/Skills/Collections/ingestion details and the other application destinations; this report does not mark those reviewed. TASK-32462 still tracks remaining footer, entry-compose and Prompts failures, plus integration into dev.

The [evidence directory](../qa/2026-09-14-library-workflow-audit/README.md) contains selected SVG/ANSI captures, measured state, test summaries and the disposable probes. Full raw logs, the complete capture matrix and private databases remain in ignored audit scratch. This change commits audit documentation and task records only.

## Compact Notes import follow-up — TASK-32600

The compact verification gap is closed with test changes only. At both 60×20 and
120×36, the journey now checks the compositor-painted source strip, selected
Library authority, and complete status/next-action text. The compact line keeps
task-32360's intentional omission of the repeated prefix; the wide line retains
it. Assertions still distinguish the Library's own database from Folder files.

The original chooser test ended at requesting the file picker, so fixing that
assertion alone would not verify an import. The adjacent real SQLite journey now
runs at both sizes. It confirms review leaves notes/folders untouched,
cancellation creates no note, and a fresh approved import creates one note and an
Inbox folder. The completed receipt paints “1 note created”; keyboard activation
of Back and Last import reopens the same receipt while the note count stays one.
The chooser's existing focus and mounted-pane retention checks also pass.

Baseline at `f69b1c8bbc`: **1 failed, 2 passed**, with the sole failure at the
obsolete compact prefix assertion. Final targeted selection: **10 passed**:

- Both sizes of the chooser and review/cancel/import/receipt journeys in
  `Tests/UI/test_library_notes_files_sync_journey.py` (four cases), plus its
  test-delegation guard.
- Compact clipping and authority-prefix boundary checks in
  `Tests/UI/test_library_crit10_notes_details.py` (three cases).
- Receipt reopening and real-file read-only-check/import-refresh cases in
  `Tests/UI/test_library_note_import_flow.py` (two cases).

Changed functions pass Ruff formatting; the complete journey file has the same
six pre-existing Ruff diagnostics as the baseline, with none added. Diff whitespace
checks pass. Pytest reports two unrelated warnings while cleaning old Kokoro test
directories. These are mounted Textual checks with production stylesheets and
disposable databases; the picker response and cancellation timing are controlled
fixtures. This follow-up makes no new native file-picker or real-server claim.
No new ADR is required: the existing ADR-086/150/161 contracts remain unchanged.

## Modal inventory follow-up — TASK-31815

Commit `b3109ba5cf` repairs the blocked inventory without production changes.
The skill import chooser was absent from the concrete modal contract table,
rather than an unsupported AST construction shape. Discovery then exposed an
unregistered review-set picker, a moved Export presenter and a renamed File Notes
root-picker presenter, in addition to the three stale Skills/Ingest entries
already described by the task.

The guard now compares **35 launch edges, 21 concrete types, and ten supported
owner scopes** in both directions. Skills, Ingest and Export controller scopes
are included. Both added dialog types run through the existing production-styled
gesture, exact-result, opener-focus and lifecycle checks. Five source-mutation
controls replace each repaired presenter's modal constructor with another known
dialog type and confirm that keeping the owner/presenter name does not conceal
the mismatch. Unknown constructors still fail discovery.

Baseline: **1 failed, 169 passed**, with discovery aborted at the skill chooser.
The first repaired discovery reached the comparison and exposed the Export/root
picker drift. Final command:

```sh
.venv/bin/python -m pytest Tests/UI/test_library_modal_dismissal.py Tests/Skills/test_skill_import_choice_modal.py Tests/UI/test_review_set_picker_dialog.py -q --tb=short
```

Result: **198 passed**. The complete modal file passes Ruff formatting and retains
exactly its two baseline Ruff diagnostics, with none added; diff whitespace
checks pass. The two warnings are the same old Kokoro-directory cleanup warnings
seen in the Notes run. The decomposition recipe's standing-failure entries are
removed and the repair commit is named. This covers the inventory's explicitly
supported owners and mounted dialog contracts, not every Library feature journey
or live external service. ADR-161 applies; no new ADR is required.
