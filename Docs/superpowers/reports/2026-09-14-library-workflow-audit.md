# Library workflow audit — 2026-09-14

Current checkpoint (2026-09-16): the initial findings and subsequent bounded
reviews have progressed through Prompts, Skills, Collections, Import and
Conversations reading, Archive and Restore. Real local Import success/restart
recovery and the caught Console sidebar startup error are now qualified and
closed. The historical
sections below retain their original scope limits; the latest continuation
record at the end links the completed follow-ups. This is not a whole-application
completion claim, and integration into current `dev` still needs review.

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

The initial product findings are repaired locally: [rail focus](../qa/2026-09-14-library-rail-focus/README.md), [Workspace refresh](../qa/2026-09-14-workspace-handoff/README.md), and [focus/resize query budgets](../qa/2026-09-14-library-query-budget/README.md). The compact Notes and modal inventory gaps are closed below. The broader feature review still includes Library Prompts/Skills/Collections/ingestion details and the other application destinations; this report does not mark those reviewed. The footer, entry-compose and Prompts failures are repaired locally in the follow-up below. TASK-32462 remains open for integration into dev.

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

## Remaining test-file repairs — TASK-32462

The footer, entry and Prompts repair is documented in the
[verification record](../qa/2026-09-14-library-remaining/README.md). It separates
three product repairs from obsolete fixtures and assertions: Notes restores
same-route focus after its canvas rebuild; a confirmed Prompt deletion owns its
single page/count refresh; and first-save count reconciliation preserves the
Prompt editor while its actions adopt the saved identity.

The test repairs preserve disabled-row, stale-result, dirty-navigation,
persistence, exactly-once and widget-identity checks. The failing teardown now
uses an actual mounted screen, stylesheet parity follows the production source
and app stylesheet union, and import worker-owner fixtures retain their Library
services. Readiness checks wait for replacement controls and a settled delete
receipt, instead of treating service entry or an initially idle flag as completion.

Verification: **342 passed** in the complete Prompts file,
**159 passed** across seven complete neighboring files (including footer, entry
and the earlier Workspace repair), plus the reusable-screen leave/return
journey. Native terminal verification with exclusive ownership of private local
databases retained Notes row focus through three refreshes and retained the
first-saved Prompt's text widget. Persisted content/version were checked, the
terminal size was measured at 170×48, and the app exited 0. Existing Console
startup and optional-service warnings remain documented; this is not a
clean-startup qualification or a full repository sweep.

TASK-32462 remains In Progress because its first four criteria explicitly require
passing on `dev`; these local commits have not been integrated. The broader
feature audit still includes remaining Prompt interactions, Skills, Collections,
ingestion details and other application destinations.


## Prompt save and browse continuity — TASK-32603

The [Prompt continuity record](../qa/2026-09-14-prompt-continuity/README.md)
closes the previously unqualified Items-loading observation. New prompt did not
dispatch its resident list request; first save refreshed only source counts.
Both now use the exact browse controller. The same journey exposed default
fractional section heights clipping Basic and Advanced text, Back leaving a
cleared editor mounted after the previous retention fix, and retained block
markers/provenance remaining draft-like after save. These are repaired without
rebuilding live text fields.

Verification: **8 passed** in the real SQLite / production CSS continuity matrix,
**88 passed** in the affected Prompts selection, **63 passed** across complete
reader/browse-controller/resize-budget files, and **31 passed** in governance.
A final reader/continuity rerun after all production edits passes **28 cases**.
Three conflict readiness cases also pass after replacing unsafe recompose-time
queries with displayed/enabled-action waits. Counts overlap across reruns; no
full repository sweep was run. Changed methods are formatted with no new Ruff
diagnostics, and the user guide describes the repaired flow.

Private native journeys at measured 170×48 and 80×24 preserved text, settled the
Items count, showed clean saved markers and reopened the same version. Wide
shutdown returned 0; the final compact run completed its UI assertions but still
appeared active when its owned terminal was closed. The evidence explicitly
excludes its stale prior exit file. Existing startup warnings remain, and neither
run qualifies a provider interaction or complete app startup/shutdown.

**Next:** TASK-32602 tracks the observed cross-size Prompt focus loss. Stable-size
Basic/Advanced coverage is complete for this journey; responsive focus and the
remaining Prompt actions, Skills, Collections, ingestion and other destinations
remain in the feature review. Integration into dev is still pending.

## Prompt resize focus — TASK-32602

The [resize evidence](../qa/2026-09-15-prompt-resize/README.md) closes the
cross-size focus gap. Prompt routes no longer apply the Notes fallback that
moved focus to the rail. The work pane scrolls its current focused descendant
into view after layout, preserving newer focus choices and existing text fields.
No source requests or reader-preference writes occur during resize.

Verification: **58 Prompt/resize checks**, **6 neighboring Notes focus checks**,
and **31 governance checks** passed. Two neighboring tests now wait for the
relevant rendered control or use an actual keyboard event; their original
behavioral assertions remain. Static review adds no Ruff diagnostics.

The private native run passed **16 actual terminal resize cases** in Basic and
Advanced, dark and light themes. It exited normally from 80×24 with a fresh
exit-0 receipt and observed shell return, resolving the outstanding compact
shutdown qualification for this run. Existing startup warnings remain recorded.

Next in the feature review: remaining Prompt actions and recovery paths, then
Skills, Collections, ingestion and other destinations. Integration into `dev`
remains pending.

## Prompt More actions and deletion recovery — TASK-32628

The [action and recovery evidence](../qa/2026-09-15-prompt-actions/README.md)
records three repairs: deletion failures reveal their existing error status;
saved New prompt editors accept confirmed deletion and its recovery result;
and a removed work pane ignores a pending resize callback. The existing
identity/version and mutation ownership checks remain, and bulk selection
continues to require the Browse route.

The new real-SQLite journeys cover all six menu controls with keyboard focus
and painted labels at both sizes and themes, detached Duplicate, Cancel,
storage failure and retry, Undo, Dismiss, and the saved Create route. Neighboring
checks also exposed two readiness gaps: a reader test ended before its
replacement dropdown mounted, and a compact focus test sampled an unfinished
scroll. A controlled compose delay reproduced the first; scroll position
observations explained the second. Their assertions remain intact with explicit
mount and focus/paint waits. The final selections pass **127 distinct targeted
checks**: 11 new journeys, 49 existing action/recovery cases, 36 neighboring
cases, and 31 governance checks. No new Ruff diagnostics were introduced.

The private native journey passes at measured 170×48 dark and 80×24 light.
Read-only post-exit SQLite checks confirm the original stays at version 1 and
the two copies remain deleted at version 4 after Undo and a second deletion.
Normal compact Ctrl+Q returns exit 0 and the shell. Existing startup/quit notices
are retained in the evidence; no provider interaction was qualified.

Next: complete the actual Export, Copy Markdown, History, Collections and
Use in Console flows, then continue through Skills and the other Library
destinations. This pass qualifies those menu controls' reachability only where
their full action is outside the recovery journey. Integration into `dev`
remains pending.

## Prompt Copy and Markdown export — TASK-32629

The [Copy/Export evidence](../qa/2026-09-15-prompt-export/README.md) closes the
actual clipboard and file-picker journeys. Native compact review exposed
stacked result notifications covering Copy and the next picker Save button.
Copy and Export now reveal their existing inline status without moving focus
or rebuilding fields. Delayed results use an application notification when the
original editor, Prompt identity or Library screen is no longer active.

The final **104 targeted checks** pass: 19 new production-CSS/SQLite journeys,
42 neighboring Copy/Export cases, 17 parser/renderer round trips, 13 token/bundle
checks and 13 controller wiring checks. Legacy and structured Prompt/Recipe
content is preserved; unavailable/throwing clipboard adapters, write failure,
retry, cancellation and focus return are covered. The stale renderer limitation
docstring and user guide now describe current behavior. No new Ruff diagnostics
were introduced, and no full repository sweep was run.

The final private native journey passes at measured 170×48 dark and 80×24 light,
including actual OSC52 clipboard handoff, a real FileNotFoundError, keyboard
Save retry and matching file output. Six rendered captures show readable
feedback and focused Export/Save. Normal compact Ctrl+Q returned exit 0 to the
observed shell; read-only SQLite inspection confirms the source remains live
at version 1 with unchanged multiline content. OS clipboard delivery and a
provider interaction were not qualified; existing startup/worker notices remain.

Next: History, Collections and Use in Console, followed by Skills and the other
Library destinations. Integration into `dev` remains pending.


## Prompt retained History — TASK-32630

The [History review](../qa/2026-09-15-prompt-history/README.md) closes the saved
Prompt History journey. More actions now reveals and focuses History from Basic
while closing its menu. Version rows retain readable multiline labels when
focused. Selection, paging, retry and modal return preserve keyboard position;
successful restore returns to the new History title. Live draft fields and
existing immutable scope, restore and concurrency contracts remain intact.

Verification: **226 targeted checks pass** (76 mounted History/journey checks,
124 controller/state/DB/normalization checks and 26 token/bundle/wiring checks).
There are no new Ruff diagnostics; no full repository sweep was run. The final
private native run passes at 170×48 dark and 80×24 light through 12 saved
versions, paging, literal read-only preview, Cancel and restoring v1 as v13.
Six rendered captures show the focused version, Restore return and updated
History title. Normal Ctrl+Q returned exit 0 to the observed shell; read-only
SQLite confirms both active v13 records and the retained version sequence.
The existing success toast can temporarily overlay lower rows; its focused
History title remains readable. Native error injection was not part of this
run; targeted tests cover those branches.

Next: Prompt Collections and Use in Console, followed by Skills and the other
Library destinations. Integration into `dev` remains pending.

## Prompt Collections — TASK-32632

The [Collections review](../qa/2026-09-15-prompt-collections/README.md) repairs
the Basic More actions entry, erased manager input drafts, offscreen modal-return
focus and lost Apply retry focus. Done stages memberships; Apply persists them
without saving or replacing the Prompt draft. Cancel preserves the prior set.
Existing local ownership, atomic membership and stale-result contracts remain.

**180 distinct targeted checks pass**, including six new keyboard journeys,
catalog and membership regressions, neighboring History/actions, and governance.
No new Ruff diagnostics or token changes; no full repository sweep. The final
native run passes at 170×48 dark and 80×24 light through creation, name collision,
rename, staging, Apply and Cancel. Six rendered captures show the final controls.
Normal Ctrl+Q returned exit 0; read-only SQLite confirms original v1 Prompt
content and the exact renamed collections and memberships. The owned session
was closed. The existing staged summary may use `Collection #ID` until Apply,
and compact status text below Apply may require scrolling. Native fault injection
and large-catalog qualification remain covered by automated tests only.

Next: Use in Console, followed by Skills and the other Library destinations.
Integration into `dev` remains pending.


## Prompt Use in Console — TASK-32638

The [handoff review](../qa/2026-09-15-prompt-console/README.md) repairs escaped
brace decoding when User-only Prompts insert directly from Library or Console.
It also removes the System checkbox's clipped duplicate label and refreshes the
Console System status chip after authorized replacement. Existing append,
snapshot replacement, explicit System authority, original-source and one-shot
handoff contracts remain intact.

**369 distinct targeted checks pass**, covering production-CSS keyboard
journeys, shared dialog/Console application, parser, claims, Recipe detachment,
missing/stale targets, recovery, wiring and governance. Four escape cases and
four paint cases reproduced the defects before repair. There are no new Ruff
diagnostics, token changes or outstanding independent review findings.

The native TldwCli journey passes at 170×48 dark and 80×24 light through direct
insertion, Cancel, System opt-in, Apply, original placeholders and repeated
navigation without duplicate insertion. Rendered confirmation shows the fixed
checkbox and wide System chip; compact variable inputs scroll above fixed
actions. The compact System chip lies beyond the initial horizontal status
viewport. Ctrl+Q returned exit 0 to the observed shell, which was then closed.
Read-only SQLite confirms all four original v1 Prompt records and zero messages.
No provider request, full-suite or durable System restart qualification is claimed.

Next: Skills and the remaining Library destinations. Integration into `dev`
remains pending.


## Skills browsing and editor — TASK-32646

The [Skills editor review](../qa/2026-09-15-skills-editor/README.md) repairs
Save focus, collapsed-list returns, and Cancel's rail/content mismatch. Back
waits for both the list result and its canvas rebuild before focusing a row.
A completed write preserves edits made during I/O as an unsaved draft with
the committed version and trust metadata, so the next Save persists them.
Name-collision warnings now cover twenty newer runtime and Console names.

**277 targeted checks pass**, including four size/theme keyboard journeys,
a held-write second-save case, forced Back/Cancel refresh races, neighboring
Skills behavior, shared list focus, token/bundle/wiring governance and the
Skills controller size checks. Nine stale canvas expectations were updated to
current grip, layout, token and footer contracts. No new Ruff diagnostics.
The inherited screen-size check still fails: 35,203 lines against 33,204,
one line smaller than HEAD. Its budget was not raised; no full suite was run.

The final private native run passes at 170×48 dark and 80×24 light. Six rendered
captures show reachable actions and returned row focus. Normal Ctrl+Q returned
exit 0 and the owned shell was closed. Read-only checks confirm both edited
SKILL.md files, exact duplicate/unavailable allowlists, no cancelled draft,
ten SQLite integrity results and zero messages. An existing dirty-veto toast
can linger over the lower pane without hiding the focused action or row.
Import, trust setup/approval, execution and restart qualification remain outside
this native journey.

Next: Skills import and trust journeys, then the remaining Library destinations.
Integration into `dev` remains pending.

## Skills import and trust — TASK-32655

The [import and trust review](../qa/2026-09-15-skills-import-trust/README.md)
repairs unreadable trust dialogs, narrow import actions, selected-Skill Import,
lost Review/Cancel focus and import receipt races. Retained Items now refreshes
after trust changes while Work is open, preserving newer editor drafts. Exact
snapshot approval still rejects files changed after capture and permits a fresh
review. Existing trust authority and storage contracts remain unchanged.

**355 distinct targeted checks pass**, including four size/theme journeys,
eight dialog cases, a forced delayed-event race, prior editor save/return tests,
Skills behavior and governance. Legacy modal harnesses now load the app CSS.
No new Ruff diagnostics or token values. Two inherited governance checks remain
red: the unchanged 22 CSS allowlist offenders and LibraryScreen's 35,202 lines
against 33,204. The screen is one line smaller than HEAD; no budget was raised.

The final private native run passes at 170×48 dark and 80×24 light. Twelve
rendered captures show readable dialogs, candidate controls, captured content
and natural focus returns. Normal Ctrl+Q returned exit 0 and the owned shell
was closed. Exact file hashes, absent unselected packages, fresh locked-to-trusted
service reopening, ten SQLite integrity checks and zero messages were verified.
An existing stale-approval notice can linger over the lower pane. Full app
restart, provider execution and full-suite qualification were not performed.

Next: Skills Files and supporting-file interactions, then the remaining Library
destinations. Integration into `dev` remains pending.

## Skills Files and supporting files — TASK-32657

The [Files review](../qa/2026-09-15-skills-files/README.md) found no production
defect in the reviewed read-only inventory. New coverage qualifies empty and
populated bundles, UTF-8 byte sizes, empty files, binary labels, nested and long
paths, and a 65-supporting-file list. Tab/Shift+Tab mode switching preserves
unsaved description and instruction fields. Real private trust records and all
66 bundle-file hashes remain unchanged. The guide now explains Files and its
keyboard navigation.

**64 distinct targeted checks pass**, including four size/theme journeys,
reader/state/service checks and token/bundle governance. The new test and native
runner pass Ruff and formatting. An initial wide-layout assertion included
neighboring panes when joining wrapped text; cropping to the inventory repaired
the test. Independent follow-up review confirmed the keyboard and trust coverage
gaps were closed. No production or stylesheet changes were required.

The private native run passes at 170×48 dark and 80×24 light. Six rendered
captures show readable empty states, long paths and the final file reached by
End. Home restores mode controls; Discard naturally returns to the bundle row.
Normal Ctrl+Q returned exit 0 and the owned shell was closed. Exact file hashes,
uninitialized trust, no trust manifest, ten SQLite integrity checks and zero
messages were verified. No full suite, provider execution or app restart was run.

Next: Library Collections, then the remaining Library destinations.
Integration into `dev` remains pending.

## Collections reader continuity — TASK-32658

The [reader review](../qa/2026-09-15-collections-reader/README.md) repairs lost
annotation drafts, highlights shown under the wrong capture, unloaded status
successors, hidden mutation errors and clipped Undo. Committed highlight writes
now remain reported as saved when the following list refresh fails; newer drafts
survive pending saves. Capture-note Save stays visible at compact size without
stealing newer focus. Existing authority and revision fences remain in force.

**95 distinct targeted checks pass.** One inherited LibraryScreen ceiling check
remains red: 35,210 lines / 1,320 methods versus 33,204 / 1,276; this slice adds
one eight-line event forwarder to the already oversized screen. Both Collections
controller size checks pass, no budgets or token values changed, and there are
no new Ruff diagnostics. Historical extraction prose is preserved in the QA
appendix with current ownership rationale in source. No full suite was run.

The final private native journey passes at 170×48 dark and 80×24 light. Six
rendered captures show visible Save, correct highlights and readable Undo.
Terminal Ctrl+Q completed normal exit 0 after a posted key did not exit; the
owned shell was observed and closed. Read-only SQLite confirms two exact saved
captures, four highlights, ten integrity checks and zero messages. These native
controls use explicit focus plus Enter and direct capture selection; complete
Tab traversal, remote execution and restart are not claimed.

Next: TASK-32659, covering Clear/search, More saved searches and repeated Archive
receipts, then the remaining Library destinations. Integration into `dev`
remains pending.


## Collections browse controls — TASK-32659

The [browse review](../qa/2026-09-15-collections-browse/README.md) repairs all
three remaining control-path findings: Clear removes text and form filters with
valid sorting, saved searches page in bounded windows with failure/retry, and
repeated Archive preserves the original Undo status. Compact filter actions are
fully readable and reachable. Paging preserves newer surviving focus and falls
back to a real destination when the focused outgoing search disappears.

**118 distinct targeted checks pass.** The unchanged LibraryScreen size check
still fails its existing ceiling; all Collections controller/inventory checks
pass without increasing existing budgets. New Python passes Ruff and formatting;
changed ranges are formatted with zero new diagnostics against base. Final
independent read-only review found no unresolved issue within this slice.

Private native journeys passed at 170×48 dark and 80×24 light. Six final rendered
captures confirm visible Clear, the next saved-search window, disabled Archived
and Undo. Read-only persistence checks confirm the exact two captures, all 21
unchanged saved searches, ten healthy SQLite databases and zero messages. Normal
terminal Quit returned to the shell with exit 0, and the owned session was closed.

Next review: the compact Collections Work pane's toolbar clipping while Items
is open, keyboard traversal, and manual text-search clearing under relevance.
The compact Clear capture records the toolbar limitation; this is not a complete
Collections qualification. Ingestion/import and the other application
destinations remain in the broader feature review. No full suite, remote/provider
request, push or integration into dev was performed.

## Compact Collections controls — TASK-32662

The [compact review](../qa/2026-09-15-collections-compact/README.md) repairs
clipped Work action and mode labels when Items is open. Bars stack according to
their measured content width and return to rows when space permits, preserving
mounted controls and unsaved notes. Empty or whitespace-only text submission
under relevance now restores unfiltered results with valid sorting and the
selected scope intact.

**137 distinct targeted checks pass**, including actual Tab/Shift+Tab and Enter,
draft retention on resize, search recovery, existing Collections journeys,
controller/query budgets and token/bundle governance. The unchanged
LibraryScreen size check still fails its existing ceiling; Collections budgets
pass without increases. Changed Python has zero new Ruff diagnostics against
base, and independent read-only review found no actionable issue.

Private native journeys passed at 170×48 dark and 80×24 light. Four rendered
captures show readable controls, visible focus and recovered search results.
Read-only persistence confirms two exact Saved captures, unchanged notes, ten
healthy SQLite databases and zero messages. Normal terminal Quit returned exit
0, and the owned session was closed.

Next review: Library ingestion/import journeys. Full form traversal, remote
authorities, extraction/provider actions and restart remain outside this
evidence. No full suite, push or integration into dev was performed.

## Import media entry and recovery — TASK-32663

The [entry review](../qa/2026-09-16-ingest-entry/README.md) repairs truncated
Start explanations. Recovery and consent text now wraps, with one reserved
row when empty, inside the existing docked commit bar. Mounted form fields,
metadata and keyboard focus survive gate updates and Clear/re-entry.

**271 distinct targeted checks pass.** Eight baseline clipping failures are now
green. One inherited Parakeet model-directory geometry test remained red at this
checkpoint. TASK-32664 subsequently traced it to omitted app-tier utility CSS,
not a production row overflow. Zero new Ruff diagnostics; budgets and
controller/screen sources are unchanged. Independent review found no actionable
issue in this repair. Test-only setup and CSS_PATH corrections are documented
in the QA report.

Private native checks pass at 170×48 dark and 80×24 light. Six rendered captures
show complete recovery messages and visible Start focus. Real local preflight
and keyboard Clear/re-entry preserve metadata without submitting an import.
Read-only checks confirm zero media, messages and jobs, ten healthy databases,
unchanged source content and normal exit 0. The owned shell was closed.

Next at this checkpoint: per-type ingest options and picker continuity,
then queue activity and recovery. File-picker operation, actual import
execution, remote/provider actions and restart remain unqualified. No full suite,
push or integration into dev was performed.

## Parakeet import folder selection — TASK-32664

The [directory-picker review](../qa/2026-09-16-ingest-options/README.md) corrects
the geometry harness to load production app CSS. The row already fits. The
actual defect was whole-form replacement after Select: it reset the title cursor
and returned compact Browse focus below the viewport. Updating the existing
folder field preserves metadata, cursor, viewport and keyboard return; Cancel
preserves the staged folder. An explicit gate refresh also clears stale Start
confirmation after selecting the same folder. **279 distinct targeted checks
pass**, with zero new Ruff diagnostics and no increased budgets. Focused review
found no remaining actionable findings after the same-folder repair.

Private native Select/Cancel journeys pass at 170×48 dark and 80×24 light;
read-only checks confirm ten healthy databases and zero media, messages and
ingest jobs, with normal exit 0. The real unavailable-package state is recorded;
enabled folder controls use an explicit UI-only availability simulation.
No installation, model execution or import was performed.

Next: compact directory-picker layout. The 80×24 capture reports one loaded
folder but has no visible list rows, and the typed path field is narrow.
Then continue remaining per-type options and queue activity/recovery. Typed-path
selection is qualified; compact list navigation, extraction, remote authorities
and restart remain unqualified. No full suite, push or integration into dev.

## Compact folder listing and validation — TASK-32665

The [compact picker review](../qa/2026-09-16-compact-picker/README.md) repairs
the missing folder rows at 80×24. Scoped compact layout gives the dialog more
space, reduces chrome and keeps its path and actions readable. Long validation
messages no longer consume the list; correcting to the current folder clears
stale errors, and overlong names return a validation message instead of raising.
Mounted inputs, selection, highlighted folder and focus survive resizing.

**154 distinct targeted checks pass**, including empty/scrolling folders,
keyboard parent/child navigation, selection/cancellation, validation recovery,
Library caller continuity and token/bundle governance. Boot CSS remains within
the unchanged limit at 615,348/634,050 bytes. No new Ruff diagnostics or increased
budgets; final independent review found no actionable issue.

Private native journeys pass at 170×48 dark and 80×24 light, including a real
terminal resize while the picker is open. Captures show visible rows and readable
errors. Ten healthy databases, zero media/messages/jobs, unchanged source content
and normal exit 0 are recorded. Parakeet availability is explicitly simulated
only to enable the UI controls; nothing installs, imports or executes a model.

Next review: remaining per-type ingest options, then queue activity/recovery.
Actual import execution, remote authorities and restart remain outside this
evidence. No full suite, push or integration into dev was performed.

## Import option editing and compact explanations — TASK-32666

The [per-type option review](../qa/2026-09-16-ingest-controls/README.md) repairs
form replacement after checkbox/select changes. Draft text, cursor/selection and
keyboard focus now survive option edits and group Reset. Dependencies, inline
validation, receipts and Start/Retry confirmation copy update together. Native
and forwarded stale edits are rejected, pending backend layouts settle safely,
and an earlier dependency refresh cannot overwrite newer sibling typing.

Long checkbox and install explanations wrap. Native inspection caught the real
compact Library shell overriding the install button with a one-row cap; a
scoped token-backed exception and two real-shell regressions repair it.
**355 targeted checks pass**, with zero new Ruff diagnostics. Existing screen
and ingest-controller size ceilings still fail at base and current; the screen
shrinks four lines and the controller is unchanged. No budget was raised.
Independent review found no remaining actionable issue.

Final private native journeys pass at 170×48 dark and 80×24 light, with ten
rendered-state captures, exact retained editor checks, ten healthy databases,
zero media/messages/ingest jobs, unchanged synthetic source bytes and normal
exit 0. Enabled optional controls use an explicit UI-only availability seam;
real unavailable-package explanations are checked first. No installation,
extraction, import submission or remote/provider operation was performed.

Next review: queue activity and recovery. An attempted fully expanded-form Retry
focus check painted the docked fold hint over its row; that keyboard overlap
remains unqualified here. Actual import execution, remote authorities and
restart remain outside this evidence. No full suite, push or dev integration.


## Import queue focus and layout continuity — TASK-32667

The [queue keyboard review](../qa/2026-09-16-ingest-queue/README.md) closes the
Retry/fold overlap carried forward from TASK-32666. Queue focus is captured at
the actual rebuild, so a late Tab survives; restoration respects newer focus and
falls back to the source field when the replacement action is unavailable.
Retry now paints its complete form-replacement confirmation.

Native investigation also exposed retained focus becoming hidden after preflight
layout contracted and grew again. The canvas rechecks current focus after
virtual-size changes, following the Library rail pattern, and stops an earlier
scroll animation before a visibility check can incorrectly do nothing. The
controlled layout and animation regressions reproduce these distinct causes.

**203 targeted checks pass**, including 16 queue journeys across both themes and
wide/compact geometry. Changed code adds no Ruff diagnostics. Existing Library
screen and ingest-controller size ceilings still fail at baseline and current;
this repair adds two label-layout lines to each without raising a budget.
Independent review has no remaining findings.

Final native run-011 passes 170×48 dark and 80×24 light, with six rendered and
inspected captures and normal exit 0. Ten private databases are healthy; no
media/messages/ingest jobs were persisted. Source bytes and default-profile hashes
are unchanged. The registry was synthetic with no store or runner; no import,
installation, provider or server request occurred. Terminal probing was primed
before stdin ownership, so ordinary startup timing is not qualified.

Next review: grouped outcomes, Clear/Recent, recovery actions and live-resize
focus ownership. A trial resize while still focused in Import transferred focus
to the rail; this broader shell behavior remains unqualified. Re-entry after
resize is covered. No full suite, push or dev integration.

## Continuation checkpoint — 2026-09-16

The gaps left after TASK-32667 are closed in their bounded follow-ups:

- [TASK-32696](../qa/2026-09-16-recent-imports/README.md) preserves Recent imports
  disclosure during queue updates.
- [TASK-32697](../qa/2026-09-16-ingest-resize/README.md) preserves Import focus
  across terminal resizing.
- [TASK-32698](../qa/2026-09-16-ingest-recovery/README.md) preserves draft and
  focus through provider recovery UI updates.
- [TASK-32699](../qa/2026-09-16-filtered-picker/README.md) keeps filtered picker
  controls usable at compact sizes.
- [TASK-32700](../qa/2026-09-16-ingest-lifecycle/resume/README.md) qualifies real
  local text/Markdown imports, duplicates, Retry, interruption recovery and
  fresh-process reopening. The host semaphore failure is no longer blocking it.
- [TASK-32701](../qa/2026-09-16-conversations-continuity/README.md) qualifies
  Conversations filtering, Read/Info, Find and resize continuity.
- [TASK-32388](../qa/2026-09-16-conversation-link-receipt/README.md) makes the
  workspace-link receipt and Undo follow the workspace they name.
- [TASK-32702](../qa/2026-09-16-sidebar-startup/README.md) fixes the caught sidebar
  startup error and verifies saved state through a fresh native process.
- [TASK-32703](../qa/2026-09-16-conversation-archive/README.md) qualifies Cancel,
  Archive, version-aware Undo and Restore only at compact/wide sizes in both
  themes, including a fresh-process restore of the same saved identity. All
  four fixture messages and the current Console context remain unchanged;
  56 targeted tests pass. No production fix was needed.

Next bounded journeys: Conversations Export and exact Resume. The later
Search/RAG, Settings and other application destinations remain governed by the
original review sequence; a completed slice
does not qualify every action in its destination. Full-suite verification has
not been requested.
