---
id: TASK-15790
title: 'Sweep inventory: file-notes arcs left ~35 Library tests behind'
status: Done
assignee: []
labels:
  - test-health
  - library
priority: medium
---

## Description

From the task-15211 full-suite sweep (`Docs/Design/2026-08-13-tests-ui-sweep-inventory.md`,
chunks 7-8): ~35 Library tests are red on dev, all consistent with the
file-notes feature arcs shipping without their module contracts re-run:

- 10x one color contract in `test_library_file_notes_git.py`
  (`Color(51,66,78)` vs `Color(81,103,126)`) — a theme change unpropagated.
- 3x push copy: `'Push checking'` vs `'Review session changes (2) · Checking push'`.
- 15x `test_library_shell.py` notes cluster: focus-target drift
  (`'library-note-preview-region'` vs `'library-note-save'`), `NoMatches` on
  notes rows, id-set diffs.
- Stragglers in workspace/export-receipt/choice-strips/multiselect/prompts
  modules, including 2x stale doubles missing new production attributes and
  2x `coroutine raised StopIteration` in `test_library_prompts_canvas.py` —
  **triage the StopIteration pair first**: a PEP-479 conversion can mask a
  real exhausted-iterator bug in production, and "possible real bug" outranks
  every stale contract here.

Per the 15512 precedent: attribute each cluster to its causing commit before
adjusting any expectation, and treat "the test is old" as evidence about the
test only after the product behaviour is confirmed intended.

## Acceptance Criteria

- [x] The StopIteration pair is attributed (real bug vs test artifact) with evidence, before any contract is updated
- [x] Each cluster is attributed to its causing commit
- [x] Genuine product breaks are fixed rather than absorbed into expectations (one found and fixed: the skills trust-panel ordering)
- [x] The listed modules pass whole on dev

## Implementation Notes (batch 1)

Re-baselined every module on CURRENT dev before touching anything -- dev had
moved so fast that the sweep's inventory was already partly stale in both
directions: `test_library_prompts_canvas.py` (the filed StopIteration pair)
now passes whole (280/280), while the file-notes cluster had GROWN to 28
failures.

**The StopIteration class: test artifact, twice over.** Both remaining
instances were `next(gen)` with no default in tests, converting "expected
tree node missing" into an opaque `RuntimeError: coroutine raised
StopIteration`. The nodes were missing because folders in the file-notes
navigator now carry `_FolderNodeData(relative_path)` (a frozen dataclass)
while the tests matched the old `("folder", value)` tuples. Both lookups now
match the dataclass field and fail with a message listing actual node data.

**One real product bug found and fixed: the skills trust panel could stick at
"not granted" for a granted skill.** task-15457 made the skill editor's
recompose CANVAS-scoped, driven by the canvas's own message pump -- which a
screen-level `call_after_refresh` has no ordering against. The grant-fetch
coroutine's deferred `_render_library_skill_trust_panel` fired before the
canvas's children existed, swallowed `NoMatches`, and never retried (measured:
grant stored True, render ran, button absent). It now rides the same canvas
post-recompose hook the editor's arming follow-up already rides. This is the
exact stuck-forever race the method's own docstring said `call_after_refresh`
prevented -- 15457 quietly invalidated its premise.

**Stale contracts updated with attribution** (each verified deliberate via
its causing commit): the file-notes-git focus color x10 (task-15509 made
focus theme-driven; the helper now asserts the active theme's
`primary-background`, the same way 15509's own test does); Protect joining
the structurally-gated set (dca0594a5); confirm-delete span under the new
`-single-editor-actions` narrow mode (a85232c37); the disabled-marker prefix
on the commit label (Library a11y convention); "Push checking" -> the
"· Checking push" phase suffix x4 (67fec3f35). Stale doubles taught 4 new
production attributes (export-quality visibility, notes/prompts mutation
in-flight). Two measurement repairs in ingest_structural: the fold's cost is
now the collapsible's own height delta (`virtual_region.y` stopped being an
absolute anchor when 15513 nested the actions), and the contrast probe
scrolls its fields on-screen first (15513's new controls pushed Language
below the 46-row viewport -- an off-screen widget paints nothing).

Green after batch 1: file_notes_git 148/148, git_push 60/60, export_receipt +
multiselect 17/17, choice_strips + skills_canvas 230/230 (incl. the product
fix), ingest_structural 22/22, prompts_canvas 280/280 (no change needed).

## Implementation Notes (batch 2 -- the focus pair; BOTH were product bugs)

The dca0594a5 suspicion was wrong -- that commit changed focus VISUALS only.
Both failures were real product defects with different mechanisms:

**1. The cancel-first confirmation focus NEVER worked.** Its feature test was
born red at 1fbd46ec6 (verified by running it at that exact commit) and
nobody saw, because the module never ran whole -- the born-red-test class
again. Mechanism: `call_after_refresh(cancel.focus)` on the WORKSPACE widget
waits for the workspace's own refresh, and `_update_controls` patches
children in place, so that refresh never comes and the callback never fires
(spy: same instance, mounted, focus still on the pressed button). Third
variant of the never-firing-deferral family in one day (screen-vs-canvas in
the skills panel, screen-vs-canvas in 15270's era, widget-vs-children here).
Fix: `call_later` -- message-queue ordering, needs no repaint.

**2. Wide->narrow hid the editor out from under its own focus.** Genuine
regression, bisected (git bisect run, 6 steps) into 4202930d6's era; the
test passed at its birth commit d642336e6. `_narrow_view` was only set to
"editor" when a document was opened WHILE ALREADY NARROW, so opening on a
wide terminal and then shrinking routed the narrow shell to the navigator --
hiding the pane under the focused editor (Textual blurs a hidden widget's
focus to None). User-visible: open a note, shrink the window, the note
vanishes into the files list. Fix: on the wide->narrow TRANSITION with an
open document, derive the view as "editor"; transition-only so Back's
explicit navigator choice, made while narrow, keeps winning.

`test_library_file_notes_workspace.py` whole: 88/88.

## Implementation Notes (batch 3 -- the shell six; module now 586/586)

All six were stale contracts against deliberate arcs, each attributed:

- metadata placeholders: 6a607b692 moved "(optional)" into PERSISTENT labels
  (placeholders vanish on typing and cannot carry field identity);
  placeholders are guidance now. Test asserts both halves of the new
  contract.
- reset-to-defaults: the collapsed-title receipt rule shows a field only
  when CHANGED from default -- asserting "Chunk size: 1000" after a reset
  contradicted the rule the panel ships. Test asserts the receipt's absence.
- worker-group baseline: 4d4dceebc consolidated create/delete onto one
  "library_note_mutation" group.
- delete-receipt count: the sectioned rail (Inspector parity) does not mount
  Browse rows in the immersive notes stage; the test exits the stage the way
  a user does (Escape) before reading the count.
- undo interlock: rows may not RENDER during an in-flight undo -- structural
  refusal is a stronger form of the same claim; when a row renders, pressing
  it must still be a no-op.
- sync-routes focus: traced -- the back-press supersedes the navigation
  generation and the CANCEL is the contract (identity discarded, no restore
  runs); the old `library-notes-filter` pin was Textual's incidental
  nearest-focusable after the sync panel unmounted, changed legitimately by
  the sectioned rail. The pin is relaxed to "a live focusable, not a
  stranded sync widget"; where the shell chooses to put focus belongs to the
  rail arc's own tests. (Left open as a UX question for that arc: after
  backing out of notes sync, focus lands on the rail's Details toggle.)
