# Library Prompt continuity — TASK-32603

Baseline: `08701c8c30`, branch `feat/component-pattern-library`.
This is a bounded create → save → Basic/Advanced → Back → reopen review.
It does not qualify every Prompt action or the complete Library workstream.

## Reproduced and repaired

- **Items never settled on New prompt.** The create route invalidated the browse
  token without dispatching a request. First save refreshed source counts only.
  A real SQLite regression remained `loading` after its deadline despite a saved
  record. Both entry and first save now request the existing exact browse scope.
- **Fields were clipped behind metadata.** Unstyled mode/section containers used
  default fractional heights. The focused Basic message field occupied rows
  42–47 while its parent ended at row 40; its painted crop contained `Saved.` and
  footer text. Advanced blocks could disappear entirely. Natural section heights
  now feed the editor's existing single scroll owner, using the existing tokens.
- **Back left the old editor mounted.** The previous retention repair excluded
  the work pane from every browse update. Back cleared state but its empty pane
  never mounted. Browse settlement now synchronizes the work pane in list mode;
  an active editor still retains its fields.
- **Saved blocks still claimed unsaved changes.** Retained Advanced cards and
  provenance kept their draft projection after persistence. The saved projection
  now updates their markers and format labels without replacing text controls.

The new regression uses production CSS and real SQLite across empty/populated
libraries, both themes and 170×48/80×24 terminals. It asserts exact persisted
content/version, entry and post-save Items settlement, retained fields and focus,
painted Basic/Advanced text, saved block/provenance state, keyboard mode changes,
Back removing the editor, and reopening the same saved version.

## Verification

Commands and results are in `verification.txt`. The complete new matrix passes
**8 cases**; the complete reader, browse-controller and resize-budget files pass
**63 cases**; design-token/component/bundle governance passes **31 cases**.
The final affected Prompts selection passes **88 cases**. A final combined
reader/continuity rerun after all production edits passes **28 cases**, and the
three repaired conflict readiness cases pass separately. Overlapping runs are
not summed as unique coverage.

Three conflict tests now wait for a displayed, enabled action through subtree
replacement. Initial runs exposed a direct `query_one` during a recompose and an
activation before the prior save worker released the action. Persistence,
exactly-two writes and history identity assertions remain. One initial Discard
focus case also failed under concurrent verification, then passed in isolation;
its assertion was not weakened. See the recorded final selection result.

Changed methods are formatted. Existing whole-file Ruff debt remains unchanged
(205 / 16 / 1 / 24 diagnostics in screen / controller / canvas / canvas tests);
the new test file passes lint and formatting. `static.json` records this scope.
No full repository sweep was run. Pytest's old Kokoro cleanup warnings remain;
governance also reports existing unknown-marker warnings.

## Native evidence and limits

The native probe used LinuxDriver in separately measured 170×48 and 80×24
terminals, with exclusive profile ownership. All ten database paths, the user DB
base and data directory were private; read-only SQLite verification is in
`persistence.json`. The final records have exact content, version 1, active state
and structured format. No external provider call was part of this journey.

`native-result-*.json` records successful create/save, Items counts, retained
fields, both modes/themes and keyboard Back/reopen. Selected SVGs show the
painted text and saved markers. Route selection and field assignment were driven
by the probe; mode changes, Save, Back and reopening used keyboard activation.

Shutdown evidence is narrower: the final wide run returned exit 0. The compact
journey completed its assertions, but its final process still appeared active
when the owned terminal session was closed. An older compact run's exit-0 file
was stale and is deliberately not used to qualify the final run. Both owned
sessions are removed and no native probe process remains. The compact result
therefore qualifies the UI journey, not clean shutdown. Existing Console sidebar
startup, optional audio and unhandled worker diagnostics were observed in the
private app logs; this is not a clean-startup claim.

The probe script is retained for inspection. Its source private profile and raw
logs/databases remain in ignored `.superpowers/sdd/2026-09-14-prompt-continuity/`.
It is not a portable user-profile setup script.

## Remaining finding

**TASK-32602:** resizing a focused Advanced message from 170×48 to 80×24 can leave
it below the viewport. Returning wide can move focus to the Create rail row.
The stable-size field clipping repair does not fix that shared resize restoration
path. A local scroll callback revealed the field but did not prevent the later
focus move, so that experiment was removed. The final regression matrix makes
no cross-size focus-retention claim. Repair this before marking Prompt responsive
interaction coverage complete.

ADR required: no. Existing ADR-086, ADR-150 and ADR-161 govern these routine
reader, retention and token repairs. No data or service boundary changed.
