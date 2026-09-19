# Collections browse controls — TASK-32659

Reviewed on `feat/component-pattern-library`, based on `0f69e0ba96`.
This completes three control-path findings from the Collections reader review.

## Changes

- Clear removes text search, domain, tags and dates together, returns to page 1
  and changes relevance to saved-desc sorting. Status/Favorites scope predicates
  remain. Expanded Items forms scroll vertically; Apply and Clear occupy separate
  rows so the complete labels fit at 80 columns. Refresh reveals the action only
  when that action still owns focus.
- Saved searches use the existing service's 20-row pages. More searches, Previous,
  a page range and Retry searches make continuation and failure explicit. Failure
  keeps the last good window. Request generation and active-authority checks
  prevent obsolete responses from publishing rows. Paging retains the capture
  scope; Enter on a saved search applies its real stored query.
- Replacing a saved-search window preserves newer focus outside it and surviving
  rows within it. If a focused row disappears, focus falls back to the continuation
  target. Logical opener IDs handle a button replaced during another refresh.
- An archived capture shows disabled **Archived** with the already-archived
  tooltip. The scope service rejects a repeat before updating the revision or
  overwriting Undo, so Undo restores the original status.

ADR required: no. Existing ADRs
[113](../../../../backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md),
[055](../../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md),
[086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) apply.
The new 49-line saved-search loading module separates pending-window state from
screen dispatch; it uses existing authority/service contracts. No schema, token
value, dependency or provider boundary changed. Source CSS was rebuilt.

## Automated evidence

**118 distinct targeted checks pass; one inherited screen-size check fails.**
[verification.json](verification.json) records the commands and counting. Twenty new checks cover production-CSS browse journeys at 170×48
and 80×24 in both themes, visible failure/retry, retained focus during held page
loads, original Archive Undo, and stale page/authority responses. Existing reader,
controller, service, geometry, wiring, token and bundle checks are included.

Initial reproductions failed because Clear retained search, More had no dispatch,
and repeat Archive replaced the prior status/revision. Compact paint assertions
then exposed an unreachable form, a clipped Clear label and overlong pagination
labels. Independent review supplied two delayed-focus cases; both failed before
repair and passed afterward. The old contextual-row count assertion was updated
for the now-actionable continuation row. One attempted combined run failed during
collection because the new paint assertion imported its helper from the wrong
test module; the import was corrected before the final run.

[Static comparison](static-comparison.json) records zero added Ruff diagnostics.
New files pass full Ruff and formatting; existing changed ranges were formatted
and all changed Python parses. Two inherited pytest garbage-directory cleanup
warnings remain. No full test suite was run.

[Size comparison](size-comparison.json): LibraryScreen is unchanged from base at 35,210 lines / 1,320 methods, so its
existing 33,204 / 1,276 ceiling check still fails. All three Collections modules
and the budget inventory pass: main controller 1,687 ≤ 1,689, capture controller
699, and new saved-search controller 49. No existing budget was increased.
Independent final read-only review found no unresolved issue within this slice.

## Native evidence

[Final run](result.json), `run-002`, used real TldwCli and LinuxDriver in an owned
tmux terminal with a [private profile](isolation.json), exclusive instance lock
and null keyring. Two captures with stored text and 21 saved searches were seeded
through Local services. No extraction, provider or server request was needed.

The runner [native_check.py](native_check.py) focuses controls explicitly and
activates them with Enter. Post-Clear and post-paging assertions inspect natural
focus without assigning it. It seeds the combined search/form/relevance request
through the normal controller request method, and selects Alpha through the screen
selection method. This establishes the repaired control paths, not complete Tab
traversal or filter-entry coverage. The four size/theme automated combinations
supplement two native configurations.

All six final SVGs were rendered and visually inspected in one confirmation pass:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Cleared form, focused Clear | [capture](clear-170.svg) | [capture](clear-80.svg) |
| Second saved-search page, focused row | [capture](searches-170.svg) | [capture](searches-80.svg) |
| Disabled Archived and focused Undo | [capture](archive-170.svg) | [capture](archive-80.svg) |

At compact size Clear remains directly above the footer. The saved-search range
wraps and Previous sits below the visible rail viewport; the subsequent keyboard
journey scrolls to and activates Previous. The Archive image precedes Undo;
service checks confirm Read is restored afterward. The first native run passed
its controls, but its disabled Archive caption clipped; the final run confirms
readable **Archived**. Raw runs remain in ignored scratch.

[Read-only persistence checks](persistence.json) confirm Alpha is Read, Beta is
Saved, both initial notes are unchanged, and all 21 saved-search names, queries
and revision-1 records remain intact under one authority. All ten SQLite integrity
checks pass; there are zero messages and no ERROR/CRITICAL app log lines. Normal
C-q shutdown returned from app.run with exit 0; the shell was observed and only
the owned verification session was closed.

## Remaining review

This slice does not qualify the entire Collections feature. With Items open at
80×24, the Work pane's existing primary and mode toolbars still clip their right
ends (visible in the compact Clear capture). Review their responsive layout and
keyboard traversal next, along with manual text-search clearing under relevance
sort. Saved-search editing/creation, large datasets, remote authority behavior,
extraction and provider actions are outside these journeys. Ingestion/import and
the other application destinations remain in the broader review workstream.
Integration into `dev` remains pending.
