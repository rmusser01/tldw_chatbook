# Collections reader continuity — TASK-32658

Reviewed on `feat/component-pattern-library`, based on `213200441a`.
The scope is Local saved-web-capture reading and annotations. Generic legacy
Collections remain a separate read-only recovery surface.

## Repairs

- Unsaved capture notes and highlight quote/note pairs survive More, mode
  changes, reader refresh and a visit to another capture. Drafts are keyed by
  capture authority and identity, retained only in memory, and pruned when the
  active authority changes. Delayed input events snapshot current widgets.
- Highlights follow the loaded capture. Identity and request-generation checks
  prevent a held response for capture A from replacing capture B's highlights.
- Archive or another status change that removes the current item from a scope
  loads its selected successor into Work. Mutation/extraction failures are
  readable and offer Refresh reader to review the current selection.
- Undo fits beside its receipt. A revision-conflicted Undo names the archived
  capture conflict and directs the user to Archived. Refreshing a successor
  does not repair the original receipt or bypass the revision check.
- A committed highlight remains reported as saved if the following list refresh
  fails. Only the submitted draft version is cleared; text typed during the
  write survives. Delete similarly separates the committed write from refresh.
- Saving a capture note retains visible Save focus at compact size. The deferred
  scroll runs only if the same capture and Save action still own focus.

Existing tokens provide error contrast and the receipt's flexible text width.
The split Library stylesheet was rebuilt from source; no token values changed.
The controller's stale byte-for-byte extraction narrative is preserved in
[controller-extraction-history.md](controller-extraction-history.md), with current
ownership and coupling rationale retained in source. This is documentation
maintenance, not a claim of architectural decomposition.

ADR required: no. Existing ADRs
[113](../../../../backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md),
[055](../../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md),
[086](../../../../backlog/decisions/086-library-adaptive-reader-shell.md),
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) govern
capture authority, reversibility, reader layout and component patterns.
Storage, service and authority contracts are unchanged.

## Automated evidence

**95 distinct targeted checks pass; one inherited screen-size check fails.**
[verification.json](verification.json) records exact commands, results and
counts without counting repeated checks twice. Twelve new journeys cover
170×48 and 80×24 in both themes, real Local service persistence, Archive/Undo,
mode changes and held-response races. Existing reader, capture controller,
service, geometry, wiring, token and bundle checks pass.

Before fixes, six initial cases reproduced draft loss, wrong-capture highlights
and the unloaded Archive successor. Three additional race cases reproduced stale
highlight publication and incorrect save receipts; the compact note-focus case
reproduced a focused but invisible Save action. Intermediate failures also
exposed clipped Undo and a submitted highlight draft being recaptured during
recomposition. These are distinct from corrected test-fixture assumptions:
CaptureSaveOutcome nests its identity, the compact rail uses the current
five-column grip, and the geometry harness must load the app's CSS utilities
using a list of stylesheet paths.

[Static comparison](static-comparison.json) shows zero new Ruff diagnostics.
The new test and native runner pass Ruff and full formatting; existing changed
ranges were formatted. Python ASTs parse. Two inherited pytest garbage-directory
cleanup warnings appear in the targeted runs. No full suite was run.

The [size comparison](size-comparison.json) records LibraryScreen at
35,210 lines / 1,320 methods against ceilings of 33,204 / 1,276. Base already
measured 35,202 / 1,319; this slice adds one eight-line event forwarding method.
Both Collections controller size/slack checks pass at 1,684 and 699 lines.
No budget was raised. Final independent read-only review found no actionable
finding after the earlier highlight and stale-Undo issues were repaired.

## Native and persistence evidence

The final run is `run-003`, driven by [native_check.py](native_check.py) through
actual TldwCli, LinuxDriver and an owned tmux terminal. The
[private profile](isolation.json) uses private databases and the null keyring.
Two captures and their initial highlights are seeded through Local services
with stored article text, so no URL extraction or provider request is needed.

Controls are explicitly focused and activated with Enter; assertions after
More, Info and Save inspect the natural focus without setting it. Capture
selection uses the screen's normal selection method directly. End scrolls the
compact Highlights body. This does not establish a complete Tab-order journey.
The four size/theme automated journeys supplement the two native configurations.

The [result](result.json) confirms preserved drafts, exact saves, visible Save,
correct highlights, the Archive successor and reachable Undo. All six SVGs were
rendered through Quick Look and visually inspected in the final confirmation:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Saved note with focused Save | [capture](note-170.svg) | [capture](note-80.svg) |
| Selected capture's highlights | [capture](highlights-170.svg) | [capture](highlights-80.svg) |
| Archive receipt and focused Undo | [capture](undo-170.svg) | [capture](undo-80.svg) |

At compact size, Work scrolls vertically; Save is visible immediately above
the footer and End reveals the selected highlight. The screenshot of Undo is
before activation; the service result confirms restoration afterward.

The runner's posted Ctrl+Q did not finish shutdown on this run. Sending C-q
through the owned terminal completed normal shutdown: app.run returned, exit
code was 0, the shell was observed, and only that owned session was closed.
[Read-only persistence checks](persistence.json) confirm both saved captures,
Alpha's exact final note and three highlights, Beta's unchanged note and sole
highlight, ten SQLite integrity checks and zero messages. The app log contains
no ERROR or CRITICAL lines.

Run-001 failed before UI startup because the private data directory had not
been created. Run-002 passed its journey but visual inspection showed compact
Save below the viewport; the final run adds an explicit visible-focus assertion
and confirms the correction. An initial persistence-check glob searched only
`data/db`; including the two databases under `data/default_user` verifies all
ten. Failed runs and raw logs remain in the ignored task scratch directory.

## Remaining scope

TASK-32659 tracks the remaining browse controls: Clear leaves the text search,
More saved searches has no handler, and repeated Archive can overwrite the
original Undo receipt. Each needs a user-journey reproduction before repair.
This review does not qualify those controls, large datasets, complete keyboard
traversal, remote authorities, extraction/provider execution, or app restart.
Integration into `dev` remains pending.
