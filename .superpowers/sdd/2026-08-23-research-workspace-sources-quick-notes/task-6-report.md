# Task 6 report — Research round trip and closeout

## Status

Complete. Review fix round 1 closed the async repaint and Server production-seam
evidence gaps. The final round-trip boundary proves that Research Sources
preserve canonical catalog ownership, captured workspace identity, staged
recovery, and authority isolation across Local and Server. TASK-21508 is Done.

ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: Task 6 verifies and documents ADR-078's accepted canonical-owner,
qualified-authority, association-only removal, and private-overlay boundaries.

## Closeout changes

- Added one real-SQLite integration boundary spanning Local Media, persisted
  ingest jobs, source operations, workspace memberships, readiness, restart,
  duplicate reuse, partial failure, and unlink-without-delete.
- Added a Server fake contract proving exact profile/principal/workspace use,
  canonical My Media ownership, server workspace-source association, no Local
  Media write, and fail-closed identity mismatch.
- Fixed the installed migration artifact chain: v40-to-v41 and v41-to-v42 are
  now included in package data and manifests, and the release checker requires
  v40-to-v43. Installed source/sdist/wheel probes exercise the chain directly.
- Replaced a midnight-sensitive date fixture with a rolling UTC timestamp and
  synchronized one mounted Library test with its existing background-worker
  boundary. Ruff-discovered unused test names were removed.
- Rewrote the Research Workspace guide around the exact F10 destination,
  Local/Server owner selector, ASCII pane handles, Sources controls, receipts,
  readiness, overlay privacy, Quick Notes ownership, conflicts, and current
  unavailable operations.

## Required inverse evidence

Each mutation failed its named guard before the source was restored:

1. Using the visible workspace instead of the captured workspace failed the
   Local round-trip guard.
2. Writing a Server media ID into Local `media_id` failed the Server no-blend
   guard.
3. Treating a `workspace:*` keyword as membership failed the projection guard.
4. Rolling back the Library item after association failure failed the partial
   failure guard.
5. Deleting canonical media during unlink failed the unlink-retention guard.
6. Reporting Hybrid with one missing retrieval path failed the readiness guard.
7. Encoding a Quick Note body into the device overlay failed the privacy guard.

The restored inverse/readiness/privacy matrix finished with 8 passing tests.

## Verification evidence

- DB, Workspace, full Research package, and integration: **444 passed**.
- App, Library, and Notes split: **600 passed**.
- Library ingest runner: **146 passed, 1 Windows-only skip**.
- Research UI: **171 passed**; shell Research wiring: **6 passed**.
- API and private-path gates: **131 passed**.
- Packaging, including installed source/sdist/wheel migration probes: **43
  passed**.
- Library ingest canvas: **136 passed**.
- CSS build, integrity, and parity: **65 passed**.
- Ruff's default selector set passed across all 105 changed Python files; that
  run did not claim the opt-in upgrade selectors. Scoped formatter,
  changed-production `compileall`, migration artifact checks, no-blend/privacy
  scans, ASCII-handle scan, Impeccable detector (`[]`), and whitespace checks
  passed.
- The broad legacy formatter inventory reports 47 pre-existing whole-file
  candidates and was not mechanically rewritten. The full pytest suite was not
  run, per repository policy.

## Review fix round 1

The closeout review found that one mounted Library test had replaced the
decorated persistence worker with a synchronous lambda, and that the original
Server fake could satisfy its own assertions without traversing production
submission and reconciliation. Both gaps are now covered through the real
production seams.

- Library backend switching now tracks pending choice without impersonating a
  durable owner, serializes preference writes, and marshals completion back to
  Textual's UI loop. Only the current generation may clear pending state or
  repaint, so a failed latest save stays on the persisted owner while a stale
  completion cannot remain final or repaint a rapid Server→Local→Server choice.
- The mounted tests use the decorated thread worker with delayed success,
  delayed failure, and two gated rapid writes. Removing the completion repaint
  failed the success guard; removing the generation fence failed the rapid
  switch guard. Both mutations were restored before the green run.
- The Server round trip now enters through the app's production intake
  preparation and dispatch, uses the real registry poll/reconciliation and
  terminal listener, then reaches the association scheduler/coordinator. The
  fake Server owns the My Media catalog created by submission; injected Local
  Media and Local registry spies prove zero calls. Mismatched profile,
  principal, or returned media identity fails closed.
- Mutating terminal reconciliation to copy a Server ID into Local `media_id`
  failed the no-blend guard. Replacing the generated catalog with a constant
  fake failed the pre-dispatch catalog guard. Sources were restored before the
  final integration run.
- The two Task-6-owned `UP017` findings now use `datetime.UTC`. The 11 remaining
  `UP` findings in the six-file Task 6 inventory are on lines unchanged from
  the pre-Task-6 base; they are recorded as baseline rather than reported as a
  clean whole-file upgrade-style lint run. A full-file `UP017` probe of the
  newly touched production screen also reports four pre-existing findings, all
  outside the fix hunks; its default-selector and changed-line gates are clean.

Fresh fix-round evidence: Library ingest canvas **138 passed**; exact backend
worker tests **3 passed**; Research Sources region **12 passed**; Server/remote
Library runner slice **20 passed**; Research source integration **7 passed**;
installed-distribution packaging **43 passed**. Default Ruff and task-owned
`UP017` gates, changed-range formatting, changed-file compileall, privacy and
no-blend scans, Impeccable detector (`[]`), and whitespace checks passed. The
full Library screen file reports **31 passed, 1 pre-existing failure**: the
stale `test_ingest_button_present` expectation fails identically at exact base
`ebf80f954`. A final combined changed-area run finished **188 passed, 1
deselected**. The full pytest suite was not run.

## Live verification

The production app was launched with isolated `TLDW_CONFIG_PATH`, XDG config,
data, and cache directories plus an explicit temporary application data path.
F10 opened route `research_workspace` as `ResearchWorkspaceScreen`; the scan
found no configured personal path or server request. No test Server API was
available, so no live Server request was attempted or claimed.
