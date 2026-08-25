# Task 6 report — Research round trip and closeout

## Status

Complete. The final round-trip boundary proves that Research Sources preserve
canonical catalog ownership, captured workspace identity, staged recovery, and
authority isolation across Local and Server. The user guide and TASK-21508 now
describe the shipped behavior and its explicit unavailable operations.

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
- Ruff lint passed across all 105 changed Python files. Scoped formatter,
  changed-production `compileall`, migration artifact checks, no-blend/privacy
  scans, ASCII-handle scan, Impeccable detector (`[]`), and whitespace checks
  passed.
- The broad legacy formatter inventory reports 47 pre-existing whole-file
  candidates and was not mechanically rewritten. The full pytest suite was not
  run, per repository policy.

## Live verification

The production app was launched with isolated `TLDW_CONFIG_PATH`, XDG config,
data, and cache directories plus an explicit temporary application data path.
F10 opened route `research_workspace` as `ResearchWorkspaceScreen`; the scan
found no configured personal path or server request. No test Server API was
available, so no live Server request was attempted or claimed.
